/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	kubedrain "k8s.io/kubectl/pkg/drain"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureWorkerUpgradeName                  confv1beta1.BKEClusterPhase = "EnsureWorkerUpgrade"
	WorkerNodeHealthCheckPollIntervalSeconds                             = 2 // 工作节点健康检查轮询间隔（秒）
	WorkerNodeHealthCheckTimeoutMinutes                                  = 5 // 工作节点健康检查超时时间（分钟）
)

type EnsureWorkerUpgrade struct {
	phaseframe.BasePhase
}

// CreateUpgradeCommandParams 创建升级命令实例函数的参数
type CreateUpgradeCommandParams struct {
	Ctx         context.Context
	Namespace   string
	Client      client.Client
	Scheme      *runtime.Scheme
	OwnerObj    *bkev1beta1.BKECluster
	ClusterName string
	Node        *confv1beta1.Node
	BKEConfig   string
	Phase       confv1beta1.BKEClusterPhase
	// ControlPlaneConfigOldHash 旧的控制面 hash 结构体（来自 Status.ControlPlaneConfigHash）
	ControlPlaneConfigOldHash confv1beta1.ComponentConfigHash
	// ControlPlaneConfigNewHash 期望的控制面 hash 结构体（来自 Spec 计算）
	ControlPlaneConfigNewHash confv1beta1.ComponentConfigHash
}

// createUpgradeCommand 创建升级命令实例
func createUpgradeCommand(params CreateUpgradeCommandParams) command.Upgrade {
	return command.Upgrade{
		BaseCommand: command.BaseCommand{
			Ctx:         params.Ctx,
			NameSpace:   params.Namespace,
			Client:      params.Client,
			Scheme:      params.Scheme,
			OwnerObj:    params.OwnerObj,
			ClusterName: params.ClusterName,
			Unique:      true,
		},
		Node:                      params.Node,
		BKEConfig:                 params.BKEConfig,
		Phase:                     params.Phase,
		ControlPlaneConfigOldHash: params.ControlPlaneConfigOldHash,
		ControlPlaneConfigNewHash: params.ControlPlaneConfigNewHash,
	}
}

func NewEnsureWorkerUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureWorkerUpgradeName)
	return &EnsureWorkerUpgrade{BasePhase: base}
}

func (e *EnsureWorkerUpgrade) ExecutePreHook() error {
	return e.BasePhase.DefaultPreHook()
}

func (e *EnsureWorkerUpgrade) Execute() (ctrl.Result, error) {
	if v, ok := annotation.HasAnnotation(e.Ctx.BKECluster, annotation.DeployActionAnnotationKey); !ok || v != annotation.DeployActionK8sUpgrade {
		//添加boc所需的注解
		patchFunc := func(bkeCluster *bkev1beta1.BKECluster) {
			annotation.SetAnnotation(bkeCluster, annotation.DeployActionAnnotationKey, annotation.DeployActionK8sUpgrade)
		}
		if err := mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster, patchFunc); err != nil {
			return ctrl.Result{}, err
		}
	}

	return e.reconcileWorkerUpgrade()
}

// Version returns the current running Kubernetes version on the cluster.
func (e *EnsureWorkerUpgrade) Version() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	return e.Ctx.BKECluster.Status.KubernetesVersion
}

func (e *EnsureWorkerUpgrade) currentKubernetesVersion() string {
	vc := e.GetVersionContext()
	if vc != nil {
		if current := strings.TrimSpace(vc.GetCurrent(upgrade.ComponentKubernetesWorker)); current != "" {
			return current
		}
		if current := strings.TrimSpace(vc.GetCurrent(upgrade.ComponentKubernetesMaster)); current != "" {
			return current
		}
		if current := strings.TrimSpace(vc.GetCurrent("kubernetes")); current != "" {
			return current
		}
	}
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	return strings.TrimSpace(e.Ctx.BKECluster.Status.KubernetesVersion)
}

func (e *EnsureWorkerUpgrade) desiredKubernetesVersion() string {
	vc := e.GetVersionContext()
	if vc != nil {
		if target := strings.TrimSpace(vc.GetTarget(upgrade.ComponentKubernetesWorker)); target != "" {
			return target
		}
		if target := strings.TrimSpace(vc.GetTarget(upgrade.ComponentKubernetesMaster)); target != "" {
			return target
		}
		if target := strings.TrimSpace(vc.GetTarget("kubernetes")); target != "" {
			return target
		}
	}
	return e.deprecatedSpecKubernetesVersion()
}

// deprecatedSpecKubernetesVersion returns the legacy target source from BKECluster spec.
// Deprecated: declarative upgrade should prefer VersionContext targets derived from ReleaseImage.
func (e *EnsureWorkerUpgrade) deprecatedSpecKubernetesVersion() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil || e.Ctx.BKECluster.Spec.ClusterConfig == nil {
		return ""
	}
	return strings.TrimSpace(e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
}

func (e *EnsureWorkerUpgrade) syncLegacyTargetKubernetesVersion(target string) error {
	target = strings.TrimSpace(target)
	if target == "" || e.Ctx == nil || e.Ctx.BKECluster == nil {
		return nil
	}
	if e.Ctx.BKECluster.Spec.ClusterConfig == nil {
		return errors.New("cluster config is nil")
	}
	if strings.TrimSpace(e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion) == target {
		return nil
	}
	e.Ctx.Log.Warn(constant.WorkerUpgradingReason,
		"sync deprecated spec kubernetesVersion from %q to %q for legacy worker upgrade execution",
		e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion, target)
	if err := mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster, func(bc *bkev1beta1.BKECluster) {
		if bc.Spec.ClusterConfig != nil {
			bc.Spec.ClusterConfig.Cluster.KubernetesVersion = target
		}
	}); err != nil {
		return err
	}
	e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion = target
	return nil
}

// isKubernetesWorkerNeedUpgrade reports whether the cluster kubernetes version differs from the upgrade target.
func (e *EnsureWorkerUpgrade) isKubernetesWorkerNeedUpgrade(_ *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if new == nil || new.Spec.ClusterConfig == nil {
		return false
	}
	target := strings.TrimSpace(e.desiredKubernetesVersion())
	current := strings.TrimSpace(e.currentKubernetesVersion())
	return target != current &&
		len(target) != 0 &&
		len(current) != 0
}

// NeedExecute determines whether the worker kubernetes upgrade phase needs to be executed.
func (e *EnsureWorkerUpgrade) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	// cluster状态不正常，不需要执行
	if new.Status.ClusterStatus == bkev1beta1.ClusterUnhealthy || new.Status.ClusterStatus == bkev1beta1.ClusterUnknown {
		return false
	}
	if !e.NeedExecuteWithVersionContext(upgrade.ComponentKubernetesWorker, old, new, e.isKubernetesWorkerNeedUpgrade) {
		return false
	}

	bkeNodes, ok := fetchBKENodesIfCPInitialized(e.Ctx, new)
	if !ok {
		return false
	}
	nodes := phaseutil.GetNeedUpgradeWorkerNodesWithBKENodes(new, bkeNodes)
	if nodes.Length() == 0 {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureWorkerUpgrade) reconcileWorkerUpgrade() (ctrl.Result, error) {
	phaseContextFields231 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields231.Context,
		phaseContextFields231.Client,
		phaseContextFields231.BKECluster,
		phaseContextFields231.Scheme,
		phaseContextFields231.Log
	targetVersion := e.desiredKubernetesVersion()
	currentVersion := e.currentKubernetesVersion()
	if targetVersion != "" && targetVersion != currentVersion {
		if err := e.syncLegacyTargetKubernetesVersion(targetVersion); err != nil {
			return ctrl.Result{}, err
		}
		ret, err := e.rolloutUpgrade()
		if err != nil {
			return ret, err
		}
	}
	log.Info(constant.WorkerUpgradedReason, "k8s version same, not need to upgrade work node")
	return ctrl.Result{}, nil
}

// PrepareUpgradeNodesParams 包含 prepareUpgradeNodes 函数的参数
type PrepareUpgradeNodesParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Log        *bkev1beta1.BKELogger
}

// prepareUpgradeNodes 准备需要升级的节点
func (e *EnsureWorkerUpgrade) prepareUpgradeNodes(params PrepareUpgradeNodesParams) (bkenode.Nodes, error) {
	needUpgradeNodes := bkenode.Nodes{}

	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(params.Ctx, params.BKECluster)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get BKENodes")
	}
	nodes := phaseutil.GetNeedUpgradeWorkerNodesWithBKENodes(params.BKECluster, bkeNodes)
	for _, node := range nodes {
		stateFlag, _ := e.Ctx.NodeFetcher().GetNodeStateFlag(params.Ctx,
			params.BKECluster.Namespace, params.BKECluster.Name, node.IP, bkev1beta1.NodeAgentReadyFlag)
		if !stateFlag {
			params.Log.Info(constant.WorkerUpgradingReason, "agent is not ready at node %s, skip upgrade", phaseutil.NodeInfo(node))
			continue
		}
		needUpgradeNodes = append(needUpgradeNodes, node)
	}

	if len(needUpgradeNodes) == 0 {
		params.Log.Info(constant.WorkerUpgradeFailedReason, "all the master node BKEAgent is not ready")
		return nil, errors.New("all the master node BKEAgent is not ready")
	}

	return needUpgradeNodes, nil
}

// ProcessNodeUpgradeParams 包含 processNodeUpgrade 函数的参数
type ProcessNodeUpgradeParams struct {
	Ctx              context.Context
	Client           client.Client
	BKECluster       *bkev1beta1.BKECluster
	ClientSet        kubernetes.Interface
	NeedUpgradeNodes bkenode.Nodes
	Drainer          *kubedrain.Helper
	Log              *bkev1beta1.BKELogger
}

// processNodeUpgrade 处理节点升级
func (e *EnsureWorkerUpgrade) processNodeUpgrade(params ProcessNodeUpgradeParams) (ctrl.Result, []string, error) {
	var failedUpgradeNodes []string
	nodeFetcher := e.Ctx.NodeFetcher()

	clientSet, _, _ := kube.GetTargetClusterClient(params.Ctx, params.Client, params.BKECluster)
	for _, node := range params.NeedUpgradeNodes {
		remoteNode, err := phaseutil.GetRemoteNodeByBKENode(params.Ctx, clientSet, node)
		if err != nil {
			return ctrl.Result{}, nil, fmt.Errorf("get remote cluster Node resource failed: %w", err)
		}
		// 已经是期望版本的节点不需要升级
		targetVersion := e.desiredKubernetesVersion()
		if remoteNode.Status.NodeInfo.KubeletVersion == targetVersion {
			params.Log.Info(constant.MasterUpgradeSucceedReason, "node %q is already the expected version %q, skip upgrade", phaseutil.NodeInfo(node), targetVersion)
			continue
		}
		// mark node as upgrading
		nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster,
			node.IP, bkev1beta1.NodeUpgrading, "Upgrading")
		if err := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
			return ctrl.Result{}, nil, err
		}

		if err := e.upgradeNode(node, remoteNode, params.Drainer); err != nil {
			failedUpgradeNodes = append(failedUpgradeNodes, phaseutil.NodeInfo(node))
			params.Log.Warn(constant.WorkerUpgradeFailedReason, "upgrade node %q failed: %v", phaseutil.NodeInfo(node), err)
			nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster,
				node.IP, bkev1beta1.NodeUpgradeFailed, err.Error())
			if err = mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
				return ctrl.Result{}, nil, err
			}
			continue
		}
		// mark node as upgrading success
		nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster,
			node.IP, bkev1beta1.NodeNotReady, "Upgrading success")
		if err := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
			return ctrl.Result{}, nil, err
		}
	}
	return ctrl.Result{}, failedUpgradeNodes, nil
}

func (e *EnsureWorkerUpgrade) rolloutUpgrade() (ctrl.Result, error) {
	phaseContextFields339 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields339.Context,
		phaseContextFields339.Client,
		phaseContextFields339.BKECluster,
		phaseContextFields339.Scheme,
		phaseContextFields339.Log

	// 准备需要升级的节点
	prepareParams := PrepareUpgradeNodesParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Log:        log,
	}
	needUpgradeNodes, err := e.prepareUpgradeNodes(prepareParams)
	if err != nil {
		return ctrl.Result{Requeue: true}, err
	}

	// 对节点依次升级
	log.Info(constant.WorkerUpgradingReason, "Start upgrade worker nodes process, upgrade policy: rollingUpgrade")

	clientSet, _, _ := kube.GetTargetClusterClient(ctx, c, bkeCluster)
	drainer := phaseutil.NewDrainer(ctx, clientSet, nil, false, log)

	// 处理节点升级
	upgradeParams := ProcessNodeUpgradeParams{
		Ctx:              ctx,
		Client:           c,
		BKECluster:       bkeCluster,
		ClientSet:        clientSet,
		NeedUpgradeNodes: needUpgradeNodes,
		Drainer:          drainer,
		Log:              log,
	}
	_, failedUpgradeNodes, err := e.processNodeUpgrade(upgradeParams)
	if err != nil {
		return ctrl.Result{}, err
	}

	if len(failedUpgradeNodes) == 0 {
		log.Info(constant.WorkerUpgradeSucceedReason, "upgrade all worker success")
		return ctrl.Result{}, nil
	} else {
		// worker node没有升级成功，不允许进入下一阶段
		return ctrl.Result{}, errors.Errorf("upgrade worker process finished, but some nodes upgrade failed, will retry later nodes: %v", failedUpgradeNodes)
	}
}

// WaitForWorkerNodeHealthCheckParams 等待节点健康检查通过函数的参数
type WaitForWorkerNodeHealthCheckParams struct {
	Ctx          context.Context
	ClientSet    kubernetes.Interface
	RemoteClient kube.RemoteKubeClient
	Node         confv1beta1.Node
	K8sVersion   string
	Logger       *bkev1beta1.BKELogger
}

// waitForWorkerNodeHealthCheck 等待节点健康检查通过
func waitForWorkerNodeHealthCheck(params WaitForWorkerNodeHealthCheckParams) error {
	return wait.PollWithContext(params.Ctx, time.Duration(WorkerNodeHealthCheckPollIntervalSeconds)*time.Second, time.Duration(WorkerNodeHealthCheckTimeoutMinutes)*time.Minute, func(ctx context.Context) (bool, error) {
		if utils.CtxDone(ctx) {
			return false, nil
		}
		remoteNode, err := params.ClientSet.CoreV1().Nodes().Get(ctx, params.Node.Hostname, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if err := params.RemoteClient.NodeHealthCheck(remoteNode, params.K8sVersion, params.Logger.NormalLogger); err != nil {
			return false, nil
		}
		return true, nil
	})
}

// executeNodeUpgrade 执行节点升级
func (e *EnsureWorkerUpgrade) executeNodeUpgrade(params ExecuteNodeUpgradeParams) error {
	createParams := CreateUpgradeCommandParams{
		Ctx:         params.Ctx,
		Namespace:   params.BKECluster.Namespace,
		Client:      params.Client,
		Scheme:      params.Scheme,
		OwnerObj:    params.BKECluster,
		ClusterName: params.BKECluster.Name,
		Node:        &params.Node,
		BKEConfig:   params.BKECluster.Name,
		Phase:       bkev1beta1.UpgradeWorker,
	}
	upgrade := createUpgradeCommand(createParams)
	upgrade.BackUpEtcd = false

	params.Log.Info(constant.WorkerUpgradingReason, "start upgrade node %s", phaseutil.NodeInfo(params.Node))
	if err := upgrade.New(); err != nil {
		return fmt.Errorf("create upgrade command，node: %q failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	params.Log.Info(constant.WorkerUpgradingReason, "wait upgrade node %s finish", phaseutil.NodeInfo(params.Node))
	err, _, failedNodes := upgrade.Wait()
	if err != nil {
		return fmt.Errorf("wait upgrade command complete failed，node: %q: %w", phaseutil.NodeInfo(params.Node), err)
	}
	if len(failedNodes) != 0 {
		commandErrs, err := phaseutil.LogCommandFailed(*upgrade.Command, failedNodes, params.Log, constant.WorkerUpgradeFailedReason)
		phaseutil.MarkNodeStatusByCommandErrs(params.Ctx, params.Client, params.BKECluster, commandErrs)
		return fmt.Errorf("upgrade node %q failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	params.Log.Info(constant.WorkerUpgradeSucceedReason, "upgrade node %q operation succeed", phaseutil.NodeInfo(params.Node))
	return nil
}

// WaitForNodeHealthParams 包含 waitForNodeHealth 函数的参数
type WaitForNodeHealthParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Node       confv1beta1.Node
	Log        *bkev1beta1.BKELogger
}

// waitForNodeHealth 等待节点健康检查
func (e *EnsureWorkerUpgrade) waitForNodeHealth(params WaitForNodeHealthParams) error {
	remoteClient, err := kube.NewRemoteClientByBKECluster(params.Ctx, params.Client, params.BKECluster)
	if err != nil {
		return fmt.Errorf("get remote client for BKECluster %q failed: %w", utils.ClientObjNS(params.BKECluster), err)
	}
	clientSet, _ := remoteClient.KubeClient()
	// wait for node pass healthy check
	params.Log.Info(constant.WorkerUpgradingReason, "wait for node %q pass healthy check", phaseutil.NodeInfo(params.Node))
	healthParams := WaitForWorkerNodeHealthCheckParams{
		Ctx:          params.Ctx,
		ClientSet:    clientSet,
		RemoteClient: remoteClient,
		Node:         params.Node,
		K8sVersion:   e.desiredKubernetesVersion(),
		Logger:       params.Log,
	}
	err = waitForWorkerNodeHealthCheck(healthParams)

	if err != nil {
		return fmt.Errorf("wait for node %q pass healthy check failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	params.Log.Info(constant.WorkerUpgradingReason, "upgrade worker node %q success", phaseutil.NodeInfo(params.Node))
	return nil
}

func (e *EnsureWorkerUpgrade) upgradeNode(node confv1beta1.Node, remoteNode *corev1.Node, drainer *kubedrain.Helper) error {
	phaseContextFields480 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields480.Context,
		phaseContextFields480.Client,
		phaseContextFields480.BKECluster,
		phaseContextFields480.Scheme,
		phaseContextFields480.Log

	// 执行节点升级
	upgradeParams := ExecuteNodeUpgradeParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Node:       node,
		Log:        log,
	}
	if err := e.executeNodeUpgrade(upgradeParams); err != nil {
		return err
	}

	// 等待节点健康检查
	healthParams := WaitForNodeHealthParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Node:       node,
		Log:        log,
	}
	return e.waitForNodeHealth(healthParams)
}
