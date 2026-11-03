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
	"net"
	"strings"
	"time"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/imagehelper"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidate "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	backupPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/backup"
	certPlugin "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	// EnsureClusterManageName is the name of the EnsureClusterManage phase
	EnsureClusterManageName confv1beta1.BKEClusterPhase = "EnsureClusterManage"
)

const (
	// indexLowLevelRuntime represents the index for low level runtime in the collected output
	indexLowLevelRuntime = iota // 0
	// indexCgroupDriver represents the index for cgroup driver in the collected output
	indexCgroupDriver // 1
	// indexDataRoot represents the index for data root in the collected output
	indexDataRoot // 2
	// indexGuessClusterType represents the index for guess cluster type in the collected output
	indexGuessClusterType // 3
	// indexKubeletRootDir represents the index for kubelet root directory in the collected output
	indexKubeletRootDir // 4
)

const (
	// masterInitTimeout represents the timeout duration for master nodes fake bootstrap operation
	masterInitTimeout = 4 * time.Minute
	// workerInitTimeout represents the timeout duration for worker nodes fake bootstrap operation
	workerInitTimeout = 4 * time.Minute
	// pollInterval represents the polling interval duration for waiting operations
	pollInterval = 2 * time.Second
	// pingAgentTimeoutMinutes represents the timeout minutes for ping agent operation
	pingAgentTimeoutMinutes = 1
	// pingAgentPollIntervalSeconds represents the polling interval seconds for ping agent operation
	pingAgentPollIntervalSeconds = 1
	// commandBackoffDelaySeconds represents the backoff delay seconds for commands
	commandBackoffDelaySeconds = 3
	// launcherDaemonSetName represents daemonset launcher pod name
	launcherDaemonSetName = "bkeagent-launcher"
	// launcherNamespace represents daemonset launcher namespace
	launcherNamespace = "kube-system"
)

type EnsureClusterManage struct {
	phaseframe.BasePhase
	remoteClient kube.RemoteKubeClient
	mockClient   kubernetes.Interface
}

// CreateBaseCommandParams 包含 createBaseCommand 函数的参数
type CreateBaseCommandParams struct {
	Ctx             context.Context
	NameSpace       string
	Client          client.Client
	Scheme          *runtime.Scheme
	OwnerObj        *bkev1beta1.BKECluster
	ClusterName     string
	Unique          bool
	RemoveAfterWait bool
}

func NewEnsureClusterManage(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureClusterManageName)
	return &EnsureClusterManage{BasePhase: base}
}

// kubeClient returns the kubernetes clientset used to query the target cluster.
// In production this is obtained from the remote client; tests may inject a
// fake clientset via mockClient. Behavior is identical when mockClient is nil.
func (e *EnsureClusterManage) kubeClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	cs, _ := e.remoteClient.KubeClient()
	return cs
}

func (e *EnsureClusterManage) Execute() (ctrl.Result, error) {
	e.Ctx.Log.Info(constant.ClusterManagingReason, "start to manage cluster %s", utils.ClientObjNS(e.Ctx.BKECluster))

	if err := e.getRemoteClient(); err != nil {
		return ctrl.Result{}, err
	}
	// 集群基础信息收集
	// include: k8s version, node info, cluster network, etc.
	if err := e.collectBaseInfo(); err != nil {
		return ctrl.Result{}, err
	}
	// 使用ds推送agent
	if err := e.pushAgent(); err != nil {
		return ctrl.Result{}, err
	}

	// 使用agent收集更多集群信息
	if err := e.collectAgentInfo(); err != nil {
		return ctrl.Result{}, err
	}

	// 其他类型的集群不需要执行后续的操作
	if !clusterutil.IsBocloudCluster(e.Ctx.BKECluster) {
		return ctrl.Result{}, nil
	}
	// cluste-api资源没有创建，不需要执行后续的操作
	if e.Ctx.BKECluster.OwnerReferences == nil {
		return ctrl.Result{Requeue: true}, nil
	}
	if err := e.Ctx.RefreshCtxCluster(); err != nil {
		return ctrl.Result{}, err
	}
	if e.Ctx.Cluster == nil {
		return ctrl.Result{}, nil
	}

	// 为后续的集群管理做准备，必须通过
	if err := e.bocloudClusterManagePrepare(); err != nil {
		return ctrl.Result{}, err
	}
	// 伪引导，完全转为bke管理，为后续的集群管理做准备
	if err := e.reconcileFakeBootstrap(); err != nil {
		return ctrl.Result{}, err
	}
	// ansible -> bke 的兼容性修改，为后续的集群管理做准备
	if err := e.compatibilityPatch(); err != nil {
		return ctrl.Result{}, err
	}
	// 纳管场景跳过节点后处理：fake bootstrap 已设置 NodePostProcessFlag，
	// 同时预设 NodesPostProcessCondition=True，使 EnsureNodesPostProcess.NeedExecute 返回 false。
	condition.ConditionMark(e.Ctx.BKECluster, bkev1beta1.NodesPostProcessCondition,
		confv1beta1.ConditionTrue, constant.NodesPostProcessReadyReason,
		"Skip postprocess for bocloud adopted cluster")
	// 纳管结束，标记full controlled,使得其他阶段能够被正常执行
	clusterutil.MarkClusterFullyControlled(e.Ctx.BKECluster)
	// requeue
	return ctrl.Result{Requeue: true}, nil
}

func (e *EnsureClusterManage) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.NormalNeedExecute(old, new) {
		return false
	}

	if clusterutil.IsBKECluster(new) {
		return false
	}
	if clusterutil.FullyControlled(new) {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureClusterManage) collectBaseInfo() error {
	phaseContextFields207 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields207.Context,
		phaseContextFields207.Client,
		phaseContextFields207.BKECluster,
		phaseContextFields207.Scheme,
		phaseContextFields207.Log
	if clusterutil.ClusterBaseInfoHasCollected(bkeCluster) {
		return nil
	}
	log.Debug("collect cluster base info")
	collectRes, warns, errs := e.remoteClient.Collect()
	if len(errs) > 0 {
		err := kerrors.NewAggregate(errs)
		return fmt.Errorf("failed to collect cluster info for BKECluster %s: %w", utils.ClientObjNS(bkeCluster), err)
	}
	if len(warns) > 0 {
		for _, warn := range warns {
			log.Warn("collect cluster info for BKECluster %s: %v", utils.ClientObjNS(bkeCluster), warn)
		}
	}

	patchFunc := func(bkeCluster *bkev1beta1.BKECluster) {
		clusterutil.MarkClusterBaseInfoCollected(bkeCluster)
		annotation.SetAnnotation(bkeCluster, annotation.ManageClusterEtcdCertDirAnnotationKey, collectRes.EtcdCertificatesDir)
		bkeCluster.Spec.ClusterConfig.Cluster.Networking = collectRes.Networking
		bkeCluster.Spec.ControlPlaneEndpoint = collectRes.ControlPlaneEndpoint
		bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion = collectRes.KubernetesVersion
		bkeCluster.Status.KubernetesVersion = collectRes.KubernetesVersion
		bkeCluster.Spec.ClusterConfig.Cluster.ContainerRuntime = collectRes.ContainerRuntime
		bkeCluster.Status.AgentStatus.Reset()
		if bkevalidate.ValidateRepo(bkeCluster.Spec.ClusterConfig.Cluster.ImageRepo) != nil {
			bkeinit.SetDefaultImageRepo(&bkeCluster.Spec.ClusterConfig.Cluster)
		}
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesEnvCondition, confv1beta1.ConditionTrue, constant.NodesEnvReadyReason, "it's necessary set true used to fake boot strap cluster for bocloud cluster")
	}

	if collectRes.Nodes != nil || len(collectRes.Nodes) > 0 {
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesInfoCondition, confv1beta1.ConditionTrue, constant.NodesInfoReadyReason, "")
	}

	return mergecluster.SyncStatusUntilComplete(c, bkeCluster, patchFunc)
}

func (e *EnsureClusterManage) pushAgent() error {
	phaseContextFields246 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields246.Context,
		phaseContextFields246.Client,
		phaseContextFields246.BKECluster,
		phaseContextFields246.Scheme,
		phaseContextFields246.Log

	nodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}

	if !e.checkAgentNeedPush(nodes) {
		return nil
	}

	log.Debug("step 2 push agent to nodes")
	// get localkubeconfig from secret
	localKubeConfig, err := phaseutil.GetLocalKubeConfig(ctx, c)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("local kubeconfig secret not found")
		}
		return fmt.Errorf("failed to get local kubeconfig secret: %w", err)
	}

	launcherAddonT, err := e.createLauncherAddon(bkeCluster, localKubeConfig, log)
	if err != nil {
		return err
	}

	if err = e.remoteClient.InstallAddon(bkeCluster, launcherAddonT, nil, nil, nodes); err != nil {
		condition.ConditionMark(bkeCluster, bkev1beta1.BKEAgentCondition, confv1beta1.ConditionFalse, constant.BKEAgentNotReadyReason, "failed create BKEAgent launcher ds")
		log.Finish(constant.BKEAgentNotReadyReason, "Failed to push bke agent to cluster, err: %v", err)
		return err
	}

	// Wait for all launcher pods Ready=True (readinessProbe /readyz => bkeagent active) before deleting DaemonSet
	log.Info(constant.ClusterManagingReason, "waiting for launcher pods to be Ready before deleting DaemonSet")
	readyNodeNames, waitErr := e.waitForLauncherPodsComplete(ctx, bkeCluster)
	if waitErr != nil {
		log.Warn(constant.ReconcileErrorReason, "wait for launcher pods Ready failed (continue anyway): %v", waitErr)
		// Continue even if wait fails, as launcher may have already completed
	}
	readyNodeSet := make(map[string]bool)
	for _, name := range readyNodeNames {
		readyNodeSet[name] = true
	}

	// delete launcher daemonset
	launcherAddonT.Operate = bkeaddon.RemoveAddon
	if err := e.remoteClient.InstallAddon(bkeCluster, launcherAddonT, nil, nil, nodes); err != nil {
		log.Warn(constant.ReconcileErrorReason, "(Ignore)Failed to delete bke agent launcher daemonset, err: %v", err)
	}

	// mark node agent pushed flag — only for nodes where launcher pod was actually Ready and not NeedSkip
	nf := e.Ctx.NodeFetcher()
	activeNodes := e.filterNeedSkipNodes(nodes)
	for _, node := range activeNodes {
		if !readyNodeSet[node.Hostname] {
			log.Warn(constant.InternalErrorReason, "Skip PushedFlag for node %s (%s): launcher pod not Ready", node.IP, node.Hostname)
			continue
		}
		if err := nf.MarkNodeStateFlagForCluster(ctx, bkeCluster, node.IP, bkev1beta1.NodeAgentPushedFlag); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to mark node state flag for %s: %v", node.IP, err)
		}
	}

	// wait concurrently for pushed flag to be visible for each marked node before pinging
	if err := e.waitAgentPushedFlagVisible(ctx, nf, bkeCluster, nodes, log); err != nil {
		return fmt.Errorf("%w", err)
	}

	// step 5 ping agent
	err, successNodes, failedNodes := phaseutil.PingBKEAgent(ctx, c, scheme, bkeCluster)
	if err != nil {
		log.Warn(constant.BKEAgentNotReadyReason,
			"(Ignore)Failed to ping bke agent in the flow nodes: %v, err: %v",
			failedNodes, err)
	}

	e.processAgentPingResults(ctx, bkeCluster, successNodes, failedNodes, log)

	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
		return err
	}

	condition.ConditionMark(bkeCluster, bkev1beta1.BKEAgentCondition, confv1beta1.ConditionTrue, constant.BKEAgentReadyReason, "")

	return nil
}

// createLauncherAddon builds the launcher addon transfer for bkeagent deployment.
func (e *EnsureClusterManage) createLauncherAddon(bkeCluster *bkev1beta1.BKECluster, localKubeConfig []byte, log *bkev1beta1.BKELogger) (*bkeaddon.AddonTransfer, error) {
	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	lancherImage := imagehelper.GetFullImageName(cfg.ImageFuyaoRepo(), "bkeagent-launcher", "1.0.1")
	if v, ok := cfg.CustomExtra["bkeagent-launcher-image"]; ok && v != "" {
		lancherImage = v
	}
	agentLogLevel := ""
	if cfg.CustomExtra != nil {
		if canonical, ok := phaseutil.NormalizeAgentLogLevel(cfg.CustomExtra[phaseutil.AgentLogLevelCustomExtraKey]); ok {
			agentLogLevel = canonical
		}
	}
	log.Info(constant.ClusterManagingReason, "BKEAgent launcher image: %s, ntpserver: %s, agentHealthPort: %s, agentLogLevel: %s",
		lancherImage, bkeCluster.Spec.ClusterConfig.Cluster.NTPServer, bkeCluster.Spec.ClusterConfig.Cluster.AgentHealthPort, agentLogLevel)

	return &bkeaddon.AddonTransfer{
		Addon: &v1beta1.Product{
			Name:    "bkeagent",
			Version: "latest",
			Param: map[string]string{
				"clusterName":     bkeCluster.Name,
				"ntpServer":       bkeCluster.Spec.ClusterConfig.Cluster.NTPServer,
				"agentHealthPort": bkeCluster.Spec.ClusterConfig.Cluster.AgentHealthPort,
				"agentLogLevel":   agentLogLevel,
				"debug":           "true",
				"kubeconfig":      string(localKubeConfig),
				"launcherImage":   lancherImage,
			},
			Block: true,
		},
		Operate: bkeaddon.CreateAddon,
	}, nil
}

func (e *EnsureClusterManage) waitAgentPushedFlagVisible(
	ctx context.Context,
	nf phaseutil.NodeStateFlagReader,
	bkeCluster *bkev1beta1.BKECluster,
	nodes bkenode.Nodes,
	log *bkev1beta1.BKELogger,
) error {
	timeOut, err := phaseutil.GetBootTimeOut(bkeCluster)
	if err != nil {
		log.Warn(constant.InternalErrorReason, "Get boot timeout failed. err: %v", err)
	}
	return phaseutil.WaitNodesStateFlagVisible(phaseutil.WaitNodesStateFlagVisibleInput{
		Context: ctx, Reader: nf, BKECluster: bkeCluster,
		Nodes: nodes, Flag: bkev1beta1.NodeAgentPushedFlag,
		Options: phaseutil.WaitNodesStateFlagVisibleOptions{
			Timeout:  timeOut,
			Interval: 1 * time.Second,
		},
	})
}

func (e *EnsureClusterManage) collectAgentInfo() error {
	phaseContextFields390 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields390.Context,
		phaseContextFields390.Client,
		phaseContextFields390.BKECluster,
		phaseContextFields390.Scheme,
		phaseContextFields390.Log

	if clusterutil.ClusterAgentInfoHasCollected(bkeCluster) || !clusterutil.ClusterBaseInfoHasCollected(bkeCluster) {
		return nil
	}
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)
	if len(bkeNodes) == 0 {
		return fmt.Errorf("no BKENode resources found for cluster %s/%s, please ensure BKENode CRDs are created", bkeCluster.Namespace, bkeCluster.Name)
	}
	if len(bkeNodes.Master()) == 0 {
		return fmt.Errorf("no master nodes found for cluster %s/%s", bkeCluster.Namespace, bkeCluster.Name)
	}

	k8sCertDir := bkeCluster.Spec.ClusterConfig.Cluster.CertificatesDir
	etcdCertDir, ok := annotation.HasAnnotation(bkeCluster, annotation.ManageClusterEtcdCertDirAnnotationKey)
	if !ok {
		etcdCertDir = k8sCertDir
	}

	// collect target cluster cert
	baseCommandParams := newBaseCommandParams(ctx, c, bkeCluster, scheme)
	collectCommand := command.Collect{
		BaseCommand:         createBaseCommand(baseCommandParams),
		Node:                &bkeNodes.Master()[0],
		EtcdCertificatesDir: etcdCertDir,
		K8sCertificatesDir:  k8sCertDir,
	}
	if err := collectCommand.New(); err != nil {
		return fmt.Errorf("failed to collect cert for cluster %s: %w", bkeCluster.Name, err)
	}
	err, _, failedNode := collectCommand.Wait()
	if err != nil {
		return fmt.Errorf("failed to wait for collect cert for cluster %s: %w", bkeCluster.Name, err)
	}
	if len(failedNode) > 0 {
		commandErrs, err := phaseutil.LogCommandFailed(*collectCommand.Command, failedNode, log, constant.CommandExecFailedReason)
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		return fmt.Errorf("failed to collect cert for cluster %s: %v, err: %w", bkeCluster.Name, failedNode, err)
	}

	if err = e.getContainerRuntimeConfigFromCollectCommand(collectCommand.Command); err != nil {
		return fmt.Errorf("failed to get container runtime config from collect command: %w", err)
	}

	log.Info(constant.CollectClusterInfoSucceedReason, "finish collect cluster info")
	return nil
}

func (e *EnsureClusterManage) reconcileFakeBootstrap() error {
	bkeCluster := e.Ctx.BKECluster
	if !clusterutil.IsBocloudCluster(bkeCluster) || clusterutil.FullyControlled(bkeCluster) {
		return nil
	}
	if bkeCluster.OwnerReferences == nil {
		return nil
	}
	err := e.Ctx.RefreshCtxCluster()
	if err != nil {
		return err
	}
	if e.Ctx.Cluster == nil {
		return nil
	}
	// 到此为之 开始伪引导了，所谓伪引导指的是将纳管的集群和相关的集群机器在当前集群中创建相关CRD进行映射
	// 使得结束伪引导后该集群的机器可以被当前集群管理

	if err := e.fakeBootstrapMaster(); err != nil {
		return fmt.Errorf("failed to fake bootstrap master: %w", err)
	}

	if err := e.fakeBootstrapWorker(); err != nil {
		return fmt.Errorf("failed to fake bootstrap worker: %w", err)
	}

	return nil
}

func (e *EnsureClusterManage) fakeBootstrapMaster() error {
	phaseContextFields472 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields472.Context,
		phaseContextFields472.Client,
		phaseContextFields472.BKECluster,
		phaseContextFields472.Scheme,
		phaseContextFields472.Log
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)

	// 第一步修改 kubeadmControlPlane.replicas = 当前集群的master节点数量
	expectKcpReplicas := int32(bkeNodes.Master().Length())
	if err := e.updateKubeadmControlPlaneReplicas(ctx, c, expectKcpReplicas); err != nil {
		return err
	}

	// 第二步 启动cluster-api,设置 当前bkecluster.status.ready = true
	if err := e.waitForClusterInfrastructureReady(ctx, c, bkeCluster); err != nil {
		return err
	}

	// 第三步等待所有master节点都结束伪引导
	successJoinMasterNodes, err := e.waitForMasterNodesBootstrap(nodesBootstrapInput{
		ctx: ctx, client: c, cluster: bkeCluster,
		nodes: bkeNodes.Master(), expectedCount: expectKcpReplicas, log: log,
	})
	if err != nil {
		return err
	}

	if err := e.Ctx.RefreshCtxBKECluster(); err != nil {
		return err
	}

	e.markNodesBootstrapSuccess(ctx, successJoinMasterNodes, log)

	return mergecluster.SyncStatusUntilComplete(c, e.Ctx.BKECluster)
}

// updateKubeadmControlPlaneReplicas 更新 KubeadmControlPlane 副本数
func (e *EnsureClusterManage) updateKubeadmControlPlaneReplicas(ctx context.Context, c client.Client, replicas int32) error {
	kcp, err := phaseutil.GetClusterAPIKubeadmControlPlane(ctx, c, e.Ctx.Cluster)
	if err != nil {
		return err
	}
	return phaseutil.ResumeAndUpdateKubeadmControlPlaneReplicas(ctx, c, kcp, replicas)
}

// waitForClusterInfrastructureReady 等待集群基础设施就绪
func (e *EnsureClusterManage) waitForClusterInfrastructureReady(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) error {
	bkeCluster.Status.Ready = true
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
		return err
	}

	ctxWithTimeout, cancel := context.WithTimeout(ctx, time.Duration(pingAgentTimeoutMinutes)*time.Minute)
	defer cancel()

	return wait.PollImmediateUntil(time.Duration(pingAgentPollIntervalSeconds)*time.Second, func() (bool, error) {
		if err := e.Ctx.RefreshCtxCluster(); err != nil {
			return false, err
		}
		return e.Ctx.Cluster.Status.InfrastructureReady, nil
	}, ctxWithTimeout.Done())
}

// waitForMasterNodesBootstrap 等待所有 master 节点伪引导完成
type nodesBootstrapInput struct {
	ctx           context.Context
	client        client.Client
	cluster       *bkev1beta1.BKECluster
	nodes         bkenode.Nodes
	successNodes  map[int]confv1beta1.Node
	expectedCount int32
	log           *bkev1beta1.BKELogger
}

func (e *EnsureClusterManage) waitForMasterNodesBootstrap(input nodesBootstrapInput) (map[int]confv1beta1.Node, error) {
	ctx := input.ctx

	ctxWithTimeout, cancel := context.WithTimeout(ctx, masterInitTimeout)
	defer cancel()

	masterInitFlag := false
	successJoinMasterNodes := map[int]confv1beta1.Node{}

	err := wait.PollImmediateUntil(pollInterval, func() (bool, error) {
		// 检查控制平面初始化条件
		if !masterInitFlag {
			if err := e.Ctx.RefreshCtxCluster(); err == nil {
				if conditions.IsTrue(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
					masterInitFlag = true
				} else {
					return false, nil
				}
			}
		}
		// 检查各个节点的引导状态
		input.successNodes = successJoinMasterNodes
		if err := waitForNodesBootstrap(input); err != nil {
			return false, nil
		}
		return true, nil
	}, ctxWithTimeout.Done())

	if err != nil {
		return nil, errors.Errorf(
			"failed to wait for cluster %s master nodes fake bootstrap ready: %v",
			utils.ClientObjNS(e.Ctx.BKECluster), err,
		)
	}
	return successJoinMasterNodes, nil
}

func (e *EnsureClusterManage) fakeBootstrapWorker() error {
	phaseContextFields578 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields578.Context,
		phaseContextFields578.Client,
		phaseContextFields578.BKECluster,
		phaseContextFields578.Scheme,
		phaseContextFields578.Log
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)

	workerNum := bkeNodes.Worker().Length()
	if workerNum == 0 {
		return nil
	}

	return e.doFakeBootstrapWorker(nodesBootstrapInput{
		ctx: ctx, client: c, cluster: bkeCluster,
		nodes: bkeNodes.Worker(), expectedCount: int32(workerNum), log: log,
	})
}

// doFakeBootstrapWorker 执行 worker 节点伪引导的核心逻辑
func (e *EnsureClusterManage) doFakeBootstrapWorker(input nodesBootstrapInput) error {
	ctx, c := input.ctx, input.client
	expectMDReplicas, log := input.expectedCount, input.log
	// 第四步修改machineDeployment.replicas = 当前集群的worker节点数量
	md, err := phaseutil.GetClusterAPIMachineDeployment(ctx, c, e.Ctx.Cluster)
	if err != nil {
		return err
	}
	if err = phaseutil.ResumeAndUpdateMachineDeploymentReplicas(ctx, c, md, expectMDReplicas); err != nil {
		return err
	}

	// 第五步等待所有worker节点都结束伪引导
	successJoinWorkerNodes := map[int]confv1beta1.Node{}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, workerInitTimeout)
	defer cancel()

	err = wait.PollImmediateUntil(pollInterval, func() (bool, error) {
		input.successNodes = successJoinWorkerNodes
		return waitForNodesBootstrap(input) == nil, nil
	}, ctxWithTimeout.Done())
	if err != nil {
		return fmt.Errorf(
			"failed to wait for cluster %s worker nodes fake bootstrap ready: %w",
			utils.ClientObjNS(e.Ctx.BKECluster), err,
		)
	}

	if err := e.Ctx.RefreshCtxBKECluster(); err != nil {
		return err
	}

	e.markNodesBootstrapSuccess(ctx, successJoinWorkerNodes, log)

	return mergecluster.SyncStatusUntilComplete(c, e.Ctx.BKECluster)
}

func (e *EnsureClusterManage) compatibilityPatch() error {
	// 设置etcd pod的注释以兼容bke的集群升级
	phaseContextFields630 := e.Ctx.Untie()
	ctx, _, _, _, log := phaseContextFields630.Context,
		phaseContextFields630.Client,
		phaseContextFields630.BKECluster,
		phaseContextFields630.Scheme,
		phaseContextFields630.Log
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	bkeNodes := bkenode.Nodes(allNodes)
	clientSet := e.kubeClient()
	etcdPods, err := clientSet.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{
		LabelSelector: "component=etcd",
	})
	if err != nil {
		return err
	}

	var failedUpdateEtcdNode []string
	for _, pod := range etcdPods.Items {
		nodeName := pod.Spec.NodeName
		nodes := bkeNodes.Filter(bkenode.FilterOptions{"Hostname": nodeName})
		if nodes.Length() == 0 {
			log.Warn(constant.ClusterManageWarningReason, "etcd pod node %s not found in BKECluster nodes fields, skip", nodeName)
			failedUpdateEtcdNode = append(failedUpdateEtcdNode, nodeName)
		}
		if _, ok := annotation.HasAnnotation(&pod, annotation.EtcdAdvertiseClientUrlsAnnotationKey); ok {
			continue
		}
		annotateValue := phaseutil.GetClientURLByIP(nodes[0].IP)
		err = phaseutil.RetryOnConflict(func() error {
			latestPod, err := clientSet.CoreV1().Pods(metav1.NamespaceSystem).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			annotation.SetAnnotation(latestPod, annotation.EtcdAdvertiseClientUrlsAnnotationKey, annotateValue)
			_, err = clientSet.CoreV1().Pods(metav1.NamespaceSystem).Update(ctx, latestPod, metav1.UpdateOptions{})
			return err
		})
		if err != nil {
			log.Warn(constant.ClusterManageWarningReason, "failed to update etcd pod %s: %v", utils.ClientObjNS(&pod), err)
			failedUpdateEtcdNode = append(failedUpdateEtcdNode, nodeName)
		}
	}
	if len(failedUpdateEtcdNode) > 0 {
		log.Warn(constant.ClusterManageWarningReason, "following etcd node failed to set annotation: %v,this will cause subsequent cluster upgrades to fail", failedUpdateEtcdNode)
		log.Warn(constant.ClusterManageWarningReason, "please manually update the etcd pod and set the annotation %q, eg: bkeagent.bocloud.com/etcd.advertise-client-urls: https://<node ip>:2379", annotation.EtcdAdvertiseClientUrlsAnnotationKey)
		log.Warn(constant.ClusterManageWarningReason, "before upgrading the cluster, please make sure that the etcd pod has been set annotation %q", annotation.EtcdAdvertiseClientUrlsAnnotationKey)
	}
	return nil
}

// waitForLauncherPodsComplete waits until all launcher DaemonSet pods report Ready=True.
// Ready depends on the pod readinessProbe (typically /readyz), which verifies host bkeagent is active.
// This is stricter than Phase=Running alone, which can become true before systemctl start finishes.
// Returns the node names (pod.Spec.NodeName) of pods that are Ready, so callers can selectively
// mark PushedFlag only for nodes where the launcher actually succeeded.
func (e *EnsureClusterManage) waitForLauncherPodsComplete(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) ([]string, error) {
	phaseContextFields684 := e.Ctx.Untie()
	log := phaseContextFields684.Log
	clientSet := e.kubeClient()

	waitTimeout := phaseutil.GetLauncherWaitTimeout(bkeCluster)
	log.Info(constant.ClusterManagingReason, "waiting for launcher pods to be Ready (timeout: %v)", waitTimeout)

	var readyNodeNames []string

	err := wait.PollUntilContextTimeout(ctx, pollInterval, waitTimeout, true, func(ctx context.Context) (bool, error) {
		status, err := launcherPodsReady(ctx, clientSet)
		if status.observed {
			readyNodeNames = status.nodeNames
		}
		if status.ready {
			log.Info(
				constant.ClusterManagingReason,
				"all launcher pods are Ready (%d/%d)", len(status.nodeNames), status.desired,
			)
		}
		return status.ready, err
	})

	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, wait.ErrWaitTimeout) {
			log.Warn(constant.ClusterManagingReason, "timeout waiting for launcher pods to be Ready, but continuing anyway")
			return readyNodeNames, nil
		}
		return nil, err
	}

	log.Info(constant.ClusterManagingReason, "launcher pods have completed their work")
	return readyNodeNames, nil
}

type launcherPodsStatus struct {
	ready     bool
	observed  bool
	nodeNames []string
	desired   int32
}

func launcherPodsReady(ctx context.Context, clientSet kubernetes.Interface) (launcherPodsStatus, error) {
	ds, err := clientSet.AppsV1().DaemonSets(launcherNamespace).Get(
		ctx, launcherDaemonSetName, metav1.GetOptions{},
	)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return launcherPodsStatus{}, nil
		}
		return launcherPodsStatus{}, err
	}

	desired := ds.Status.DesiredNumberScheduled
	if desired == 0 || ds.Status.NumberReady < desired {
		return launcherPodsStatus{desired: desired}, nil
	}

	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return launcherPodsStatus{desired: desired}, err
	}
	pods, err := clientSet.CoreV1().Pods(launcherNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return launcherPodsStatus{desired: desired}, err
	}

	status := launcherPodsStatus{observed: true, desired: desired}
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		if !phaseutil.PodIsReady(pod) {
			return status, nil
		}
		status.nodeNames = append(status.nodeNames, pod.Spec.NodeName)
	}
	status.ready = int32(len(status.nodeNames)) >= desired
	return status, nil
}

func (e *EnsureClusterManage) getRemoteClient() error {
	phaseContextFields755 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields755.Context,
		phaseContextFields755.Client,
		phaseContextFields755.BKECluster,
		phaseContextFields755.Scheme,
		phaseContextFields755.Log
	remoteClient, err := kube.NewRemoteClientByBKECluster(ctx, c, bkeCluster)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster client: %w", utils.ClientObjNS(bkeCluster), err)
	}
	e.remoteClient = remoteClient
	e.remoteClient.SetBKELogger(log)
	e.remoteClient.SetLogger(log.NormalLogger)
	return nil
}

// bocloudClusterManagePrepare 升级前准备
// 备份数据、分发证 等
func (e *EnsureClusterManage) bocloudClusterManagePrepare() error {
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	// Filter out NeedSkip nodes so backup/cert distribution only target active nodes
	bkeNodes := e.filterNeedSkipNodes(bkenode.Nodes(allNodes))
	// todo  reconcile load balancer or other

	masterNode := bkeNodes.Master()
	//补全证书
	certsGenerator := certs.NewKubernetesCertGeneratorWithCache(e.Ctx.Context, e.Ctx.Client, e.Ctx.Cache, e.Ctx.BKECluster)
	certsGenerator.SetNodes(bkeNodes)
	certsGenerator.ConfigKubeConfig(net.JoinHostPort(masterNode[0].IP, "6443"))
	if err := certsGenerator.LookUpOrGenerate(); err != nil {
		return err
	}

	// backup data
	if err := e.backupBocloudClusterData(bkeNodes); err != nil {
		e.Ctx.Log.Warn(constant.BocloudClusterDataBackupFailedReason, "failed to backup cluster %s data: %v", utils.ClientObjNS(e.Ctx.BKECluster), err)
	}

	// distribute target cluster certs
	if err := e.distributeTargetClusterCerts(bkeNodes); err != nil {
		return err
	}

	// 环境初始化，hosts文件、运行时配置等
	if err := e.initBocloudClusterEnv(); err != nil {
		return err
	}
	return nil
}

// distributeTargetClusterCerts 分发证书给目标集群所有节点
func (e *EnsureClusterManage) distributeTargetClusterCerts(bkeNodes bkenode.Nodes) error {
	// 如果推断集群是bke集群，那么就不需要分发证书了
	if v, ok := condition.HasCondition(bkev1beta1.TypeOfManagementClusterGuessCondition, e.Ctx.BKECluster); ok {
		clusterType := v.Reason
		if clusterType == common.BKEClusterFromAnnotationValueBKE {
			return nil
		}
	}

	// 分发证书给master节点
	if bkeNodes.Master().Length() != 0 {
		if err := e.distributeMasterNodesCerts(bkeNodes); err != nil {
			return err
		}
	}
	// 分发证书给worker节点
	if bkeNodes.Worker().Length() != 0 {
		if err := e.distributeWorkerNodesCerts(bkeNodes); err != nil {
			return err
		}
	}
	return nil
}

// distributeMasterNodesCerts 分发证书给master节点
func (e *EnsureClusterManage) distributeMasterNodesCerts(bkeNodes bkenode.Nodes) error {
	phaseContextFields830 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields830.Context,
		phaseContextFields830.Client,
		phaseContextFields830.BKECluster,
		phaseContextFields830.Scheme,
		phaseContextFields830.Log
	if _, ok := condition.HasCondition(bkev1beta1.BocloudClusterMasterCertDistributionCondition, bkeCluster); ok {
		return nil
	}

	log.Info(constant.ClusterManagingReason, "Distribute master nodes certs")
	certCommandName := fmt.Sprintf("distribute-master-cert-%s", bkeCluster.Name)
	certCommandSpec := createCertCommandSpec(CreateCertCommandSpecParams{
		CertPluginName:  certPlugin.Name,
		ClusterName:     bkeCluster.Name,
		Namespace:       bkeCluster.Namespace,
		CertificatesDir: bkeCluster.Spec.ClusterConfig.Cluster.CertificatesDir,
		AdditionalParams: []string{
			"generate=false",
			"generateKubeConfig=true",
			"loadCACert=false",
			"loadTargetClusterCert=true",
			"loadAdminKubeconfig=false",
			"uploadCerts=false",
		},
	})
	choseNodes := bkeNodes.Master()
	baseCommandParams1 := newBaseCommandParams(ctx, c, bkeCluster, scheme)
	certCommand := createCustomCommand(CreateCustomCommandParams{
		BaseCommand:  createBaseCommand(baseCommandParams1),
		Nodes:        choseNodes,
		CommandName:  certCommandName,
		CommandSpec:  certCommandSpec,
		CommandLabel: command.BKEClusterLabel,
	})

	executeParams := ExecuteCommandAndWaitParams{
		Ctx:            ctx,
		Client:         c,
		Command:        certCommand,
		CommandName:    certCommandName,
		BKECluster:     bkeCluster,
		Log:            log,
		ConditionType:  bkev1beta1.BocloudClusterMasterCertDistributionCondition,
		SuccessReason:  constant.BocloudClusterMasterCertDistributionSuccessReason,
		FailedReason:   constant.BocloudClusterMasterCertDistributionFailedReason,
		SuccessMessage: "Distribute master nodes certs success",
	}
	if err := executeCommandAndWait(executeParams); err != nil {
		return err
	}
	return nil
}

// distributeWorkerNodesCerts 分发证书给worker节点
func (e *EnsureClusterManage) distributeWorkerNodesCerts(bkeNodes bkenode.Nodes) error {
	phaseContextFields881 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields881.Context,
		phaseContextFields881.Client,
		phaseContextFields881.BKECluster,
		phaseContextFields881.Scheme,
		phaseContextFields881.Log
	if _, ok := condition.HasCondition(bkev1beta1.BocloudClusterWorkerCertDistributionCondition, bkeCluster); ok {
		return nil
	}

	log.Info(constant.ClusterManagingReason, "Distribute worker nodes certs")
	certCommandName := fmt.Sprintf("distribute-worker-cert-%s", bkeCluster.Name)
	certCommandSpec := createCertCommandSpec(CreateCertCommandSpecParams{
		CertPluginName:  certPlugin.Name,
		ClusterName:     bkeCluster.Name,
		Namespace:       bkeCluster.Namespace,
		CertificatesDir: bkeCluster.Spec.ClusterConfig.Cluster.CertificatesDir,
		AdditionalParams: []string{
			"generate=false",
			"generateKubeConfig=false",
			"loadCACert=true",
			"caCertNames=ca,proxy",
			"loadTargetClusterCert=false",
			"loadAdminKubeconfig=true",
			"uploadCerts=false",
		},
	})
	choseNodes := bkeNodes.Worker()
	baseCommandParams2 := newBaseCommandParams(ctx, c, bkeCluster, scheme)
	certCommand := createCustomCommand(CreateCustomCommandParams{
		BaseCommand:  createBaseCommand(baseCommandParams2),
		Nodes:        choseNodes,
		CommandName:  certCommandName,
		CommandSpec:  certCommandSpec,
		CommandLabel: command.BKEClusterLabel,
	})

	executeParams2 := ExecuteCommandAndWaitParams{
		Ctx:            ctx,
		Client:         c,
		Command:        certCommand,
		CommandName:    certCommandName,
		BKECluster:     bkeCluster,
		Log:            log,
		ConditionType:  bkev1beta1.BocloudClusterWorkerCertDistributionCondition,
		SuccessReason:  constant.BocloudClusterWorkerCertDistributionSuccessReason,
		FailedReason:   constant.BocloudClusterWorkerCertDistributionFailedReason,
		SuccessMessage: "Distribute worker nodes certs success",
	}
	if err := executeCommandAndWait(executeParams2); err != nil {
		return err
	}
	return nil
}

// backupBocloudClusterData backup bocloud cluster data
// include: old pki dir、etcd pki dir, etc
func (e *EnsureClusterManage) backupBocloudClusterData(bkeNodes bkenode.Nodes) error {
	phaseContextFields934 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields934.Context,
		phaseContextFields934.Client,
		phaseContextFields934.BKECluster,
		phaseContextFields934.Scheme,
		phaseContextFields934.Log
	if _, ok := condition.HasCondition(bkev1beta1.BocloudClusterDataBackupCondition, bkeCluster); ok {
		return nil
	}

	log.Info(constant.ClusterManagingReason, "backup boCloud cluster data")
	dirs := []string{
		// kubernetes dir
		"/etc/kubernetes",
		// etcd cert dir
		"/etc/etcd/ssl",
	}
	files := []string{
		"",
	}

	backupCommandName := fmt.Sprintf("backup-bocloud-cluster-data-%s", bkeCluster.Name)
	backupCommandSpec := command.GenerateDefaultCommandSpec()
	backupCommandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "backup",
			Command: []string{
				backupPlugin.Name,
				fmt.Sprintf("backupDirs=%s", strings.Join(dirs, ",")),
				fmt.Sprintf("backupFiles=%s", strings.Join(files, ",")),
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
	}

	choseNodes := bkeNodes
	baseCommandParams3 := newBaseCommandParams(ctx, c, bkeCluster, scheme)
	backupCommand := command.Custom{
		BaseCommand:  createBaseCommand(baseCommandParams3),
		Nodes:        choseNodes,
		CommandName:  backupCommandName,
		CommandSpec:  backupCommandSpec,
		CommandLabel: command.BKEClusterLabel,
	}

	if err := backupCommand.New(); err != nil {
		return err
	}
	err, _, failed := backupCommand.Wait()
	if err != nil {
		return fmt.Errorf("failed to wait command %q: %w", backupCommandName, err)
	}
	if failed != nil || len(failed) > 0 {
		commandErrs, cmdErr := phaseutil.LogCommandFailed(*backupCommand.Command, failed, log, "BackupBocloudClusterDataFailed")
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		if cmdErr == nil {
			cmdErr = fmt.Errorf("command timed out on nodes %q", failed)
		}
		return fmt.Errorf("failed to backup bocloud cluster data on nodes %q: %w", failed, cmdErr)
	}
	condition.ConditionMark(bkeCluster, bkev1beta1.BocloudClusterDataBackupCondition, confv1beta1.ConditionTrue, constant.BocloudClusterDataBackupSuccessReason, "backup bocloud cluster data success")
	log.Info(constant.CommandExecSuccessReason, "backup bocloud cluster data success")
	return nil
}

// initBocloudClusterEnv init bocloud cluster env
func (e *EnsureClusterManage) initBocloudClusterEnv() error {
	phaseContextFields997 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields997.Context,
		phaseContextFields997.Client,
		phaseContextFields997.BKECluster,
		phaseContextFields997.Scheme,
		phaseContextFields997.Log

	// Use NodeFetcher to get BKENodes from API server
	allBkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		log.Warn(constant.ClusterManagingReason, "failed to get BKENodes: %v", err)
		return nil
	}
	bkeNodes := phaseutil.GetNeedInitEnvNodesWithBKENodes(bkeCluster, allBkeNodes)
	if bkeNodes.Length() == 0 {
		return nil
	}

	log.Info(constant.ClusterManagingReason, "init bocloud cluster env, scope: hosts file, http repo")
	envCommandName := fmt.Sprintf("init-bocloud-cluster-env-%s", bkeCluster.Name)
	envCommandSpec := command.GenerateDefaultCommandSpec()
	envCommandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "init bocloud cluster env",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				"scope=hosts,httpRepo,registry",
				fmt.Sprintf("bkeConfig=%s:%s", bkeCluster.Namespace, bkeCluster.Name),
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  commandBackoffDelaySeconds,
			BackoffIgnore: false,
		},
	}
	baseCommandParams4 := newBaseCommandParams(ctx, c, bkeCluster, scheme)
	envCommand := command.Custom{
		BaseCommand:  createBaseCommand(baseCommandParams4),
		Nodes:        bkeNodes,
		CommandName:  envCommandName,
		CommandSpec:  envCommandSpec,
		CommandLabel: command.BKEClusterLabel,
	}

	if err := envCommand.New(); err != nil {
		return err
	}

	err, success, failed := envCommand.Wait()
	if err != nil {
		return fmt.Errorf("failed to wait command %q: %w", envCommandName, err)
	}

	nf := e.Ctx.NodeFetcher()
	// 标记成功节点状态
	for _, node := range success {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		if err := nf.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, nodeIP, func(status *confv1beta1.BKENodeStatus) {
			status.StateCode |= bkev1beta1.NodeEnvFlag
			status.Message = "Nodes env is ready"
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update env success status for %s: %v", nodeIP, err)
		}
	}

	// 标记失败节点状态
	for _, node := range failed {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		if err := e.Ctx.SetNodeStateWithMessage(nodeIP, bkev1beta1.NodeInitFailed, "Failed to check k8s env"); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to set node state for %s: %v", nodeIP, err)
		}
	}

	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
		return err
	}

	// 只要有失败的就返回，不同于集群部署
	if len(failed) > 0 {
		commandErrs, cmdErr := phaseutil.LogCommandFailed(*envCommand.Command, failed, log, constant.BocloudClusterEnvInitFailedReason)
		phaseutil.MarkNodeStatusByCommandErrs(ctx, c, bkeCluster, commandErrs)
		if cmdErr == nil {
			cmdErr = fmt.Errorf("command timed out on nodes %q", failed)
		}
		errInfo := fmt.Sprintf("failed to init bocloud cluster env on flow nodes %q, err: %v", failed, cmdErr)
		condition.ConditionMark(bkeCluster, bkev1beta1.BocloudClusterEnvInitCondition, confv1beta1.ConditionFalse, constant.BocloudClusterEnvInitFailedReason, errInfo)
		return fmt.Errorf("failed to init bocloud cluster env on flow nodes %q: %w", failed, cmdErr)
	}

	condition.ConditionMark(bkeCluster, bkev1beta1.BocloudClusterEnvInitCondition, confv1beta1.ConditionTrue, constant.BocloudClusterEnvInitSuccessReason, "init bocloud cluster env success")
	for _, node := range bkeNodes {
		if err := nf.MarkNodeStateFlagForCluster(ctx, bkeCluster, node.IP, bkev1beta1.NodeEnvFlag); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to mark node state flag for %s: %v", node.IP, err)
		}
	}

	log.Info(constant.BocloudClusterEnvInitSuccessReason, "init bocloud cluster env success")
	return nil
}

func (e *EnsureClusterManage) getContainerRuntimeConfigFromCollectCommand(collectCommand *agentv1beta1.Command) error {
	phaseContextFields1094 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields1094.Context,
		phaseContextFields1094.Client,
		phaseContextFields1094.BKECluster,
		phaseContextFields1094.Scheme,
		phaseContextFields1094.Log

	finalLowLevelRuntime := bkeinit.DefaultRuntime
	finalCgroupDriver := bkeinit.DefaultCgroupDriver
	finalDataRoot := bkeinit.DefaultCRIDockerDataRootDir
	finalClusterType := common.BKEClusterFromAnnotationValueBocloud
	finalKubeletRootDir := bkeinit.DefaultKubeletRootDir

	var lowLevelRuntimes []string
	var cgroupDrivers []string
	var dataRoots []string
	var guessClusterTypes []string
	var kubeletRootDirs []string

	for _, cmd := range collectCommand.Status {
		condition := cmd.Conditions[0]
		if condition.StdOut == nil {
			continue
		}

		// 环境中的agent 由于版本不同，可能会有不同的输出，所以这里需要做一下兼容
		for i, v := range condition.StdOut {
			if v == "" {
				continue
			}

			switch i {
			case indexLowLevelRuntime:
				lowLevelRuntimes = append(lowLevelRuntimes, v)
			case indexCgroupDriver:
				cgroupDrivers = append(cgroupDrivers, v)
			case indexDataRoot:
				dataRoots = append(dataRoots, v)
			case indexGuessClusterType:
				guessClusterTypes = append(guessClusterTypes, v)
			case indexKubeletRootDir:
				kubeletRootDirs = append(kubeletRootDirs, v)
			default:
			}
		}

	}

	finalLowLevelRuntime = utils.MostCommonChar(lowLevelRuntimes)
	finalCgroupDriver = utils.MostCommonChar(cgroupDrivers)
	finalDataRoot = utils.MostCommonChar(dataRoots)
	finalClusterType = utils.MostCommonChar(guessClusterTypes)
	finalKubeletRootDir = utils.MostCommonChar(kubeletRootDirs)

	log.Info(constant.ClusterManagingReason, "bocloud cluster container runtime is %q. low level runtime: %q, cgroup driver: %q, data root: %q",
		bkeCluster.Spec.ClusterConfig.Cluster.ContainerRuntime.CRI, finalLowLevelRuntime, finalCgroupDriver, finalDataRoot)
	log.Info(constant.ClusterManagingReason, "bocloud cluster kubelet root dir is %q", finalKubeletRootDir)

	log.Info(constant.ClusterManagingReason, "infer the original cluster was created by %q", finalClusterType)

	containerRuntime := confv1beta1.ContainerRuntime{
		CRI:     bkeCluster.Spec.ClusterConfig.Cluster.ContainerRuntime.CRI,
		Runtime: finalLowLevelRuntime,
		Param: map[string]string{
			"cgroupDriver": finalCgroupDriver,
			"data-root":    finalDataRoot,
		},
	}

	patchFunc := func(bkeCluster *bkev1beta1.BKECluster) {
		// mark cluster info collected
		clusterutil.MarkClusterAgentInfoCollected(bkeCluster)
		annotation.RemoveAnnotation(bkeCluster, annotation.ManageClusterEtcdCertDirAnnotationKey)

		bkeCluster.Spec.ClusterConfig.Cluster.ContainerRuntime = containerRuntime
		if bkeCluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes != nil {
			for i, v := range bkeCluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes {
				tmp := v
				if tmp.Name == "kubelet-root-dir" {
					bkeCluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes[i].HostPath = finalKubeletRootDir
				}
			}
		} else {
			bkeCluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes = []confv1beta1.HostPathMount{
				{
					Name:     "kubelet-root-dir",
					HostPath: finalKubeletRootDir,
				},
			}
		}
		condition.ConditionMark(bkeCluster, bkev1beta1.TypeOfManagementClusterGuessCondition, confv1beta1.ConditionTrue, finalClusterType, "")
	}

	return mergecluster.SyncStatusUntilComplete(c, bkeCluster, patchFunc)
}

// CreateCertCommandSpecParams 包含 createCertCommandSpec 函数的参数
type CreateCertCommandSpecParams struct {
	CertPluginName   string
	ClusterName      string
	Namespace        string
	CertificatesDir  string
	AdditionalParams []string // 额外的参数
}

// createCertCommandSpec 创建证书命令规范
func createCertCommandSpec(params CreateCertCommandSpecParams) *agentv1beta1.CommandSpec {
	commandSpec := command.GenerateDefaultCommandSpec()

	// 构建基本命令
	cmd := []string{
		params.CertPluginName,
		fmt.Sprintf("clusterName=%s", params.ClusterName),
		fmt.Sprintf("namespace=%s", params.Namespace),
		fmt.Sprintf("certificatesDir=%s", params.CertificatesDir),
	}

	// 添加额外参数
	cmd = append(cmd, params.AdditionalParams...)

	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID:            "cert",
			Command:       cmd,
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  commandBackoffDelaySeconds,
			BackoffIgnore: false,
		},
	}

	return commandSpec
}

// CreateCustomCommandParams 包含 createCustomCommand 函数的参数
type CreateCustomCommandParams struct {
	BaseCommand  command.BaseCommand
	Nodes        bkenode.Nodes
	CommandName  string
	CommandSpec  *agentv1beta1.CommandSpec
	CommandLabel string
}

// createCustomCommand 创建自定义命令
func createCustomCommand(params CreateCustomCommandParams) command.Custom {
	return command.Custom{
		BaseCommand:  params.BaseCommand,
		Nodes:        params.Nodes,
		CommandName:  params.CommandName,
		CommandSpec:  params.CommandSpec,
		CommandLabel: params.CommandLabel,
	}
}

// newBaseCommandParams 创建基础命令参数
func newBaseCommandParams(ctx context.Context, client client.Client, bkeCluster *bkev1beta1.BKECluster, scheme *runtime.Scheme) CreateBaseCommandParams {
	return CreateBaseCommandParams{
		Ctx:             ctx,
		Client:          client,
		Scheme:          scheme,
		OwnerObj:        bkeCluster,
		NameSpace:       bkeCluster.Namespace,
		ClusterName:     bkeCluster.Name,
		RemoveAfterWait: true,
		Unique:          true,
	}
}

// ExecuteCommandAndWaitParams 包含 executeCommandAndWait 函数的参数
type ExecuteCommandAndWaitParams struct {
	Ctx            context.Context
	Client         client.Client
	Command        command.Custom
	CommandName    string
	BKECluster     *bkev1beta1.BKECluster
	Log            *bkev1beta1.BKELogger
	ConditionType  confv1beta1.ClusterConditionType
	SuccessReason  string
	FailedReason   string
	SuccessMessage string
}

// executeCommandAndWait 执行命令并等待完成
func executeCommandAndWait(params ExecuteCommandAndWaitParams) error {
	if err := params.Command.New(); err != nil {
		return err
	}

	err, _, failed := params.Command.Wait()
	if err != nil {
		return fmt.Errorf("failed to wait command %q: %w", params.CommandName, err)
	}
	if failed != nil || len(failed) > 0 {
		commandErrs, err := phaseutil.LogCommandFailed(*params.Command.Command, failed, params.Log, params.FailedReason)
		phaseutil.MarkNodeStatusByCommandErrs(params.Ctx, params.Client, params.BKECluster, commandErrs)
		return fmt.Errorf("failed to execute command on nodes %q: %w", failed, err)
	}
	condition.ConditionMark(params.BKECluster, params.ConditionType, confv1beta1.ConditionTrue, params.SuccessReason, params.SuccessMessage)
	params.Log.Info(params.SuccessReason, params.SuccessMessage)
	return nil
}

// markNodesBootstrapSuccess 标记节点伪引导成功状态
// 用于减少 fakeBootstrapMaster 和 fakeBootstrapWorker 中的重复代码
func (e *EnsureClusterManage) markNodesBootstrapSuccess(ctx context.Context, nodes map[int]confv1beta1.Node, log *bkev1beta1.BKELogger) {
	nf := e.Ctx.NodeFetcher()
	for _, node := range nodes {
		if err := nf.UpdateNodeStatusByIPForCluster(ctx, e.Ctx.BKECluster, node.IP, func(status *confv1beta1.BKENodeStatus) {
			status.StateCode |= bkev1beta1.NodeBootFlag
			status.StateCode |= bkev1beta1.NodePostProcessFlag
			status.State = bkev1beta1.NodeNotReady
			status.Message = "Fake bootstrap success, postprocess skipped"
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update fake bootstrap status for %s: %v", node.IP, err)
		}
	}
}

// checkAgentNeedPush 检查是否需要推送 Agent
// 只有当所有节点都已设置 NodeAgentPushedFlag 时才跳过推送；
// 任一节点缺少 PushedFlag（如 retry 后或新节点加入）都需要重新推送。
func (e *EnsureClusterManage) checkAgentNeedPush(nodes bkenode.Nodes) bool {
	activeNodes := e.filterNeedSkipNodes(nodes)
	for _, node := range activeNodes {
		hasPushedFlag, _ := e.Ctx.GetNodeStateFlag(node.IP, bkev1beta1.NodeAgentPushedFlag)
		if !hasPushedFlag {
			return true
		}
	}
	return false
}

// filterNeedSkipNodes returns nodes that are not marked as NeedSkip.
func (e *EnsureClusterManage) filterNeedSkipNodes(nodes bkenode.Nodes) bkenode.Nodes {
	nf := e.Ctx.NodeFetcher()
	phaseContextFields1323 := e.Ctx.Untie()
	ctx, _, bkeCluster, _, _ := phaseContextFields1323.Context,
		phaseContextFields1323.Client,
		phaseContextFields1323.BKECluster,
		phaseContextFields1323.Scheme,
		phaseContextFields1323.Log
	var result bkenode.Nodes
	for _, node := range nodes {
		needSkip, _ := nf.GetNodeStateNeedSkip(ctx, bkeCluster.Namespace, bkeCluster.Name, node.IP)
		if needSkip {
			continue
		}
		result = append(result, node)
	}
	return result
}

// processAgentPingResults 处理 Agent ping 结果
func (e *EnsureClusterManage) processAgentPingResults(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	successNodes, failedNodes []string, log *bkev1beta1.BKELogger) {
	nf := e.Ctx.NodeFetcher()
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		log.Warn(constant.InternalErrorReason, "Failed to get nodes for role check: %v", err)
	}
	workerNodes := allNodes.Worker()

	for _, node := range failedNodes {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		isWorker := workerNodes.Filter(bkenode.FilterOptions{"IP": nodeIP}).Length() > 0
		if err := nf.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, nodeIP, func(status *confv1beta1.BKENodeStatus) {
			e.markFailedPingNode(status, isWorker)
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update failed ping status for %s: %v", nodeIP, err)
		}
	}

	for _, node := range successNodes {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		if err := nf.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, nodeIP, func(status *confv1beta1.BKENodeStatus) {
			markSuccessPingNode(status)
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update ping success status for %s: %v", nodeIP, err)
		}
	}
}

// markSuccessPingNode sets success status for a node that passed agent ping.
func markSuccessPingNode(status *confv1beta1.BKENodeStatus) {
	status.Message = "BKEAgent is ready"
	status.StateCode |= bkev1beta1.NodeAgentPushedFlag
	status.StateCode |= bkev1beta1.NodeAgentReadyFlag
	status.RetryCount = 0
}

// markFailedPingNode sets failure status for a node that failed agent ping.
// Worker nodes are marked NeedSkip after exceeding the allowed retry count.
func (e *EnsureClusterManage) markFailedPingNode(status *confv1beta1.BKENodeStatus, isWorker bool) {
	status.State = bkev1beta1.NodeInitFailed
	status.StateCode &= ^bkev1beta1.NodeAgentPushedFlag
	status.RetryCount++
	if isWorker && status.RetryCount >= statusmanage.ReconcileAllowedFailedCount {
		status.NeedSkip = true
		status.Message = fmt.Sprintf("Failed ping bkeagent after %d retries, marked as NeedSkip", status.RetryCount)
	} else {
		status.Message = "Failed ping bkeagent"
	}
}

// waitForNodesBootstrap 等待节点伪引导完成的通用逻辑
func waitForNodesBootstrap(input nodesBootstrapInput) error {
	ctx, c, bkeCluster := input.ctx, input.client, input.cluster
	nodes, successNodes, expectedCount, log := input.nodes, input.successNodes, input.expectedCount, input.log
	if successNodes == nil {
		return errors.New("successNodes map cannot be nil")
	}
	for i, node := range nodes {
		if _, ok := successNodes[i]; ok {
			continue
		}
		machine, err := phaseutil.NodeToMachine(ctx, c, bkeCluster, node)
		if err != nil {
			continue
		}
		if machine.Status.NodeRef != nil {
			log.Info(constant.ClusterManagingReason, "node %s fake bootstrap success", phaseutil.NodeInfo(node))
			successNodes[i] = node
		}
	}
	if len(successNodes) != int(expectedCount) {
		return errors.New("not all nodes ready")
	}
	return nil
}
