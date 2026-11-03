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

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	containerutil "sigs.k8s.io/cluster-api/util/container"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureMasterUpgradeName confv1beta1.BKEClusterPhase = "EnsureMasterUpgrade"
	// MasterUpgradePollIntervalSeconds 等待master升级时轮询间隔秒数
	MasterUpgradePollIntervalSeconds = 2
	// MasterUpgradeTimeoutMinutes 等待master升级超时分钟数
	MasterUpgradeTimeoutMinutes = 5
)

type EnsureMasterUpgrade struct {
	phaseframe.BasePhase
	// mockTargetClient allows tests to inject a fake target-cluster clientset
	// for the kube-proxy/etcd clientset operations. nil in production.
	mockTargetClient kubernetes.Interface
}

// targetClientSet returns the target-cluster kubernetes clientset. In production
// it is obtained via kube.GetTargetClusterClient; tests may inject mockTargetClient.
func (e *EnsureMasterUpgrade) targetClientSet(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (kubernetes.Interface, error) {
	if e.mockTargetClient != nil {
		return e.mockTargetClient, nil
	}
	cs, _, err := kube.GetTargetClusterClient(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}
	return cs, nil
}

func NewEnsureMasterUpgrade(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureMasterUpgradeName)
	return &EnsureMasterUpgrade{BasePhase: base}
}

func (e *EnsureMasterUpgrade) Execute() (ctrl.Result, error) {
	if v, ok := annotation.HasAnnotation(e.Ctx.BKECluster, annotation.DeployActionAnnotationKey); !ok || v != annotation.DeployActionK8sUpgrade {
		//添加boc所需的注解
		patchFunc := func(bkeCluster *bkev1beta1.BKECluster) {
			annotation.SetAnnotation(bkeCluster, annotation.DeployActionAnnotationKey, annotation.DeployActionK8sUpgrade)
		}
		if err := mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster, patchFunc); err != nil {
			return ctrl.Result{}, err
		}
	}

	return e.reconcileMasterUpgrade()
}

// Version returns the current running Kubernetes version on the cluster.
func (e *EnsureMasterUpgrade) Version() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil {
		return ""
	}
	return e.Ctx.BKECluster.Status.KubernetesVersion
}

func (e *EnsureMasterUpgrade) currentKubernetesVersion() string {
	vc := e.GetVersionContext()
	if vc != nil {
		if current := strings.TrimSpace(vc.GetCurrent(upgrade.ComponentKubernetesMaster)); current != "" {
			return current
		}
		if current := strings.TrimSpace(vc.GetCurrent(upgrade.ComponentKubernetesWorker)); current != "" {
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

func (e *EnsureMasterUpgrade) desiredKubernetesVersion() string {
	vc := e.GetVersionContext()
	if vc != nil {
		if target := strings.TrimSpace(vc.GetTarget(upgrade.ComponentKubernetesMaster)); target != "" {
			return target
		}
		if target := strings.TrimSpace(vc.GetTarget(upgrade.ComponentKubernetesWorker)); target != "" {
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
func (e *EnsureMasterUpgrade) deprecatedSpecKubernetesVersion() string {
	if e.Ctx == nil || e.Ctx.BKECluster == nil || e.Ctx.BKECluster.Spec.ClusterConfig == nil {
		return ""
	}
	return strings.TrimSpace(e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
}

func (e *EnsureMasterUpgrade) syncLegacyTargetKubernetesVersion(target string) error {
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
	e.Ctx.Log.Warn(constant.MasterUpgradingReason,
		"sync deprecated spec kubernetesVersion from %q to %q for legacy master upgrade execution",
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

// isKubernetesMasterNeedUpgrade reports whether the cluster kubernetes version differs from the upgrade target.
func (e *EnsureMasterUpgrade) isKubernetesMasterNeedUpgrade(_ *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if new == nil || new.Spec.ClusterConfig == nil {
		return false
	}
	target := strings.TrimSpace(e.desiredKubernetesVersion())
	current := strings.TrimSpace(e.currentKubernetesVersion())
	return target != current &&
		len(target) != 0 &&
		len(current) != 0
}

// NeedExecute determines whether the master kubernetes upgrade phase needs to be executed.
func (e *EnsureMasterUpgrade) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	if new.Status.ClusterStatus == bkev1beta1.ClusterUnhealthy ||
		new.Status.ClusterStatus == bkev1beta1.ClusterUnknown {
		return false
	}
	if !e.NeedExecuteWithVersionContext(upgrade.ComponentKubernetesMaster, old, new, e.isKubernetesMasterNeedUpgrade) {
		return false
	}

	bkeNodes, ok := fetchBKENodesIfCPInitialized(e.Ctx, new)
	if !ok {
		return false
	}
	nodes := phaseutil.GetNeedUpgradeMasterNodesWithBKENodes(new, bkeNodes)
	if nodes.Length() == 0 {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureMasterUpgrade) reconcileMasterUpgrade() (ctrl.Result, error) {
	phaseContextFields208 := e.Ctx.Untie()
	_, _, _, _, log := phaseContextFields208.Context,
		phaseContextFields208.Client,
		phaseContextFields208.BKECluster,
		phaseContextFields208.Scheme,
		phaseContextFields208.Log
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
	log.Info(constant.MasterUpgradedReason, "k8s version same, not need to upgrade master node")
	return ctrl.Result{}, nil
}

func (e *EnsureMasterUpgrade) rolloutUpgrade() (ctrl.Result, error) {
	phaseContextFields225 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields225.Context,
		phaseContextFields225.Client,
		phaseContextFields225.BKECluster,
		phaseContextFields225.Scheme,
		phaseContextFields225.Log

	needUpgradeNodes, err := e.getNeedUpgradeNodes(bkeCluster, log)
	if err != nil {
		return ctrl.Result{Requeue: true}, err
	}

	// 检查etcd配置
	specNodes, _ := e.Ctx.NodeFetcher().GetNodesForBKECluster(e.Ctx, bkeCluster)
	needBackupEtcd := false
	backEtcdNode := confv1beta1.Node{}
	etcdNodes := specNodes.Etcd()
	if etcdNodes.Length() != 0 {
		needBackupEtcd = true
		backEtcdNode = etcdNodes[0]
		log.Info(constant.MasterUpgradingReason, "backup etcd data to node %s", phaseutil.NodeInfo(backEtcdNode))
	}

	if err := e.ensureEtcdAdvertiseClientUrlsAnnotation(etcdNodes); err != nil {
		return ctrl.Result{}, fmt.Errorf("ensure etcd advertise client urls annotation failed: %w", err)
	}

	// 升级节点
	upgradeParams := UpgradeMasterNodesParams{
		Ctx:              ctx,
		Client:           c,
		BKECluster:       bkeCluster,
		NeedUpgradeNodes: needUpgradeNodes,
		NeedBackupEtcd:   needBackupEtcd,
		BackEtcdNode:     backEtcdNode,
		Log:              log,
	}
	if err := e.upgradeMasterNodesWithParams(upgradeParams); err != nil {
		return ctrl.Result{}, err
	}

	log.Info(constant.MasterUpgradeSucceedReason, "upgrade all master success")
	// master 始终是最后更新完的,这时候更改status的版本
	bkeCluster.Status.KubernetesVersion = e.desiredKubernetesVersion()

	// 更新addon版本
	return e.updateAddonVersions(c, bkeCluster, log)
}

// getNeedUpgradeNodes 获取需要升级的节点
func (e *EnsureMasterUpgrade) getNeedUpgradeNodes(bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) (bkenode.Nodes, error) {
	needUpgradeNodes := bkenode.Nodes{}

	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get BKENodes")
	}
	nodes := phaseutil.GetNeedUpgradeMasterNodesWithBKENodes(bkeCluster, bkeNodes)
	for _, node := range nodes {
		nodeState, _ := e.Ctx.NodeFetcher().GetNodeStateFlagForCluster(e.Ctx, bkeCluster, node.IP, bkev1beta1.NodeAgentReadyFlag)
		if !nodeState {
			log.Info(constant.MasterUpgradingReason, "agent is not ready at node %s, skip upgrade", phaseutil.NodeInfo(node))
			continue
		}
		needUpgradeNodes = append(needUpgradeNodes, node)
	}

	if len(needUpgradeNodes) == 0 {
		log.Info(constant.MasterUpgradingReason, "all the master node BKEAgent is not ready")
		return nil, errors.New("all the master node BKEAgent is not ready")
	}

	return needUpgradeNodes, nil
}

// UpgradeMasterNodesParams 包含升级master节点所需的参数
type UpgradeMasterNodesParams struct {
	Ctx              context.Context
	Client           client.Client
	BKECluster       *bkev1beta1.BKECluster
	NeedUpgradeNodes bkenode.Nodes
	NeedBackupEtcd   bool
	BackEtcdNode     confv1beta1.Node
	Log              *bkev1beta1.BKELogger
}

// upgradeMasterNodes 升级master节点
// upgradeMasterNodesWithParams 使用参数结构体升级master节点
func (e *EnsureMasterUpgrade) upgradeMasterNodesWithParams(params UpgradeMasterNodesParams) error {
	params.Log.Info(constant.MasterUpgradingReason, "Start upgrade master nodes process, upgrade policy: rollingUpgrade")

	clientSet, _, _ := kube.GetTargetClusterClient(params.Ctx, params.Client, params.BKECluster)
	nodeFetcher := e.Ctx.NodeFetcher()
	for _, node := range params.NeedUpgradeNodes {
		remoteNode, err := phaseutil.GetRemoteNodeByBKENode(params.Ctx, clientSet, node)
		if err != nil {
			return fmt.Errorf("get remote cluster Node resource failed: %w", err)
		}
		// 已经是期望版本的节点不需要升级
		if remoteNode.Status.NodeInfo.KubeletVersion == params.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion {
			params.Log.Info(constant.MasterUpgradeSucceedReason, "node %q is already the expected version %q,skip upgrade", phaseutil.NodeInfo(node), params.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
			continue
		}

		// mark node as upgrading
		nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeUpgrading, "Upgrading")
		if err := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
			return err
		}

		if err := e.upgradeNode(params.NeedBackupEtcd, params.BackEtcdNode, node, remoteNode); err != nil {
			// master node block until upgrade success
			nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeUpgradeFailed, err.Error())
			if syncErr := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); syncErr != nil {
				return fmt.Errorf("upgrade node %q failed: %w", phaseutil.NodeInfo(node), syncErr)
			}
			return fmt.Errorf("upgrade node %q failed: %w", phaseutil.NodeInfo(node), err)
		}
		// mark node as upgrading success
		nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeNotReady, "Upgrading success")
		if err := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
			return err
		}
	}
	return nil
}

// imageTagFromReference returns the tag segment of a container image reference (supports host:port/repo/name:tag).
func normalizeKubeComponentVersion(s string) string {
	return strings.TrimPrefix(strings.TrimSpace(s), "v")
}

func (e *EnsureMasterUpgrade) getKubeProxyImageTagFromCluster(c client.Client, bkeCluster *bkev1beta1.BKECluster) (string, error) {
	phaseContextFields354 := e.Ctx.Untie()
	ctx, _, _, _, _ := phaseContextFields354.Context,
		phaseContextFields354.Client,
		phaseContextFields354.BKECluster,
		phaseContextFields354.Scheme,
		phaseContextFields354.Log
	clientSet, err := e.targetClientSet(ctx, c, bkeCluster)
	if err != nil {
		return "", err
	}
	ds, err := clientSet.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(ctx, "kube-proxy", metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	var img string
	for _, co := range ds.Spec.Template.Spec.Containers {
		if co.Name == "kube-proxy" {
			img = co.Image
			break
		}
	}
	if img == "" && len(ds.Spec.Template.Spec.Containers) > 0 {
		img = ds.Spec.Template.Spec.Containers[0].Image
	}
	if img == "" {
		return "", errors.New("kube-proxy daemonset has no containers")
	}
	tag := utils.ImageTagFromReference(img)
	if tag == "" {
		return "", errors.Errorf("could not parse image tag from %q", img)
	}
	return tag, nil
}

func scanKubeproxyKubectlUpgradeFromAddons(addons []confv1beta1.Product, k8sVer string, log *bkev1beta1.BKELogger) (kubeproxyInSpec, kubeproxyNeedUpgrade, kubectlNeedUpgrade bool) {
	for _, addon := range addons {
		if addon.Name == "kubeproxy" {
			kubeproxyInSpec = true
			if addon.Version != k8sVer {
				log.Info(constant.MasterUpgradingReason, "kubeproxy need upgrade")
				kubeproxyNeedUpgrade = true
			}
		}
		if addon.Name == "kubectl" && addon.Version != "v1.25" {
			log.Info(constant.MasterUpgradingReason, "kubectl need upgrade")
			kubectlNeedUpgrade = true
		}
	}
	return kubeproxyInSpec, kubeproxyNeedUpgrade, kubectlNeedUpgrade
}

func (e *EnsureMasterUpgrade) logKubeProxyTagProbeFailure(log *bkev1beta1.BKELogger, err error) {
	if apierrors.IsNotFound(err) {
		log.Info(constant.MasterUpgradingReason, "no kubeproxy addon and kube-proxy daemonset not found, skip kube-proxy upgrade check")
		return
	}
	log.Warn(constant.MasterUpgradingReason, "no kubeproxy addon, failed to read kube-proxy image tag: %v, skip kube-proxy upgrade", err)
}

type kubeProxyUpgradeProbe struct {
	client            client.Client
	cluster           *bkev1beta1.BKECluster
	log               *bkev1beta1.BKELogger
	inSpec            bool
	needUpgrade       bool
	kubernetesVersion string
}

func (e *EnsureMasterUpgrade) augmentKubeproxyUpgradeNeedFromDaemonSet(probe kubeProxyUpgradeProbe) bool {
	if probe.inSpec {
		return probe.needUpgrade
	}
	tag, err := e.getKubeProxyImageTagFromCluster(probe.client, probe.cluster)
	if err != nil {
		e.logKubeProxyTagProbeFailure(probe.log, err)
		return probe.needUpgrade
	}
	if normalizeKubeComponentVersion(tag) == normalizeKubeComponentVersion(probe.kubernetesVersion) {
		return probe.needUpgrade
	}
	probe.log.Info(
		constant.MasterUpgradingReason,
		"kubeproxy need upgrade (daemonset image tag %q vs cluster kubernetesVersion %q)",
		tag,
		probe.kubernetesVersion,
	)
	return true
}

func patchKubeproxyAddonVersions(current *bkev1beta1.BKECluster) {
	ver := current.Spec.ClusterConfig.Cluster.KubernetesVersion
	for i, d := range current.Spec.ClusterConfig.Addons {
		if d.Name == "kubeproxy" {
			d.Version = ver
			current.Spec.ClusterConfig.Addons[i] = d
		}
	}
	for i, d := range current.Status.AddonStatus {
		addon := d.DeepCopy()
		if addon.Name == "kubeproxy" {
			addon.Version = ver
			current.Status.AddonStatus[i] = *addon
		}
	}
}

func patchKubectlAddonToV125(current *bkev1beta1.BKECluster) {
	found := false
	for i, d := range current.Spec.ClusterConfig.Addons {
		if d.Name == "kubectl" {
			found = true
			d.Version = "v1.25"
			current.Spec.ClusterConfig.Addons[i] = d
		}
	}
	if !found {
		current.Spec.ClusterConfig.Addons = append(current.Spec.ClusterConfig.Addons, confv1beta1.Product{
			Name:    "kubectl",
			Version: "v1.25",
			Block:   false,
		})
	}
}

// updateAddonVersions 更新 addon 版本。
// kube-proxy 升级由声明式 manifest 链路处理，此处仅保留 kubectl 版本同步。
func (e *EnsureMasterUpgrade) updateAddonVersions(c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) (ctrl.Result, error) {
	k8sVer := bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion
	_, _, kubectlNeed := scanKubeproxyKubectlUpgradeFromAddons(bkeCluster.Spec.ClusterConfig.Addons, k8sVer, log)

	var patchFuncs []mergecluster.PatchFunc
	if kubectlNeed {
		patchFuncs = append(patchFuncs, patchKubectlAddonToV125)
	}
	if len(patchFuncs) == 0 {
		return ctrl.Result{}, nil
	}
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster, patchFuncs...); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to upgrade addon version, err: %w", err)
	}
	return ctrl.Result{}, nil
}

func (e *EnsureMasterUpgrade) upgradeNode(needBackupEtcd bool, backEtcdNode confv1beta1.Node, node confv1beta1.Node, remoteNode *corev1.Node) error {
	phaseContextFields488 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields488.Context,
		phaseContextFields488.Client,
		phaseContextFields488.BKECluster,
		phaseContextFields488.Scheme,
		phaseContextFields488.Log

	// 执行节点升级
	upgradeParams := ExecuteNodeUpgradeParams{
		Ctx:            ctx,
		Client:         c,
		BKECluster:     bkeCluster,
		Scheme:         scheme,
		Log:            log,
		NeedBackupEtcd: needBackupEtcd,
		BackEtcdNode:   backEtcdNode,
		Node:           node,
	}
	if err := e.executeNodeUpgradeWithParams(upgradeParams); err != nil {
		return err
	}

	log.Info(constant.MasterUpgradingReason, "upgrade node %q operation succeed", phaseutil.NodeInfo(node))

	// 等待节点健康检查通过
	healthCheckParams := WaitForNodeHealthCheckParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Log:        log,
		Node:       node,
	}
	return e.waitForNodeHealthCheckWithParams(healthCheckParams)
}

// ExecuteNodeUpgradeParams 包含执行节点升级所需的参数
type ExecuteNodeUpgradeParams struct {
	Ctx            context.Context
	Client         client.Client
	BKECluster     *bkev1beta1.BKECluster
	Scheme         *runtime.Scheme
	Log            *bkev1beta1.BKELogger
	NeedBackupEtcd bool
	BackEtcdNode   confv1beta1.Node
	Node           confv1beta1.Node
}

// executeNodeUpgrade 执行节点升级
// executeNodeUpgradeWithParams 使用参数结构体执行节点升级
func (e *EnsureMasterUpgrade) executeNodeUpgradeWithParams(params ExecuteNodeUpgradeParams) error {
	masterParams := CreateUpgradeCommandParams{
		Ctx:         params.Ctx,
		Namespace:   params.BKECluster.Namespace,
		Client:      params.Client,
		Scheme:      params.Scheme,
		OwnerObj:    params.BKECluster,
		ClusterName: params.BKECluster.Name,
		Node:        &params.Node,
		BKEConfig:   params.BKECluster.Name,
		Phase:       bkev1beta1.UpgradeControlPlane,
	}
	upgrade := createUpgradeCommand(masterParams)

	if params.NeedBackupEtcd && params.Node.IP == params.BackEtcdNode.IP {
		upgrade.BackUpEtcd = true
	}

	params.Log.Info(constant.MasterUpgradingReason, "start upgrade node %s", phaseutil.NodeInfo(params.Node))
	if err := upgrade.New(); err != nil {
		return fmt.Errorf("create upgrade command，node: %q failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	params.Log.Info(constant.MasterUpgradingReason, "wait upgrade node %s finish", phaseutil.NodeInfo(params.Node))
	err, _, failedNodes := upgrade.Wait()
	if err != nil {
		return fmt.Errorf("wait upgrade command complete failed，node: %q: %w", phaseutil.NodeInfo(params.Node), err)
	}
	if len(failedNodes) != 0 {
		commandErrs, err := phaseutil.LogCommandFailed(*upgrade.Command, failedNodes, params.Log, constant.MasterUpgradeFailedReason)
		phaseutil.MarkNodeStatusByCommandErrs(params.Ctx, params.Client, params.BKECluster, commandErrs)
		return fmt.Errorf("upgrade node %q failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	return nil
}

// WaitForNodeHealthCheckParams 包含等待节点健康检查所需的参数
type WaitForNodeHealthCheckParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Log        *bkev1beta1.BKELogger
	Node       confv1beta1.Node
}

// waitForNodeHealthCheck 等待节点健康检查通过
func (e *EnsureMasterUpgrade) waitForNodeHealthCheck(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger, node confv1beta1.Node) error {
	params := WaitForNodeHealthCheckParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Log:        log,
		Node:       node,
	}
	return e.waitForNodeHealthCheckWithParams(params)
}

// waitForNodeHealthCheckWithParams 使用参数结构体等待节点健康检查通过
func (e *EnsureMasterUpgrade) waitForNodeHealthCheckWithParams(params WaitForNodeHealthCheckParams) error {
	remoteClient, err := kube.NewRemoteClientByBKECluster(params.Ctx, params.Client, params.BKECluster)
	if err != nil {
		return fmt.Errorf("get remote client for BKECluster %q failed: %w", utils.ClientObjNS(params.BKECluster), err)
	}
	clientSet, _ := remoteClient.KubeClient()

	// wait for node pass healthy check
	params.Log.Info(constant.MasterUpgradingReason, "wait for node %q pass healthy check", phaseutil.NodeInfo(params.Node))
	masterParams := WaitForWorkerNodeHealthCheckParams{
		Ctx:          params.Ctx,
		ClientSet:    clientSet,
		RemoteClient: remoteClient,
		Node:         params.Node,
		K8sVersion:   params.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
		Logger:       params.Log,
	}
	err = waitForWorkerNodeHealthCheck(masterParams)

	if err != nil {
		return fmt.Errorf("wait for node %q pass healthy check failed: %w", phaseutil.NodeInfo(params.Node), err)
	}
	params.Log.Info(constant.MasterUpgradingReason, "upgrade master node %q success", phaseutil.NodeInfo(params.Node))
	return nil
}

func (e *EnsureMasterUpgrade) upgradeKubeProxy(expectVersion string) error {
	phaseContextFields616 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields616.Context,
		phaseContextFields616.Client,
		phaseContextFields616.BKECluster,
		phaseContextFields616.Scheme,
		phaseContextFields616.Log
	clientSet, err := e.targetClientSet(ctx, c, bkeCluster)
	if err != nil {
		return err
	}

	cfg := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)
	imageRepo := cfg.ImageFuyaoRepo()

	var dstImage string
	err = phaseutil.RetryOnConflict(func() error {
		ds, err := clientSet.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(ctx, "kube-proxy", metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("get kube-proxy ds failed: %w", err)
		}

		srcImage := ds.Spec.Template.Spec.Containers[0].Image
		srcImage, err = containerutil.ModifyImageRepository(srcImage, imageRepo)
		if err != nil {
			return err
		}
		dstImage, err = containerutil.ModifyImageTag(srcImage, expectVersion)
		if err != nil {
			return err
		}
		ds.Spec.Template.Spec.Containers[0].Image = dstImage

		_, err = clientSet.AppsV1().DaemonSets(metav1.NamespaceSystem).Update(ctx, ds, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		return err
	}
	log.Info(constant.MasterUpgradingReason, "update kube-proxy image to %q success", dstImage)
	return nil
}

// ensureEtcdAdvertiseClientUrlsAnnotation
// ensure EtcdAdvertiseClientUrlsAnnotation exit in etcd pod annotations
func (e *EnsureMasterUpgrade) ensureEtcdAdvertiseClientUrlsAnnotation(etcdNodes bkenode.Nodes) error {
	phaseContextFields656 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields656.Context,
		phaseContextFields656.Client,
		phaseContextFields656.BKECluster,
		phaseContextFields656.Scheme,
		phaseContextFields656.Log
	clientSet, err := e.targetClientSet(ctx, c, bkeCluster)
	if err != nil {
		return err
	}
	for _, n := range etcdNodes {
		etcdPodName := kube.StaticPodName(mfutil.Etcd, n.Hostname)
		podClient := clientSet.CoreV1().Pods(metav1.NamespaceSystem)
		if err := phaseutil.RetryOnConflict(func() error {
			pod, err := podClient.Get(ctx, etcdPodName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			annotations := pod.GetAnnotations()
			if v, ok := annotations[annotation.EtcdAdvertiseClientUrlsAnnotationKey]; ok && v != "" {
				return nil
			}
			annotations[annotation.EtcdAdvertiseClientUrlsAnnotationKey] = phaseutil.GetClientURLByIP(n.IP)
			pod.SetAnnotations(annotations)
			_, err = podClient.Update(ctx, pod, metav1.UpdateOptions{})
			return err
		}); err != nil {
			return err
		}
	}
	return nil
}
