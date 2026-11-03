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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	metricrecord "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics/record"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
)

const (
	EnsureClusterName confv1beta1.BKEClusterPhase = "EnsureCluster"
	LabelReadyTimeout                             = 10 * time.Second
)

const (
	quickRequeueInterval  = 10 * time.Second
	periodicCheckInterval = 5 * time.Minute
)

type EnsureCluster struct {
	phaseframe.BasePhase
	remoteClient   kube.RemoteKubeClient
	healthCheckErr error
	// mockClient allows tests to inject a fake clientset for the CoreV1()
	// node label/token operations. nil in production.
	mockClient kubernetes.Interface
}

// kubeClient returns the kubernetes clientset used for node label/token ops.
// In production this is obtained from the remote cluster client; tests may
// inject a fake clientset via mockClient.
func (e *EnsureCluster) kubeClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	cs, _ := e.remoteClient.KubeClient()
	return cs
}

func NewEnsureCluster(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureClusterName)
	return &EnsureCluster{BasePhase: base}
}

func (e *EnsureCluster) Execute() (_ ctrl.Result, err error) {
	var errs []error

	if e.Ctx.Cluster == nil {
		return ctrl.Result{}, errors.Errorf("cluster is nil")
	}

	if !conditions.IsTrue(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		return ctrl.Result{}, errors.Errorf("cluster is not init")
	}

	if err = e.getRemoteClient(); err != nil {
		errs = append(errs, err)
		return ctrl.Result{}, kerrors.NewAggregate(errs)
	}

	if clusterutil.IsBKECluster(e.Ctx.BKECluster) {
		// ignore error
		if err = e.setAlertLabel(); err != nil {
			errs = append(errs, err)
		}
	}

	if e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.ContainerRuntime.Runtime == "richrunc" {
		// ignore error
		if err = e.setBareMetalLabel(); err != nil {
			errs = append(errs, err)
		}
	}

	// set node label
	if err = e.setNodeLabel(); err != nil {
		errs = append(errs, err)
	}

	if err = e.ensureK8sToken(); err != nil {
		errs = append(errs, err)
	}

	if err = e.ensureRemoteHealthCheckConfigCM(); err != nil {
		errs = append(errs, err)
	}

	phaseContextFields127 := e.Ctx.Untie()
	_, _, bkeCluster, _, log := phaseContextFields127.Context,
		phaseContextFields127.Client,
		phaseContextFields127.BKECluster,
		phaseContextFields127.Scheme,
		phaseContextFields127.Log
	if err != nil {
		return ctrl.Result{}, kerrors.NewAggregate(errs)
	}

	// 如果集群处于特殊状态，则暂不执行定时检查
	if isClusterInSpecialState(bkeCluster) {
		log.Info(constant.ClusterManagingReason, "cluster is in special state, skip health check")
		return ctrl.Result{RequeueAfter: periodicCheckInterval}, nil
	}

	// 后置处理未完成时，不进入健康检查，避免误判为已完成
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		return ctrl.Result{}, err
	}
	if phaseutil.GetNeedPostProcessNodesWithBKENodes(bkeCluster, bkeNodes).Length() > 0 {
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition, confv1beta1.ConditionFalse, constant.NodesPostProcessNotReadyReason, "")
		return ctrl.Result{RequeueAfter: quickRequeueInterval}, errors.Errorf("postprocess not finished")
	}

	if err = e.ensureClusterReady(); err != nil {
		interval := kube.GetHealthCheckRequeueInterval(err)
		e.healthCheckErr = err
		log.Warn(constant.TargetClusterNotReadyReason, "cluster health check failed, requeue after %s: %v", interval, err)
		// 健康检查失败已写入集群健康状态，返回 nil error 才能让 RequeueAfter 严格生效。
		return ctrl.Result{RequeueAfter: interval}, nil
	}

	// 正常状态下，定时5min 来检查重新调谐
	return ctrl.Result{RequeueAfter: periodicCheckInterval}, kerrors.NewAggregate(errs)
}

// 校验函数: 判断集群是否处于特殊状态
func isClusterInSpecialState(bkeCluster *bkev1beta1.BKECluster) bool {
	// 将相关状态集中在一个数组中进行判断
	specialStates := []confv1beta1.ClusterStatus{
		bkev1beta1.ClusterMasterScalingUp,
		bkev1beta1.ClusterMasterScalingDown,
		bkev1beta1.ClusterWorkerScalingUp,
		bkev1beta1.ClusterWorkerScalingDown,
		bkev1beta1.ClusterInitializing,
		bkev1beta1.ClusterPaused,
		bkev1beta1.ClusterUpgrading,
	}

	// 获取集群的当前状态
	clusterStatus := bkeCluster.Status.ClusterStatus

	// 判断当前状态是否在特殊状态数组中
	for _, status := range specialStates {
		if clusterStatus == status {
			return true
		}
	}
	return false
}

func (e *EnsureCluster) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureCluster) ExecutePostHook(err error) error {
	defer func() { e.healthCheckErr = nil }()
	if err == nil && e.healthCheckErr != nil {
		// 健康检查失败不作为 reconcile error 返回，但 phase 状态仍按失败上报。
		return e.DefaultPostHook(e.healthCheckErr)
	}
	return e.DefaultPostHook(err)
}

// getRemoteClient get remote cluster client
func (e *EnsureCluster) getRemoteClient() error {
	phaseContextFields204 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields204.Context,
		phaseContextFields204.Client,
		phaseContextFields204.BKECluster,
		phaseContextFields204.Scheme,
		phaseContextFields204.Log
	remoteClient, err := kube.NewRemoteClientByBKECluster(ctx, c, bkeCluster)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster client: %w", utils.ClientObjNS(bkeCluster), err)
	}
	e.remoteClient = remoteClient
	e.remoteClient.SetLogger(log.NormalLogger)
	return nil
}

// setAlertLabel set alert label
func (e *EnsureCluster) setAlertLabel() (err error) {

	alterNodeFilter := &metav1.ListOptions{
		LabelSelector: labelhelper.AlertLabelKey,
	}
	alterNode, err := e.remoteClient.ListNodes(alterNodeFilter)
	if err != nil {
		return fmt.Errorf("failed to list nodes: %w", err)
	}
	if len(alterNode.Items) > 0 {
		return
	}

	workerNodeFilter := &metav1.ListOptions{
		LabelSelector: labelhelper.NodeRoleNodeLabel,
	}

	nodes, err := e.remoteClient.ListNodes(workerNodeFilter)
	if err != nil {
		e.Ctx.Log.Warn("SetAlertLabelFailed", "failed to list nodes, err: %v", err)
		return
	}
	if len(nodes.Items) == 0 {
		e.Ctx.Log.Debug("(ignore)no worker role node found, skip set alert label")
		return
	}
	availableNode := &nodes.Items[0]
	clientSet := e.kubeClient()
	err = phaseutil.RetryOnConflict(func() error {
		latestNode, err := clientSet.CoreV1().Nodes().Get(e.Ctx, availableNode.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		labelhelper.SetLabel(latestNode, labelhelper.AlertLabelKey, labelhelper.AlertLabelValue)
		_, err = clientSet.CoreV1().Nodes().Update(e.Ctx, latestNode, metav1.UpdateOptions{})
		return err
	})
	if err != nil {
		return fmt.Errorf("failed to set alert label to node %s, err: %w", availableNode.Name, err)
	}
	return
}

func (e *EnsureCluster) setBareMetalLabel() error {
	nodes, err := e.remoteClient.ListNodes(nil)
	if err != nil {
		return err
	}
	clientSet := e.kubeClient()
	for i := range nodes.Items {
		node := &nodes.Items[i]
		if !labelhelper.HasLabel(node, labelhelper.BareMetalLabelKey) {
			if updateErr := phaseutil.RetryOnConflict(func() error {
				latestNode, err := clientSet.CoreV1().Nodes().Get(e.Ctx, node.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				labelhelper.SetLabel(latestNode, labelhelper.BareMetalLabelKey, "true")
				_, err = clientSet.CoreV1().Nodes().Update(e.Ctx, latestNode, metav1.UpdateOptions{})
				return err
			}); updateErr != nil {
				e.Ctx.Log.Warn("SetBareMetalLabelFailed",
					"(ignore)failed to set baremetal label to node %s, err: %v: ",
					node.Name, updateErr)
				continue
			}
		}
	}
	return nil
}

// ensureK8sToken ensure remote cluster k8s token
func (e *EnsureCluster) ensureK8sToken() error {
	phaseContextFields288 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, _ := phaseContextFields288.Context,
		phaseContextFields288.Client,
		phaseContextFields288.BKECluster,
		phaseContextFields288.Scheme,
		phaseContextFields288.Log
	var secret *v1.Secret
	if err := phaseutil.RetryOnConflict(func() error {
		var err error
		secret, err = phaseutil.GetK8sTokenSecret(ctx, c, bkeCluster)
		if err != nil {
			return err
		}
		if secret.OwnerReferences == nil || len(secret.OwnerReferences) == 0 {
			if err = controllerutil.SetControllerReference(bkeCluster, secret, scheme); err != nil {
				return err
			}
			if err = c.Update(ctx, secret); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		if strings.Contains(err.Error(), "not found") {
			token, err := e.remoteClient.NewK8sToken()
			if err = phaseutil.NewK8sTokenSecret(ctx, token, c, bkeCluster); err != nil {
				return fmt.Errorf("failed to create BKECluster %q remote cluster k8sToken: %w", utils.ClientObjNS(bkeCluster), err)
			}
			condition.ConditionMark(bkeCluster, "k8sTokenCreated", confv1beta1.ConditionTrue, "k8sTokenCreated", "")
			return nil
		}
		return fmt.Errorf("failed to get BKECluster %q remote cluster k8sToken secret: %w", utils.ClientObjNS(bkeCluster), err)
	}

	if secret == nil {
		return fmt.Errorf("k8sToken secret is nil for BKECluster %q", utils.ClientObjNS(bkeCluster))
	}
	if v, ok := secret.Data["token"]; !ok || string(v) == "" {
		token, err := e.remoteClient.NewK8sToken()
		if err != nil {
			return fmt.Errorf("failed to create BKECluster %q remote cluster k8sToken: %w", utils.ClientObjNS(bkeCluster), err)
		}
		if err = phaseutil.NewK8sTokenSecret(ctx, token, c, bkeCluster); err != nil {
			return fmt.Errorf("failed to create BKECluster %q remote cluster k8sToken: %w", utils.ClientObjNS(bkeCluster), err)
		}
	}
	condition.ConditionMark(bkeCluster, "k8sTokenCreated", confv1beta1.ConditionTrue, "k8sTokenCreated", "")
	return nil
}

// ensureRemoteBKEConfigCM
// Deprecated bkeconfig cm will be created before deploy addon cluster-api and bocoperator
func (e *EnsureCluster) ensureRemoteBKEConfigCM() error {
	phaseContextFields336 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, _ := phaseContextFields336.Context,
		phaseContextFields336.Client,
		phaseContextFields336.BKECluster,
		phaseContextFields336.Scheme,
		phaseContextFields336.Log

	clientSet, _ := e.remoteClient.KubeClient()
	config, err := phaseutil.GetRemoteBKEConfigCM(ctx, clientSet)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster bke-config cm: %w",
			utils.ClientObjNS(bkeCluster), err)
	}
	if config == nil {
		if err = phaseutil.MigrateBKEConfigCM(ctx, c, clientSet); err != nil {
			return fmt.Errorf("failed to migrate BKECluster %q bke-config cm to remote cluster: %w",
				utils.ClientObjNS(bkeCluster), err)
		}
	}
	return nil
}

// ensureRemoteHealthCheckConfigCM 在检查目标集群健康状态前同步健康检查配置。
func (e *EnsureCluster) ensureRemoteHealthCheckConfigCM() error {
	phaseContextFields355 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields355.Context,
		phaseContextFields355.Client,
		phaseContextFields355.BKECluster,
		phaseContextFields355.Scheme,
		phaseContextFields355.Log

	clientSet, _ := e.remoteClient.KubeClient()
	config, err := phaseutil.GetRemoteHealthCheckConfigCM(ctx, clientSet)
	if err != nil {
		return fmt.Errorf("failed to get BKECluster %q remote cluster health-check-config cm: %w",
			utils.ClientObjNS(bkeCluster), err)
	}
	if config == nil {
		log.Info("HealthCheckConfigSync", "remote health-check-config cm missing, migrate from management cluster")
		if err = phaseutil.MigrateHealthCheckConfigCM(ctx, c, clientSet); err != nil {
			return fmt.Errorf("failed to migrate BKECluster %q health-check-config cm to remote cluster: %w",
				utils.ClientObjNS(bkeCluster), err)
		}
		log.Info("HealthCheckConfigSync", "remote health-check-config cm migrated successfully")
		return nil
	}
	log.Info("HealthCheckConfigSync", "remote health-check-config cm already exists")
	return nil
}

// ensureClusterReady check cluster health status
func (e *EnsureCluster) ensureClusterReady() error {
	phaseContextFields378 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields378.Context,
		phaseContextFields378.Client,
		phaseContextFields378.BKECluster,
		phaseContextFields378.Scheme,
		phaseContextFields378.Log
	// 在首次部署但未完成之前，不检查
	bkeNodes, _ := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(ctx, bkeCluster)
	if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterInitializing && !phaseutil.ClusterEndDeployedWithContext(ctx, c, e.Ctx.Cluster, bkeCluster, bkeNodes) {
		return errors.Errorf("cluster %s is deploying, can not check health", bkeCluster.Name)
	}

	err := e.runHealthChecks(ctx, c, bkeCluster, log)

	return e.handleClusterReadyPostCheck(bkeCluster, c, log, err)
}

func (e *EnsureCluster) runHealthChecks(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	for i := 0; i < 3; i++ {
		if err := e.performHealthCheck(ctx, c, bkeCluster, log); err != nil {
			return err
		}
	}

	log.Finish(constant.ClusterReadyReason, "cluster is ready")
	return nil
}

// handleClusterReadyPostCheck handles operations after ensureClusterReady checks
func (e *EnsureCluster) handleClusterReadyPostCheck(bkeCluster *bkev1beta1.BKECluster, c client.Client, log *bkev1beta1.BKELogger, err error) error {
	metricrecord.ClusterHealthyCountRecord(bkeCluster, bkeCluster.Status.ClusterStatus)

	trackerBkeNodes, trackerErr := e.Ctx.GetBKENodes()
	if trackerErr != nil {
		log.Warn(constant.InternalErrorReason, "failed to get BKENodes for tracker check: %v", trackerErr)
	}
	if phaseutil.ClusterAllowTrackerWithBKENodes(trackerBkeNodes, e.Ctx.Cluster) {
		condition.ConditionMark(bkeCluster, bkev1beta1.TargetClusterReadyCondition, confv1beta1.ConditionTrue, "", "")
	} else {
		condition.ConditionMark(bkeCluster, bkev1beta1.TargetClusterReadyCondition, confv1beta1.ConditionFalse, "", "")
	}
	// 移除追踪器添加的触发调谐的注解
	_, ok := annotation.HasAnnotation(bkeCluster, annotation.ClusterTrackerHealthyCheckFailedAnnotationKey)
	if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterStatusReady && ok {
		patchFunc := func(combinedBKECluster *bkev1beta1.BKECluster) {
			annotation.RemoveAnnotation(combinedBKECluster, annotation.ClusterTrackerHealthyCheckFailedAnnotationKey)
		}
		return mergecluster.SyncStatusUntilComplete(c, bkeCluster, patchFunc)
	}
	return err
}

// performHealthCheck performs a single health check iteration
func (e *EnsureCluster) performHealthCheck(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	rawNodes, err := e.Ctx.NodeFetcher().GetBKENodesForBKECluster(e.Ctx.Context, bkeCluster)
	if err != nil {
		return fmt.Errorf("failed to get nodes for cluster health check: %w", err)
	}
	bkeNodes := bkev1beta1.NewBKENodes(rawNodes)
	if err := e.remoteClient.CheckClusterHealth(bkeCluster, bkeCluster.Status.KubernetesVersion, bkeNodes); err != nil {
		phaseutil.SetClusterStatus(bkeCluster, bkev1beta1.ClusterUnhealthy)

		if updateErr := mergecluster.UpdateModifiedBKENodes(ctx, c, bkeNodes); updateErr != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update BKENode status: %v", updateErr)
		}
		return fmt.Errorf("CheckClusterHealth failed: %w", err)
	}

	// Update BKENode status after successful health check
	if err := mergecluster.UpdateModifiedBKENodes(ctx, c, bkeNodes); err != nil {
		log.Warn(constant.InternalErrorReason, "Failed to update BKENode status: %v", err)
	}

	e.updateClusterVersionStatus(bkeCluster)
	if err := e.initializeConfigHashes(bkeCluster); err != nil {
		return err
	}
	shouldSyncCV, err := clusterversion.ShouldSyncClusterVersionInstallStatus(ctx, c, bkeCluster)
	if err != nil {
		log.Warn(constant.InternalErrorReason, "check ClusterVersion install sync failed: %v", err)
	}
	if err == nil && shouldSyncCV {
		if err := clusterversion.SyncClusterVersionInstallStatus(ctx, c, bkeCluster); err != nil {
			log.Warn(constant.InternalErrorReason, "sync ClusterVersion install status failed: %v", err)
		}
	}

	phaseutil.SetClusterStatus(bkeCluster, bkev1beta1.ClusterStatusReady)

	if err := e.Report("", false); err != nil {
		return fmt.Errorf("ensureCluster report failed: %w", err)
	}
	return nil
}

// updateClusterVersionStatus updates version fields in cluster status if empty
func (e *EnsureCluster) updateClusterVersionStatus(bkeCluster *bkev1beta1.BKECluster) {
	if bkeCluster.Status.KubernetesVersion == "" {
		bkeCluster.Status.KubernetesVersion = bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion
	}
	if bkeCluster.Status.EtcdVersion == "" {
		bkeCluster.Status.EtcdVersion = bkeCluster.Spec.ClusterConfig.Cluster.EtcdVersion
	}
	if bkeCluster.Status.OpenFuyaoVersion == "" {
		bkeCluster.Status.OpenFuyaoVersion = bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion
	}
	if bkeCluster.Status.ContainerdVersion == "" {
		bkeCluster.Status.ContainerdVersion = bkeCluster.Spec.ClusterConfig.Cluster.ContainerdVersion
	}
	// Freshly installed nginx clusters get no status stamp anywhere else: the
	// EnsureNginxUpgrade phase only records the version after an upgrade, so
	// backfill it here to give the declarative upgrade a current baseline.
	if clusterutil.LoadBalanceType(bkeCluster.Spec.ClusterConfig) == "nginx" && bkeCluster.Status.NginxProxyVersion == "" {
		bkeCluster.Status.NginxProxyVersion = bkeCluster.Spec.ClusterConfig.Cluster.NginxProxyVersion
	}
}

// initializeConfigHashes 初始化 Status 中的配置 hash 字段。
// 当 Status.ControlPlaneConfigHash / Status.EtcdConfigHash 为空时，
// 用当前 Spec 计算并回写，作为后续配置变更检测的基线。
// 这样新部署的集群在部署完成后即拥有正确的基线 hash，
// 后续用户修改配置时 EnsureMasterConfigUpdate / EnsureEtcdConfigUpdate
// 才能准确检测到差异并触发更新。
// 任一 hash 计算失败（如配置序列化错误）时返回 error，由上层调谐快速重新入队重试，
// 避免基线 hash 缺失导致后续配置变更检测失效。
func (e *EnsureCluster) initializeConfigHashes(bkeCluster *bkev1beta1.BKECluster) error {
	cluster := &bkeCluster.Spec.ClusterConfig.Cluster

	// 初始化控制面配置 hash（APIServer / ControllerManager / Scheduler）
	if bkeCluster.Status.ControlPlaneConfigHash == (confv1beta1.ComponentConfigHash{}) {
		desiredHash, err := phaseutil.CalculateComponentConfigHash(cluster)
		if err != nil {
			return fmt.Errorf("calculate component config hash failed: %w", err)
		}
		if desiredHash != (confv1beta1.ComponentConfigHash{}) {
			bkeCluster.Status.ControlPlaneConfigHash = desiredHash
		}
	}

	// 初始化 etcd 配置 hash
	if bkeCluster.Status.EtcdConfigHash == "" {
		desiredHash, err := phaseutil.CalculateEtcdConfigHash(cluster)
		if err != nil {
			return fmt.Errorf("calculate etcd config hash failed: %w", err)
		}
		if desiredHash != "" {
			bkeCluster.Status.EtcdConfigHash = desiredHash
		}
	}
	return nil
}

// ensureAgentStatus check bkeagent status
func (e *EnsureCluster) ensureAgentStatus() error {
	phaseContextFields518 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields518.Context,
		phaseContextFields518.Client,
		phaseContextFields518.BKECluster,
		phaseContextFields518.Scheme,
		phaseContextFields518.Log
	if condition.HasConditionStatus(bkev1beta1.SwitchBKEAgentCondition, bkeCluster, confv1beta1.ConditionTrue) {
		log.Info(constant.BKEAgentUnknownReason, "unknown bkeagent status, already switch bkeagent, skip check bkeagent status")
		return nil
	}

	if bkeCluster.Status.AgentStatus.Replies != 0 {
		err, _, failedNodes := phaseutil.PingBKEAgent(ctx, c, scheme, bkeCluster)
		if err != nil {
			return fmt.Errorf("failed to ping BKEAgent: %w", err)
		}

		if len(failedNodes) != 0 {
			return fmt.Errorf("failed to ping bkeagent on flow Nodes: %v", failedNodes)
		}
		log.Info(constant.BKEAgentReadyReason, "BKEAgent is ready")
		return nil
	}
	return nil
}

// setNodeLabel is the main function that coordinates the process of setting node labels.
func (e *EnsureCluster) setNodeLabel() error {
	globalLabels := e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.Labels
	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}
	nodes, err := e.remoteClient.ListNodes(nil)
	if err != nil {
		return fmt.Errorf("failed to list nodes: %w", err)
	}

	setNodeLablesMap := e.buildNodeLabelsMap(globalLabels, allNodes)

	clientSet := e.kubeClient()
	for i := range nodes.Items {
		node := &nodes.Items[i]
		if err = e.applyLabelsToNode(clientSet, node, setNodeLablesMap); err != nil {
			return err
		}
	}

	return nil
}

// buildNodeLabelsMap builds a map of node labels to apply, merging global labels.
func (e *EnsureCluster) buildNodeLabelsMap(
	globalLabels []confv1beta1.Label,
	setNodeLabels []confv1beta1.Node,
) map[string]map[string]string {
	setNodeLablesMap := make(map[string]map[string]string)
	for _, node := range setNodeLabels {
		labelMap := mergeLabels(node.Labels, globalLabels)
		if len(labelMap) > 0 {
			setNodeLablesMap[node.Hostname] = labelMap
		}
	}
	return setNodeLablesMap
}

// mergeLabels merges the node-specific labels with the global labels.
func mergeLabels(nodeLabels []confv1beta1.Label, globalLabels []confv1beta1.Label) map[string]string {
	labelMap := make(map[string]string)
	for _, label := range nodeLabels {
		labelMap[label.Key] = label.Value
	}
	for _, label := range globalLabels {
		if _, ok := labelMap[label.Key]; !ok {
			labelMap[label.Key] = label.Value
		}
	}
	return labelMap
}

// applyLabelsToNode applies the labels to a given node if necessary.
func (e *EnsureCluster) applyLabelsToNode(
	clientSet kubernetes.Interface,
	node *v1.Node,
	setNodeLabelsMap map[string]map[string]string,
) error {
	labels, found := getNodeLabels(node.Name, setNodeLabelsMap)
	if !found {
		return nil
	}
	return e.applyNecessaryLabels(clientSet, node, labels)
}

// getNodeLabels retrieves the labels for a given node from the label map.
func getNodeLabels(nodeName string, setNodeLabelsMap map[string]map[string]string) (map[string]string, bool) {
	labels, ok := setNodeLabelsMap[nodeName]
	return labels, ok
}

// applyNecessaryLabels checks and applies labels to the node if necessary.
func (e *EnsureCluster) applyNecessaryLabels(
	clientSet kubernetes.Interface,
	node *v1.Node,
	labels map[string]string,
) error {
	for key, value := range labels {
		if !labelhelper.IsLabelEqual(node, key, value) {
			if err := e.waitLabelReady(clientSet, node, key, value); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *EnsureCluster) waitLabelReady(clientSet kubernetes.Interface, node *v1.Node, k, v string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	err := wait.PollImmediateUntil(LabelReadyTimeout, func() (bool, error) {
		// Get the latest version of the node
		latestNode, err := clientSet.CoreV1().Nodes().Get(e.Ctx, node.Name, metav1.GetOptions{})
		if err != nil {
			// retry
			return false, fmt.Errorf("failed to get node %s: %w", node.Name, err)
		}

		// Create a deep copy of the node to avoid modifying the original object
		nodeCopy := latestNode.DeepCopy()

		// Set the label on the latest node copy
		labelhelper.SetLabel(nodeCopy, k, v)

		// Try to update the node with the latest version
		_, err = clientSet.CoreV1().Nodes().Update(e.Ctx, nodeCopy, metav1.UpdateOptions{
			FieldManager: "node-label-updater",
		})

		if err != nil {
			e.Ctx.Log.Warn("SetNodeLabelConFailed", "failed to set label %s=%s on node %s, error: %v", k, v, node.Name, err)
			return false, nil
		}

		e.Ctx.Log.Info("SetNodeLabelSuccess", "Successfully set label %s=%s on node %s", k, v, node.Name)
		return true, nil // Success

	}, ctx.Done())

	if err != nil {
		return fmt.Errorf("failed to set label %s=%s on node %s after retries: %w", k, v, node.Name, err)
	}

	return nil
}
