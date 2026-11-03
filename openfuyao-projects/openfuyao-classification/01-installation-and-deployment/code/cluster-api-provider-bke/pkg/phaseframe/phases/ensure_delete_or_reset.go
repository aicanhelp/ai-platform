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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	agentutils "gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureDeleteOrResetName confv1beta1.BKEClusterPhase = "EnsureDeleteOrReset"
	// DeleteOrResetTimeoutMinutes 控制删除或重置操作的超时时间（分钟）
	DeleteOrResetTimeoutMinutes = 5
	// DeleteOrResetPollIntervalSeconds 控制删除或重置操作的轮询间隔（秒）
	DeleteOrResetPollIntervalSeconds = 10
)

type EnsureDeleteOrReset struct {
	phaseframe.BasePhase
}

func NewEnsureDeleteOrReset(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureDeleteOrResetName)
	p := &EnsureDeleteOrReset{BasePhase: base}
	p.RegisterPostHooks(ensureDeleteOrResetPostHook)
	return p
}

func (e *EnsureDeleteOrReset) Execute() (ctrl.Result, error) {
	// 不使用PhaseCtx.Context，因为此时已经被cancel了
	baseCtx := context.Background()
	phaseContextFields76 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields76.Context,
		phaseContextFields76.Client,
		phaseContextFields76.BKECluster,
		phaseContextFields76.Scheme,
		phaseContextFields76.Log

	if e.Ctx.BKECluster.Spec.Pause {
		log.Info(constant.ClusterDeletingReason, "BKECluster %s is paused, resume it first", utils.ClientObjNS(e.Ctx.BKECluster))
		patchF := func(currentCombinedBKECluster *bkev1beta1.BKECluster) {
			currentCombinedBKECluster.Spec.Pause = false
		}
		if err := mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster, patchF); err != nil {
			return ctrl.Result{}, err
		}
		if err := e.Ctx.RefreshCtxBKECluster(baseCtx); err != nil {
			return ctrl.Result{}, err
		}

		e.removeCommandsForPausedCluster(baseCtx, c, bkeCluster, log)

		return ctrl.Result{Requeue: true}, nil
	}

	ctx, cancel := context.WithTimeout(baseCtx, DeleteOrResetTimeoutMinutes*time.Minute)
	defer cancel()
	maxAttempts := int(
		(DeleteOrResetTimeoutMinutes * time.Minute) /
			(DeleteOrResetPollIntervalSeconds * time.Second),
	)
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	attempt := 0
	err := wait.PollImmediateUntil(DeleteOrResetPollIntervalSeconds*time.Second, func() (bool, error) {
		attempt++
		if err := e.reconcileDelete(ctx); err != nil {
			log.Warn("RetryDelete", "reconcileDelete failed (attempt %d/%d): %v", attempt, maxAttempts, err)
			return false, nil
		}
		return true, nil
	}, ctx.Done())

	if errors.Is(err, wait.ErrWaitTimeout) {
		return ctrl.Result{}, errors.Errorf("Wait delete timeout")
	}
	return ctrl.Result{}, nil
}

func (e *EnsureDeleteOrReset) removeCommandsForPausedCluster(
	ctx context.Context,
	c client.Client,
	bkeCluster *bkev1beta1.BKECluster,
	log *bkev1beta1.BKELogger,
) {
	log.Info(constant.ClusterDeletingReason, "cluster is paused, remove all agent command now")
	commandList := &agentv1beta1.CommandList{}
	filters := phaseutil.GetListFiltersByBKECluster(bkeCluster)
	if err := c.List(ctx, commandList, filters...); err != nil {
		log.Warn(constant.ReconcileErrorReason, "failed to list command: %v", err)
		return
	}
	for i := range commandList.Items {
		cmd := &commandList.Items[i]
		if err := command.PurgeCommand(ctx, c, cmd); err != nil && !apierrors.IsNotFound(err) {
			log.Warn(constant.ReconcileErrorReason, "failed to purge command %s: %v", cmd.GetName(), err)
		}
	}
}

func (e *EnsureDeleteOrReset) NeedExecute(_ *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !new.DeletionTimestamp.IsZero() || new.Spec.Reset {
		e.SetStatus(bkev1beta1.PhaseWaiting)
		return true
	}
	return false
}

func (e *EnsureDeleteOrReset) reconcileDelete(ctx context.Context) error {
	phaseContextFields153 := e.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields153.Context,
		phaseContextFields153.Client,
		phaseContextFields153.BKECluster,
		phaseContextFields153.Scheme,
		phaseContextFields153.Log

	if err := e.Ctx.RefreshCtxCluster(ctx); err != nil {
		if !strings.Contains(err.Error(), "owner cluster is nil") {
			return err
		}
	}

	if err := e.ensureClusterStatusDeleting(c, bkeCluster, log); err != nil {
		return err
	}

	if err := e.handleClusterDeletion(ctx, c, log); err != nil {
		return err
	}

	if err := e.handleBKEMachineDeletion(ctx, c, bkeCluster, log); err != nil {
		return err
	}

	if err := e.deleteRelatedResources(ctx, c, bkeCluster, log); err != nil {
		return err
	}

	if err := e.cleanupClusterResources(ctx, c, bkeCluster, log); err != nil {
		return err
	}

	return e.handleNamespaceDeletion(ctx, c, bkeCluster, log)
}

// ensureClusterStatusDeleting 确保集群状态为删除中
func (e *EnsureDeleteOrReset) ensureClusterStatusDeleting(c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	if bkeCluster.Status.ClusterStatus != bkev1beta1.ClusterDeleting {
		log.Debug("mark bkeCluster as deleting")
		phaseutil.SetClusterStatus(bkeCluster, bkev1beta1.ClusterDeleting)
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			return fmt.Errorf("failed to update bkeCluster Status: %w", err)
		}
	}
	return nil
}

// handleClusterDeletion 处理集群删除
func (e *EnsureDeleteOrReset) handleClusterDeletion(ctx context.Context, c client.Client, log *bkev1beta1.BKELogger) error {
	if e.Ctx.Cluster != nil && e.Ctx.Cluster.Status.Phase != string(clusterv1.ClusterPhaseDeleting) {
		log.Debug("delete relation cluster-api obj cluster")
		// delete cluster api obj cluster will delete all relation obj
		if err := c.Delete(ctx, e.Ctx.Cluster); err != nil {
			return fmt.Errorf("failed to delete cluster: %w", err)
		}
		// return now, Wait for the deletion of cluster api obj to trigger the deletion of bkeCluster
		return errors.New("wait for the deletion of cluster api obj to trigger the deletion of bkeCluster")
	}
	return nil
}

// handleBKEMachineDeletion 处理BKEMachine删除
func (e *EnsureDeleteOrReset) handleBKEMachineDeletion(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	// need wait all bkeMachine delete
	log.Debug("start delete bkeMachine process")
	bkeMachines := &bkev1beta1.BKEMachineList{}
	if err := c.List(ctx, bkeMachines, client.InNamespace(bkeCluster.Namespace)); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to list bkeMachine: %w", err)
		}
		return fmt.Errorf("failed to list bkeMachine: %w", err)
	}
	if len(bkeMachines.Items) == 0 {
		log.Debug("all bkeMachine deleted")
		return nil
	}
	for i := range bkeMachines.Items {
		if err := deleteOrDetachBKEMachine(ctx, c, &bkeMachines.Items[i]); err != nil {
			return err
		}
	}
	return errors.New("wait for bkeMachine delete")
}

func deleteOrDetachBKEMachine(ctx context.Context, c client.Client, bkeMachine *bkev1beta1.BKEMachine) error {
	if !bkeMachine.Status.Bootstrapped {
		if err := c.Delete(ctx, bkeMachine); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	if len(bkeMachine.OwnerReferences) != 0 {
		return nil
	}
	return phaseutil.RetryOnConflict(func() error {
		latestBKEMachine := &bkev1beta1.BKEMachine{}
		if err := c.Get(ctx, client.ObjectKeyFromObject(bkeMachine), latestBKEMachine); err != nil {
			return err
		}
		controllerutil.RemoveFinalizer(latestBKEMachine, bkev1beta1.BKEMachineFinalizer)
		return c.Update(ctx, latestBKEMachine)
	})
}

// deleteRelatedResources 删除相关资源
func (e *EnsureDeleteOrReset) deleteRelatedResources(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	// remove all secret type == bke.bocloud.com/secret
	log.Debug("start delete related secret resource")
	if err := c.DeleteAllOf(ctx, &corev1.Secret{}, client.InNamespace(bkeCluster.Namespace), client.MatchingFields{"type": string(agentutils.BKESecretType)}); err != nil {
		log.Warn(constant.ReconcileErrorReason, "failed to delete secret: %v", err)
	}

	// remove all agent Command
	log.Debug("start delete related agent command resource")
	commandList := &agentv1beta1.CommandList{}
	if err := c.List(ctx, commandList, client.InNamespace(bkeCluster.Namespace)); err != nil {
		log.Warn(constant.ReconcileErrorReason, "failed to list command: %v", err)
	} else {
		for i := range commandList.Items {
			cmd := &commandList.Items[i]
			if err := command.PurgeCommand(ctx, c, cmd); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				log.Warn(constant.ReconcileErrorReason, "failed to purge command %s: %v", cmd.GetName(), err)
			}
		}
	}
	return nil
}

// cleanupClusterResources 清理集群资源
func (e *EnsureDeleteOrReset) cleanupClusterResources(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	log.Debug("start delete associated BKENode resources")
	if err := markBKENodesDeleting(ctx, c, bkeCluster, log); err != nil {
		return err
	}
	if err := deleteBKENodes(ctx, c, bkeCluster, log); err != nil {
		return err
	}

	if bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		host := fmt.Sprintf("https://%s", bkeCluster.Spec.ControlPlaneEndpoint.String())
		log.Debug("delete RESTMapper cache for target cluster host %s", host)
		kube.DeleteRESTMapperCacheByHost(host)
	}

	log.Debug("remove bkeCluster finalizer")
	controllerutil.RemoveFinalizer(bkeCluster, bkev1beta1.ClusterFinalizer)
	// maybe we not need to do anything
	log.Finish(constant.TargetClusterDeletedReason, "bkeCluster deleted successfully")
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
		return fmt.Errorf("failed to update bkeCluster Status: %w", err)
	}
	// remove all event created by bkeCluster
	log.Debug("remove all event created by bkeCluster")
	eventFilters := []client.DeleteAllOfOption{
		client.MatchingFieldsSelector{Selector: fields.AndSelectors(
			fields.OneTermEqualSelector("involvedObject.name", bkeCluster.Name),
			fields.OneTermEqualSelector("involvedObject.namespace", bkeCluster.Namespace),
		)},
		client.InNamespace(bkeCluster.Namespace),
	}
	if err := c.DeleteAllOf(ctx, &corev1.Event{}, eventFilters...); err != nil {
		log.Warn(constant.ReconcileErrorReason, "failed to delete event: %v", err)
	}
	return nil
}

func markBKENodesDeleting(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster,
	log *bkev1beta1.BKELogger) error {
	bnList := &confv1beta1.BKENodeList{}
	if err := c.List(ctx, bnList, bkeNodeListOptions(bkeCluster)...); err != nil {
		return err
	}
	for i := range bnList.Items {
		bn := &bnList.Items[i]
		if bn.Status.State == confv1beta1.NodeDeleting {
			continue
		}
		base := bn.DeepCopy()
		bn.Status.State = confv1beta1.NodeDeleting
		bn.Status.Message = "Cluster is being deleted"
		if err := c.Status().Patch(ctx, bn, client.MergeFrom(base)); err != nil && !apierrors.IsNotFound(err) {
			log.Warn(constant.ReconcileErrorReason, "failed to set BKENode %s to Deleting state: %v", bn.Name, err)
		}
	}
	return nil
}

func deleteBKENodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster,
	log *bkev1beta1.BKELogger) error {
	err := c.DeleteAllOf(ctx, &confv1beta1.BKENode{}, bkeNodeDeleteOptions(bkeCluster)...)
	switch {
	case err == nil, apierrors.IsNotFound(err):
		return nil
	case apierrors.IsForbidden(err):
		log.Warn(constant.ReconcileErrorReason,
			"deletecollection forbidden, fallback to delete BKENodes individually: %v", err)
		return deleteBKENodesIndividually(ctx, c, bkeCluster)
	default:
		return err
	}
}

func deleteBKENodesIndividually(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) error {
	bnList := &confv1beta1.BKENodeList{}
	if err := c.List(ctx, bnList, bkeNodeListOptions(bkeCluster)...); err != nil {
		return err
	}
	for i := range bnList.Items {
		err := c.Delete(ctx, &bnList.Items[i])
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func bkeNodeListOptions(bkeCluster *bkev1beta1.BKECluster) []client.ListOption {
	return []client.ListOption{
		client.InNamespace(bkeCluster.Namespace),
		client.MatchingLabels{"cluster.x-k8s.io/cluster-name": bkeCluster.Name},
	}
}

func bkeNodeDeleteOptions(bkeCluster *bkev1beta1.BKECluster) []client.DeleteAllOfOption {
	return []client.DeleteAllOfOption{
		client.InNamespace(bkeCluster.Namespace),
		client.MatchingLabels{"cluster.x-k8s.io/cluster-name": bkeCluster.Name},
	}
}

// handleNamespaceDeletion 处理命名空间删除
func (e *EnsureDeleteOrReset) handleNamespaceDeletion(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, log *bkev1beta1.BKELogger) error {
	// 不存在DeleteIgnoreNamespaceAnnotationKey注解，或者有DeleteIgnoreNamespaceAnnotationKey注解，且值为false不忽略，则删除namespace
	if v, ok := annotation.HasAnnotation(bkeCluster, annotation.DeleteIgnoreNamespaceAnnotationKey); ok && v == "false" {
		// remove ns if not have other bkeCluster
		bkeClusters := &bkev1beta1.BKEClusterList{}
		if err := c.List(ctx, bkeClusters, client.InNamespace(bkeCluster.Namespace)); err != nil {
			log.Warn(constant.ReconcileErrorReason, "failed to list bkeCluster: %v", err)
		}
		// 没有其他bkeCluster了，且有DeleteIgnoreNamespaceAnnotationKey注解，且值为false不忽略，则删除namespace
		if len(bkeClusters.Items) == 0 {
			ns := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: bkeCluster.Namespace,
				},
			}
			log.Info("DeleteNamespace", "remove namespace %q", ns.Name)
			if err := c.Delete(ctx, ns); err != nil {
				log.Warn(constant.ReconcileErrorReason, "failed to delete namespace: %v", err)
			}
		}
	} else {
		if !ok {
			log.Info("DeleteNamespace", "no %s annotation, ignore delete namespace", annotation.DeleteIgnoreNamespaceAnnotationKey)
		}
		if ok && v == "true" {
			log.Info("DeleteNamespace", "%s = %s, ignore delete namespace", annotation.DeleteIgnoreNamespaceAnnotationKey, v)
		}
	}
	// 清理status cache
	statusmanage.DefaultStatusManager().RemoveBKEClusterStatusCache(bkeCluster)

	return nil
}

func ensureDeleteOrResetPostHook(p phaseframe.Phase, err error) error {
	if err == nil {
		// remove bkeCluster metrics
		bkemetrics.DefaultRegister().Unregister(utils.ClientObjNS(p.GetPhaseContext().BKECluster))
	}
	return nil
}
