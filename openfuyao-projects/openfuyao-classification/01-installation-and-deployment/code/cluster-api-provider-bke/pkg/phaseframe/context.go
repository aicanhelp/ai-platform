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

package phaseframe

import (
	"context"
	"sync"
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type PhaseContext struct {
	BKECluster *bkev1beta1.BKECluster
	Cluster    *clusterv1.Cluster
	client.Client
	context.Context
	Log        *bkev1beta1.BKELogger
	Scheme     *runtime.Scheme
	RestConfig *rest.Config
	Cache      cache.Cache // Informer cache; pass to client wrappers that need write-through
	cancelFunc context.CancelFunc

	mux            sync.RWMutex
	nodeFetcher    *nodeutil.NodeFetcher
	VersionContext *upgrade.VersionContext

	// DeclarativeDAGCompleted is set when the declarative upgrade DAG finished successfully
	// in the same reconcile; PhaseFlow must not re-run inline upgrade phases from PostDeploy.
	DeclarativeDAGCompleted bool
}

func NewReconcilePhaseCtx(ctx context.Context) *PhaseContext {
	phaseCancelCtx, phaseCancel := context.WithCancel(ctx)
	return &PhaseContext{
		Context:    phaseCancelCtx,
		cancelFunc: phaseCancel,
		mux:        sync.RWMutex{},
	}
}

func (pc *PhaseContext) SetBKECluster(bkeCluster *bkev1beta1.BKECluster) *PhaseContext {
	pc.mux.Lock()
	pc.BKECluster = bkeCluster
	pc.mux.Unlock()
	return pc
}

func (pc *PhaseContext) SetCluster(cluster *clusterv1.Cluster) *PhaseContext {
	pc.mux.Lock()
	pc.Cluster = cluster
	pc.mux.Unlock()
	return pc
}

func (pc *PhaseContext) SetClient(client client.Client) *PhaseContext {
	pc.Client = client
	return pc
}

// SetCache attaches the Informer cache to the phase context. Downstream code
// that needs write-through caching (e.g. the cert generator) reads this value
// and wraps the controller-runtime client accordingly. Passing nil is safe
// and means "no write-through" (the previous behaviour).
func (pc *PhaseContext) SetCache(c cache.Cache) *PhaseContext {
	pc.Cache = c
	return pc
}

func (pc *PhaseContext) SetLogger(log *bkev1beta1.BKELogger) *PhaseContext {
	pc.Log = log
	return pc
}

// BindPhaseLogger attaches structured phase fields to the context logger.
func (pc *PhaseContext) BindPhaseLogger(phaseName confv1beta1.BKEClusterPhase) {
	if pc == nil || pc.Log == nil {
		return
	}
	pc.Log.SetNormalLogger(
		log.With("phase", phaseName.String()).
			With("bkecluster", utils.ClientObjNS(pc.BKECluster)),
	)
}

func (pc *PhaseContext) SetScheme(scheme *runtime.Scheme) *PhaseContext {
	pc.Scheme = scheme
	return pc
}

func (pc *PhaseContext) SetRestConfig(restConfig *rest.Config) *PhaseContext {
	pc.RestConfig = restConfig
	return pc
}

// SetVersionContext attaches a version context for declarative upgrade decisions.
func (pc *PhaseContext) SetVersionContext(vc *upgrade.VersionContext) *PhaseContext {
	pc.VersionContext = vc
	return pc
}

// BuildAndSetVersionContext builds VersionContext from BKECluster and attaches it.
// Prefer BuildAndSetVersionContextFromBundle during declarative upgrade when a release bundle is available.
func (pc *PhaseContext) BuildAndSetVersionContext() *PhaseContext {
	if pc.VersionContext != nil {
		return pc
	}
	if pc.BKECluster == nil {
		return pc
	}
	pc.VersionContext = upgrade.BuildVersionContextFromBKECluster(pc.BKECluster)
	return pc
}

// BuildAndSetVersionContextFromBundle builds VersionContext from ReleaseImage bundle(s) and attaches it.
func (pc *PhaseContext) BuildAndSetVersionContextFromBundle(
	targetBundle, currentBundle *releasemanifest.Bundle,
) *PhaseContext {
	pc.VersionContext = upgrade.BuildVersionContextForUpgrade(targetBundle, currentBundle, pc.BKECluster)
	return pc
}

// FinishDeclarativeDAGForPhaseFlow clears stale VersionContext, refreshes BKECluster from the API,
// and marks that PostDeploy must not repeat declarative inline upgrade phases this reconcile.
func (pc *PhaseContext) FinishDeclarativeDAGForPhaseFlow() error {
	if pc == nil {
		return nil
	}
	pc.VersionContext = nil
	pc.DeclarativeDAGCompleted = true
	return pc.RefreshCtxBKECluster()
}

// PhaseContextFields is a concurrency-safe snapshot of the values shared by phases.
type PhaseContextFields struct {
	Context    context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Scheme     *runtime.Scheme
	Log        *bkev1beta1.BKELogger
}

// Untie returns a concurrency-safe snapshot of the fields shared by phases.
func (pc *PhaseContext) Untie() PhaseContextFields {
	pc.mux.RLock()
	defer pc.mux.RUnlock()
	return PhaseContextFields{
		Context:    pc.Context,
		Client:     pc.Client,
		BKECluster: pc.BKECluster,
		Scheme:     pc.Scheme,
		Log:        pc.Log,
	}
}

func (pc *PhaseContext) fetchNewestBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (*bkev1beta1.BKECluster, error) {
	if bkeCluster == nil {
		return nil, errors.New("BKECluster is nil")
	}
	return mergecluster.GetCombinedBKECluster(ctx, pc.Client, bkeCluster.Namespace, bkeCluster.Name)
}

func (pc *PhaseContext) GetNewestBKECluster(customCtx ...context.Context) (*bkev1beta1.BKECluster, error) {
	var getCtx context.Context
	pc.mux.RLock()
	if customCtx != nil && len(customCtx) != 0 {
		getCtx = customCtx[0]
	} else {
		getCtx = pc.Context
	}
	bkeCluster := pc.BKECluster
	pc.mux.RUnlock()
	return pc.fetchNewestBKECluster(getCtx, bkeCluster)
}

func (pc *PhaseContext) RefreshCtxBKECluster(customCtx ...context.Context) error {
	pc.mux.Lock()
	defer pc.mux.Unlock()

	var ctx context.Context
	if customCtx != nil && len(customCtx) != 0 {
		ctx = customCtx[0]
	} else {
		ctx = pc.Context
	}
	newBKECluster, err := pc.fetchNewestBKECluster(ctx, pc.BKECluster)
	if err != nil {
		return err
	}
	pc.BKECluster = newBKECluster
	return nil
}

func (pc *PhaseContext) RefreshCtxCluster(customCtx ...context.Context) error {
	var refreshCtx context.Context
	if customCtx != nil && len(customCtx) != 0 {
		refreshCtx = customCtx[0]
	} else {
		refreshCtx = pc.Context
	}
	err := pc.RefreshCtxBKECluster(refreshCtx)
	if err != nil {
		return err
	}
	pc.mux.RLock()
	bkeCluster := pc.BKECluster
	pc.mux.RUnlock()
	cluster, err := util.GetOwnerCluster(refreshCtx, pc.Client, bkeCluster.ObjectMeta)
	if err != nil {
		return errors.Wrapf(err, "failed to get owner cluster")
	}
	if cluster == nil {
		return errors.New("owner cluster is nil")
	}
	pc.mux.Lock()
	pc.Cluster = cluster
	pc.mux.Unlock()
	return nil
}

func (pc *PhaseContext) Cancel() {
	pc.cancelFunc()
}

func (pc *PhaseContext) WatchBKEClusterStatus() {
	refreshTicker := time.NewTicker(2 * time.Second)
	defer refreshTicker.Stop()
	pausedTicker := time.NewTicker(10 * time.Second)
	defer pausedTicker.Stop()

	pc.mux.RLock()
	if pc.BKECluster == nil {
		pc.mux.RUnlock()
		if pc.Log != nil {
			pc.Log.Warn("", "BKECluster is nil, cannot watch status")
		}
		return
	}
	pc.mux.RUnlock()

	bkeCluster, err := pc.GetNewestBKECluster()
	if err != nil {
		if pc.Log != nil {
			pc.Log.Warn(constant.ReconcileErrorReason, "failed to get newest BKECluster: %v", err)
		}
		return
	}

	select {
	case <-refreshTicker.C:
		cluster, ok := pc.refreshWatchedCluster()
		if !ok {
			return
		}
		bkeCluster = cluster

	case <-pausedTicker.C:
		bkeCluster = pc.currentBKEClusterOrDefault(bkeCluster)
		pc.logPausedRunningPhase(bkeCluster)

	case <-pc.Done():
		return
	default:
		bkeCluster = pc.currentBKEClusterOrDefault(bkeCluster)
		if pc.handleDeletingBKECluster(bkeCluster) {
			return
		}

	}
}

func (pc *PhaseContext) refreshWatchedCluster() (*bkev1beta1.BKECluster, bool) {
	cluster, err := pc.GetNewestBKECluster()
	if err != nil {
		return nil, false
	}
	return cluster, true
}

func (pc *PhaseContext) currentBKEClusterOrDefault(defaultCluster *bkev1beta1.BKECluster) *bkev1beta1.BKECluster {
	pc.mux.RLock()
	current := pc.BKECluster
	pc.mux.RUnlock()
	if current != nil {
		return current
	}
	return defaultCluster
}

func (pc *PhaseContext) logPausedRunningPhase(bkeCluster *bkev1beta1.BKECluster) {
	v, ok := annotation.HasAnnotation(bkeCluster, annotation.BKEClusterPauseAnnotationKey)
	flag := ok && v == "true"
	// 外部设置了暂停但是，还在运行phase，给个日志提示下吧
	if bkeCluster.Spec.Pause && !flag {
		// get running phase
		for _, phase := range bkeCluster.Status.PhaseStatus {
			if phase.Status == bkev1beta1.PhaseRunning && pc.Log != nil {
				pc.Log.Info(constant.PhaseRunningReason, "BKECluster is paused, but status %q is running, "+
					"waiting for phase to complete", bkeCluster.Status.ClusterStatus)
			}
		}
	}
}

func (pc *PhaseContext) handleDeletingBKECluster(bkeCluster *bkev1beta1.BKECluster) bool {
	if bkeCluster.DeletionTimestamp == nil || bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
		return false
	}
	// mark bkeCluster as deleting
	phaseutil.SetClusterStatus(bkeCluster, bkev1beta1.ClusterDeleting)
	if err := mergecluster.SyncStatusUntilComplete(pc.Client, bkeCluster); err != nil {
		if pc.Log != nil {
			pc.Log.Warn(constant.ReconcileErrorReason, "failed to update bkeCluster Status: %v", err)
		}
	}
	if pc.Log != nil {
		pc.Log.Info(constant.ClusterDeletingReason, "BKECluster is deleted, canceling phase context")
	}
	pc.Cancel()
	return true
}

// NodeFetcher 返回懒加载的 NodeFetcher 实例
func (pc *PhaseContext) NodeFetcher() *nodeutil.NodeFetcher {
	if pc.nodeFetcher == nil {
		pc.nodeFetcher = nodeutil.NewNodeFetcher(pc.Client)
	}
	return pc.nodeFetcher
}

// GetNodes 获取当前 BKECluster 关联的节点列表
func (pc *PhaseContext) GetNodes() (bkenode.Nodes, error) {
	return pc.NodeFetcher().GetNodesForBKECluster(pc.Context, pc.BKECluster)
}

// GetBKENodes 获取当前 BKECluster 关联的 BKENode 包装列表
func (pc *PhaseContext) GetBKENodes() (bkev1beta1.BKENodes, error) {
	return pc.NodeFetcher().GetBKENodesWrapperForCluster(pc.Context, pc.BKECluster)
}

// HasNodes 检查当前 BKECluster 是否有关联的节点
func (pc *PhaseContext) HasNodes() (bool, error) {
	count, err := pc.NodeFetcher().GetNodeCountForCluster(pc.Context, pc.BKECluster)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// GetNodeStateFlag 检查节点是否设置了指定标志
func (pc *PhaseContext) GetNodeStateFlag(ip string, flag int) (bool, error) {
	return pc.NodeFetcher().GetNodeStateFlagForCluster(pc.Context, pc.BKECluster, ip, flag)
}

// SetNodeStateWithMessage 设置节点状态和消息
func (pc *PhaseContext) SetNodeStateWithMessage(ip string, state confv1beta1.NodeState, message string) error {
	return pc.NodeFetcher().SetNodeStateWithMessageForCluster(pc.Context, pc.BKECluster, ip, state, message)
}

// SetNodeStateMessage 只设置节点消息（不改变状态）
func (pc *PhaseContext) SetNodeStateMessage(ip string, message string) error {
	return pc.NodeFetcher().SetBKENodeStateMessage(pc.Context, pc.BKECluster.Namespace, pc.BKECluster.Name, ip, message)
}
