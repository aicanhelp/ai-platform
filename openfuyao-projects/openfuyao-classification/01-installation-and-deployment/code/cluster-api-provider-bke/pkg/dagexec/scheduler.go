/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	"context"
	"fmt"
	"sync/atomic"

	"golang.org/x/sync/errgroup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

const (
	defaultComponentVersion    = "v1.0.0"
	defaultMaxParallelPerBatch = 8
)

// ensureProviderWebhook registers mbkenode/vbkenode after provider upgrade/skip.
// Tests may override this hook.
var ensureProviderWebhook = upgrade.EnsureBKENodeWebhooks

// Scheduler executes upgrade components according to a topological DAG.
type Scheduler struct {
	InlineRunner        InlineRunner
	ManifestStore       manifest.Store
	ManifestApplier     manifest.Applier
	MaxParallelPerBatch int
	Registry            *ExecutorRegistry
	CVStore             ComponentVersionStore
}

// Config holds dependencies for DAG execution.
type Config struct {
	InlineRunner    InlineRunner
	ManifestStore   manifest.Store
	ManifestApplier manifest.Applier
	// MaxParallelPerBatch limits concurrent components within one batch; 0 uses defaultMaxParallelPerBatch.
	MaxParallelPerBatch int
	// YamlExecutor / HelmExecutor are optional; registered only when non-nil.
	YamlExecutor ComponentExecutor
	HelmExecutor ComponentExecutor
	CVStore      ComponentVersionStore
}

// NewScheduler creates a scheduler with the given dependencies.
// Inline is always registered; yaml/helm are registered only when the corresponding executor is non-nil.
func NewScheduler(cfg Config) *Scheduler {
	maxParallel := cfg.MaxParallelPerBatch
	if maxParallel == 0 {
		maxParallel = defaultMaxParallelPerBatch
	}
	reg := NewExecutorRegistry()
	registerExecutorOrLog(reg, &InlineComponentExecutor{Runner: cfg.InlineRunner}, ComponentTypeInline)
	if cfg.YamlExecutor != nil {
		registerExecutorOrLog(reg, cfg.YamlExecutor, ComponentTypeYAML)
	}
	if cfg.HelmExecutor != nil {
		registerExecutorOrLog(reg, cfg.HelmExecutor, ComponentTypeHelm)
	}
	return &Scheduler{
		InlineRunner:        cfg.InlineRunner,
		ManifestStore:       cfg.ManifestStore,
		ManifestApplier:     cfg.ManifestApplier,
		MaxParallelPerBatch: maxParallel,
		Registry:            reg,
		CVStore:             cfg.CVStore,
	}
}

func registerExecutorOrLog(reg *ExecutorRegistry, executor ComponentExecutor, expectedType ComponentType) {
	if err := reg.Register(executor); err != nil {
		NewLogger(nil).Info("skip registering component executor %s; unregistered types fall back to legacy: %v", expectedType, err)
	}
}

type componentResult struct {
	name        string
	node        *topology.ComponentNode
	err         error
	viaRegistry bool // true when executed via ComponentExecutor (writes clusterComponentStatuses)
}

// ExecuteDAG runs all components in topological batches using ExecutionContext.
func (s *Scheduler) ExecuteDAG(
	ctx context.Context,
	execCtx *ExecutionContext,
	dag *topology.UpgradeDAG,
) error {
	if s == nil {
		return fmt.Errorf("dag scheduler is nil")
	}
	if dag == nil {
		return fmt.Errorf("upgrade DAG is nil")
	}
	if execCtx == nil {
		return fmt.Errorf("execution context is required")
	}
	if execCtx.Cluster == nil {
		return fmt.Errorf("execution context cluster is required")
	}
	if execCtx.TemplateContext.ClusterName == "" && execCtx.TemplateContext.Namespace == "" {
		execCtx.TemplateContext = buildTemplateContext(execCtx.Cluster)
	}

	batches, err := dag.TopologicalBatches()
	if err != nil {
		return err
	}

	oldCluster := execCtx.OldCluster
	newCluster := execCtx.Cluster
	tmpl := execCtx.TemplateContext

	var agg []error
	for batchIdx, batch := range batches {
		batchErrs, failFastStop := s.executeBatchParallel(batchExecution{
			ctx: ctx, execCtx: execCtx, oldCluster: oldCluster, newCluster: newCluster,
			batchIdx: batchIdx, batch: batch, dag: dag, tmpl: tmpl,
		})
		if len(batchErrs) > 0 {
			agg = append(agg, batchErrs...)
		}
		if failFastStop {
			return kerrors.NewAggregate(agg)
		}
	}
	return kerrors.NewAggregate(agg)
}

type batchExecution struct {
	ctx                    context.Context
	execCtx                *ExecutionContext
	oldCluster, newCluster *bkev1beta1.BKECluster
	batchIdx               int
	batch                  []string
	dag                    *topology.UpgradeDAG
	tmpl                   manifest.TemplateContext
}

type batchWorkItem struct {
	name        string
	node        *topology.ComponentNode
	viaRegistry bool
}

func (s *Scheduler) executeBatchParallel(input batchExecution) ([]error, bool) {
	items, batchErrs, failFastStop := s.prepareBatchItems(input)
	if failFastStop || len(items) == 0 {
		return batchErrs, failFastStop
	}

	results := s.executeBatchItems(input, items)
	return s.persistBatchResults(input.ctx, input.execCtx, results, batchErrs)
}

func (s *Scheduler) prepareBatchItems(input batchExecution) ([]batchWorkItem, []error, bool) {
	var items []batchWorkItem
	var batchErrs []error
	for _, compName := range input.batch {
		item, err := s.prepareBatchItem(input, compName)
		if err != nil {
			batchErrs = append(batchErrs, err)
			if item.node != nil && item.node.FailurePolicy == topology.FailurePolicyFailFast {
				return items, batchErrs, true
			}
			continue
		}
		if item.node != nil {
			items = append(items, item)
		}
	}
	return items, batchErrs, false
}

func (s *Scheduler) prepareBatchItem(input batchExecution, compName string) (batchWorkItem, error) {
	node, ok := input.dag.GetNode(compName)
	if !ok {
		return batchWorkItem{}, fmt.Errorf("batch %d: component %q not found", input.batchIdx, compName)
	}
	if s.shouldSkipComponent(input.execCtx, node) || !s.componentNeedsUpgrade(input.execCtx, node) {
		if err := s.ensureProviderWebhookAfterTouch(input.ctx, input.execCtx, node); err != nil {
			return batchWorkItem{name: compName, node: node}, err
		}
		return batchWorkItem{}, nil
	}

	item := batchWorkItem{
		name:        compName,
		node:        node,
		viaRegistry: s.usesRegistryExecutor(input.ctx, node),
	}
	if item.viaRegistry {
		if err := s.markLifecyclePending(input.ctx, input.execCtx, node); err != nil {
			return item, fmt.Errorf("%s: mark pending: %w", compName, err)
		}
	}
	return item, nil
}

func (s *Scheduler) executeBatchItems(input batchExecution, items []batchWorkItem) []componentResult {
	results := make([]componentResult, len(items))
	g, batchCtx := errgroup.WithContext(input.ctx)
	parallelLimit := s.maxParallel(len(items))
	sem := make(chan struct{}, parallelLimit)
	var activeWorkers atomic.Int32

	loggerFrom(input.execCtx).Info(
		"batch start, index=%d, batch_size=%d, runnable=%d, parallel_limit=%d",
		input.batchIdx, len(input.batch), len(items), parallelLimit,
	)

	for i, item := range items {
		i, item := i, item
		g.Go(func() error {
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-batchCtx.Done():
				return batchCtx.Err()
			}

			active := activeWorkers.Add(1)
			loggerFrom(input.execCtx).Info(
				"component start, batch=%d, component=%s, active_workers=%d",
				input.batchIdx, item.name, active,
			)
			err := s.executeComponent(componentExecution{
				ctx:        batchCtx,
				execCtx:    input.execCtx,
				oldCluster: input.oldCluster,
				newCluster: input.newCluster,
				node:       item.node,
				tmpl:       input.tmpl,
			})
			if err == nil {
				if ensureErr := s.ensureProviderWebhookAfterTouch(batchCtx, input.execCtx, item.node); ensureErr != nil {
					err = ensureErr
				}
			}
			results[i] = componentResult{name: item.name, node: item.node, err: err, viaRegistry: item.viaRegistry}
			active = activeWorkers.Add(-1)
			loggerFrom(input.execCtx).Info(
				"component done, batch=%d, component=%s, active_workers=%d, has_error=%t",
				input.batchIdx, item.name, active, err != nil,
			)

			if err == nil || manifest.IsSkipNotInstalled(err) {
				return nil
			}
			if item.node.FailurePolicy == topology.FailurePolicyFailFast {
				return err
			}
			return nil
		})
	}

	_ = g.Wait()
	loggerFrom(input.execCtx).Info(
		"batch done, index=%d, batch_size=%d, runnable=%d",
		input.batchIdx, len(input.batch), len(items),
	)
	return results
}

func (s *Scheduler) persistBatchResults(
	ctx context.Context,
	execCtx *ExecutionContext,
	results []componentResult,
	batchErrs []error,
) ([]error, bool) {
	var failFastStop bool
	var successes []componentResult

	for _, r := range results {
		if r.node == nil {
			continue
		}
		if r.err != nil {
			stop, errs := s.persistBatchFailure(ctx, execCtx, r)
			batchErrs = append(batchErrs, errs...)
			if stop {
				failFastStop = true
			}
			continue
		}
		successes = append(successes, r)
	}

	stop, errs := s.persistBatchSuccesses(ctx, execCtx, successes)
	batchErrs = append(batchErrs, errs...)
	if stop {
		failFastStop = true
	}
	return batchErrs, failFastStop
}

func (s *Scheduler) persistBatchFailure(
	ctx context.Context,
	execCtx *ExecutionContext,
	r componentResult,
) (bool, []error) {
	if manifest.IsSkipNotInstalled(r.err) {
		if r.viaRegistry {
			if persistErr := s.markLifecycleClear(ctx, execCtx, r.node); persistErr != nil {
				return false, []error{fmt.Errorf("%s: clear lifecycle after skip: %w", r.name, persistErr)}
			}
		}
		return false, nil
	}
	compName := r.name
	var errs []error
	if persistErr := s.markComponentFailed(ctx, execCtx, r.node, r.err); persistErr != nil {
		errs = append(errs, fmt.Errorf("%s: persist failure: %w", compName, persistErr))
	}
	if r.viaRegistry {
		if persistErr := s.markLifecycleFailed(ctx, execCtx, r.node, r.err); persistErr != nil {
			errs = append(errs, fmt.Errorf("%s: persist lifecycle failure: %w", compName, persistErr))
		}
	}
	errs = append(errs, fmt.Errorf("%s: %w", compName, r.err))
	return isFailFast(r.node), errs
}

// persistBatchSuccesses records all successful components in one DeclarativeUpgrade Status
// patch, then writes per-component lifecycle status when using the registry path.
func (s *Scheduler) persistBatchSuccesses(
	ctx context.Context,
	execCtx *ExecutionContext,
	successes []componentResult,
) (bool, []error) {
	if len(successes) == 0 {
		return false, nil
	}

	nodes := make([]*topology.ComponentNode, 0, len(successes))
	for _, r := range successes {
		nodes = append(nodes, r.node)
	}

	var errs []error
	var failFastStop bool
	if err := s.markComponentsCompleted(ctx, execCtx, nodes); err != nil {
		for _, r := range successes {
			errs = append(errs, fmt.Errorf("%s: persist completion: %w", r.name, err))
			if isFailFast(r.node) {
				failFastStop = true
			}
		}
	}

	for _, r := range successes {
		if !r.viaRegistry {
			continue
		}
		if err := s.markLifecycleInstalled(ctx, execCtx, r.node); err != nil {
			errs = append(errs, fmt.Errorf("%s: persist lifecycle completion: %w", r.name, err))
			if isFailFast(r.node) {
				failFastStop = true
			}
		}
	}
	return failFastStop, errs
}

func isFailFast(node *topology.ComponentNode) bool {
	return node != nil && node.FailurePolicy == topology.FailurePolicyFailFast
}

func (s *Scheduler) maxParallel(batchLen int) int {
	if batchLen <= 0 {
		return 1
	}
	limit := s.MaxParallelPerBatch
	if limit <= 0 {
		limit = defaultMaxParallelPerBatch
	}
	if limit > batchLen {
		return batchLen
	}
	return limit
}

func (s *Scheduler) nodeVersionKey(node *topology.ComponentNode) string {
	if node == nil {
		return defaultComponentVersion
	}
	if node.Inline != nil {
		if node.Inline.Version != "" {
			return node.Inline.Version
		}
		return defaultComponentVersion
	}
	if node.Version != "" {
		return node.Version
	}
	return defaultComponentVersion
}

func (s *Scheduler) shouldSkipComponent(execCtx *ExecutionContext, node *topology.ComponentNode) bool {
	if execCtx == nil || execCtx.Cluster == nil || node == nil {
		return false
	}
	st := execCtx.Cluster.Status.DeclarativeUpgrade
	if st == nil {
		return false
	}
	return st.IsCompleted(node.Name, s.nodeVersionKey(node))
}

// componentNeedsUpgrade skips components whose current version already matches target.
func (s *Scheduler) componentNeedsUpgrade(execCtx *ExecutionContext, node *topology.ComponentNode) bool {
	if node == nil {
		return false
	}
	var vc *upgrade.VersionContext
	if execCtx != nil {
		vc = execCtx.VersionContext
	}
	return upgrade.NeedsExecution(vc, node.Name)
}

// ensureProviderWebhookAfterTouch runs after provider is applied or skipped so that
// mbkenode/vbkenode are registered once the new reconciler Deployment is Ready (06→09 resume).
func (s *Scheduler) ensureProviderWebhookAfterTouch(
	ctx context.Context,
	execCtx *ExecutionContext,
	node *topology.ComponentNode,
) error {
	if node == nil || node.Name != upgrade.ComponentProvider {
		return nil
	}
	if execCtx == nil || execCtx.Client == nil {
		return fmt.Errorf("provider webhook ensure: execution client is nil")
	}
	if err := ensureProviderWebhook(ctx, execCtx.Client); err != nil {
		return fmt.Errorf("%s: ensure bkenode/vbkenode webhooks: %w", upgrade.ComponentProvider, err)
	}
	loggerFrom(execCtx).Info("provider webhooks ensured, component=%s", node.Name)
	return nil
}

// markComponentsCompleted persists Completed records for all nodes in one Status patch.
func (s *Scheduler) markComponentsCompleted(
	ctx context.Context,
	execCtx *ExecutionContext,
	nodes []*topology.ComponentNode,
) error {
	if execCtx == nil || execCtx.Cluster == nil || execCtx.Client == nil || len(nodes) == 0 {
		return nil
	}
	now := metav1.Now()
	return patchDeclarativeUpgrade(ctx, execCtx.Client, execCtx.Cluster, func(st *confv1beta1.DeclarativeUpgradeStatus) {
		for _, node := range nodes {
			if node == nil {
				continue
			}
			st.MarkCompleted(node.Name, s.nodeVersionKey(node), now)
		}
		st.LastError = ""
		st.ClearFailure()
	})
}

func (s *Scheduler) markComponentCompleted(
	ctx context.Context,
	execCtx *ExecutionContext,
	node *topology.ComponentNode,
) error {
	if node == nil {
		return nil
	}
	return s.markComponentsCompleted(ctx, execCtx, []*topology.ComponentNode{node})
}

func (s *Scheduler) markComponentFailed(
	ctx context.Context,
	execCtx *ExecutionContext,
	node *topology.ComponentNode,
	err error,
) error {
	if execCtx == nil || execCtx.Cluster == nil || execCtx.Client == nil || node == nil || err == nil {
		return nil
	}
	now := metav1.Now()
	return patchDeclarativeUpgrade(ctx, execCtx.Client, execCtx.Cluster, func(st *confv1beta1.DeclarativeUpgradeStatus) {
		st.MarkFailure(node.Name, s.nodeVersionKey(node), err.Error(), now)
	})
}

type componentExecution struct {
	ctx                    context.Context
	execCtx                *ExecutionContext
	oldCluster, newCluster *bkev1beta1.BKECluster
	node                   *topology.ComponentNode
	tmpl                   manifest.TemplateContext
}

func (s *Scheduler) executeComponent(input componentExecution) error {
	ctx, execCtx, node := input.ctx, input.execCtx, input.node
	if typ, ok := s.resolveComponentType(ctx, node); ok {
		if s.Registry != nil {
			if executor, found := s.Registry.Get(typ); found {
				return executor.ExecuteComponent(ctx, node, execCtx)
			}
		}
	}
	return s.executeComponentLegacy(ctx, legacyComponentArgs{
		execCtx:    execCtx,
		oldCluster: input.oldCluster,
		newCluster: input.newCluster,
		node:       node,
		tmpl:       input.tmpl,
	})
}

// usesRegistryExecutor reports whether executeComponent would dispatch to a registered executor.
// Legacy path must not write clusterComponentStatuses.
func (s *Scheduler) usesRegistryExecutor(ctx context.Context, node *topology.ComponentNode) bool {
	typ, ok := s.resolveComponentType(ctx, node)
	if !ok || s == nil || s.Registry == nil {
		return false
	}
	_, found := s.Registry.Get(typ)
	return found
}

func (s *Scheduler) markLifecyclePending(ctx context.Context, execCtx *ExecutionContext, node *topology.ComponentNode) error {
	updater, cluster := lifecycleUpdater(execCtx)
	if updater == nil || cluster == nil || node == nil {
		return nil
	}
	return updater.MarkPending(ctx, lifecycleMarkRef(cluster, node))
}

func (s *Scheduler) markLifecycleInstalled(ctx context.Context, execCtx *ExecutionContext, node *topology.ComponentNode) error {
	updater, cluster := lifecycleUpdater(execCtx)
	if updater == nil || cluster == nil || node == nil {
		return nil
	}
	return updater.MarkInstalled(ctx, lifecycleMarkRef(cluster, node), componentStatusVersion(node))
}

func (s *Scheduler) markLifecycleFailed(ctx context.Context, execCtx *ExecutionContext, node *topology.ComponentNode, err error) error {
	updater, cluster := lifecycleUpdater(execCtx)
	if updater == nil || cluster == nil || node == nil || err == nil {
		return nil
	}
	return updater.MarkFailed(ctx, lifecycleMarkRef(cluster, node), err)
}

func (s *Scheduler) markLifecycleClear(ctx context.Context, execCtx *ExecutionContext, node *topology.ComponentNode) error {
	updater, cluster := lifecycleUpdater(execCtx)
	if updater == nil || cluster == nil || node == nil {
		return nil
	}
	return updater.ClearComponentStatus(ctx, lifecycleMarkRef(cluster, node))
}

func lifecycleMarkRef(cluster *bkev1beta1.BKECluster, node *topology.ComponentNode) ComponentMarkRef {
	return ComponentMarkRef{
		Cluster:       cluster,
		Name:          node.Name,
		ComponentType: StatusComponentTypeCluster,
	}
}

func lifecycleUpdater(execCtx *ExecutionContext) (ComponentStatusUpdater, *bkev1beta1.BKECluster) {
	if execCtx == nil {
		return nil, nil
	}
	return execCtx.ComponentStatusUpdater, execCtx.Cluster
}

func componentStatusVersion(node *topology.ComponentNode) string {
	if node == nil {
		return defaultComponentVersion
	}
	if node.Version != "" {
		return node.Version
	}
	if node.Inline != nil && node.Inline.Version != "" {
		return node.Inline.Version
	}
	return defaultComponentVersion
}

// resolveComponentType returns the executor type from CVStore (cv.Spec.Type).
// When unresolved, callers fall back to Legacy.
func (s *Scheduler) resolveComponentType(ctx context.Context, node *topology.ComponentNode) (ComponentType, bool) {
	if node == nil || s == nil || s.CVStore == nil {
		return "", false
	}
	version := node.Version
	if version == "" {
		version = defaultComponentVersion
	}
	cv, err := s.CVStore.GetComponentVersion(ctx, node.Name, version)
	if err != nil || cv == nil || cv.Spec.Type == "" {
		return "", false
	}
	return ComponentType(cv.Spec.Type), true
}

// legacyComponentArgs groups inputs for the pre-registry Inline / Manifest path.
type legacyComponentArgs struct {
	execCtx                *ExecutionContext
	oldCluster, newCluster *bkev1beta1.BKECluster
	node                   *topology.ComponentNode
	tmpl                   manifest.TemplateContext
}

// executeComponentLegacy keeps the pre-registry Inline / Manifest paths.
// It does not write clusterComponentStatuses via ComponentStatusUpdater.
func (s *Scheduler) executeComponentLegacy(ctx context.Context, args legacyComponentArgs) error {
	if args.node != nil && args.node.Inline != nil {
		return s.executeInline(ctx, args.oldCluster, args.newCluster, args.node)
	}
	return s.executeManifest(ctx, args.execCtx, args.node, args.tmpl)
}

func (s *Scheduler) executeInline(
	ctx context.Context,
	oldCluster, newCluster *bkev1beta1.BKECluster,
	node *topology.ComponentNode,
) error {
	if s.InlineRunner == nil {
		return fmt.Errorf("inline runner is nil")
	}
	handler := node.Inline.Handler
	version := node.Inline.Version
	if handler == "" {
		return fmt.Errorf("inline component %q missing handler", node.Name)
	}
	if version == "" {
		version = defaultComponentVersion
	}
	return s.InlineRunner.Execute(ctx, oldCluster, newCluster, handler, version)
}

// manifestNeedsUpgrade skips apply when VersionContext has a target and current already matches.
func manifestNeedsUpgrade(vc *upgrade.VersionContext, componentName string) bool {
	return upgrade.NeedsExecution(vc, componentName)
}

func (s *Scheduler) executeManifest(
	ctx context.Context,
	execCtx *ExecutionContext,
	node *topology.ComponentNode,
	tmpl manifest.TemplateContext,
) error {
	if node == nil {
		return fmt.Errorf("component node is nil")
	}
	var vc *upgrade.VersionContext
	if execCtx != nil {
		vc = execCtx.VersionContext
	}
	if !manifestNeedsUpgrade(vc, node.Name) {
		return nil
	}
	version := node.Version
	if version == "" {
		version = defaultComponentVersion
	}
	if s.ManifestStore == nil {
		return fmt.Errorf("manifest store is not configured")
	}
	pkg, err := s.ManifestStore.GetComponentManifests(ctx, node.Name, version, tmpl)
	if err != nil {
		return err
	}
	if len(pkg.Manifests) == 0 {
		return fmt.Errorf("component %q version %q has no manifests to apply", node.Name, version)
	}
	if s.ManifestApplier == nil {
		return fmt.Errorf("manifest applier is not configured")
	}
	return s.ManifestApplier.ApplyComponent(ctx, pkg)
}

// RequeueAwareError reports whether the reconcile should requeue.
func RequeueAwareError(err error) (ctrl.Result, bool) {
	if err == nil {
		return ctrl.Result{}, false
	}
	return ctrl.Result{}, true
}
