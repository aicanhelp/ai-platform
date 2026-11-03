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

package capbke

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/componentfactory"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/featuregate"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/helminstaller"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phases"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/yamlinstaller"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

const releaseImageRequeueInterval = 30 * time.Second

// executeUpgradeDAG runs declarative upgrade components in topological order.
// The bool is true only when the hop completed successfully (completeDeclarativeUpgrade).
func (r *BKEClusterReconciler) executeUpgradeDAG(
	ctx context.Context,
	phaseCtx *phaseframe.PhaseContext,
	oldCluster, newCluster *bkev1beta1.BKECluster,
	bkeLogger *bkev1beta1.BKELogger,
) (bool, ctrl.Result, error) {
	hopTarget, err := r.declarativeUpgradeTargetVersion(newCluster)
	if err != nil {
		return false, ctrl.Result{}, err
	}

	bundle, releaseImage, err := r.resolveUpgradeBundle(ctx, newCluster, hopTarget)
	if err != nil {
		if isReleaseImageNotReady(err) {
			bkeLogger.Info("waiting for release image", "reason: %s", err.Error())
			return false, ctrl.Result{RequeueAfter: releaseImageRequeueInterval}, nil
		}
		return false, ctrl.Result{}, err
	}

	currentBundle, _ := r.resolveCurrentReleaseBundle(ctx, newCluster, hopTarget)
	phaseCtx.BuildAndSetVersionContextFromBundle(bundle, currentBundle)
	if err := phaseCtx.SyncUpgradeTargetsToClusterSpec(); err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "sync upgrade targets to cluster spec")
	}
	if err := r.patchClusterOpenFuyaoVersionSpecBeforeDAG(newCluster, hopTarget); err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "sync openFuyaoVersion to cluster spec before DAG")
	}
	if err := phaseCtx.RefreshCtxBKECluster(); err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "refresh phase context after openFuyaoVersion spec sync")
	}

	dag, err := upgrade.BuildDAGFromBundle(bundle, upgrade.BundleDependencyResolver(bundle))
	if err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "build upgrade DAG")
	}

	bkeLogger.Info("declarative upgrade",
		"hopTarget=%s releaseImage=%s phase=%s components=%d source=%s",
		hopTarget, releaseImage.Name, releaseImage.Status.Phase, len(dag.NodeNames()), bundle.Source,
	)

	if err := r.patchClusterStatus(newCluster, bkev1beta1.ClusterUpgrading); err != nil {
		return false, ctrl.Result{}, err
	}

	if err := r.ensureDeclarativeUpgradeProgress(newCluster, hopTarget); err != nil {
		return false, ctrl.Result{}, err
	}

	factory, err := componentfactory.NewFactoryFromBundle(bundle)
	if err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "build component factory from release bundle")
	}

	bundleStore := manifest.NewBundleStore(bundle)
	applier := r.buildManifestApplier(ctx, phaseCtx, newCluster, bkeLogger)
	targetAccess, err := r.targetClusterAccessForDAG(ctx, newCluster)
	if err != nil {
		return false, ctrl.Result{}, errors.Wrap(err, "get target cluster access for declarative upgrade")
	}

	yamlExec := &yamlinstaller.YamlComponentExecutor{
		Installer: yamlinstaller.NewYamlInstaller(yamlinstaller.YamlInstallerConfig{
			Store:   bundleStore,
			Applier: applier,
			Logger:  bkeLogger,
		}),
		CVStore: bundleStore,
	}
	helmExec := &helminstaller.HelmComponentExecutor{
		Installer: helminstaller.NewHelmInstaller(helminstaller.HelmInstallerConfig{
			Store:        bundleStore,
			LocalClient:  r.Client,
			BKECluster:   newCluster,
			TargetClient: targetAccess.Clientset,
			RestConfig:   targetAccess.RestConfig,
			Logger:       bkeLogger,
		}),
		CVStore: bundleStore,
	}
	sched := dagexec.NewScheduler(r.buildDAGSchedulerConfig(newCluster, dagexec.Config{
		InlineRunner:        NewInlinePhaseRunnerAdapter(phaseCtx, &componentfactory.PhaseRunner{Factory: factory}),
		ManifestStore:       bundleStore,
		CVStore:             bundleStore,
		ManifestApplier:     applier,
		YamlExecutor:        yamlExec,
		HelmExecutor:        helmExec,
		MaxParallelPerBatch: 0, // 0 = dagexec.defaultMaxParallelPerBatch (8)
	}))

	execCtx := buildExecutionContext(executionContextBuildOptions{
		phaseCtx:         phaseCtx,
		oldCluster:       oldCluster,
		newCluster:       newCluster,
		log:              bkeLogger,
		targetClient:     targetAccess.Clientset,
		targetRestConfig: targetAccess.RestConfig,
	})
	if err := sched.ExecuteDAG(ctx, execCtx, dag); err != nil {
		if handleErr := r.handleDeclarativeUpgradeDAGFailure(ctx, newCluster, err); handleErr != nil {
			return false, ctrl.Result{}, handleErr
		}
		return false, ctrl.Result{}, nil
	}

	// Do not re-GET DeclarativeUpgrade from the informer cache here: DAG Status patches already
	// synced Completed into newCluster memory, and a stale cache read would drop the last batch.
	if err := r.completeDeclarativeUpgrade(ctx, newCluster); err != nil {
		return false, ctrl.Result{}, err
	}

	return true, ctrl.Result{}, nil
}

func (r *BKEClusterReconciler) ensureDeclarativeUpgradeProgress(bkeCluster *bkev1beta1.BKECluster, targetVersion string) error {
	if bkeCluster == nil {
		return nil
	}
	now := metav1.Now()
	return mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster, func(bc *bkev1beta1.BKECluster) {
		if bc.Status.DeclarativeUpgrade == nil {
			bc.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{}
		}
		// EnsureInitialized resets completion when target changes and clears FinishedAt.
		bc.Status.DeclarativeUpgrade.EnsureInitialized(targetVersion, now)
	})
}

// shouldUseDeclarativeUpgrade reports whether BKECluster should run the declarative upgrade DAG.
// ClusterVersion sets cvo.openfuyao.cn/upgrade-ready when a hop is ready; that is the only gate.
func (r *BKEClusterReconciler) shouldUseDeclarativeUpgrade(bkeCluster *bkev1beta1.BKECluster) bool {
	_, ok := featuregate.UpgradeReady(bkeCluster)
	return ok
}

// declarativeUpgradeTargetVersion returns the openFuyao version for the current upgrade hop
// (cvo.openfuyao.cn/upgrade-ready), not cv.spec.desiredVersion which may be the final target.
func (r *BKEClusterReconciler) declarativeUpgradeTargetVersion(bkeCluster *bkev1beta1.BKECluster) (string, error) {
	if bkeCluster == nil {
		return "", fmt.Errorf("bke cluster is nil")
	}
	target, ok := featuregate.UpgradeReady(bkeCluster)
	if !ok {
		return "", fmt.Errorf("missing %s annotation", featuregate.UpgradeReadyAnnotationKey)
	}
	return target, nil
}

// resolveUpgradeBundle loads and validates the release bundle for the given hop target version.
func (r *BKEClusterReconciler) resolveUpgradeBundle(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	hopTarget string,
) (*releasemanifest.Bundle, *cvv1alpha1.ReleaseImage, error) {
	ri, err := r.resolveReleaseImageCR(ctx, bkeCluster, hopTarget)
	if err != nil {
		return nil, nil, err
	}

	switch ri.Status.Phase {
	case cvv1alpha1.ReleaseImagePhaseValid:
		// continue
	case cvv1alpha1.ReleaseImagePhaseInvalid,
		cvv1alpha1.ReleaseImagePhaseManifestMissing,
		cvv1alpha1.ReleaseImagePhaseCompatibilityFailed:
		return nil, ri, fmt.Errorf("release image %s/%s phase %s: %s",
			ri.Namespace, ri.Name, ri.Status.Phase, ri.Status.Message)
	default:
		return nil, ri, &releaseImagePendingError{
			msg: fmt.Sprintf("release image %s/%s phase %s", ri.Namespace, ri.Name, ri.Status.Phase),
		}
	}

	bundle, err := r.releaseStore().ResolveRelease(ctx, releaseRefFromCR(ri))
	if err != nil {
		return nil, ri, errors.Wrapf(err, "resolve release %s/%s", ri.Namespace, ri.Name)
	}
	return bundle, ri, nil
}

// resolveCurrentReleaseBundle loads the release bundle for the cluster's current version.
// Even when current version equals the hop target (same-version retrigger), we still resolve
// the current bundle so VersionContext can populate current component versions and let the DAG
// scheduler consistently skip already-aligned components.
func (r *BKEClusterReconciler) resolveCurrentReleaseBundle(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	hopTarget string,
) (*releasemanifest.Bundle, error) {
	currentVer, err := r.clusterCurrentOpenFuyaoVersion(ctx, bkeCluster)
	if err != nil || currentVer == "" {
		return nil, err
	}

	ri, err := clusterversion.ResolveReleaseImageForVersion(ctx, r.Client, bkeCluster.Namespace, currentVer)
	if err != nil {
		return nil, nil
	}
	if ri.Status.Phase != cvv1alpha1.ReleaseImagePhaseValid {
		return nil, nil
	}
	return r.releaseStore().ResolveRelease(ctx, releaseRefFromCR(ri))
}

func (r *BKEClusterReconciler) clusterCurrentOpenFuyaoVersion(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
) (string, error) {
	cv, err := clusterversion.GetClusterVersionForBKECluster(ctx, r.Client, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return clusterversion.OpenFuyaoVersionForBKECluster(bkeCluster), nil
		}
		return "", err
	}
	current := cv.Status.CurrentVersion
	if current == "" {
		current = clusterversion.OpenFuyaoVersionForBKECluster(bkeCluster)
	}
	return current, nil
}

func (r *BKEClusterReconciler) resolveReleaseImageCR(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	hopTarget string,
) (*cvv1alpha1.ReleaseImage, error) {
	if bkeCluster == nil {
		return nil, fmt.Errorf("bke cluster is nil")
	}
	return clusterversion.ResolveReleaseImageForVersion(ctx, r.Client, bkeCluster.Namespace, hopTarget)
}

func releaseRefFromCR(ri *cvv1alpha1.ReleaseImage) releasemanifest.ReleaseRef {
	return releasemanifest.ReleaseRef{
		Version:            ri.Spec.Version,
		Digest:             strings.TrimSpace(ri.Spec.Digest),
		VerifySignature:    ri.Spec.VerifySignature,
		SignatureKey:       ri.Spec.SignatureKey,
		AllowCacheFallback: ri.Spec.AllowCacheFallback,
	}
}

func (r *BKEClusterReconciler) releaseStore() *releasemanifest.Store {
	if r.ReleaseStore != nil {
		return r.ReleaseStore
	}
	return releasemanifest.NewStore(releasemanifest.ReleaseCacheDir(), nil, nil)
}

type dagTargetAccess struct {
	Clientset  kubernetes.Interface
	RestConfig *rest.Config
}

// targetClusterAccessForDAG returns the workload-cluster access used by DAG executors.
// YAML health checks need Clientset, while Helm SDK action.Configuration needs RestConfig.
func (r *BKEClusterReconciler) targetClusterAccessForDAG(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
) (*dagTargetAccess, error) {
	if r == nil || r.Client == nil || bkeCluster == nil {
		return nil, fmt.Errorf("reconciler client and bke cluster are required")
	}
	remoteClient, err := kube.GetTargetRemoteClient(ctx, r.Client, bkeCluster)
	if err != nil {
		return nil, err
	}
	clientset, _ := remoteClient.KubeClient()
	if clientset == nil {
		return nil, fmt.Errorf("target cluster clientset is nil")
	}
	if remoteClient.RestConfig == nil {
		return nil, fmt.Errorf("target cluster rest config is nil")
	}
	return &dagTargetAccess{Clientset: clientset, RestConfig: remoteClient.RestConfig}, nil
}

func (r *BKEClusterReconciler) buildManifestApplier(
	ctx context.Context,
	phaseCtx *phaseframe.PhaseContext,
	bkeCluster *bkev1beta1.BKECluster,
	bkeLogger *bkev1beta1.BKELogger,
) manifest.Applier {
	if r.ManifestApplier != nil {
		return r.ManifestApplier
	}
	var nodes bkenode.Nodes
	if phaseCtx != nil {
		if n, err := phaseCtx.GetNodes(); err == nil {
			nodes = n
		}
	}
	return manifest.NewClusterApplier(manifest.ClusterApplierConfig{
		Context:    ctx,
		Client:     r.Client,
		BKECluster: bkeCluster,
		Logger:     bkeLogger,
		Nodes:      nodes,
	})
}

func (r *BKEClusterReconciler) patchClusterStatus(
	bkeCluster *bkev1beta1.BKECluster,
	status confv1beta1.ClusterStatus,
) error {
	return mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster, func(bc *bkev1beta1.BKECluster) {
		phaseutil.SetClusterStatus(bc, status)
		if status == bkev1beta1.ClusterUpgrading {
			annotation.SetAnnotation(bc, annotation.StatusRecordAnnotationKey, "")
		}
	})
}

// handleDeclarativeUpgradeDAGFailure records DAG failure, retries via StatusManager
// (ALLOWED_FAILED_COUNT), and aborts the hop when the retry budget is exhausted.
func (r *BKEClusterReconciler) handleDeclarativeUpgradeDAGFailure(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	dagErr error,
) error {
	if bkeCluster == nil {
		return nil
	}
	if err := mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster, func(bc *bkev1beta1.BKECluster) {
		phaseutil.SetClusterStatus(bc, bkev1beta1.ClusterUpgradeFailed)
		annotation.SetAnnotation(bc, annotation.StatusRecordAnnotationKey, "")
		if dagErr != nil && bc.Status.DeclarativeUpgrade != nil {
			bc.Status.DeclarativeUpgrade.LastError = dagErr.Error()
		}
	}); err != nil {
		return err
	}

	if statusmanage.DefaultStatusManager().GetCtrlResult(bkeCluster).Requeue {
		return nil
	}
	return r.abortDeclarativeUpgrade(ctx, bkeCluster, dagErr)
}

// abortDeclarativeUpgrade stops retrying a declarative upgrade hop after retries are exhausted.
func (r *BKEClusterReconciler) abortDeclarativeUpgrade(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	cause error,
) error {
	if bkeCluster == nil {
		return nil
	}

	hopTarget, _ := r.declarativeUpgradeTargetVersion(bkeCluster)

	_ = mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster, func(bc *bkev1beta1.BKECluster) {
		phaseutil.SetClusterStatus(bc, bkev1beta1.ClusterUpgradeFailed)
		if cause != nil && bc.Status.DeclarativeUpgrade != nil {
			bc.Status.DeclarativeUpgrade.LastError = cause.Error()
		}
	})

	orig := bkeCluster.DeepCopy()
	ann := bkeCluster.GetAnnotations()
	if ann == nil {
		ann = make(map[string]string)
	}
	delete(ann, featuregate.UpgradeReadyAnnotationKey)
	delete(ann, annotation.CVOUpgradePathAnnotationKey)
	delete(ann, annotation.CVOClusterVersionAnnotationKey)
	bkeCluster.SetAnnotations(ann)
	if err := r.Patch(ctx, bkeCluster, client.MergeFrom(orig)); err != nil {
		return err
	}

	if hopTarget == "" {
		return nil
	}
	cv, err := clusterversion.GetClusterVersionForBKECluster(ctx, r.Client, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return clusterversion.FailUpgradeHop(ctx, r.Client, cv, hopTarget, cause)
}

func (r *BKEClusterReconciler) patchClusterOpenFuyaoVersionSpecBeforeDAG(
	bkeCluster *bkev1beta1.BKECluster,
	hopTarget string,
) error {
	return mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster, func(bc *bkev1beta1.BKECluster) {
		upgrade.ApplyUpgradeHopToClusterSpec(bc, hopTarget)
	})
}

type releaseImagePendingError struct {
	msg string
}

func (e *releaseImagePendingError) Error() string {
	return e.msg
}

func isReleaseImageNotReady(err error) bool {
	var pending *releaseImagePendingError
	return errors.As(err, &pending)
}

// completeDeclarativeUpgrade clears upgrade-ready and syncs ClusterVersion status per design.
// DeclarativeUpgrade.Completed is taken from in-memory bkeCluster (authoritative after DAG
// Status patches), not from an informer re-GET, to avoid dropping the last batch.
func (r *BKEClusterReconciler) completeDeclarativeUpgrade(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
) error {
	if bkeCluster == nil {
		return nil
	}

	hopTarget, err := r.declarativeUpgradeTargetVersion(bkeCluster)
	if err != nil {
		return err
	}

	if err := r.persistDeclarativeUpgradeFinished(ctx, bkeCluster, hopTarget); err != nil {
		return err
	}

	orig := bkeCluster.DeepCopy()
	ann := bkeCluster.GetAnnotations()
	if ann != nil {
		delete(ann, featuregate.UpgradeReadyAnnotationKey)
		delete(ann, annotation.CVOUpgradePathAnnotationKey)
		delete(ann, annotation.CVOClusterVersionAnnotationKey)
		bkeCluster.SetAnnotations(ann)
	}
	if err := r.Patch(ctx, bkeCluster, client.MergeFrom(orig)); err != nil {
		return err
	}

	cv, err := clusterversion.GetClusterVersionForBKECluster(ctx, r.Client, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return clusterversion.CompleteUpgradeHop(ctx, r.Client, cv, hopTarget)
}

// persistDeclarativeUpgradeFinished writes FinishedAt and hop version status fields.
// Completed records come from memory (DAG), while ResourceVersion comes from a Get for the patch.
func (r *BKEClusterReconciler) persistDeclarativeUpgradeFinished(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	hopTarget string,
) error {
	if r == nil || r.Client == nil || bkeCluster == nil {
		return nil
	}

	progress := bkeCluster.Status.DeclarativeUpgrade.DeepCopy()
	if progress != nil {
		now := metav1.Now()
		progress.FinishedAt = &now
		progress.LastError = ""
		progress.ClearFailure()
	}

	key := client.ObjectKeyFromObject(bkeCluster)
	current := &bkev1beta1.BKECluster{}
	if err := r.Get(ctx, key, current); err != nil {
		return err
	}
	orig := current.DeepCopy()

	// Spec on current may lag memory after pre-DAG sync; prefer memory Spec for version status.
	if bkeCluster.Spec.ClusterConfig != nil {
		current.Spec.ClusterConfig = bkeCluster.Spec.ClusterConfig.DeepCopy()
	}
	upgrade.ApplyUpgradeHopToClusterStatus(current, hopTarget)
	if progress != nil {
		current.Status.DeclarativeUpgrade = progress
	}

	if err := r.Status().Patch(ctx, current, client.MergeFrom(orig)); err != nil {
		return err
	}

	bkeCluster.Status.OpenFuyaoVersion = current.Status.OpenFuyaoVersion
	bkeCluster.Status.KubernetesVersion = current.Status.KubernetesVersion
	bkeCluster.Status.EtcdVersion = current.Status.EtcdVersion
	bkeCluster.Status.ContainerdVersion = current.Status.ContainerdVersion
	if progress != nil {
		bkeCluster.Status.DeclarativeUpgrade = progress.DeepCopy()
	}
	return nil
}

// declarativeUpgradePhaseName maps a DAG node to a BKECluster phase name for status reporting.
func declarativeUpgradePhaseName(node *topology.ComponentNode) confv1beta1.BKEClusterPhase {
	if node.Inline != nil && node.Inline.Handler != "" {
		return confv1beta1.BKEClusterPhase(node.Inline.Handler)
	}
	switch node.Name {
	case upgrade.ComponentProvider:
		return phases.EnsureProviderSelfUpgradeName
	case upgrade.ComponentCoreDNS:
		return phases.EnsureComponentUpgradeName
	default:
		return confv1beta1.BKEClusterPhase(node.Name)
	}
}
