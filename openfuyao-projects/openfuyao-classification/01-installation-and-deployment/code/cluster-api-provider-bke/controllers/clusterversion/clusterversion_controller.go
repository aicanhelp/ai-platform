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

package clusterversion

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	cvensure "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
	pathstore "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const clusterVersionControllerName = "clusterversion"

// ClusterVersionReconciler reconciles a ClusterVersion object.
type ClusterVersionReconciler struct {
	client.Client
	Scheme      *runtime.Scheme
	Recorder    record.EventRecorder
	PathService pathstore.Loader
	OCIClient   *oci.Client
}

// +kubebuilder:rbac:groups=config.openfuyao.com,resources=clusterversions,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=clusterversions/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=releaseimages,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=upgradepaths,verbs=get;list;watch
// +kubebuilder:rbac:groups=bke.bocloud.com,resources=bkeclusters,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile pulls ri OCI, ensures ReleaseImage CRs, validates upgrade paths, and marks BKECluster upgrade-ready.
// ClusterVersion CR creation and status writes are owned by BKECluster Reconciler per design.
func (r *ClusterVersionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.ControllerLogger(clusterVersionControllerName).With("clusterversion", req.NamespacedName)

	cv := &cvv1beta1.ClusterVersion{}
	if err := r.Get(ctx, req.NamespacedName, cv); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	if desired == "" {
		r.event(cv, corev1.EventTypeWarning, "DesiredVersionMissing", "spec.desiredVersion is required")
		return ctrl.Result{}, nil
	}

	bc := &bkev1beta1.BKECluster{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: cv.Namespace, Name: cv.Name}, bc); err != nil {
		msg := fmt.Sprintf("get BKECluster %s/%s: %v", cv.Namespace, cv.Name, err)
		r.event(cv, corev1.EventTypeWarning, "BKEClusterNotFound", msg)
		return ctrl.Result{}, err
	}

	ensurer := &cvensure.ReleaseImageEnsurer{
		Client:    r.Client,
		Scheme:    r.Scheme,
		OCIClient: r.OCIClient,
	}

	if r.isInstallPhase(ctx, bc, cv) {
		if err := r.setStatus(ctx, cv, statusPatch{
			Phase: phasePtr(cvv1beta1.ClusterVersionPhaseInstalling),
		}); err != nil {
			return ctrl.Result{}, err
		}
		ri, res, err := ensurer.Ensure(ctx, bc, cv, desired)
		if err != nil {
			r.event(cv, corev1.EventTypeWarning, "ReleaseImageEnsureFailed", err.Error())
			if statusErr := r.setStatus(ctx, cv, statusPatch{
				Phase:      phasePtr(cvv1beta1.ClusterVersionPhaseFailed),
				Conditions: conditionPtr(failureCondition("ReleaseImageEnsureFailed", err.Error())),
			}); statusErr != nil {
				return ctrl.Result{}, statusErr
			}
			return ctrl.Result{}, err
		}
		if !res.IsZero() {
			logger.Info("waiting for install ReleaseImage to become Valid",
				"releaseImage", ri.Name, "phase", ri.Status.Phase)
			return res, nil
		}
		logger.Info("install ReleaseImage ready", "releaseImage", ri.Name, "version", desired)
		return ctrl.Result{}, nil
	}

	return r.reconcileUpgrade(upgradeReconcileInput{
		ctx: ctx, logger: logger, clusterVersion: cv,
		bkeCluster: bc, ensurer: ensurer, desired: desired,
	})
}

type upgradeReconcileInput struct {
	ctx    context.Context
	logger interface {
		Info(msg string, keysAndValues ...interface{})
	}
	clusterVersion *cvv1beta1.ClusterVersion
	bkeCluster     *bkev1beta1.BKECluster
	ensurer        *cvensure.ReleaseImageEnsurer
	desired        string
}

func (r *ClusterVersionReconciler) reconcileUpgrade(input upgradeReconcileInput) (ctrl.Result, error) {
	ctx, logger, cv, bc := input.ctx, input.logger, input.clusterVersion, input.bkeCluster
	desired := input.desired
	current := firstNonEmpty(cv.Status.CurrentVersion, cvensure.OpenFuyaoVersionForBKECluster(bc))
	if current == "" {
		msg := "current version is empty in ClusterVersion.status and BKECluster"
		r.event(cv, corev1.EventTypeWarning, "CurrentVersionMissing", msg)
		return ctrl.Result{}, nil
	}

	if current == desired {
		if err := r.clearUpgradeReady(ctx, bc); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	if cv.Status.Phase == cvv1beta1.ClusterVersionPhaseFailed {
		logger.Info("upgrade blocked: ClusterVersion in Failed phase, waiting for spec change or manual retry",
			"current", current, "desired", desired)
		return ctrl.Result{}, nil
	}

	path, stop, err := r.findUpgradePath(input, current)
	if stop || err != nil {
		return ctrl.Result{}, err
	}
	return r.prepareUpgradeHop(input, current, path)
}

func (r *ClusterVersionReconciler) findUpgradePath(
	input upgradeReconcileInput,
	current string,
) ([]cvv1beta1.UpgradePathRule, bool, error) {
	ctx, cv, desired := input.ctx, input.clusterVersion, input.desired
	if _, err := r.resolveUpgradePath(ctx); err != nil {
		msg := fmt.Sprintf("resolve UpgradePath failed: %v", err)
		r.event(cv, corev1.EventTypeWarning, "UpgradePathResolveFailed", msg)
		if statusErr := r.setStatus(ctx, cv, statusPatch{
			Phase:      phasePtr(cvv1beta1.ClusterVersionPhasePreCheckFailed),
			Conditions: conditionPtr(failureCondition("UpgradePathResolveFailed", msg)),
		}); statusErr != nil {
			return nil, true, statusErr
		}
		return nil, true, nil
	}

	path, err := r.pathService().FindPath(current, desired)
	if err != nil {
		msg := fmt.Sprintf("no valid upgrade path from %s to %s: %v", current, desired, err)
		r.event(cv, corev1.EventTypeWarning, "UpgradePathBlocked", msg)
		if statusErr := r.setStatus(ctx, cv, statusPatch{
			Phase:      phasePtr(cvv1beta1.ClusterVersionPhasePreCheckFailed),
			Conditions: conditionPtr(failureCondition("UpgradePathBlocked", msg)),
		}); statusErr != nil {
			return nil, true, statusErr
		}
		return nil, true, nil
	}
	if len(path) == 0 {
		return nil, true, nil
	}
	return path, false, nil
}

func (r *ClusterVersionReconciler) prepareUpgradeHop(
	input upgradeReconcileInput,
	current string,
	path []cvv1beta1.UpgradePathRule,
) (ctrl.Result, error) {
	ctx, logger, cv, bc := input.ctx, input.logger, input.clusterVersion, input.bkeCluster
	hopTarget := path[0].To
	ri, res, err := input.ensurer.Ensure(ctx, bc, cv, hopTarget)
	if err != nil {
		msg := fmt.Sprintf("ensure ReleaseImage for %s failed: %v", hopTarget, err)
		r.event(cv, corev1.EventTypeWarning, "ReleaseImageEnsureFailed", msg)
		if statusErr := r.setStatus(ctx, cv, statusPatch{
			Phase:      phasePtr(cvv1beta1.ClusterVersionPhasePreCheckFailed),
			Conditions: conditionPtr(failureCondition("ReleaseImageEnsureFailed", msg)),
		}); statusErr != nil {
			return ctrl.Result{}, statusErr
		}
		return ctrl.Result{}, err
	}
	if !res.IsZero() {
		logger.Info("waiting for upgrade ReleaseImage to become Valid",
			"releaseImage", ri.Name, "hop", hopTarget)
		return res, nil
	}

	if readyTarget, ok := bc.Annotations[annotation.CVOUpgradeReadyAnnotationKey]; ok && readyTarget == hopTarget {
		logger.Info("upgrade precheck already applied", "hop", hopTarget)
		return ctrl.Result{}, nil
	}

	if err := r.setStatus(ctx, cv, statusPatch{
		Phase: phasePtr(cvv1beta1.ClusterVersionPhasePreChecking),
	}); err != nil {
		return ctrl.Result{}, err
	}

	if err := r.markUpgradeReady(ctx, bc, cv, path, hopTarget); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("upgrade precheck passed, BKECluster marked upgrade-ready",
		"cluster", client.ObjectKeyFromObject(bc), "from", current, "to", hopTarget, "desired", input.desired)
	r.event(cv, corev1.EventTypeNormal, "UpgradeReady",
		fmt.Sprintf(
			"BKECluster %s/%s marked %s=%s",
			bc.Namespace, bc.Name, annotation.CVOUpgradeReadyAnnotationKey, hopTarget,
		))
	return ctrl.Result{}, nil
}

// isInstallPhase decides whether CV should ensure the install ReleaseImage.
// BC may stamp status.currentVersion during first install before ClusterReady;
// when currentVersion equals desiredVersion during an in-progress install, ensure
// ReleaseImage exists. After the cluster is Ready, manual ReleaseImage deletion
// must not trigger recreation.
func (r *ClusterVersionReconciler) isInstallPhase(
	ctx context.Context,
	bc *bkev1beta1.BKECluster,
	cv *cvv1beta1.ClusterVersion,
) bool {
	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	current := strings.TrimSpace(cv.Status.CurrentVersion)

	if current != "" && current != desired {
		return false
	}

	if current != "" && current == desired {
		if bc.Status.ClusterStatus == bkev1beta1.ClusterStatusReady {
			return false
		}
		hasRI, err := r.hasReleaseImageForVersion(ctx, cv.Namespace, current)
		if err != nil || !hasRI {
			return true
		}
		return false
	}

	if bc.Status.ClusterStatus == bkev1beta1.ClusterStatusReady {
		return false
	}
	return true
}

func (r *ClusterVersionReconciler) hasReleaseImageForVersion(
	ctx context.Context,
	namespace, version string,
) (bool, error) {
	list := &cvv1beta1.ReleaseImageList{}
	if err := r.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return false, err
	}
	for i := range list.Items {
		if strings.TrimSpace(list.Items[i].Spec.Version) == version {
			return true, nil
		}
	}
	return false, nil
}

// statusPatch carries optional ClusterVersion.status fields to update.
// Fields omitted from the patch are left unchanged on the API object.
type statusPatch struct {
	Phase      *cvv1beta1.ClusterVersionPhase
	Conditions *[]cvv1beta1.ClusterVersionCondition
}

func (r *ClusterVersionReconciler) setStatus(
	ctx context.Context,
	cv *cvv1beta1.ClusterVersion,
	patch statusPatch,
) error {
	if patch.Phase == nil && patch.Conditions == nil {
		return nil
	}
	orig := cv.DeepCopy()
	if patch.Phase != nil {
		cv.Status.Phase = *patch.Phase
	}
	if patch.Conditions != nil {
		cv.Status.Conditions = *patch.Conditions
	}
	return r.Status().Patch(ctx, cv, client.MergeFrom(orig))
}

func phasePtr(phase cvv1beta1.ClusterVersionPhase) *cvv1beta1.ClusterVersionPhase {
	return &phase
}

func conditionPtr(conditions []cvv1beta1.ClusterVersionCondition) *[]cvv1beta1.ClusterVersionCondition {
	return &conditions
}

func failureCondition(reason, message string) []cvv1beta1.ClusterVersionCondition {
	return []cvv1beta1.ClusterVersionCondition{{
		Type:               "Ready",
		Status:             metav1.ConditionFalse,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
	}}
}

func (r *ClusterVersionReconciler) resolveUpgradePath(ctx context.Context) (*cvv1beta1.UpgradePath, error) {
	upList := &cvv1beta1.UpgradePathList{}
	if err := r.List(ctx, upList); err != nil {
		return nil, err
	}
	if len(upList.Items) == 0 {
		return nil, fmt.Errorf("UpgradePath not found")
	}
	selected := &upList.Items[0]
	for i := range upList.Items {
		if upList.Items[i].Status.Phase == cvv1beta1.UpgradePathPhaseActive {
			selected = &upList.Items[i]
			break
		}
	}
	return selected, r.pathService().Load(selected.Spec.Paths, selected.Spec.Versions, selected.Status.LastDigest)
}

func (r *ClusterVersionReconciler) markUpgradeReady(
	ctx context.Context,
	bc *bkev1beta1.BKECluster,
	cv *cvv1beta1.ClusterVersion,
	path []cvv1beta1.UpgradePathRule,
	target string,
) error {
	orig := bc.DeepCopy()
	if bc.Annotations == nil {
		bc.Annotations = map[string]string{}
	}
	bc.Annotations[annotation.CVOUpgradeReadyAnnotationKey] = target
	bc.Annotations[annotation.CVOClusterVersionAnnotationKey] = cv.Name
	bc.Annotations[annotation.CVOUpgradePathAnnotationKey] = formatPath(path)
	return r.Patch(ctx, bc, client.MergeFrom(orig))
}

func (r *ClusterVersionReconciler) clearUpgradeReady(ctx context.Context, bc *bkev1beta1.BKECluster) error {
	if bc.Annotations == nil {
		return nil
	}
	if _, ok := bc.Annotations[annotation.CVOUpgradeReadyAnnotationKey]; !ok {
		return nil
	}
	orig := bc.DeepCopy()
	delete(bc.Annotations, annotation.CVOUpgradeReadyAnnotationKey)
	delete(bc.Annotations, annotation.CVOUpgradePathAnnotationKey)
	delete(bc.Annotations, annotation.CVOClusterVersionAnnotationKey)
	return r.Patch(ctx, bc, client.MergeFrom(orig))
}

func (r *ClusterVersionReconciler) pathService() pathstore.Loader {
	if r.PathService == nil {
		r.PathService = pathstore.NewService()
	}
	return r.PathService
}

func (r *ClusterVersionReconciler) event(obj client.Object, eventType, reason, message string) {
	if r.Recorder != nil {
		r.Recorder.Event(obj, eventType, reason, message)
	}
}

func formatPath(path []cvv1beta1.UpgradePathRule) string {
	segments := make([]string, 0, len(path))
	for _, edge := range path {
		segments = append(segments, edge.From+"->"+edge.To)
	}
	return strings.Join(segments, ",")
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// SetupWithManager registers the controller with the Manager.
func (r *ClusterVersionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cvv1beta1.ClusterVersion{}).
		Named("clusterversion").
		Complete(r)
}

var _ reconcile.Reconciler = &ClusterVersionReconciler{}
