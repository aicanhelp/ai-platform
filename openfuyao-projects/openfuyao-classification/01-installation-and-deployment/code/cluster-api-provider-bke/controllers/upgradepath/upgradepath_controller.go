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

package upgradepath

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	pathstore "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const upgradePathControllerName = "upgradepath"

var upgradePathLogger = log.ControllerLogger(upgradePathControllerName)

type UpgradePathReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	// PathService is the in-memory upgrade path graph loader. If nil, a default Service
	// is created lazily via pathService().
	PathService pathstore.Loader
}

// +kubebuilder:rbac:groups=config.openfuyao.com,resources=upgradepaths,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=upgradepaths/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=upgradepaths/finalizers,verbs=update

func (r *UpgradePathReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := upgradePathLogger.With("upgradepath", req.NamespacedName.String())

	up := &upv1alpha1.UpgradePath{}
	if err := r.Get(ctx, req.NamespacedName, up); err != nil {
		if apierrors.IsNotFound(err) {
			r.pathService().Clear()
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	digest := up.Annotations[pathstore.OCIDigestAnnotation]

	var phase upv1alpha1.UpgradePathPhase
	var message string
	if err := r.pathService().Load(up.Spec.Paths, up.Spec.Versions, digest); err != nil {
		logger.Errorf("invalid upgrade path rules: %v", err)
		phase = upv1alpha1.UpgradePathPhaseInvalid
		message = err.Error()
	} else {
		logger.Infof("upgrade path graph loaded, paths=%d versions=%d digest=%s",
			len(up.Spec.Paths), len(up.Spec.Versions), digest)
		phase = upv1alpha1.UpgradePathPhaseActive
	}

	if len(up.Spec.Paths) > 0 &&
		up.Status.LastDigest == digest &&
		up.Status.Phase == phase &&
		up.Status.PathCount == len(up.Spec.Paths) {
		logger.Debugf("upgrade path status unchanged, skipping status patch")
		return ctrl.Result{}, nil
	}

	return ctrl.Result{}, r.updateStatus(ctx, up, phase, digest, message)
}

// FindPath delegates to the PathService to find the shortest upgrade path between
// two versions. This is a convenience method for external callers (e.g. other controllers).
func (r *UpgradePathReconciler) FindPath(from, to string) ([]upv1alpha1.UpgradePathRule, error) {
	return r.pathService().FindPath(from, to)
}

func (r *UpgradePathReconciler) pathService() pathstore.Loader {
	if r.PathService == nil {
		r.PathService = pathstore.NewService()
	}
	return r.PathService
}

func (r *UpgradePathReconciler) updateStatus(
	ctx context.Context,
	up *upv1alpha1.UpgradePath,
	phase upv1alpha1.UpgradePathPhase,
	digest string,
	message string) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest := &upv1alpha1.UpgradePath{}
		if err := r.Get(ctx, client.ObjectKeyFromObject(up), latest); err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}

		now := metav1.Now()
		latest.Status.Phase = phase
		latest.Status.LastDigest = digest
		latest.Status.PathCount = len(latest.Spec.Paths)
		latest.Status.LastCheckedAt = &now

		cond := metav1.Condition{
			Type:               "Validated",
			LastTransitionTime: now,
		}
		if message != "" {
			cond.Status = metav1.ConditionFalse
			cond.Reason = "InvalidRules"
			cond.Message = message
		} else {
			cond.Status = metav1.ConditionTrue
			cond.Reason = "ValidRules"
			cond.Message = "Upgrade path rules validated successfully"
		}
		meta.SetStatusCondition(&latest.Status.Conditions, cond)

		return r.Status().Update(ctx, latest)
	})
}

// SetupWithManager registers the reconciler with the controller manager to watch
// UpgradePath CRs.
func (r *UpgradePathReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&upv1alpha1.UpgradePath{}).
		Named("upgradepath").
		Complete(r)
}
