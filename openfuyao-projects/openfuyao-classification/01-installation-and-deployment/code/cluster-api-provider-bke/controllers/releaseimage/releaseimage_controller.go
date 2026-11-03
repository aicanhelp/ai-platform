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

package releaseimage

import (
	"context"
	"fmt"
	"sort"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	riv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/compatibility"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/imageref"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const releaseImageFinalizer = "releaseimage.config.openfuyao.com/finalizer"

const releaseImageControllerName = "releaseimage"

// ReleaseImageReconciler reconciles a ReleaseImage object.
type ReleaseImageReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	Store         *manifest.Store
	Compatibility *compatibility.Engine
	OCIClient     *oci.Client
}

// +kubebuilder:rbac:groups=config.openfuyao.com,resources=releaseimages,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=releaseimages/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=config.openfuyao.com,resources=releaseimages/finalizers,verbs=update
// +kubebuilder:rbac:groups=bke.bocloud.com,resources=bkeclusters,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop.
func (r *ReleaseImageReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.ControllerLogger(releaseImageControllerName).With("releaseImage", req.NamespacedName)

	ri := &riv1beta1.ReleaseImage{}
	if err := r.Get(ctx, req.NamespacedName, ri); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !ri.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(ctx, logger, ri)
	}

	if !controllerutil.ContainsFinalizer(ri, releaseImageFinalizer) {
		orig := ri.DeepCopy()
		controllerutil.AddFinalizer(ri, releaseImageFinalizer)
		if err := r.Patch(ctx, ri, client.MergeFrom(orig)); err != nil {
			return ctrl.Result{}, fmt.Errorf("add release image finalizer: %w", err)
		}
	}

	store := r.releaseStore()
	engine := r.compatibilityEngine()

	refs, err := r.buildReleaseRefs(ctx, ri)
	if err != nil {
		logger.Error(err, "release image repository resolve failed", "releaseImage", req.NamespacedName)
		return ctrl.Result{}, r.updateStatus(ctx, ri, releaseImageStatusUpdate{
			phase: riv1beta1.ReleaseImagePhaseInvalid, message: err.Error(),
		})
	}

	var (
		bundle  *manifest.Bundle
		files   *manifest.BundleFiles
		usedRef manifest.ReleaseRef
		lastErr error
	)
	for _, ref := range refs {
		bundle, files, err = store.RefreshRelease(ctx, ref)
		if err == nil {
			usedRef = ref
			logger.Info("release image refreshed from OCI", "ociRef", ref.OCIRef)
			break
		}
		lastErr = fmt.Errorf("refresh release from %s failed: %w", ref.OCIRef, err)
		logger.Warn("release image refresh failed, trying next OCI ref", "ociRef", ref.OCIRef, "error", err.Error())
	}
	if bundle == nil {
		msg := "release image refresh failed"
		if lastErr != nil {
			msg = lastErr.Error()
		}
		logger.Errorf("release image refresh failed: %v", lastErr)
		return ctrl.Result{}, r.updateStatus(ctx, ri, releaseImageStatusUpdate{
			phase: riv1beta1.ReleaseImagePhaseInvalid, message: msg,
		})
	}

	report := engine.Check(ctx, bundle)
	if !report.Allowed {
		return ctrl.Result{}, r.updateStatus(ctx, ri, releaseImageStatusUpdate{
			phase:  riv1beta1.ReleaseImagePhaseCompatibilityFailed,
			digest: bundle.Digest, message: report.Detail(), bundle: bundle,
		})
	}

	if err := store.CommitRelease(usedRef, bundle, files); err != nil {
		logger.Warn("commit validated release bundle to cache failed", "error", err.Error())
	}

	return ctrl.Result{}, r.updateStatus(ctx, ri, releaseImageStatusUpdate{
		phase: riv1beta1.ReleaseImagePhaseValid, digest: bundle.Digest,
		message: "release image validated", bundle: bundle,
	})
}

func (r *ReleaseImageReconciler) reconcileDelete(
	ctx context.Context,
	logger interface {
		Info(msg string, keysAndValues ...interface{})
		Warn(msg string, keysAndValues ...interface{})
	},
	ri *riv1beta1.ReleaseImage,
) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(ri, releaseImageFinalizer) {
		return ctrl.Result{}, nil
	}

	store := r.releaseStore()
	evicted := map[string]struct{}{}
	for _, ref := range cacheRefsForReleaseImage(ri) {
		key := ref.CacheKey()
		if _, ok := evicted[key]; ok {
			continue
		}
		if err := store.EvictRelease(ref); err != nil {
			logger.Warn("evict release bundle cache failed", "cacheKey", key, "error", err.Error())
		} else {
			logger.Info("evicted release bundle cache", "cacheKey", key)
		}
		evicted[key] = struct{}{}
	}

	orig := ri.DeepCopy()
	controllerutil.RemoveFinalizer(ri, releaseImageFinalizer)
	if err := r.Patch(ctx, ri, client.MergeFrom(orig)); err != nil {
		return ctrl.Result{}, fmt.Errorf("remove release image finalizer: %w", err)
	}
	return ctrl.Result{}, nil
}

func (r *ReleaseImageReconciler) buildReleaseRefs(
	ctx context.Context,
	ri *riv1beta1.ReleaseImage,
) ([]manifest.ReleaseRef, error) {
	ociRefs, err := imageref.ReleaseImageRefs(
		ctx,
		r.Client,
		ri.Namespace,
		ri.Annotations[imageref.AnnotationBKEClusterName],
		ri.Spec.Version,
	)
	if err != nil {
		return nil, err
	}
	refs := make([]manifest.ReleaseRef, 0, len(ociRefs))
	for _, ociRef := range ociRefs {
		refs = append(refs, releaseRefFromSpec(ri, ociRef))
	}
	if len(refs) == 0 {
		return nil, fmt.Errorf("no release OCI refs resolved for version %q", ri.Spec.Version)
	}
	return refs, nil
}

func releaseRefFromSpec(ri *riv1beta1.ReleaseImage, ociRef string) manifest.ReleaseRef {
	return manifest.ReleaseRef{
		Version:            ri.Spec.Version,
		OCIRef:             ociRef,
		Digest:             ri.Spec.Digest,
		VerifySignature:    ri.Spec.VerifySignature,
		SignatureKey:       ri.Spec.SignatureKey,
		AllowCacheFallback: ri.Spec.AllowCacheFallback,
	}
}

func cacheRefsForReleaseImage(ri *riv1beta1.ReleaseImage) []manifest.ReleaseRef {
	if ri == nil {
		return nil
	}
	return []manifest.ReleaseRef{{
		Version:            ri.Spec.Version,
		Digest:             strings.TrimSpace(ri.Spec.Digest),
		VerifySignature:    ri.Spec.VerifySignature,
		SignatureKey:       ri.Spec.SignatureKey,
		AllowCacheFallback: ri.Spec.AllowCacheFallback,
	}}
}

func (r *ReleaseImageReconciler) releaseStore() *manifest.Store {
	if r.Store != nil {
		return r.Store
	}
	return manifest.NewStore(manifest.ReleaseCacheDir(), manifest.OCIPuller{Client: r.OCIClient}, nil)
}

func (r *ReleaseImageReconciler) compatibilityEngine() *compatibility.Engine {
	if r.Compatibility != nil {
		return r.Compatibility
	}
	return compatibility.NewEngine()
}

type releaseImageStatusUpdate struct {
	phase   riv1beta1.ReleaseImagePhase
	digest  string
	message string
	bundle  *manifest.Bundle
}

func (r *ReleaseImageReconciler) updateStatus(
	ctx context.Context,
	ri *riv1beta1.ReleaseImage,
	update releaseImageStatusUpdate,
) error {
	orig := ri.DeepCopy()
	ri.Status.Phase = update.phase
	ri.Status.Digest = update.digest
	ri.Status.Message = update.message
	ri.Status.CompatibilityReport = update.message
	ri.Status.ComponentCount = 0
	ri.Status.Components = nil
	ri.Status.Source = ""
	ri.Status.CacheFallback = false
	if update.bundle != nil {
		ri.Status.ComponentCount = len(update.bundle.Components)
		ri.Status.Components = componentStatuses(update.bundle)
		ri.Status.Source = update.bundle.Source
		ri.Status.CacheFallback = update.bundle.CacheFallback
	}
	now := metav1.Now()
	ri.Status.ValidatedAt = &now
	if err := r.Status().Patch(ctx, ri, client.MergeFrom(orig)); err != nil {
		return fmt.Errorf("update release image status: %w", err)
	}
	return nil
}

func componentStatuses(bundle *manifest.Bundle) []riv1beta1.ComponentStatus {
	if bundle == nil || len(bundle.Components) == 0 {
		return nil
	}
	components := make([]riv1beta1.ComponentStatus, 0, len(bundle.Components))
	for _, component := range bundle.Components {
		components = append(components, riv1beta1.ComponentStatus{
			Name:    component.Spec.Name,
			Version: component.Spec.Version,
			Type:    component.Spec.Type,
		})
	}
	sort.Slice(components, func(i, j int) bool {
		if components[i].Name == components[j].Name {
			return components[i].Version < components[j].Version
		}
		return components[i].Name < components[j].Name
	})
	return components
}

// SetupWithManager sets up the controller with the Manager.
func (r *ReleaseImageReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&riv1beta1.ReleaseImage{}).
		Named("releaseimage").
		Complete(r)
}
