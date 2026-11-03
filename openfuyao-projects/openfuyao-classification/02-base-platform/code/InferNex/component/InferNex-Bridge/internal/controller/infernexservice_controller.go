/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	infernexRuntimeLabel         = "infernex.io/runtime"
	infernexRuntimeValue         = "true"
	defaultAggregateTemplateName = "infernex-default-aggregate-template"
	defaultPDTemplateName        = "infernex-default-pd-template"
	infernexServiceFinalizer     = "infernex.infernex.io/infernexservice"
)

// ErrTemplateNotFound is returned when a baseRefs template (InferNexServiceConfig)
// is not found in the template namespace. Reconcilers should treat this as a persistent
// configuration error — update status and stop retrying — rather than retrying forever.
var ErrTemplateNotFound = errors.New("infernexserviceconfig template not found")

// InferNexServiceReconciler reconciles a InferNexService object
type InferNexServiceReconciler struct {
	client.Client
	Scheme            *runtime.Scheme
	TemplateNamespace string
	Recorder          record.EventRecorder
}

// +kubebuilder:rbac:groups=infernex.infernex.io,resources=infernexservices,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=infernex.infernex.io,resources=infernexservices/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=infernex.infernex.io,resources=infernexservices/finalizers,verbs=update
// +kubebuilder:rbac:groups=infernex.infernex.io,resources=infernexserviceconfigs,verbs=create;get;list;watch
// +kubebuilder:rbac:groups=apps,resources=deployments;daemonsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=rolebindings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterrolebindings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=serving.kserve.io,resources=llminferenceservices,verbs=get;list;watch
// PD orchestrator sub-controllers embed ClusterRoles; the manager must hold the same rules (RBAC escalation).
// +kubebuilder:rbac:groups=*,resources=*,verbs=get;list;watch;patch;update
// +kubebuilder:rbac:groups=apps,resources=deployments/scale,verbs=get;patch;update
// +kubebuilder:rbac:groups=apps,resources=statefulsets/scale,verbs=get;patch;update
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling.openfuyao.com,resources=resourcescalinggroups,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling.openfuyao.com,resources=resourcescalinggroups/finalizers,verbs=update
// +kubebuilder:rbac:groups=autoscaling.openfuyao.com,resources=resourcescalinggroups/status,verbs=get;patch;update
// +kubebuilder:rbac:groups=autoscaling.openfuyao.com,resources=resourcescalinggroups/scale,verbs=get;patch;update
// +kubebuilder:rbac:groups=elasticscaler.io,resources=elasticscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=elasticscaler.io,resources=elasticscalers/finalizers,verbs=update
// +kubebuilder:rbac:groups=elasticscaler.io,resources=elasticscalers/status,verbs=get;patch;update
// +kubebuilder:rbac:groups=tidal.io,resources=tidals,verbs=get;list;watch
// +kubebuilder:rbac:groups=tidal.io,resources=tidals/finalizers,verbs=update
// +kubebuilder:rbac:groups=tidal.io,resources=tidals/status,verbs=get;update
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=replicasets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=controllerrevisions,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=leaderworkerset.x-k8s.io,resources=leaderworkersets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=workloads.x-k8s.io,resources=rolebasedgroups,verbs=get;list;watch
// Namespaced leader-election Role template created in InferNexService namespaces.
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// Gateway API (managed Gateway / HTTPRoute / InferencePool; deletes ownership-gated).
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=gateways,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=httproutes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.envoyproxy.io,resources=envoyproxies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=inference.networking.k8s.io,resources=inferencepools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=inference.networking.x-k8s.io,resources=inferenceobjectives,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *InferNexServiceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	infsvc, err := r.reconcileFetchInferNexService(ctx, req)
	if err != nil {
		return ctrl.Result{}, err
	}
	if infsvc == nil {
		return ctrl.Result{}, nil
	}
	// Migrate legacy canonical Mooncake Services before any per-owner cleanup
	// can observe their old infernex.io/owner labels or OwnerReferences.
	if err := r.migrateLegacySingletonServices(ctx, infsvc.Namespace); err != nil {
		return ctrl.Result{}, err
	}
	if !infsvc.DeletionTimestamp.IsZero() {
		return r.reconcileInferNexServiceDeletion(ctx, infsvc)
	}
	if !controllerutil.ContainsFinalizer(infsvc, infernexServiceFinalizer) {
		controllerutil.AddFinalizer(infsvc, infernexServiceFinalizer)
		if err := r.Update(ctx, infsvc); err != nil {
			if apierrors.IsConflict(err) {
				return ctrl.Result{Requeue: true}, nil
			}
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
	deleted, err := r.syncSourceRefLifecycle(ctx, infsvc)
	if err != nil {
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}
	desired, effectiveSpec, mode, err := r.reconcileWorkloadsAndRBAC(ctx, infsvc)
	if err != nil {
		r.recordWarn(infsvc, "ReconcileFailed", "reconcile workloads and rbac failed: %v", err)
		if errors.Is(err, ErrTemplateNotFound) {
			// Persistent configuration error: baseRefs template (InferNexServiceConfig)
			// not found. Update status to surface the error and stop retrying, otherwise
			// the controller would loop forever on an unrecoverable condition.
			if statusErr := r.persistReconcileFailureStatus(ctx, req, err); statusErr != nil {
				return ctrl.Result{}, statusErr
			}
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	return r.reconcilePersistStatus(ctx, req, desired, effectiveSpec, mode)
}

// reconcileInferNexServiceDeletion removes Gateway API objects created by this controller (ownership / infernex labels),
// then drops the InferNexService finalizer so garbage collection can complete.
func (r *InferNexServiceReconciler) reconcileInferNexServiceDeletion(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (ctrl.Result, error) {
	// Keep this migration in the deletion path as well as the normal reconcile
	// path. It is idempotent and protects callers/tests that invoke deletion
	// reconciliation directly during upgrades.
	if err := r.migrateLegacySingletonServices(ctx, infsvc.Namespace); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.syncMooncakeSingletonReplicas(ctx, infsvc); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.deleteManagedGatewayRouting(ctx, infsvc); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.releaseSingletonOwnerReferences(ctx, infsvc); err != nil {
		return ctrl.Result{}, err
	}
	if !controllerutil.ContainsFinalizer(infsvc, infernexServiceFinalizer) {
		return ctrl.Result{}, nil
	}
	controllerutil.RemoveFinalizer(infsvc, infernexServiceFinalizer)
	if err := r.Update(ctx, infsvc); err != nil {
		if apierrors.IsConflict(err) {
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	return ctrl.Result{}, nil
}

func (r *InferNexServiceReconciler) releaseSingletonOwnerReferences(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	// Mooncake Deployments and Services are namespace singletons and are not
	// owner-referenced by an InferNexService. ServiceAccounts remain shared and
	// use additive non-controller owner references.
	return r.syncSingletonComponentSAOwnerRefsCleanup(ctx, owner, nil)
}

// reconcileFetchInferNexService loads the InferNexService or ensures one from a linked LLMInferenceService when missing.
func (r *InferNexServiceReconciler) reconcileFetchInferNexService(
	ctx context.Context,
	req ctrl.Request,
) (*infernexv1alpha1.InferNexService, error) {
	infsvc := &infernexv1alpha1.InferNexService{}
	if err := r.Get(ctx, req.NamespacedName, infsvc); err != nil {
		if apierrors.IsNotFound(err) {
			if ensureErr := r.buildInferNexServiceFromLLMInferenceService(ctx, req.NamespacedName); ensureErr != nil {
				return nil, ensureErr
			}
			return nil, nil
		}
		return nil, client.IgnoreNotFound(err)
	}
	return infsvc, nil
}

// reconcileWorkloadsAndRBAC applies spec templates, reconciles Deployments/Services, prunes extras, and syncs controller RBAC.
func (r *InferNexServiceReconciler) reconcileWorkloadsAndRBAC(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (map[string]componentPlan, infernexv1alpha1.InferNexServiceSpec, string, error) {
	effectiveSpec, mode, err := r.buildEffectiveSpec(ctx, infsvc)
	if err != nil {
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	desired, err := r.buildDesiredComponents(ctx, infsvc, effectiveSpec)
	if err != nil {
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	// Mirror KServe delete-on-stop: when the linked LLMInferenceService carries
	// serving.kserve.io/stop=true, KServe deletes its own serving workloads. Empty the
	// desired set so the existing pruners release every per-owner sidecar workload/Service/RBAC.
	// Namespace singleton Mooncake Deployments/Services are kept independently of
	// any one InferNexService and are scaled by syncMooncakeSingletonReplicas;
	// shared ServiceAccounts use additive owner references.
	stopped, err := r.linkedLLMIsStopped(ctx, infsvc)
	if err != nil {
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	if stopped {
		if len(desired) > 0 {
			r.recordNormal(infsvc, "Stopped",
				"linked LLMInferenceService is stopped; releasing %d managed component(s)", len(desired))
		}
		desired = map[string]componentPlan{}
	}
	// Per-service opt-out: drop components named in the infernex.io/disabled-components
	// annotation so users can disable e.g. eagle-eye for one inference service without
	// editing the namespace-wide default template. The pruners then release any already
	// running per-service instance; namespace singleton Mooncake resources are managed
	// independently and are only scaled by the namespace activity check.
	if !stopped {
		disabled, err := r.disabledComponents(ctx, infsvc)
		if err != nil {
			return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
		}
		for name := range disabled {
			if _, ok := desired[name]; ok {
				delete(desired, name)
				r.recordNormal(infsvc, "ComponentDisabled",
					"component %q disabled via %s annotation", name, infernexDisabledComponentsAnnotationKey)
			}
		}
	}
	for name, comp := range desired {
		if err := r.reconcileComponentWorkload(ctx, infsvc, name, comp, effectiveSpec.Engine); err != nil {
			r.recordWarn(infsvc, "ComponentReconcileFailed", "workload for component %q failed: %v", name, err)
			return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
		}
		if err := r.reconcileComponentService(ctx, infsvc, name, comp); err != nil {
			r.recordWarn(infsvc, "ComponentReconcileFailed", "service for component %q failed: %v", name, err)
			return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
		}
	}
	if err := r.pruneOrphanedComponents(ctx, infsvc, desired); err != nil {
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	if err := r.syncMooncakeSingletonReplicas(ctx, infsvc); err != nil {
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	if err := r.reconcileGatewayRouting(ctx, infsvc, effectiveSpec); err != nil {
		r.recordWarn(infsvc, "GatewayRoutingFailed", "gateway routing reconcile failed: %v", err)
		return nil, infernexv1alpha1.InferNexServiceSpec{}, "", err
	}
	return desired, effectiveSpec, mode, nil
}

// pruneOrphanedComponents releases component Deployments/LeaderWorkerSets/DaemonSets/Services
// and syncs controller/proxy RBAC cleanup for everything no longer in the desired set.
func (r *InferNexServiceReconciler) pruneOrphanedComponents(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	if err := r.pruneComponentDeployments(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.pruneComponentLeaderWorkerSets(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.pruneCacheIndexerConfigMap(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.pruneComponentDaemonSets(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.pruneComponentServices(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.syncProxyRBACCleanup(ctx, infsvc, desired); err != nil {
		return err
	}
	if err := r.syncComponentControllerRBACCleanup(ctx, infsvc, desired); err != nil {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) reconcilePersistStatus(
	ctx context.Context,
	req ctrl.Request,
	desired map[string]componentPlan,
	effectiveSpec infernexv1alpha1.InferNexServiceSpec,
	mode string,
) (ctrl.Result, error) {
	log := logf.FromContext(ctx)
	fresh := &infernexv1alpha1.InferNexService{}
	if err := r.Get(ctx, req.NamespacedName, fresh); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	st, err := r.computeInferNexServiceStatus(ctx, fresh, desired, effectiveSpec, mode)
	if err != nil {
		return ctrl.Result{}, err
	}
	oldConditions := append([]metav1.Condition(nil), fresh.Status.Conditions...)
	fresh.Status = st
	if err := r.Status().Update(ctx, fresh); err != nil {
		if apierrors.IsConflict(err) {
			return ctrl.Result{Requeue: true}, nil
		}
		log.Error(err, "failed to update status")
		return ctrl.Result{}, err
	}
	r.emitGatewayRoutingConditionEvent(fresh, oldConditions, st.Conditions)
	r.recordNormal(fresh, "ReconcileSucceeded", "reconcile completed: mode=%s ready=%t", mode, st.Ready)
	return ctrl.Result{}, nil
}

func (r *InferNexServiceReconciler) syncSourceRefLifecycle(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (bool, error) {
	if infsvc.Spec.SourceRef == nil {
		return false, nil
	}
	if strings.TrimSpace(infsvc.Spec.SourceRef.Kind) != "LLMInferenceService" {
		return false, nil
	}

	sourceNS := strings.TrimSpace(infsvc.Spec.SourceRef.Namespace)
	if sourceNS == "" {
		sourceNS = infsvc.Namespace
	}
	sourceName := strings.TrimSpace(infsvc.Spec.SourceRef.Name)
	if sourceName == "" {
		sourceName = infsvc.Name
	}

	llm, err := r.getLLMInferenceService(ctx, types.NamespacedName{
		Namespace: sourceNS,
		Name:      sourceName,
	}, infsvc.Spec.SourceRef.APIVersion)
	if err != nil {
		if apierrors.IsNotFound(err) {
			if delErr := r.Delete(ctx, infsvc); delErr != nil && !apierrors.IsNotFound(delErr) {
				return false, delErr
			}
			return true, nil
		}
		return false, err
	}

	if llm.GetLabels()[infernexRuntimeLabel] != infernexRuntimeValue {
		if err := r.Delete(ctx, infsvc); err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}

	return false, nil
}

// getInferNexServiceConfig loads an InferNexServiceConfig by name from controller TemplateNamespace only.
// Templates are control-plane assets and must be installed in controller namespace.
func (r *InferNexServiceReconciler) getInferNexServiceConfig(
	ctx context.Context,
	name string,
) (*infernexv1alpha1.InferNexServiceConfig, error) {
	cfg := &infernexv1alpha1.InferNexServiceConfig{}
	templateNS := strings.TrimSpace(r.TemplateNamespace)
	if templateNS == "" {
		return nil, fmt.Errorf("template namespace is empty: set INFERNEX_TEMPLATE_NAMESPACE for controller")
	}
	if err := r.Get(ctx, types.NamespacedName{Namespace: templateNS, Name: name}, cfg); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("%w: %w", ErrTemplateNotFound, err)
		}
		return nil, fmt.Errorf("load InferNexServiceConfig %q from namespace %q: %w", name, templateNS, err)
	}
	return cfg, nil
}

func (r *InferNexServiceReconciler) buildEffectiveSpec(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (infernexv1alpha1.InferNexServiceSpec, string, error) {
	effective := infsvc.Spec.DeepCopy()
	mode, err := r.computeServingMode(ctx, infsvc)
	if err != nil {
		return *effective, "", err
	}

	baseRefs := effective.BaseRefs
	// Backward compatible path: no explicit baseRefs => use mode-based default template.
	if len(baseRefs) == 0 {
		templateName := defaultAggregateTemplateName
		if mode == "pd" {
			templateName = defaultPDTemplateName
		}
		baseRefs = []infernexv1alpha1.NamedRef{{Name: templateName}}
	}

	for _, ref := range baseRefs {
		name := strings.TrimSpace(ref.Name)
		if name == "" {
			continue
		}
		cfg, err := r.getInferNexServiceConfig(ctx, name)
		if err != nil {
			if errors.Is(err, ErrTemplateNotFound) {
				templateNS := strings.TrimSpace(r.TemplateNamespace)
				return *effective, mode, fmt.Errorf(
					"load InferNexServiceConfig template %q from namespace %q: %w",
					name,
					templateNS,
					err,
				)
			}
			return *effective, mode, err
		}
		mergeSpecFromTemplate(effective, cfg.Spec.InferNexServiceSpec)
	}
	// Direct mode: mode was derived from infsvc.Spec before template merge; align with merged engine.
	if infsvc.Spec.SourceRef == nil {
		if effective.Engine != nil && EngineIsPDMode(effective.Engine) {
			mode = "pd"
		} else {
			mode = "aggregate"
		}
	}
	platformName := defaultAggregateTemplateName
	if mode == "pd" {
		platformName = defaultPDTemplateName
	}
	if err := r.mergePlatformDefaultTemplates(ctx, infsvc, effective, platformName); err != nil {
		return *effective, mode, err
	}
	return *effective, mode, nil
}

func (r *InferNexServiceReconciler) computeServingMode(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
) (string, error) {
	// Linked llmisvc mode may come from raw spec.prefill, merged baseRefs templates, or
	// already-materialized prefill workloads; avoid relying on one raw field only.
	if infsvc.Spec.SourceRef != nil {
		hasPD, err := r.linkedLLMHasPDMode(ctx, infsvc)
		if err != nil {
			return "", err
		}
		if hasPD {
			return "pd", nil
		}
		return "aggregate", nil
	}

	// Direct InferNexService mode derives from its own engine block.
	if infsvc.Spec.Engine != nil && EngineIsPDMode(infsvc.Spec.Engine) {
		return "pd", nil
	}
	return "aggregate", nil
}

// SetupWithManager sets up the controller with the Manager.F
func (r *InferNexServiceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.Recorder == nil {
		r.Recorder = mgr.GetEventRecorderFor("infernexservice-controller")
	}
	enqueueFromLinkedLLM := func(_ context.Context, obj client.Object) []reconcile.Request {
		return []reconcile.Request{{
			NamespacedName: types.NamespacedName{
				Name:      obj.GetName(),
				Namespace: obj.GetNamespace(),
			},
		}}
	}
	llmV1 := &unstructured.Unstructured{}
	llmV1.SetGroupVersionKind(schema.GroupVersionKind{
		Group: llmInferenceServiceGroup, Version: "v1alpha1", Kind: llmInferenceServiceKind,
	})
	llmV2 := &unstructured.Unstructured{}
	llmV2.SetGroupVersionKind(schema.GroupVersionKind{
		Group: llmInferenceServiceGroup, Version: "v1alpha2", Kind: llmInferenceServiceKind,
	})
	enqueueFromMooncakeSingleton := handler.EnqueueRequestsFromMapFunc(r.enqueueFromMooncakeSingleton)
	mooncakeSingletonPredicate := builder.WithPredicates(predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return isMooncakeSingletonObject(obj)
	}))

	return ctrl.NewControllerManagedBy(mgr).
		For(&infernexv1alpha1.InferNexService{}).
		Watches(llmV1, handler.EnqueueRequestsFromMapFunc(enqueueFromLinkedLLM)).
		Watches(llmV2, handler.EnqueueRequestsFromMapFunc(enqueueFromLinkedLLM)).
		Watches(&infernexv1alpha1.InferNexServiceConfig{}, handler.EnqueueRequestsFromMapFunc(r.enqueueFromConfig)).
		Watches(&appsv1.Deployment{}, enqueueFromMooncakeSingleton, mooncakeSingletonPredicate).
		Watches(&corev1.Service{}, enqueueFromMooncakeSingleton, mooncakeSingletonPredicate).
		Owns(&appsv1.Deployment{}).
		Owns(&appsv1.DaemonSet{}).
		Owns(&lwsv1.LeaderWorkerSet{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		Named("infernexservice").
		Complete(r)
}

func isMooncakeSingletonObject(obj client.Object) bool {
	if obj == nil {
		return false
	}
	if obj.GetLabels()["infernex.io/singleton"] == "mooncake" {
		return true
	}
	switch obj.GetName() {
	case "mooncake-master-deployment", "redis-server-deployment",
		"mooncake-master-service", "redis-service":
		return true
	default:
		return false
	}
}

// enqueueFromMooncakeSingleton requeues every InferNexService in the affected
// namespace. Singleton resources have no InferNexService owner reference, so
// Owns(...) cannot provide this relationship.
func (r *InferNexServiceReconciler) enqueueFromMooncakeSingleton(ctx context.Context, obj client.Object) []reconcile.Request {
	if obj == nil {
		return nil
	}
	var services infernexv1alpha1.InferNexServiceList
	if err := r.List(ctx, &services, client.InNamespace(obj.GetNamespace())); err != nil {
		logf.FromContext(ctx).Error(err, "failed to list InferNexServices for singleton enqueue", "namespace", obj.GetNamespace(), "name", obj.GetName())
		return nil
	}
	requests := make([]reconcile.Request, 0, len(services.Items))
	for i := range services.Items {
		requests = append(requests, reconcile.Request{NamespacedName: types.NamespacedName{
			Namespace: services.Items[i].Namespace,
			Name:      services.Items[i].Name,
		}})
	}
	return requests
}

// enqueueFromConfig maps InferNexServiceConfig create/update events to the
// InferNexServices that reference it via baseRefs or default template names.
// This ensures that after a deleted template is restored, the affected
// InferNexServices are automatically re-reconciled.
func (r *InferNexServiceReconciler) enqueueFromConfig(ctx context.Context, obj client.Object) []reconcile.Request {
	cfg, ok := obj.(*infernexv1alpha1.InferNexServiceConfig)
	if !ok || cfg == nil || cfg.Namespace != strings.TrimSpace(r.TemplateNamespace) {
		return nil
	}

	var list infernexv1alpha1.InferNexServiceList
	if err := r.List(ctx, &list); err != nil {
		return nil
	}

	var reqs []reconcile.Request
	for _, svc := range list.Items {
		if svcRefersToConfig(&svc, cfg.Name) {
			reqs = append(reqs, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      svc.Name,
					Namespace: svc.Namespace,
				},
			})
		}
	}
	return reqs
}

// isDefaultTemplateName reports whether cfgName matches one of the mode-based
// default template names used when an InferNexService has no explicit baseRefs.
func isDefaultTemplateName(cfgName string) bool {
	return cfgName == defaultAggregateTemplateName || cfgName == defaultPDTemplateName
}

// svcRefersToConfig returns true when the InferNexService references cfgName via
// spec.baseRefs, or when baseRefs is empty and cfgName matches the mode-based
// default template name.
func svcRefersToConfig(svc *infernexv1alpha1.InferNexService, cfgName string) bool {
	if svc == nil || cfgName == "" {
		return false
	}
	for _, ref := range svc.Spec.BaseRefs {
		if strings.TrimSpace(ref.Name) == cfgName {
			return true
		}
	}
	if len(svc.Spec.BaseRefs) == 0 && isDefaultTemplateName(cfgName) {
		return true
	}
	return false
}

func (r *InferNexServiceReconciler) recordNormal(obj *infernexv1alpha1.InferNexService, reason, messageFmt string, args ...interface{}) {
	if r.Recorder == nil || obj == nil {
		return
	}
	r.Recorder.Eventf(obj, corev1.EventTypeNormal, reason, messageFmt, args...)
}

func (r *InferNexServiceReconciler) recordWarn(obj *infernexv1alpha1.InferNexService, reason, messageFmt string, args ...interface{}) {
	if r.Recorder == nil || obj == nil {
		return
	}
	r.Recorder.Eventf(obj, corev1.EventTypeWarning, reason, messageFmt, args...)
}

func (r *InferNexServiceReconciler) emitGatewayRoutingConditionEvent(
	obj *infernexv1alpha1.InferNexService,
	oldConds []metav1.Condition,
	newConds []metav1.Condition,
) {
	newCond := apimeta.FindStatusCondition(newConds, "GatewayRoutingReady")
	if newCond == nil {
		return
	}
	oldCond := apimeta.FindStatusCondition(oldConds, "GatewayRoutingReady")
	if oldCond != nil &&
		oldCond.Status == newCond.Status &&
		oldCond.Reason == newCond.Reason &&
		oldCond.Message == newCond.Message {
		return
	}
	if newCond.Status == metav1.ConditionTrue {
		r.recordNormal(obj, "GatewayRoutingReady", "%s", newCond.Message)
		return
	}
	r.recordWarn(obj, newCond.Reason, "%s", newCond.Message)
}

// persistReconcileFailureStatus updates the InferNexService status with a Ready=False
// condition when reconcileWorkloadsAndRBAC returns a persistent (non-retryable) error.
// This prevents the controller from entering an infinite retry loop on unrecoverable
// configuration errors such as a missing baseRefs template.
func (r *InferNexServiceReconciler) persistReconcileFailureStatus(
	ctx context.Context,
	req ctrl.Request,
	reconcileErr error,
) error {
	fresh := &infernexv1alpha1.InferNexService{}
	if err := r.Get(ctx, req.NamespacedName, fresh); err != nil {
		return client.IgnoreNotFound(err)
	}
	cond := metav1.Condition{
		Type:               conditionTypeReady,
		Status:             metav1.ConditionFalse,
		Reason:             "TemplateNotFound",
		Message:            reconcileErr.Error(),
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: fresh.Generation,
	}
	apimeta.SetStatusCondition(&fresh.Status.Conditions, cond)
	fresh.Status.Ready = false
	return r.Status().Update(ctx, fresh)
}
