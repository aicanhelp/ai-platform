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
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	managedGatewaySuffix       = "infernex-gateway"
	managedEnvoyProxySuffix    = "infernex-nodeport"
	managedHTTPRouteSuffix     = "infernex-route"
	managedInferencePoolSuffix = "inference-pool"
	defaultEndpointPickerPort  = 9002
	defaultGatewayClassName    = "envoy"
	defaultGatewayListenerName = "http"
	defaultGatewayPort         = 80
)

func (r *InferNexServiceReconciler) reconcileGatewayRouting(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
) error {
	// Linked LLMInferenceService mode: KServe owns Gateway API objects; do not touch them during reconcile.
	if owner.Spec.SourceRef != nil {
		return nil
	}
	if spec.IntelligentGatewayRouting == nil {
		return r.deleteManagedGatewayRouting(ctx, owner)
	}
	igr := spec.IntelligentGatewayRouting
	if igr.Router == nil || !enabled(igr.Router.Enabled) {
		return r.deleteManagedGatewayRouting(ctx, owner)
	}
	parentRefs, err := r.reconcileGateway(ctx, owner, igr.Gateway)
	if err != nil {
		return err
	}
	poolName, err := r.reconcileInferencePool(ctx, owner, spec, igr.InferencePool)
	if err != nil {
		return err
	}
	if err := r.reconcileHTTPRoute(ctx, owner, spec, igr.HTTPRoute, parentRefs, poolName); err != nil {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) reconcileGateway(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	cfg *infernexv1alpha1.GatewayRefSpec,
) ([]gwapiv1.ParentReference, error) {
	if cfg != nil && cfg.Ref != nil {
		if err := r.deleteManagedGateway(ctx, owner); err != nil {
			r.recordWarn(owner, "GatewayRefFailed", "failed to cleanup managed gateway before using ref %q: %v", cfg.Ref.Name, err)
			return nil, err
		}
		if err := r.deleteManagedEnvoyProxy(ctx, owner); err != nil {
			r.recordWarn(owner, "GatewayRefFailed", "failed to cleanup managed envoyproxy before using ref %q: %v", cfg.Ref.Name, err)
			return nil, err
		}
		r.recordNormal(owner, "GatewayReferenced", "use gateway ref %s/%s", owner.Namespace, cfg.Ref.Name)
		return toHTTPRouteParentRef(owner.Namespace, *cfg.Ref), nil
	}
	spec := defaultGatewaySpec(owner)
	if cfg != nil && cfg.Spec != nil {
		spec = cfg.Spec.DeepCopy()
	}
	if ensureDefaultNodePortEnvoyProxyRef(spec, owner) {
		if err := r.reconcileManagedEnvoyProxy(ctx, owner); err != nil {
			return nil, err
		}
	}
	name := fmt.Sprintf("%s-%s", owner.Name, managedGatewaySuffix)
	gw := &gwapiv1.Gateway{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	op, err := controllerutil.CreateOrUpdate(ctx, r.Client, gw, func() error {
		gw.Labels = managedGatewayLabels(owner.Name)
		gw.Spec = *spec
		return controllerutil.SetControllerReference(owner, gw, r.Scheme)
	})
	if err != nil {
		r.recordWarn(owner, "GatewayReconcileFailed", "reconcile gateway %s/%s failed: %v", owner.Namespace, name, err)
		return nil, err
	}
	if op != controllerutil.OperationResultNone {
		r.recordNormal(owner, "GatewayReconciled", "gateway %s/%s %s", owner.Namespace, name, op)
	}
	return managedInfernexGatewayParentRefs(owner), nil
}

func (r *InferNexServiceReconciler) reconcileManagedEnvoyProxy(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	name := managedEnvoyProxyName(owner)
	proxy := &unstructured.Unstructured{}
	proxy.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	proxy.SetKind("EnvoyProxy")
	proxy.SetName(name)
	proxy.SetNamespace(owner.Namespace)
	op, err := controllerutil.CreateOrUpdate(ctx, r.Client, proxy, func() error {
		proxy.SetLabels(managedGatewayLabels(owner.Name))
		proxy.Object["spec"] = map[string]interface{}{
			"provider": map[string]interface{}{
				"type": "Kubernetes",
				"kubernetes": map[string]interface{}{
					"envoyService": map[string]interface{}{
						"type": "NodePort",
					},
				},
			},
		}
		return controllerutil.SetControllerReference(owner, proxy, r.Scheme)
	})
	if err != nil {
		r.recordWarn(owner, "EnvoyProxyReconcileFailed", "reconcile envoyproxy %s/%s failed: %v", owner.Namespace, name, err)
		return err
	}
	if op != controllerutil.OperationResultNone {
		r.recordNormal(owner, "EnvoyProxyReconciled", "envoyproxy %s/%s %s", owner.Namespace, name, op)
	}
	return nil
}

func (r *InferNexServiceReconciler) reconcileInferencePool(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
	cfg *infernexv1alpha1.InferencePoolRefSpec,
) (string, error) {
	if cfg != nil && cfg.Ref != nil {
		if err := r.deleteManagedInferencePool(ctx, owner); err != nil {
			r.recordWarn(owner, "InferencePoolRefFailed", "failed to cleanup managed inferencepool before using ref %q: %v", cfg.Ref.Name, err)
			return "", err
		}
		r.recordNormal(owner, "InferencePoolReferenced", "use inferencepool ref %s/%s", owner.Namespace, cfg.Ref.Name)
		return cfg.Ref.Name, nil
	}
	name := fmt.Sprintf("%s-%s", owner.Name, managedInferencePoolSuffix)
	pool := &igwapiv1.InferencePool{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	op, err := controllerutil.CreateOrUpdate(ctx, r.Client, pool, func() error {
		pool.Labels = managedGatewayLabels(owner.Name)
		pool.Spec = infernexManagedInferencePoolSpec(owner, spec)
		if cfg != nil && cfg.Spec != nil {
			pool.Spec = *cfg.Spec.DeepCopy()
		}
		return controllerutil.SetControllerReference(owner, pool, r.Scheme)
	})
	if err != nil {
		r.recordWarn(owner, "InferencePoolReconcileFailed", "reconcile inferencepool %s/%s failed: %v", owner.Namespace, name, err)
		return "", err
	}
	if op != controllerutil.OperationResultNone {
		r.recordNormal(owner, "InferencePoolReconciled", "inferencepool %s/%s %s", owner.Namespace, name, op)
	}
	return name, nil
}

func (r *InferNexServiceReconciler) reconcileHTTPRoute(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
	cfg *infernexv1alpha1.HTTPRouteRefSpec,
	parentRefs []gwapiv1.ParentReference,
	poolName string,
) error {
	if cfg != nil && cfg.Ref != nil {
		r.recordNormal(owner, "HTTPRouteReferenced", "use httproute ref %s/%s", owner.Namespace, cfg.Ref.Name)
		return r.deleteManagedHTTPRoute(ctx, owner)
	}
	name := fmt.Sprintf("%s-%s", owner.Name, managedHTTPRouteSuffix)
	route := &gwapiv1.HTTPRoute{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	op, err := controllerutil.CreateOrUpdate(ctx, r.Client, route, func() error {
		route.Labels = managedGatewayLabels(owner.Name)
		route.Spec = infernexManagedHTTPRouteSpec(owner, spec, parentRefs, poolName)
		if cfg != nil && cfg.Spec != nil {
			route.Spec = *cfg.Spec.DeepCopy()
			if len(route.Spec.ParentRefs) == 0 {
				route.Spec.ParentRefs = parentRefs
			}
		}
		return controllerutil.SetControllerReference(owner, route, r.Scheme)
	})
	if err != nil {
		r.recordWarn(owner, "HTTPRouteReconcileFailed", "reconcile httproute %s/%s failed: %v", owner.Namespace, name, err)
		return err
	}
	if op != controllerutil.OperationResultNone {
		r.recordNormal(owner, "HTTPRouteReconciled", "httproute %s/%s %s", owner.Namespace, name, op)
	}
	return err
}

func defaultGatewaySpec(owner *infernexv1alpha1.InferNexService) *gwapiv1.GatewaySpec {
	return &gwapiv1.GatewaySpec{
		GatewayClassName: gwapiv1.ObjectName(defaultGatewayClassName),
		Infrastructure: &gwapiv1.GatewayInfrastructure{
			ParametersRef: defaultNodePortEnvoyProxyRef(owner),
		},
		Listeners: []gwapiv1.Listener{{
			Name:     gwapiv1.SectionName(defaultGatewayListenerName),
			Port:     gwapiv1.PortNumber(defaultGatewayPort),
			Protocol: gwapiv1.HTTPProtocolType,
		}},
	}
}

func ensureDefaultNodePortEnvoyProxyRef(spec *gwapiv1.GatewaySpec, owner *infernexv1alpha1.InferNexService) bool {
	if spec == nil || owner == nil {
		return false
	}
	if spec.Infrastructure == nil {
		spec.Infrastructure = &gwapiv1.GatewayInfrastructure{}
	}
	if spec.Infrastructure.ParametersRef != nil {
		return isDefaultNodePortEnvoyProxyRef(spec.Infrastructure.ParametersRef, owner)
	}
	spec.Infrastructure.ParametersRef = defaultNodePortEnvoyProxyRef(owner)
	return true
}

func defaultNodePortEnvoyProxyRef(owner *infernexv1alpha1.InferNexService) *gwapiv1.LocalParametersReference {
	return &gwapiv1.LocalParametersReference{
		Group: gwapiv1.Group("gateway.envoyproxy.io"),
		Kind:  gwapiv1.Kind("EnvoyProxy"),
		Name:  managedEnvoyProxyName(owner),
	}
}

func isDefaultNodePortEnvoyProxyRef(ref *gwapiv1.LocalParametersReference, owner *infernexv1alpha1.InferNexService) bool {
	if ref == nil || owner == nil {
		return false
	}
	return ref.Group == gwapiv1.Group("gateway.envoyproxy.io") &&
		ref.Kind == gwapiv1.Kind("EnvoyProxy") &&
		ref.Name == managedEnvoyProxyName(owner)
}

func managedEnvoyProxyName(owner *infernexv1alpha1.InferNexService) string {
	return fmt.Sprintf("%s-%s", owner.Name, managedEnvoyProxySuffix)
}

func toHTTPRouteParentRef(defaultNS string, ref infernexv1alpha1.NamedRef) []gwapiv1.ParentReference {
	return []gwapiv1.ParentReference{{
		Name:      gwapiv1.ObjectName(ref.Name),
		Namespace: ptr.To[gwapiv1.Namespace](gwapiv1.Namespace(defaultNS)),
		Group:     ptr.To[gwapiv1.Group](gwapiv1.Group(gwapiv1.GroupName)),
		Kind:      ptr.To[gwapiv1.Kind](gwapiv1.Kind("Gateway")),
	}}
}

func managedGatewayLabels(ownerName string) map[string]string {
	return map[string]string{
		"infernex.io/owner":              ownerName,
		"infernex.io/managed-gw-routing": "true",
	}
}

// bridgeOwnsGatewayAPIObject returns true only for Gateway API objects created by this reconciler
// (controller ref to this InferNexService, or infernex-managed labels). Avoids deleting KServe-owned routes.
func bridgeOwnsGatewayAPIObject(owner *infernexv1alpha1.InferNexService, obj metav1.Object) bool {
	if owner == nil || obj == nil {
		return false
	}
	if metav1.IsControlledBy(obj, owner) {
		return true
	}
	lab := obj.GetLabels()
	if lab == nil {
		return false
	}
	return lab["infernex.io/managed-gw-routing"] == "true" && lab["infernex.io/owner"] == owner.Name
}

func (r *InferNexServiceReconciler) deleteManagedGatewayRouting(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	if err := r.deleteManagedHTTPRoute(ctx, owner); err != nil {
		return err
	}
	if err := r.deleteManagedInferencePool(ctx, owner); err != nil {
		return err
	}
	if err := r.deleteManagedGateway(ctx, owner); err != nil {
		return err
	}
	if err := r.deleteManagedEnvoyProxy(ctx, owner); err != nil {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) deleteManagedHTTPRoute(ctx context.Context, owner *infernexv1alpha1.InferNexService) error {
	name := fmt.Sprintf("%s-%s", owner.Name, managedHTTPRouteSuffix)
	route := &gwapiv1.HTTPRoute{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, route); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !bridgeOwnsGatewayAPIObject(owner, route) {
		return nil
	}
	if err := r.Delete(ctx, route); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) deleteManagedInferencePool(ctx context.Context, owner *infernexv1alpha1.InferNexService) error {
	name := fmt.Sprintf("%s-%s", owner.Name, managedInferencePoolSuffix)
	pool := &igwapiv1.InferencePool{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, pool); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !bridgeOwnsGatewayAPIObject(owner, pool) {
		return nil
	}
	if err := r.Delete(ctx, pool); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) deleteManagedGateway(ctx context.Context, owner *infernexv1alpha1.InferNexService) error {
	name := fmt.Sprintf("%s-%s", owner.Name, managedGatewaySuffix)
	gw := &gwapiv1.Gateway{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, gw); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !bridgeOwnsGatewayAPIObject(owner, gw) {
		return nil
	}
	if err := r.Delete(ctx, gw); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) deleteManagedEnvoyProxy(ctx context.Context, owner *infernexv1alpha1.InferNexService) error {
	name := managedEnvoyProxyName(owner)
	proxy := &unstructured.Unstructured{}
	proxy.SetAPIVersion("gateway.envoyproxy.io/v1alpha1")
	proxy.SetKind("EnvoyProxy")
	if err := r.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, proxy); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !bridgeOwnsGatewayAPIObject(owner, proxy) {
		return nil
	}
	if err := r.Delete(ctx, proxy); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}
