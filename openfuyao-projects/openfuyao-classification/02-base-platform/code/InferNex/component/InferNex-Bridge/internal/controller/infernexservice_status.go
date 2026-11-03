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
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	readyReasonAllManagedResourcesReady = "AllManagedResourcesReady"
	readyReasonInferenceBackendNotReady = "InferenceBackendNotReady"
	readyReasonProxyServerNotReady      = "ProxyServerNotReady"
	readyReasonHermesRouterNotReady     = "HermesRouterNotReady"
	readyReasonCacheIndexerNotReady     = "CacheIndexerNotReady"
	readyReasonMooncakeNotReady         = "MooncakeNotReady"
	readyReasonPDOrchestratorNotReady   = "PDOrchestratorNotReady"
	readyReasonEagleEyeNotReady         = "EagleEyeNotReady"
	conditionTypeReady                  = "Ready"
)

var infernexEngineKeys = []string{"engine-aggregate", "engine-pd-prefill", "engine-pd-decode"}

var infernexComponentGroups = struct {
	InferenceEngine []string
	HermesRouter    []string
	ProxyServer     []string
	CacheIndexer    []string
	Mooncake        []string
	PDOrchestrator  []string
	EagleEye        []string
}{
	InferenceEngine: infernexEngineKeys,
	HermesRouter:    []string{hermesRouterComponent},
	ProxyServer:     []string{"proxy-server"},
	CacheIndexer:    []string{"cache-indexer"},
	Mooncake:        []string{"mooncake-master", "mooncake-metadata"},
	PDOrchestrator:  []string{"pd-orchestrator-elastic-scaler", "pd-orchestrator-tidal", "pd-orchestrator-rsg"},
	EagleEye:        []string{"eagle-eye-hardware-monitor", "eagle-eye-hardware-diagnosis", "eagle-eye-network-performance-exporter"},
}

type workloadReadyFamily struct {
	status *infernexv1alpha1.ComponentStatus
	reason string
}

func keysPresentInDesired(desired map[string]componentPlan, keys []string) []string {
	var out []string
	for _, k := range keys {
		if _, ok := desired[k]; ok {
			out = append(out, k)
		}
	}
	return out
}

func deploymentRolloutReady(d *appsv1.Deployment) (bool, string) {
	for i := range d.Status.Conditions {
		c := d.Status.Conditions[i]
		if c.Type == appsv1.DeploymentProgressing && c.Status == corev1.ConditionFalse {
			return false, strings.TrimSpace(c.Message)
		}
	}
	if d.Spec.Replicas == nil || *d.Spec.Replicas == 0 {
		return true, ""
	}
	want := *d.Spec.Replicas
	if d.Generation > d.Status.ObservedGeneration {
		return false, "waiting for controller to observe latest spec"
	}
	if d.Status.ReadyReplicas < want {
		return false, fmt.Sprintf("ready pods %d/%d", d.Status.ReadyReplicas, want)
	}
	return true, ""
}

func workloadKeyRequired(plan componentPlan) bool {
	return plan.Replicas == nil || *plan.Replicas > 0
}

func gatewayRoutingRequired(infsvc *infernexv1alpha1.InferNexService, effectiveSpec infernexv1alpha1.InferNexServiceSpec) bool {
	if infsvc.Spec.SourceRef != nil {
		return false
	}
	igr := effectiveSpec.IntelligentGatewayRouting
	return igr != nil && igr.Router != nil && enabled(igr.Router.Enabled)
}

func mooncakeOrDeploymentReady(component string, dep *appsv1.Deployment) (bool, string) {
	if isSingletonComponentService(component) && (dep.Spec.Replicas == nil || *dep.Spec.Replicas == 0) {
		return false, "Mooncake singleton Deployment is scaled to zero"
	}
	return deploymentRolloutReady(dep)
}

func (r *InferNexServiceReconciler) externalMooncakeServiceReady(ctx context.Context, ns, component string) (bool, error) {
	if !isSingletonComponentService(component) {
		return false, nil
	}
	svc := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Namespace: ns, Name: mooncakeSingletonServiceName(component)}, svc)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return !isBridgeManagedMooncakeService(svc), nil
}

func daemonSetRolloutReady(ds *appsv1.DaemonSet) (bool, string) {
	if ds.Generation > ds.Status.ObservedGeneration {
		return false, "waiting for controller to observe latest spec"
	}
	if ds.Status.DesiredNumberScheduled == 0 {
		return true, ""
	}
	if ds.Status.NumberReady < ds.Status.DesiredNumberScheduled {
		return false, fmt.Sprintf("ready pods %d/%d", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled)
	}
	if ds.Status.UpdatedNumberScheduled < ds.Status.DesiredNumberScheduled {
		return false, fmt.Sprintf("updated pods %d/%d", ds.Status.UpdatedNumberScheduled, ds.Status.DesiredNumberScheduled)
	}
	return true, ""
}

func (r *InferNexServiceReconciler) statusForWorkloadKeys(
	ctx context.Context,
	ns, owner string,
	keys []string,
	desired map[string]componentPlan,
) (*infernexv1alpha1.ComponentStatus, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	var msgs []string
	allReady := true
	for _, k := range keys {
		name := fmt.Sprintf("%s-%s", owner, k)
		if isSingletonComponentService(k) {
			name = mooncakeSingletonWorkloadName(k)
		}
		plan := desired[k]
		ok := false
		msg := ""
		switch componentWorkloadKind(plan) {
		case workloadKindDaemonSet:
			ds := &appsv1.DaemonSet{}
			if err := r.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, ds); err != nil {
				if apierrors.IsNotFound(err) {
					allReady = false
					msgs = append(msgs, fmt.Sprintf("%s: DaemonSet %q not found", k, name))
					continue
				}
				return nil, err
			}
			ok, msg = daemonSetRolloutReady(ds)
		case workloadKindLeaderWorkerSet:
			lws := &lwsv1.LeaderWorkerSet{}
			if err := r.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, lws); err != nil {
				if apierrors.IsNotFound(err) {
					allReady = false
					msgs = append(msgs, fmt.Sprintf("%s: LeaderWorkerSet %q not found", k, name))
					continue
				}
				return nil, err
			}
			ok, msg = lwsRolloutReady(lws)
		default:
			external, extErr := r.externalMooncakeServiceReady(ctx, ns, k)
			if extErr != nil {
				return nil, extErr
			}
			if external {
				// The canonical external Service is authoritative, even if a
				// stale local Deployment with the canonical name remains.
				ok = true
				continue
			}
			dep := &appsv1.Deployment{}
			if err := r.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, dep); err != nil {
				if apierrors.IsNotFound(err) {
					allReady = false
					msgs = append(msgs, fmt.Sprintf("%s: Deployment %q not found", k, name))
					continue
				}
				return nil, err
			}
			ok, msg = mooncakeOrDeploymentReady(k, dep)
		}
		if !ok {
			allReady = false
			if msg != "" {
				msgs = append(msgs, fmt.Sprintf("%s: %s", k, msg))
			} else {
				msgs = append(msgs, fmt.Sprintf("%s: not ready", k))
			}
		}
	}
	out := &infernexv1alpha1.ComponentStatus{Ready: allReady}
	if len(msgs) > 0 {
		out.Message = strings.Join(msgs, "; ")
	}
	return out, nil
}

func (r *InferNexServiceReconciler) statusForWorkloadFamily(
	ctx context.Context,
	ns, owner string,
	keys []string,
	desired map[string]componentPlan,
) (*infernexv1alpha1.ComponentStatus, error) {
	required := make([]string, 0, len(keys))
	for _, k := range keys {
		plan, ok := desired[k]
		if !ok || !workloadKeyRequired(plan) {
			continue
		}
		required = append(required, k)
	}
	return r.statusForWorkloadKeys(ctx, ns, owner, required, desired)
}

func (r *InferNexServiceReconciler) evaluateInferNexServiceReady(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	effectiveSpec infernexv1alpha1.InferNexServiceSpec,
	comp *infernexv1alpha1.InferNexComponentStatuses,
) (bool, string, string, error) {
	families := []workloadReadyFamily{
		{comp.InferenceEngine, readyReasonInferenceBackendNotReady},
		{comp.ProxyServer, readyReasonProxyServerNotReady},
		{comp.HermesRouter, readyReasonHermesRouterNotReady},
		{comp.CacheIndexer, readyReasonCacheIndexerNotReady},
		{comp.Mooncake, readyReasonMooncakeNotReady},
		{comp.PDOrchestrator, readyReasonPDOrchestratorNotReady},
		{comp.EagleEye, readyReasonEagleEyeNotReady},
	}
	for _, family := range families {
		if family.status != nil && !family.status.Ready {
			msg := strings.TrimSpace(family.status.Message)
			if msg == "" {
				msg = "one or more managed workloads are not ready"
			}
			return false, family.reason, msg, nil
		}
	}
	if gatewayRoutingRequired(infsvc, effectiveSpec) {
		gwCond, err := r.buildGatewayRoutingReadyCondition(ctx, infsvc, effectiveSpec)
		if err != nil {
			return false, "", "", err
		}
		if gwCond.Status != metav1.ConditionTrue {
			return false, gwCond.Reason, gwCond.Message, nil
		}
	}
	return true, readyReasonAllManagedResourcesReady, "all managed resources are ready", nil
}

func buildReadyCondition(infsvc *infernexv1alpha1.InferNexService, ready bool, reason, message string) metav1.Condition {
	cond := metav1.Condition{
		Type:               conditionTypeReady,
		Status:             metav1.ConditionTrue,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
	if !ready {
		cond.Status = metav1.ConditionFalse
	}
	return cond
}

func (r *InferNexServiceReconciler) computeInferNexServiceStatus(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
	effectiveSpec infernexv1alpha1.InferNexServiceSpec,
	mode string,
) (infernexv1alpha1.InferNexServiceStatus, error) {
	ns := infsvc.Namespace
	owner := infsvc.Name

	comp := &infernexv1alpha1.InferNexComponentStatuses{}
	var err error
	if comp.InferenceEngine, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.InferenceEngine, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.ProxyServer, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.ProxyServer, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.HermesRouter, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.HermesRouter, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.CacheIndexer, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.CacheIndexer, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.Mooncake, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.Mooncake, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.PDOrchestrator, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.PDOrchestrator, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	if comp.EagleEye, err = r.statusForWorkloadFamily(ctx, ns, owner, infernexComponentGroups.EagleEye, desired); err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}

	overall, readyReason, readyMessage, err := r.evaluateInferNexServiceReady(ctx, infsvc, effectiveSpec, comp)
	if err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}

	conditions := make([]metav1.Condition, len(infsvc.Status.Conditions))
	copy(conditions, infsvc.Status.Conditions)
	summary := statusSummaryFromComponents(comp, overall)
	avail := metav1.Condition{
		Type:               "Available",
		Status:             metav1.ConditionTrue,
		Reason:             readyReasonAllManagedResourcesReady,
		Message:            summary,
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
	if !overall {
		avail.Status = metav1.ConditionFalse
		avail.Reason = readyReason
		if readyMessage != "" {
			avail.Message = readyMessage
		} else {
			avail.Message = summary
		}
	}
	apimeta.SetStatusCondition(&conditions, avail)
	apimeta.SetStatusCondition(&conditions, buildSpecValidatedCondition(infsvc))
	apimeta.SetStatusCondition(&conditions, buildBackendReadyCondition(infsvc, comp))
	gwCond, err := r.buildGatewayRoutingReadyCondition(ctx, infsvc, effectiveSpec)
	if err != nil {
		return infernexv1alpha1.InferNexServiceStatus{}, err
	}
	apimeta.SetStatusCondition(&conditions, gwCond)
	apimeta.SetStatusCondition(&conditions, buildComponentsReadyCondition(infsvc, comp, overall, readyReason, readyMessage))
	apimeta.SetStatusCondition(&conditions, buildReadyCondition(infsvc, overall, readyReason, readyMessage))

	st := infernexv1alpha1.InferNexServiceStatus{
		Mode:               mode,
		Ready:              overall,
		ObservedGeneration: infsvc.Generation,
		Conditions:         conditions,
	}
	if hasAnyComponentStatus(comp) {
		st.Components = comp
	}
	return st, nil
}

func buildSpecValidatedCondition(infsvc *infernexv1alpha1.InferNexService) metav1.Condition {
	return metav1.Condition{
		Type:               "SpecValidated",
		Status:             metav1.ConditionTrue,
		Reason:             "WebhookValidated",
		Message:            "InferNexService spec validation passed",
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
}

func buildBackendReadyCondition(infsvc *infernexv1alpha1.InferNexService, comp *infernexv1alpha1.InferNexComponentStatuses) metav1.Condition {
	cond := metav1.Condition{
		Type:               "BackendReady",
		Status:             metav1.ConditionTrue,
		Reason:             "BackendConfigured",
		Message:            "inference backend workloads are configured",
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
	if infsvc.Spec.SourceRef != nil {
		cond.Reason = "SourceRefManagedByLLMInferenceService"
		cond.Message = "inference backend is managed by linked LLMInferenceService"
		return cond
	}
	if comp == nil || comp.InferenceEngine == nil {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "BackendNotConfigured"
		cond.Message = "no inference backend workload is configured in desired components"
		return cond
	}
	if !comp.InferenceEngine.Ready {
		cond.Status = metav1.ConditionFalse
		cond.Reason = readyReasonInferenceBackendNotReady
		cond.Message = comp.InferenceEngine.Message
	}
	return cond
}

func (r *InferNexServiceReconciler) buildGatewayRoutingReadyCondition(
	ctx context.Context,
	infsvc *infernexv1alpha1.InferNexService,
	effectiveSpec infernexv1alpha1.InferNexServiceSpec,
) (metav1.Condition, error) {
	cond := metav1.Condition{
		Type:               "GatewayRoutingReady",
		Status:             metav1.ConditionTrue,
		Reason:             "GatewayRoutingReady",
		Message:            "gateway, httproute and inferencepool are configured",
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
	if infsvc.Spec.SourceRef != nil {
		cond.Reason = "SourceRefManagedByLLMInferenceService"
		cond.Message = "gateway routing is managed by linked LLMInferenceService"
		return cond, nil
	}
	igr := effectiveSpec.IntelligentGatewayRouting
	if igr == nil || igr.Router == nil || !enabled(igr.Router.Enabled) {
		cond.Reason = "RouterDisabled"
		cond.Message = "router is disabled; gateway routing is not required"
		return cond, nil
	}
	gwKey := r.resolvedGatewayNamespacedName(infsvc, igr)
	gwName := gwKey.Name
	if gwName == "" {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "GatewayRefInvalid"
		cond.Message = "gateway reference name is empty"
		return cond, nil
	}
	gw := &gwapiv1.Gateway{}
	if err := r.Get(ctx, gwKey, gw); err != nil {
		if apierrors.IsNotFound(err) {
			cond.Status = metav1.ConditionFalse
			cond.Reason = "GatewayNotFound"
			cond.Message = fmt.Sprintf("gateway %s/%s not found", gwKey.Namespace, gwName)
			return cond, nil
		}
		return metav1.Condition{}, err
	}
	if ok, msg := gatewayProgrammed(gw); !ok {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "GatewayNotProgrammed"
		cond.Message = msg
		return cond, nil
	}
	routeName := fmt.Sprintf("%s-%s", infsvc.Name, managedHTTPRouteSuffix)
	if igr.HTTPRoute != nil && igr.HTTPRoute.Ref != nil {
		routeName = strings.TrimSpace(igr.HTTPRoute.Ref.Name)
	}
	if routeName == "" {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "HTTPRouteRefInvalid"
		cond.Message = "httproute reference name is empty"
		return cond, nil
	}
	route := &gwapiv1.HTTPRoute{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: infsvc.Namespace, Name: routeName}, route); err != nil {
		if apierrors.IsNotFound(err) {
			cond.Status = metav1.ConditionFalse
			cond.Reason = "HTTPRouteNotFound"
			cond.Message = fmt.Sprintf("httproute %s/%s not found", infsvc.Namespace, routeName)
			return cond, nil
		}
		return metav1.Condition{}, err
	}
	if ok, msg := httpRouteAcceptedByGateway(route, gwName); !ok {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "HTTPRouteNotAccepted"
		cond.Message = msg
		return cond, nil
	}
	poolName := fmt.Sprintf("%s-%s", infsvc.Name, managedInferencePoolSuffix)
	if igr.InferencePool != nil && igr.InferencePool.Ref != nil {
		poolName = strings.TrimSpace(igr.InferencePool.Ref.Name)
	}
	if poolName == "" {
		cond.Status = metav1.ConditionFalse
		cond.Reason = "InferencePoolRefInvalid"
		cond.Message = "inferencepool reference name is empty"
		return cond, nil
	}
	pool := &igwapiv1.InferencePool{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: infsvc.Namespace, Name: poolName}, pool); err != nil {
		if apierrors.IsNotFound(err) {
			cond.Status = metav1.ConditionFalse
			cond.Reason = "InferencePoolNotFound"
			cond.Message = fmt.Sprintf("inferencepool %s/%s not found", infsvc.Namespace, poolName)
			return cond, nil
		}
		return metav1.Condition{}, err
	}
	if ok, msg, reason := r.inferencePoolReady(ctx, pool, gwName); !ok {
		cond.Status = metav1.ConditionFalse
		cond.Reason = reason
		cond.Message = msg
		return cond, nil
	}
	cond.Message = fmt.Sprintf(
		"gateway routing linked: gateway=%s/%s, httproute=%s/%s, inferencepool=%s/%s",
		infsvc.Namespace, gwName, infsvc.Namespace, routeName, infsvc.Namespace, poolName,
	)
	return cond, nil
}

func gatewayProgrammed(gw *gwapiv1.Gateway) (bool, string) {
	accepted := apimeta.FindStatusCondition(gw.Status.Conditions, "Accepted")
	if accepted != nil {
		if accepted.Status == metav1.ConditionFalse {
			return false, fmt.Sprintf("gateway %s/%s not accepted: %s", gw.Namespace, gw.Name, strings.TrimSpace(accepted.Message))
		}
		if accepted.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("gateway %s/%s accepted condition is %s", gw.Namespace, gw.Name, accepted.Status)
		}
	}
	programmed := apimeta.FindStatusCondition(gw.Status.Conditions, "Programmed")
	if programmed != nil {
		if programmed.Status == metav1.ConditionFalse {
			return false, fmt.Sprintf("gateway %s/%s not programmed: %s", gw.Namespace, gw.Name, strings.TrimSpace(programmed.Message))
		}
		if programmed.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("gateway %s/%s programmed condition is %s", gw.Namespace, gw.Name, programmed.Status)
		}
	}
	return true, ""
}

func httpRouteAcceptedByGateway(route *gwapiv1.HTTPRoute, gatewayName string) (bool, string) {
	for _, parent := range route.Status.Parents {
		if string(parent.ParentRef.Name) != gatewayName {
			continue
		}
		accepted := apimeta.FindStatusCondition(parent.Conditions, "Accepted")
		if accepted == nil {
			return false, fmt.Sprintf("httproute %s/%s missing Accepted condition for gateway %q", route.Namespace, route.Name, gatewayName)
		}
		if accepted.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("httproute %s/%s not accepted by gateway %q: %s", route.Namespace, route.Name, gatewayName, strings.TrimSpace(accepted.Message))
		}
		resolvedRefs := apimeta.FindStatusCondition(parent.Conditions, "ResolvedRefs")
		if resolvedRefs != nil && resolvedRefs.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("httproute %s/%s has unresolved refs for gateway %q: %s", route.Namespace, route.Name, gatewayName, strings.TrimSpace(resolvedRefs.Message))
		}
		return true, ""
	}
	return false, fmt.Sprintf("httproute %s/%s has no parent status for gateway %q", route.Namespace, route.Name, gatewayName)
}

func (r *InferNexServiceReconciler) inferencePoolReady(
	ctx context.Context,
	pool *igwapiv1.InferencePool,
	gatewayName string,
) (bool, string, string) {
	acceptedByGateway := false
	for _, p := range pool.Status.Parents {
		if string(p.ParentRef.Name) != gatewayName {
			continue
		}
		accepted := apimeta.FindStatusCondition(p.Conditions, "Accepted")
		if accepted == nil {
			return false, fmt.Sprintf("inferencepool %s/%s missing Accepted condition for gateway %q", pool.Namespace, pool.Name, gatewayName), "InferencePoolNotAccepted"
		}
		if accepted.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("inferencepool %s/%s not accepted by gateway %q: %s", pool.Namespace, pool.Name, gatewayName, strings.TrimSpace(accepted.Message)), "InferencePoolNotAccepted"
		}
		resolvedRefs := apimeta.FindStatusCondition(p.Conditions, "ResolvedRefs")
		if resolvedRefs != nil && resolvedRefs.Status != metav1.ConditionTrue {
			return false, fmt.Sprintf("inferencepool %s/%s has unresolved refs for gateway %q: %s", pool.Namespace, pool.Name, gatewayName, strings.TrimSpace(resolvedRefs.Message)), "InferencePoolRefsUnresolved"
		}
		acceptedByGateway = true
		break
	}
	if acceptedByGateway {
		return true, "", ""
	}
	labelQuery := make(map[string]string, len(pool.Spec.Selector.MatchLabels))
	for k, v := range pool.Spec.Selector.MatchLabels {
		labelQuery[string(k)] = string(v)
	}
	if len(labelQuery) == 0 {
		return false, fmt.Sprintf("inferencepool %s/%s matchLabels is empty", pool.Namespace, pool.Name), "InferencePoolMatchLabelsEmpty"
	}
	var pods corev1.PodList
	if err := r.listPodsByMatchLabels(ctx, pool.Namespace, labelQuery, &pods); err != nil {
		return false, fmt.Sprintf("list pods for inferencepool %s/%s with matchLabels failed: %v", pool.Namespace, pool.Name, err), "InferencePoolMatchLabelsCheckFailed"
	}
	if len(pods.Items) == 0 {
		return false, fmt.Sprintf("inferencepool %s/%s matchLabels matched 0 pods", pool.Namespace, pool.Name), "InferencePoolNoEndpoints"
	}
	return true, "", ""
}

func (r *InferNexServiceReconciler) listPodsByMatchLabels(
	ctx context.Context,
	namespace string,
	matchLabels map[string]string,
	pods *corev1.PodList,
) error {
	return r.List(ctx, pods, client.InNamespace(namespace), client.MatchingLabels(matchLabels))
}

func buildComponentsReadyCondition(
	infsvc *infernexv1alpha1.InferNexService,
	comp *infernexv1alpha1.InferNexComponentStatuses,
	overall bool,
	readyReason, readyMessage string,
) metav1.Condition {
	cond := metav1.Condition{
		Type:               "ComponentsReady",
		Status:             metav1.ConditionTrue,
		Reason:             readyReasonAllManagedResourcesReady,
		Message:            statusSummaryFromComponents(comp, overall),
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: infsvc.Generation,
	}
	if !overall {
		cond.Status = metav1.ConditionFalse
		cond.Reason = readyReason
		if readyMessage != "" {
			cond.Message = readyMessage
		}
	}
	return cond
}

func containsAnyKey(m map[string]componentPlan, keys ...string) bool {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

func hasAnyComponentStatus(comp *infernexv1alpha1.InferNexComponentStatuses) bool {
	if comp == nil {
		return false
	}
	for _, st := range []*infernexv1alpha1.ComponentStatus{
		comp.InferenceEngine,
		comp.HermesRouter,
		comp.ProxyServer,
		comp.CacheIndexer,
		comp.Mooncake,
		comp.PDOrchestrator,
		comp.EagleEye,
	} {
		if st != nil {
			return true
		}
	}
	return false
}

func statusSummaryFromComponents(comp *infernexv1alpha1.InferNexComponentStatuses, overall bool) string {
	if overall {
		return "all managed resources are ready"
	}
	if comp == nil {
		return "one or more managed resources are not ready"
	}
	var parts []string
	appendNotReady := func(name string, s *infernexv1alpha1.ComponentStatus) {
		if s != nil && !s.Ready && s.Message != "" {
			parts = append(parts, name+": "+s.Message)
		}
	}
	appendNotReady("inferenceEngine", comp.InferenceEngine)
	appendNotReady("hermesRouter", comp.HermesRouter)
	appendNotReady("proxyServer", comp.ProxyServer)
	appendNotReady("cacheIndexer", comp.CacheIndexer)
	appendNotReady("mooncake", comp.Mooncake)
	appendNotReady("pdOrchestrator", comp.PDOrchestrator)
	appendNotReady("eagleEye", comp.EagleEye)
	if len(parts) == 0 {
		return "one or more managed resources are not ready"
	}
	return strings.Join(parts, " | ")
}
