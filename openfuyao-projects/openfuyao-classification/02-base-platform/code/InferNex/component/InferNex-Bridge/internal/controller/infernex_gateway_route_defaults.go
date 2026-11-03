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
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// resolvedGatewayNamespacedName returns the Gateway used as HTTPRoute parent and for readiness checks.
func (r *InferNexServiceReconciler) resolvedGatewayNamespacedName(
	infsvc *infernexv1alpha1.InferNexService,
	igr *infernexv1alpha1.IntelligentGatewayRoutingSpec,
) types.NamespacedName {
	if igr != nil && igr.Gateway != nil {
		if ref := igr.Gateway.Ref; ref != nil && strings.TrimSpace(ref.Name) != "" {
			return types.NamespacedName{Namespace: infsvc.Namespace, Name: strings.TrimSpace(ref.Name)}
		}
		if igr.Gateway.Spec != nil {
			return types.NamespacedName{
				Namespace: infsvc.Namespace,
				Name:      fmt.Sprintf("%s-%s", infsvc.Name, managedGatewaySuffix),
			}
		}
	}
	return types.NamespacedName{
		Namespace: infsvc.Namespace,
		Name:      fmt.Sprintf("%s-%s", infsvc.Name, managedGatewaySuffix),
	}
}

func managedInfernexGatewayParentRefs(owner *infernexv1alpha1.InferNexService) []gwapiv1.ParentReference {
	name := fmt.Sprintf("%s-%s", owner.Name, managedGatewaySuffix)
	return []gwapiv1.ParentReference{{
		Name:      gwapiv1.ObjectName(name),
		Namespace: ptr.To[gwapiv1.Namespace](gwapiv1.Namespace(owner.Namespace)),
		Group:     ptr.To[gwapiv1.Group](gwapiv1.Group(gwapiv1.GroupName)),
		Kind:      ptr.To[gwapiv1.Kind](gwapiv1.Kind("Gateway")),
	}}
}

func infernexHTTPRoutePathBase(owner *infernexv1alpha1.InferNexService) string {
	return "/" + owner.Namespace + "/" + owner.Name
}

func inferBackendComponent(spec infernexv1alpha1.InferNexServiceSpec) string {
	if spec.Engine != nil && EngineIsPDMode(spec.Engine) {
		if EngineDecodeWorkload(spec.Engine) != nil {
			return "engine-pd-decode"
		}
		if EnginePrefillWorkload(spec.Engine) != nil {
			return "engine-pd-prefill"
		}
	}
	return "engine-aggregate"
}

func inferBackendServiceName(owner *infernexv1alpha1.InferNexService, spec infernexv1alpha1.InferNexServiceSpec) string {
	if owner != nil && spec.Engine != nil && EngineIsPDMode(spec.Engine) {
		return pdWorkloadServiceName(owner.Name)
	}
	if owner != nil && spec.Engine != nil {
		if wl := EngineAggregateWorkload(spec.Engine); wl != nil {
			if groupSize, err := EngineWorkloadGroupSize(wl); err == nil && groupSize > 1 {
				return pdWorkloadServiceName(owner.Name)
			}
		}
	}
	if owner == nil {
		return inferBackendComponent(spec)
	}
	return fmt.Sprintf("%s-%s", owner.Name, inferBackendComponent(spec))
}

// infernexManagedInferencePoolMatchLabels builds InferencePool.spec.selector.matchLabels.
// Upstream InferencePool LabelSelector only supports matchLabels (AND); it cannot OR multiple
// infernex.io/component values, and openfuyao.com/pdRole cannot be one literal for prefill, decode,
// and leader at once. For PD we use labels shared by P/D/proxy and add extra AND keys to avoid
// unrelated pods (Hermes, cache-indexer, mooncake share infernex.io/owner but lack PD group).
//
// PD uses either openfuyao.com/pdGroupID + infernex.io/owner (direct) or app.kubernetes.io/name + app.kubernetes.io/part-of=llminferenceservice (KServe-linked).
// Aggregate / non-PD uses infernex.io/owner + infernex.io/component.
func infernexManagedInferencePoolMatchLabels(owner *infernexv1alpha1.InferNexService, spec infernexv1alpha1.InferNexServiceSpec) map[igwapiv1.LabelKey]igwapiv1.LabelValue {
	if owner == nil {
		return nil
	}
	wl, _ := pdWorkloadIdentity(owner)
	if spec.Engine != nil && EngineIsPDMode(spec.Engine) {
		if owner.Spec.SourceRef == nil {
			return map[igwapiv1.LabelKey]igwapiv1.LabelValue{
				igwapiv1.LabelKey(labelOpenFuyaoPDGroup): igwapiv1.LabelValue(wl),
				igwapiv1.LabelKey(labelInfernexOwner):    igwapiv1.LabelValue(owner.Name),
			}
		}
		return map[igwapiv1.LabelKey]igwapiv1.LabelValue{
			igwapiv1.LabelKey(labelAppKubernetesIOName):   igwapiv1.LabelValue(wl),
			igwapiv1.LabelKey(labelAppKubernetesIOPartOf): igwapiv1.LabelValue(valueKServeAppPartOf),
		}
	}
	comp := inferBackendComponent(spec)
	return map[igwapiv1.LabelKey]igwapiv1.LabelValue{
		"infernex.io/owner":     igwapiv1.LabelValue(owner.Name),
		"infernex.io/component": igwapiv1.LabelValue(comp),
	}
}

func infernexManagedInferencePoolSpec(owner *infernexv1alpha1.InferNexService, spec infernexv1alpha1.InferNexServiceSpec) igwapiv1.InferencePoolSpec {
	// Hermes / gateway-api-inference-extension EPP is exposed as the per-InferNexService router Service
	// (<infsvc-name>-hermes-router), not a cluster-wide "inference-gateway" Service.
	pickerSvc := fmt.Sprintf("%s-%s", owner.Name, hermesRouterComponent)
	return igwapiv1.InferencePoolSpec{
		Selector: igwapiv1.LabelSelector{
			MatchLabels: infernexManagedInferencePoolMatchLabels(owner, spec),
		},
		TargetPorts: []igwapiv1.Port{{
			Number: igwapiv1.PortNumber(defaultWorkloadServicePort),
		}},
		EndpointPickerRef: igwapiv1.EndpointPickerRef{
			Name: igwapiv1.ObjectName(pickerSvc),
			Port: &igwapiv1.Port{
				Number: igwapiv1.PortNumber(defaultEndpointPickerPort),
			},
		},
	}
}

// infernexManagedHTTPRouteSpec builds a two-rule HTTPRoute for LLM ingress at /<namespace>/<service>/...
// Rule 1: single InferencePool prefix /v1 (covers completions, chat/completions, responses) so ai-gateway
// installs one ext_proc filter per listener; multiple per-path Pool rules stack ext_proc and duplicate EPP traffic.
// Rule 2: catch-all to workload Service for health, /v1/models, etc. (bypasses EPP).
// InferNexServiceConfig merge may replace parentRefs, rules, or backends via intelligentGatewayRouting.httpRoute.spec.
func infernexManagedHTTPRouteSpec(
	owner *infernexv1alpha1.InferNexService,
	spec infernexv1alpha1.InferNexServiceSpec,
	parentRefs []gwapiv1.ParentReference,
	poolName string,
) gwapiv1.HTTPRouteSpec {
	base := infernexHTTPRoutePathBase(owner)
	infGroup := gwapiv1.Group(igwapiv1.GroupName)
	poolNS := gwapiv1.Namespace(owner.Namespace)
	workloadSvc := inferBackendServiceName(owner, spec)
	zero := gwapiv1.Duration("0s")
	timeouts := &gwapiv1.HTTPRouteTimeouts{
		Request:        ptr.To(zero),
		BackendRequest: ptr.To(zero),
	}
	poolBackend := []gwapiv1.HTTPBackendRef{{
		BackendRef: gwapiv1.BackendRef{
			BackendObjectReference: gwapiv1.BackendObjectReference{
				Group:     &infGroup,
				Kind:      ptr.To[gwapiv1.Kind]("InferencePool"),
				Name:      gwapiv1.ObjectName(poolName),
				Namespace: &poolNS,
				Port:      ptr.To[gwapiv1.PortNumber](8000),
			},
			Weight: ptr.To[int32](1),
		},
	}}
	rewrite := func(prefix string) []gwapiv1.HTTPRouteFilter {
		return []gwapiv1.HTTPRouteFilter{{
			Type: gwapiv1.HTTPRouteFilterURLRewrite,
			URLRewrite: &gwapiv1.HTTPURLRewriteFilter{
				Path: &gwapiv1.HTTPPathModifier{
					Type:               gwapiv1.PrefixMatchHTTPPathModifier,
					ReplacePrefixMatch: ptr.To(prefix),
				},
			},
		}}
	}
	svcGroup := gwapiv1.Group("")
	return gwapiv1.HTTPRouteSpec{
		CommonRouteSpec: gwapiv1.CommonRouteSpec{
			ParentRefs: parentRefs,
		},
		Rules: []gwapiv1.HTTPRouteRule{
			{
				Matches: []gwapiv1.HTTPRouteMatch{{
					Path: &gwapiv1.HTTPPathMatch{
						Type:  ptr.To(gwapiv1.PathMatchPathPrefix),
						Value: ptr.To(base + "/v1"),
					},
				}},
				BackendRefs: poolBackend,
				Filters:     rewrite("/v1"),
				Timeouts:    timeouts,
			},
			{
				Matches: []gwapiv1.HTTPRouteMatch{{
					Path: &gwapiv1.HTTPPathMatch{
						Type:  ptr.To(gwapiv1.PathMatchPathPrefix),
						Value: ptr.To(base),
					},
				}},
				BackendRefs: []gwapiv1.HTTPBackendRef{{
					BackendRef: gwapiv1.BackendRef{
						BackendObjectReference: gwapiv1.BackendObjectReference{
							Group: &svcGroup,
							Kind:  ptr.To[gwapiv1.Kind]("Service"),
							Name:  gwapiv1.ObjectName(workloadSvc),
							Port:  ptr.To[gwapiv1.PortNumber](8000),
						},
						Weight: ptr.To[int32](1),
					},
				}},
				Filters:  rewrite("/"),
				Timeouts: timeouts,
			},
		},
	}
}
