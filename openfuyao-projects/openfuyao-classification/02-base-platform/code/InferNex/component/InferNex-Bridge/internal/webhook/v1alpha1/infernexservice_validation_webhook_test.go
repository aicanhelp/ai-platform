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


package v1alpha1

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
	"gitcode.com/openFuyao/InferNex/internal/controller"
)

const testTemplateNS = "default"

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := infernexv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	return s
}

func defaultAggregateConfigNoEngine(ns string) *infernexv1alpha1.InferNexServiceConfig {
	return &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "infernex-default-aggregate-template", Namespace: ns},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
					Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)},
				},
			},
		},
	}
}

func defaultPDConfigNoEngine(ns string) *infernexv1alpha1.InferNexServiceConfig {
	return &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "infernex-default-pd-template", Namespace: ns},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
					Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)},
				},
				Engine: &infernexv1alpha1.InferenceEngineSpec{
					Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{},
				},
			},
		},
	}
}

func buildEffective(t *testing.T, insvc *infernexv1alpha1.InferNexService, extra ...client.Object) infernexv1alpha1.InferNexServiceSpec {
	t.Helper()
	s := testScheme(t)
	objs := []client.Object{
		defaultAggregateConfigNoEngine(testTemplateNS),
		defaultPDConfigNoEngine(testTemplateNS),
	}
	objs = append(objs, extra...)
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).Build()
	eff, _, err := controller.BuildEffectiveSpecForWebhook(context.Background(), cl, testTemplateNS, insvc)
	if err != nil {
		t.Fatal(err)
	}
	return eff
}

func minimalAggregateEngine() *infernexv1alpha1.InferenceEngineSpec {
	return &infernexv1alpha1.InferenceEngineSpec{
		InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
			Replicas: ptr.To(int32(1)),
			Template: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "main",
						Image: "infer/engine:v1",
					}},
				},
			},
		},
	}
}

func TestValidateDirectInferNexService_ExplicitEnabled(t *testing.T) {
	t.Parallel()
	base := func() *infernexv1alpha1.InferNexService {
		return &infernexv1alpha1.InferNexService{
			ObjectMeta: metav1.ObjectMeta{Name: "x", Namespace: testTemplateNS},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: minimalAggregateEngine(),
			},
		}
	}

	t.Run("allows no components", func(t *testing.T) {
		t.Parallel()
		ins := base()
		if err := validateDirectInferNexServiceExplicitComponents(ins); err != nil {
			t.Fatal(err)
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("denies cacheIndexer without explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Replicas: 1},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err == nil {
			t.Fatal("expected error for missing cacheIndexer.enabled")
		}
	})

	t.Run("denies pd orchestrator elasticScaler without explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			PDOrchestrator: &infernexv1alpha1.PDOrchestratorComponentSpec{
				ElasticScaler: &infernexv1alpha1.EnabledComponentSpec{},
			},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err == nil {
			t.Fatal("expected error for missing elasticScaler.enabled")
		}
	})

	t.Run("denies eagleEye hardwareMonitor without explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			EagleEye: &infernexv1alpha1.EagleEyeComponentSpec{
				HardwareMonitor: &infernexv1alpha1.EnabledComponentSpec{},
			},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err == nil {
			t.Fatal("expected error for missing hardwareMonitor.enabled")
		}
	})

	t.Run("denies eagleEye networkPerformanceExporter without explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			EagleEye: &infernexv1alpha1.EagleEyeComponentSpec{
				NetworkPerformanceExporter: &infernexv1alpha1.EnabledComponentSpec{},
			},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err == nil {
			t.Fatal("expected error for missing networkPerformanceExporter.enabled")
		}
	})

	t.Run("allows cacheIndexer with enabled true", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{
				Enabled:  ptr.To(true),
				Replicas: 1,
			},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err != nil {
			t.Fatal(err)
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("denies router block on CR without explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.IntelligentGatewayRouting = &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "router:v1"}},
					},
				},
			},
		}
		if err := validateRawRouterExplicitEnabledIfPresent(ins); err == nil {
			t.Fatal("expected error for missing router.enabled on submitted CR")
		}
	})

	t.Run("allows router with enabled true and template", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.IntelligentGatewayRouting = &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled: ptr.To(true),
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "router:v1",
							Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 9002}},
						}},
					},
				},
			},
		}
		if err := validateRawRouterExplicitEnabledIfPresent(ins); err != nil {
			t.Fatal(err)
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("denies router without main EPP container", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.IntelligentGatewayRouting = &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled: ptr.To(true),
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "tokenizer", Image: "tok:v1"},
							{Name: "epp", Image: "router:v1", Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 9002}}},
						},
					},
				},
			},
		}
		eff := buildEffective(t, ins)
		err := validateMergedDirectInferNexService(eff)
		if err == nil || !strings.Contains(err.Error(), `container named "main"`) {
			t.Fatalf("expected main container error, got %v", err)
		}
	})

	t.Run("denies router when main lacks grpc port", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.IntelligentGatewayRouting = &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled: ptr.To(true),
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "tokenizer", Image: "tok:v1"},
							{Name: "main", Image: "router:v1"},
						},
					},
				},
			},
		}
		eff := buildEffective(t, ins)
		err := validateMergedDirectInferNexService(eff)
		if err == nil || !strings.Contains(err.Error(), `named port "grpc"`) {
			t.Fatalf("expected grpc port error, got %v", err)
		}
	})

	t.Run("allows router with tokenizer before main", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.IntelligentGatewayRouting = &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled: ptr.To(true),
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "tokenizer", Image: "tok:v1"},
							{Name: "main", Image: "router:v1", Ports: []corev1.ContainerPort{{Name: "grpc", ContainerPort: 9002}}},
						},
					},
				},
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("allows mooncake master without nested enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			Mooncake: &infernexv1alpha1.MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master: &infernexv1alpha1.TemplateComponentSpec{
					Replicas: 1,
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "mooncake-master", Image: "m:v1"}},
						},
					},
				},
			},
		}
		if err := validateDirectInferNexServiceExplicitComponents(ins); err != nil {
			t.Fatalf("expected mooncake.master without nested enabled to pass validation, got %v", err)
		}
	})

	t.Run("merged PD validates without proxyServer on spec", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "d:v1"}},
					},
				},
			},
			Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
				Replicas: ptr.To(int32(1)),
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "p:v1"}},
					},
				},
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("rejects indivisible dataParallelSize", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "e:v1"}},
					},
				},
				DataParallelSize:      ptr.To(int32(3)),
				DataParallelSizeLocal: ptr.To(int32(2)),
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err == nil {
			t.Fatal("expected validation error for indivisible dataParallelSize")
		}
	})

	t.Run("rejects worker when groupSize is 1", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "e:v1"}},
					},
				},
				Worker: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "w", Image: "e:v1"}},
					},
				},
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err == nil {
			t.Fatal("expected validation error for worker with groupSize 1")
		}
	})

	t.Run("allows empty worker placeholder when groupSize is 1", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "e:v1"}},
					},
				},
				Worker: &corev1.PodTemplateSpec{},
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatalf("expected empty worker allowed for Deployment path, got %v", err)
		}
	})

	t.Run("allows omitted worker when groupSize greater than 1", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "e:v1"}},
					},
				},
				DataParallelSize:      ptr.To(int32(2)),
				DataParallelSizeLocal: ptr.To(int32(1)),
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err != nil {
			t.Fatalf("expected LWS without worker allowed, got %v", err)
		}
	})

	t.Run("validates worker template when groupSize greater than 1", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "e:v1"}},
					},
				},
				Worker: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "w", Image: ""}},
					},
				},
				DataParallelSize:      ptr.To(int32(2)),
				DataParallelSizeLocal: ptr.To(int32(1)),
			},
		}
		eff := buildEffective(t, ins)
		if err := validateMergedDirectInferNexService(eff); err == nil {
			t.Fatal("expected validation error for worker missing image")
		}
	})
}

func TestValidateMergedDirectInferNexService_EngineFromBaseRefsOnly(t *testing.T) {
	t.Parallel()
	cfg := &infernexv1alpha1.InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "my-engine", Namespace: testTemplateNS},
		Spec: infernexv1alpha1.InferNexServiceConfigSpec{
			InferNexServiceSpec: infernexv1alpha1.InferNexServiceSpec{
				Engine: minimalAggregateEngine(),
			},
		},
	}
	ins := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "svc", Namespace: testTemplateNS},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			BaseRefs: []infernexv1alpha1.NamedRef{{Name: "my-engine"}},
		},
	}
	eff := buildEffective(t, ins, cfg)
	if err := validateMergedDirectInferNexService(eff); err != nil {
		t.Fatal(err)
	}
}

func TestValidatePodTemplate_ProxyTemplateRequiresContainers(t *testing.T) {
	t.Parallel()
	err := validatePodTemplate("spec.engine.prefill.template", &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Containers: []corev1.Container{}},
	})
	if err == nil {
		t.Fatal("expected error for empty containers")
	}
}

func TestInferNexServiceValidationWebhook_Handle_TerminatingSkipsMissingBaseRefTemplate(t *testing.T) {
	t.Parallel()
	s := testScheme(t)
	cl := fake.NewClientBuilder().WithScheme(s).
		WithObjects(defaultAggregateConfigNoEngine(testTemplateNS), defaultPDConfigNoEngine(testTemplateNS)).
		Build()

	dec := admission.NewDecoder(s)
	h := &inferNexServiceValidatingHandler{
		decoder:           dec,
		apiClient:         cl,
		templateNamespace: testTemplateNS,
	}

	baseInfsvc := func() *infernexv1alpha1.InferNexService {
		return &infernexv1alpha1.InferNexService{
			TypeMeta: metav1.TypeMeta{
				APIVersion: infernexv1alpha1.GroupVersion.String(),
				Kind:       "InferNexService",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:       "q",
				Namespace:  testTemplateNS,
				Finalizers: []string{"infernex.infernex.io/infernexservice"},
			},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				BaseRefs: []infernexv1alpha1.NamedRef{{Name: "vllm-ascend-aggregate-engine-with-mooncake-template"}},
				Engine:   minimalAggregateEngine(),
			},
		}
	}

	t.Run("new object has deletionTimestamp", func(t *testing.T) {
		t.Parallel()
		ins := baseInfsvc()
		now := metav1.Now()
		ins.DeletionTimestamp = &now
		raw, err := json.Marshal(ins)
		if err != nil {
			t.Fatal(err)
		}
		req := admission.Request{
			AdmissionRequest: admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Object:    runtime.RawExtension{Raw: raw},
			},
		}
		resp := h.Handle(context.Background(), req)
		if !resp.Allowed {
			t.Fatalf("expected Allowed, got %+v", resp)
		}
	})

	t.Run("update old object was terminating", func(t *testing.T) {
		t.Parallel()
		oldIns := baseInfsvc()
		now := metav1.Now()
		oldIns.DeletionTimestamp = &now
		oldRaw, err := json.Marshal(oldIns)
		if err != nil {
			t.Fatal(err)
		}

		newIns := baseInfsvc()
		newRaw, err := json.Marshal(newIns)
		if err != nil {
			t.Fatal(err)
		}

		req := admission.Request{
			AdmissionRequest: admissionv1.AdmissionRequest{
				Operation: admissionv1.Update,
				Object:    runtime.RawExtension{Raw: newRaw},
				OldObject: runtime.RawExtension{Raw: oldRaw},
			},
		}
		resp := h.Handle(context.Background(), req)
		if !resp.Allowed {
			t.Fatalf("expected Allowed, got %+v", resp)
		}
	})
}

func TestValidateGatewayRefSpec_RefSpecMutualExclusion(t *testing.T) {
	t.Parallel()
	enabled := ptr.To(true)
	base := &infernexv1alpha1.IntelligentGatewayRoutingSpec{
		Router: &infernexv1alpha1.ComponentSpec{Enabled: enabled},
	}
	t.Run("gateway ref+spec denied", func(t *testing.T) {
		t.Parallel()
		igr := base.DeepCopy()
		igr.Gateway = &infernexv1alpha1.GatewayRefSpec{
			Ref:  &infernexv1alpha1.NamedRef{Name: "gw"},
			Spec: &gwapiv1.GatewaySpec{},
		}
		err := validateGatewayRefSpec(igr)
		if err == nil || !strings.Contains(err.Error(), "gateway.ref and spec.intelligentGatewayRouting.gateway.spec are mutually exclusive") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("httpRoute ref+spec denied", func(t *testing.T) {
		t.Parallel()
		igr := base.DeepCopy()
		igr.HTTPRoute = &infernexv1alpha1.HTTPRouteRefSpec{
			Ref:  &infernexv1alpha1.NamedRef{Name: "route"},
			Spec: &gwapiv1.HTTPRouteSpec{},
		}
		err := validateGatewayRefSpec(igr)
		if err == nil || !strings.Contains(err.Error(), "httpRoute.ref and spec.intelligentGatewayRouting.httpRoute.spec are mutually exclusive") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("inferencePool ref+spec denied", func(t *testing.T) {
		t.Parallel()
		igr := base.DeepCopy()
		igr.InferencePool = &infernexv1alpha1.InferencePoolRefSpec{
			Ref:  &infernexv1alpha1.NamedRef{Name: "pool"},
			Spec: &igwapiv1.InferencePoolSpec{},
		}
		err := validateGatewayRefSpec(igr)
		if err == nil || !strings.Contains(err.Error(), "inferencePool.ref and spec.intelligentGatewayRouting.inferencePool.spec are mutually exclusive") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestValidateGatewayRefSpec_SkipsWhenRouterDisabled(t *testing.T) {
	t.Parallel()
	igr := &infernexv1alpha1.IntelligentGatewayRoutingSpec{
		Router: &infernexv1alpha1.ComponentSpec{Enabled: ptr.To(false)},
		Gateway: &infernexv1alpha1.GatewayRefSpec{
			Ref:  &infernexv1alpha1.NamedRef{Name: "gw"},
			Spec: &gwapiv1.GatewaySpec{},
		},
	}
	if err := validateGatewayRefSpec(igr); err != nil {
		t.Fatalf("router disabled should skip ref/spec exclusivity, got %v", err)
	}
}

func TestInferNexServiceValidationWebhook_Handle_SourceRefSkipsDirectValidation(t *testing.T) {
	t.Parallel()
	s := testScheme(t)
	cl := fake.NewClientBuilder().WithScheme(s).
		WithObjects(defaultAggregateConfigNoEngine(testTemplateNS), defaultPDConfigNoEngine(testTemplateNS)).
		Build()
	dec := admission.NewDecoder(s)
	h := &inferNexServiceValidatingHandler{
		decoder:           dec,
		apiClient:         cl,
		templateNamespace: testTemplateNS,
	}

	ins := &infernexv1alpha1.InferNexService{
		TypeMeta: metav1.TypeMeta{
			APIVersion: infernexv1alpha1.GroupVersion.String(),
			Kind:       "InferNexService",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "linked",
			Namespace: testTemplateNS,
		},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				Kind: "LLMInferenceService",
				Name: "llm-a",
			},
		},
	}
	raw, err := json.Marshal(ins)
	if err != nil {
		t.Fatal(err)
	}
	req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: raw},
	}}
	resp := h.Handle(context.Background(), req)
	if !resp.Allowed {
		t.Fatalf("expected allowed for sourceRef linked object, got %+v", resp)
	}
}

func TestInferNexServiceValidationWebhook_Handle_DirectCreate(t *testing.T) {
	t.Parallel()
	s := testScheme(t)
	cl := fake.NewClientBuilder().WithScheme(s).
		WithObjects(defaultAggregateConfigNoEngine(testTemplateNS), defaultPDConfigNoEngine(testTemplateNS)).
		Build()
	dec := admission.NewDecoder(s)
	h := &inferNexServiceValidatingHandler{
		decoder:           dec,
		apiClient:         cl,
		templateNamespace: testTemplateNS,
	}

	marshalReq := func(ins *infernexv1alpha1.InferNexService) admission.Request {
		t.Helper()
		raw, err := json.Marshal(ins)
		if err != nil {
			t.Fatal(err)
		}
		return admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Object:    runtime.RawExtension{Raw: raw},
		}}
	}

	base := func() *infernexv1alpha1.InferNexService {
		return &infernexv1alpha1.InferNexService{
			TypeMeta: metav1.TypeMeta{
				APIVersion: infernexv1alpha1.GroupVersion.String(),
				Kind:       "InferNexService",
			},
			ObjectMeta: metav1.ObjectMeta{Name: "direct", Namespace: testTemplateNS},
			Spec: infernexv1alpha1.InferNexServiceSpec{
				Engine: minimalAggregateEngine(),
			},
		}
	}

	t.Run("allowed for valid direct aggregate", func(t *testing.T) {
		t.Parallel()
		resp := h.Handle(context.Background(), marshalReq(base()))
		if !resp.Allowed {
			t.Fatalf("expected allowed, got %+v", resp)
		}
	})

	t.Run("denied when cacheIndexer missing explicit enabled", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Components = &infernexv1alpha1.InfernexComponentsSpec{
			CacheIndexer: &infernexv1alpha1.CacheIndexerComponentSpec{Replicas: 1},
		}
		resp := h.Handle(context.Background(), marshalReq(ins))
		if resp.Allowed || !strings.Contains(resp.AdmissionResponse.Result.Message, "cacheIndexer") {
			t.Fatalf("expected denied for missing enabled, got %+v", resp)
		}
	})

	t.Run("denied when effective spec merge fails", func(t *testing.T) {
		t.Parallel()
		ins := base()
		ins.Spec.Engine = nil
		ins.Spec.BaseRefs = []infernexv1alpha1.NamedRef{{Name: "missing-config"}}
		resp := h.Handle(context.Background(), marshalReq(ins))
		if resp.Allowed || !strings.Contains(resp.AdmissionResponse.Result.Message, "effective spec") {
			t.Fatalf("expected denied for merge failure, got %+v", resp)
		}
	})

	t.Run("errored on invalid object payload", func(t *testing.T) {
		t.Parallel()
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Object:    runtime.RawExtension{Raw: []byte("{not-json")},
		}}
		resp := h.Handle(context.Background(), req)
		if resp.Allowed || resp.Result.Code != 400 {
			t.Fatalf("expected 400 errored response, got %+v", resp)
		}
	})
}
