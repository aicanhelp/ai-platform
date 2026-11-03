package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestMergeHelpers_CoverLowBranches(t *testing.T) {
	t.Parallel()
	enabledTrue := true

	dst := &infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			PDOrchestrator: &infernexv1alpha1.PDOrchestratorComponentSpec{},
			EagleEye:       &infernexv1alpha1.EagleEyeComponentSpec{},
		},
		Engine: &infernexv1alpha1.InferenceEngineSpec{
			Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{},
		},
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue},
		},
	}
	src := infernexv1alpha1.InferNexServiceSpec{
		Components: &infernexv1alpha1.InfernexComponentsSpec{
			PDOrchestrator: &infernexv1alpha1.PDOrchestratorComponentSpec{
				ElasticScaler:        &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
				Tidal:                &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
				ResourceScalingGroup: &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
			},
			EagleEye: &infernexv1alpha1.EagleEyeComponentSpec{
				HardwareMonitor:   &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
				HardwareDiagnosis: &infernexv1alpha1.EnabledComponentSpec{Enabled: &enabledTrue},
			},
		},
		Engine: &infernexv1alpha1.InferenceEngineSpec{
			InferenceEngineWorkloadSpec: infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: testTemplate("agg:v1", 8000),
			},
			Prefill: &infernexv1alpha1.InferenceEngineWorkloadSpec{
				Template: testTemplate("prefill:v1", 8001),
			},
		},
		IntelligentGatewayRouting: &infernexv1alpha1.IntelligentGatewayRoutingSpec{
			Router: &infernexv1alpha1.ComponentSpec{
				Enabled:  &enabledTrue,
				Replicas: 2,
				Template: testTemplate("router:v1", 9000),
			},
			Gateway: &infernexv1alpha1.GatewayRefSpec{
				Ref:  &infernexv1alpha1.NamedRef{Name: "gw"},
				Spec: &gwapiv1.GatewaySpec{},
			},
			HTTPRoute: &infernexv1alpha1.HTTPRouteRefSpec{
				Ref:  &infernexv1alpha1.NamedRef{Name: "route"},
				Spec: &gwapiv1.HTTPRouteSpec{},
			},
			InferencePool: &infernexv1alpha1.InferencePoolRefSpec{
				Ref: &infernexv1alpha1.NamedRef{Name: "pool"},
			},
		},
	}
	mergeSpecFromTemplate(dst, src)
	if dst.IntelligentGatewayRouting == nil || dst.IntelligentGatewayRouting.Gateway == nil || dst.IntelligentGatewayRouting.HTTPRoute == nil || dst.IntelligentGatewayRouting.InferencePool == nil {
		t.Fatalf("expected gateway/route/pool refs merged, got %#v", dst.IntelligentGatewayRouting)
	}
	if dst.Components == nil || dst.Components.PDOrchestrator == nil || dst.Components.PDOrchestrator.Tidal == nil || dst.Components.EagleEye == nil || dst.Components.EagleEye.HardwareMonitor == nil {
		t.Fatalf("expected component subtrees merged, got %#v", dst.Components)
	}
	if dst.Engine == nil || dst.Engine.Prefill == nil || dst.Engine.Prefill.Template == nil {
		t.Fatalf("expected engine prefill template merged, got %#v", dst.Engine)
	}

	compDst := &infernexv1alpha1.ComponentSpec{}
	compSrc := &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue, Replicas: 2, ServicePort: 8080, Template: testTemplate("comp:v1", 8080)}
	mergeComponentSpecFields(compDst, compSrc)
	if compDst.Enabled == nil || compDst.Replicas != 2 || compDst.ServicePort != 8080 || compDst.Template == nil {
		t.Fatalf("expected component fields merged, got %#v", compDst)
	}

	var ptrComp *infernexv1alpha1.ComponentSpec
	mergePtrComponentSpec(&ptrComp, compSrc)
	if ptrComp == nil || ptrComp.Template == nil {
		t.Fatalf("expected ptr component merged, got %#v", ptrComp)
	}

	dstTplComp := &infernexv1alpha1.TemplateComponentSpec{}
	srcTplComp := &infernexv1alpha1.TemplateComponentSpec{Replicas: 1, ServicePort: 8000, Template: testTemplate("tpl:v1", 8000)}
	mergeTemplateComponentSpecFields(dstTplComp, srcTplComp)
	if dstTplComp.Replicas != 1 || dstTplComp.ServicePort != 8000 || dstTplComp.Template == nil {
		t.Fatalf("expected template component fields merged, got %#v", dstTplComp)
	}

	var ptrTplComp *infernexv1alpha1.TemplateComponentSpec
	mergePtrTemplateComponentSpec(&ptrTplComp, srcTplComp)
	if ptrTplComp == nil || ptrTplComp.Template == nil {
		t.Fatalf("expected ptr template component merged, got %#v", ptrTplComp)
	}

	// mergePodTemplateSpecs should fill missing labels/annotations/images without overriding existing values.
	left := &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      map[string]string{"keep": "1"},
			Annotations: map[string]string{"a": "x"},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "c1"}}},
	}
	right := &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      map[string]string{"fill": "1", "keep": "2"},
			Annotations: map[string]string{"b": "y"},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "c1", Image: "img:v1", ImagePullPolicy: corev1.PullIfNotPresent}}},
	}
	mergePodTemplateSpecs(left, right)
	if left.Labels["keep"] != "1" || left.Labels["fill"] != "1" || left.Annotations["b"] != "y" {
		t.Fatalf("expected labels/annotations merged without overwrite, got labels=%v annotations=%v", left.Labels, left.Annotations)
	}
	if left.Spec.Containers[0].Image != "img:v1" || left.Spec.Containers[0].ImagePullPolicy != corev1.PullIfNotPresent {
		t.Fatalf("expected image fields filled from source template, got %#v", left.Spec.Containers[0])
	}

	t.Run("mergeIfMissing via gateway ref partial merge", func(t *testing.T) {
		dstGateway := &infernexv1alpha1.GatewayRefSpec{
			Ref: &infernexv1alpha1.NamedRef{Name: "keep-gw"},
		}
		srcGateway := &infernexv1alpha1.GatewayRefSpec{
			Ref:  &infernexv1alpha1.NamedRef{Name: "other-gw"},
			Spec: &gwapiv1.GatewaySpec{GatewayClassName: gwapiv1.ObjectName("gc")},
		}
		mergeGatewayRefSpec(&dstGateway, srcGateway)
		if dstGateway.Ref.Name != "keep-gw" {
			t.Fatalf("expected existing ref preserved, got %q", dstGateway.Ref.Name)
		}
		if dstGateway.Spec == nil || dstGateway.Spec.GatewayClassName != "gc" {
			t.Fatalf("expected spec filled from src, got %#v", dstGateway.Spec)
		}

		dstPool := &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool"}}
		srcPool := &infernexv1alpha1.InferencePoolRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "pool2"}}
		mergeInferencePoolRefSpec(&dstPool, srcPool)
		if dstPool.Ref.Name != "pool" {
			t.Fatalf("expected pool ref preserved, got %q", dstPool.Ref.Name)
		}

		var dstRoute *infernexv1alpha1.HTTPRouteRefSpec
		srcRoute := &infernexv1alpha1.HTTPRouteRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "route"}}
		mergeHTTPRouteRefSpec(&dstRoute, srcRoute)
		if dstRoute == nil || dstRoute.Ref.Name != "route" {
			t.Fatalf("expected nil dst route filled, got %#v", dstRoute)
		}
	})

	t.Run("fillComponentTemplateIfNeeded", func(t *testing.T) {
		enabledTrue := true
		enabledFalse := false
		dst := &infernexv1alpha1.ComponentSpec{Enabled: &enabledTrue}
		src := &infernexv1alpha1.ComponentSpec{Template: testTemplate("fill:v1", 8080)}
		fillComponentTemplateIfNeeded(dst, src)
		if dst.Template == nil {
			t.Fatal("expected template filled for enabled component")
		}
		disabled := &infernexv1alpha1.ComponentSpec{Enabled: &enabledFalse}
		fillComponentTemplateIfNeeded(disabled, src)
		if disabled.Template != nil {
			t.Fatal("disabled component should not receive template fill")
		}
	})
}
