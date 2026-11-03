package v1alpha1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
)

func TestDeepCopyInto_Matrix(t *testing.T) {
	t.Parallel()
	tpl := &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"k": "v"}},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "img:v1"}},
		},
	}

	cases := []struct {
		name string
		run  func()
	}{
		{"SourceRef", func() {
			in := &SourceRef{APIVersion: "v1", Kind: "LLMInferenceService", Name: "x", Namespace: "ns"}
			var out SourceRef
			in.DeepCopyInto(&out)
			if out.Name != "x" {
				t.Fatalf("SourceRef copy failed")
			}
		}},
		{"HTTPRouteRefSpec", func() {
			in := &HTTPRouteRefSpec{Ref: &NamedRef{Name: "route"}, Spec: &gwapiv1.HTTPRouteSpec{}}
			var out HTTPRouteRefSpec
			in.DeepCopyInto(&out)
			if out.Ref == nil || out.Spec == nil {
				t.Fatal("HTTPRouteRefSpec copy failed")
			}
		}},
		{"InferNexServiceConfig", func() {
			in := &InferNexServiceConfig{
				ObjectMeta: metav1.ObjectMeta{Name: "cfg", Namespace: "ns"},
				Spec: InferNexServiceConfigSpec{
					InferNexServiceSpec: InferNexServiceSpec{
						Engine: &InferenceEngineSpec{
							InferenceEngineWorkloadSpec: InferenceEngineWorkloadSpec{Template: tpl.DeepCopy()},
						},
					},
				},
			}
			var out InferNexServiceConfig
			in.DeepCopyInto(&out)
			if out.Spec.Engine == nil || out.Spec.Engine.Template == nil {
				t.Fatal("InferNexServiceConfig copy failed")
			}
		}},
		{"InferNexServiceList", func() {
			in := &InferNexServiceList{
				Items: []InferNexService{{
					ObjectMeta: metav1.ObjectMeta{Name: "demo"},
					Spec: InferNexServiceSpec{
						Components: &InfernexComponentsSpec{
							CacheIndexer: &CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
						},
					},
					Status: InferNexServiceStatus{Mode: "aggregate", Ready: true},
				}},
			}
			var out InferNexServiceList
			in.DeepCopyInto(&out)
			if len(out.Items) != 1 || out.Items[0].Status.Mode != "aggregate" {
				t.Fatal("InferNexServiceList copy failed")
			}
		}},
		{"MooncakeComponentSpec", func() {
			in := &MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master:  &TemplateComponentSpec{Replicas: 1, Template: tpl.DeepCopy()},
			}
			var out MooncakeComponentSpec
			in.DeepCopyInto(&out)
			if out.Master == nil || out.Master.Template == nil {
				t.Fatal("MooncakeComponentSpec copy failed")
			}
		}},
		{"PDOrchestratorComponentSpec", func() {
			in := &PDOrchestratorComponentSpec{
				ElasticScaler:        &EnabledComponentSpec{Enabled: ptr.To(true)},
				Tidal:                &EnabledComponentSpec{Enabled: ptr.To(false)},
				ResourceScalingGroup: &EnabledComponentSpec{Enabled: ptr.To(true)},
			}
			var out PDOrchestratorComponentSpec
			in.DeepCopyInto(&out)
			if out.ElasticScaler == nil || out.Tidal == nil {
				t.Fatal("PDOrchestratorComponentSpec copy failed")
			}
		}},
		{"GatewayRefSpec", func() {
			in := &GatewayRefSpec{Ref: &NamedRef{Name: "gw"}, Spec: &gwapiv1.GatewaySpec{}}
			var out GatewayRefSpec
			in.DeepCopyInto(&out)
			if out.Ref == nil || out.Spec == nil {
				t.Fatal("GatewayRefSpec copy failed")
			}
		}},
		{"InferNexComponentStatuses", func() {
			in := &InferNexComponentStatuses{
				ProxyServer:  &ComponentStatus{Ready: true, Message: "ok"},
				CacheIndexer: &ComponentStatus{Ready: false, Message: "starting"},
			}
			var out InferNexComponentStatuses
			in.DeepCopyInto(&out)
			if out.ProxyServer == nil || out.CacheIndexer == nil {
				t.Fatal("InferNexComponentStatuses copy failed")
			}
		}},
		{"InferNexServiceConfigList", func() {
			in := &InferNexServiceConfigList{Items: []InferNexServiceConfig{{ObjectMeta: metav1.ObjectMeta{Name: "cfg"}}}}
			var out InferNexServiceConfigList
			in.DeepCopyInto(&out)
			if len(out.Items) != 1 {
				t.Fatal("InferNexServiceConfigList copy failed")
			}
		}},
		{"InferencePoolRefSpec", func() {
			in := &InferencePoolRefSpec{Ref: &NamedRef{Name: "pool"}}
			var out InferencePoolRefSpec
			in.DeepCopyInto(&out)
			if out.Ref == nil {
				t.Fatal("InferencePoolRefSpec copy failed")
			}
		}},
		{"IntelligentGatewayRoutingSpec", func() {
			in := &IntelligentGatewayRoutingSpec{
				Router:  &ComponentSpec{Enabled: ptr.To(true), Replicas: 1, Template: tpl.DeepCopy()},
				Gateway: &GatewayRefSpec{Ref: &NamedRef{Name: "gw"}},
			}
			var out IntelligentGatewayRoutingSpec
			in.DeepCopyInto(&out)
			if out.Router == nil || out.Gateway == nil {
				t.Fatal("IntelligentGatewayRoutingSpec copy failed")
			}
		}},
		{"InfernexComponentsSpec", func() {
			in := &InfernexComponentsSpec{
				CacheIndexer: &CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
				EagleEye: &EagleEyeComponentSpec{
					HardwareMonitor: &EnabledComponentSpec{Enabled: ptr.To(true)},
				},
			}
			var out InfernexComponentsSpec
			in.DeepCopyInto(&out)
			if out.CacheIndexer == nil || out.EagleEye == nil {
				t.Fatal("InfernexComponentsSpec copy failed")
			}
		}},
		{"InferenceEngineSpec PD prefill", func() {
			in := &InferenceEngineSpec{
				InferenceEngineWorkloadSpec: InferenceEngineWorkloadSpec{
					Replicas: ptr.To(int32(2)),
					Template: tpl.DeepCopy(),
				},
				Prefill: &InferenceEngineWorkloadSpec{Replicas: ptr.To(int32(1)), Template: tpl.DeepCopy()},
			}
			var out InferenceEngineSpec
			in.DeepCopyInto(&out)
			if out.Prefill == nil || out.Template == nil || out.Prefill.Template == nil {
				t.Fatal("InferenceEngineSpec PD copy failed")
			}
		}},
		{"InferNexService full object", func() {
			in := &InferNexService{
				ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns"},
				Spec: InferNexServiceSpec{
					Model:  &LLMModelSpec{URI: "hf://m", Name: "m"},
					SourceRef: &SourceRef{Kind: "LLMInferenceService", Name: "llm"},
					Engine: &InferenceEngineSpec{
						Prefill: &InferenceEngineWorkloadSpec{Template: tpl.DeepCopy()},
					},
				},
				Status: InferNexServiceStatus{
					Mode: "pd",
					Conditions: []metav1.Condition{{Type: "Available", Status: metav1.ConditionTrue}},
				},
			}
			var out InferNexService
			in.DeepCopyInto(&out)
			if out.Spec.Model == nil || out.Spec.Engine == nil || out.Status.Mode != "pd" {
				t.Fatal("InferNexService copy failed")
			}
		}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tc.run()
		})
	}
}
