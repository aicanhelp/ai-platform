package v1alpha1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

func TestAddToSchemeAndDeepCopy(t *testing.T) {
	t.Parallel()
	s := runtime.NewScheme()
	if err := AddToScheme(s); err != nil {
		t.Fatalf("AddToScheme error: %v", err)
	}

	insvc := &InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
		Spec: InferNexServiceSpec{
			Model: &LLMModelSpec{URI: "hf://demo/model", Name: "demo"},
			Engine: &InferenceEngineSpec{
				InferenceEngineWorkloadSpec: InferenceEngineWorkloadSpec{
					Replicas: ptr.To(int32(1)),
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "model:v1"}},
						},
					},
				},
			},
			Components: &InfernexComponentsSpec{
				CacheIndexer: &CacheIndexerComponentSpec{Enabled: ptr.To(true), Replicas: 1},
			},
			IntelligentGatewayRouting: &IntelligentGatewayRoutingSpec{
				Router: &ComponentSpec{Enabled: ptr.To(true), Replicas: 1},
			},
		},
		Status: InferNexServiceStatus{
			Mode:  "aggregate",
			Ready: true,
			Components: &InferNexComponentStatuses{
				CacheIndexer: &ComponentStatus{Ready: true},
			},
		},
	}

	copyObj := insvc.DeepCopy()
	if copyObj == nil {
		t.Fatal("DeepCopy returned nil")
	}
	copyObj.Spec.Model.Name = "changed"
	if insvc.Spec.Model.Name == "changed" {
		t.Fatal("expected deep copy to not mutate source")
	}

	list := &InferNexServiceList{Items: []InferNexService{*insvc}}
	listCopy := list.DeepCopy()
	if len(listCopy.Items) != 1 {
		t.Fatalf("expected one item in list copy, got %d", len(listCopy.Items))
	}
	listCopy.Items[0].Name = "another"
	if list.Items[0].Name == "another" {
		t.Fatal("expected list deepcopy to isolate item mutation")
	}

	cfg := &InferNexServiceConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "cfg", Namespace: "ns-a"},
		Spec: InferNexServiceConfigSpec{
			InferNexServiceSpec: InferNexServiceSpec{
				Engine: &InferenceEngineSpec{
					Prefill: &InferenceEngineWorkloadSpec{Replicas: ptr.To(int32(1))},
				},
			},
		},
	}
	cfgCopy := cfg.DeepCopy()
	if cfgCopy == nil || cfgCopy.Spec.Engine == nil || cfgCopy.Spec.Engine.Prefill == nil {
		t.Fatal("expected config deep copy with prefill engine preserved")
	}
}

func TestDeepCopy_AllCoreTypes(t *testing.T) {
	t.Parallel()
	_ = (&SourceRef{Kind: "LLMInferenceService"}).DeepCopy()
	_ = (&LLMModelSpec{URI: "hf://demo"}).DeepCopy()
	_ = (&InferenceEngineWorkloadSpec{}).DeepCopy()
	_ = (&ComponentSpec{}).DeepCopy()
	_ = (&TemplateComponentSpec{}).DeepCopy()
	_ = (&EnabledComponentSpec{}).DeepCopy()
	_ = (&InferenceEngineSpec{}).DeepCopy()
	_ = (&NamedRef{Name: "x"}).DeepCopy()
	_ = (&HTTPRouteRefSpec{}).DeepCopy()
	_ = (&GatewayRefSpec{}).DeepCopy()
	_ = (&InferencePoolRefSpec{}).DeepCopy()
	_ = (&IntelligentGatewayRoutingSpec{}).DeepCopy()
	_ = (&CacheIndexerComponentSpec{}).DeepCopy()
	_ = (&MooncakeComponentSpec{}).DeepCopy()
	_ = (&PDOrchestratorComponentSpec{}).DeepCopy()
	_ = (&EagleEyeComponentSpec{}).DeepCopy()
	_ = (&InfernexComponentsSpec{}).DeepCopy()
	_ = (&InferNexServiceSpec{}).DeepCopy()
	_ = (&ComponentStatus{Ready: true}).DeepCopy()
	_ = (&InferNexComponentStatuses{}).DeepCopy()
	_ = (&InferNexServiceStatus{}).DeepCopy()
	_ = (&InferNexServiceConfigSpec{}).DeepCopy()
	_ = (&InferNexServiceConfigList{}).DeepCopy()
}

func TestDeepCopy_NilReceiversAndObjectMethods(t *testing.T) {
	t.Parallel()
	var svc *InferNexService
	if svc.DeepCopy() != nil {
		t.Fatal("expected nil DeepCopy for nil InferNexService receiver")
	}
	var svcList *InferNexServiceList
	if svcList.DeepCopy() != nil {
		t.Fatal("expected nil DeepCopy for nil InferNexServiceList receiver")
	}
	var cfg *InferNexServiceConfig
	if cfg.DeepCopy() != nil {
		t.Fatal("expected nil DeepCopy for nil InferNexServiceConfig receiver")
	}
	var cfgList *InferNexServiceConfigList
	if cfgList.DeepCopy() != nil {
		t.Fatal("expected nil DeepCopy for nil InferNexServiceConfigList receiver")
	}

	svcObj := &InferNexService{}
	if got := svcObj.DeepCopyObject(); got == nil {
		t.Fatal("expected non-nil DeepCopyObject for InferNexService")
	}
	svcListObj := &InferNexServiceList{}
	if got := svcListObj.DeepCopyObject(); got == nil {
		t.Fatal("expected non-nil DeepCopyObject for InferNexServiceList")
	}
	cfgObj := &InferNexServiceConfig{}
	if got := cfgObj.DeepCopyObject(); got == nil {
		t.Fatal("expected non-nil DeepCopyObject for InferNexServiceConfig")
	}
	cfgListObj := &InferNexServiceConfigList{}
	if got := cfgListObj.DeepCopyObject(); got == nil {
		t.Fatal("expected non-nil DeepCopyObject for InferNexServiceConfigList")
	}
}

func TestDeepCopyInto_PreservesNestedFields(t *testing.T) {
	t.Parallel()
	src := &InferNexServiceSpec{
		Engine: &InferenceEngineSpec{
			InferenceEngineWorkloadSpec: InferenceEngineWorkloadSpec{
				Replicas: ptr.To(int32(2)),
				Template: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "engine"}},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "img:v1"}},
					},
				},
			},
		},
		Components: &InfernexComponentsSpec{
			Mooncake: &MooncakeComponentSpec{
				Enabled: ptr.To(true),
				Master: &TemplateComponentSpec{
					Replicas: 1,
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "mooncake-master", Image: "m:v1"}},
						},
					},
				},
			},
		},
		IntelligentGatewayRouting: &IntelligentGatewayRoutingSpec{
			Router: &ComponentSpec{Enabled: ptr.To(true), Replicas: 1},
			Gateway: &GatewayRefSpec{
				Ref: &NamedRef{Name: "gw"},
			},
		},
	}
	var dst InferNexServiceSpec
	src.DeepCopyInto(&dst)
	if dst.Engine == nil || dst.Engine.Template == nil {
		t.Fatal("expected engine template copied")
	}
	if dst.Engine.Template.Labels["app"] != "engine" {
		t.Fatalf("expected nested labels copied, got %#v", dst.Engine.Template.Labels)
	}
	if dst.Components == nil || dst.Components.Mooncake == nil || dst.Components.Mooncake.Master == nil {
		t.Fatal("expected mooncake master copied")
	}
	if dst.IntelligentGatewayRouting == nil || dst.IntelligentGatewayRouting.Gateway == nil {
		t.Fatal("expected gateway ref copied")
	}
	src.Engine.Replicas = ptr.To(int32(99))
	if dst.Engine.Replicas != nil && *dst.Engine.Replicas == 99 {
		t.Fatal("expected deep copy isolation for nested pointer fields")
	}
}
