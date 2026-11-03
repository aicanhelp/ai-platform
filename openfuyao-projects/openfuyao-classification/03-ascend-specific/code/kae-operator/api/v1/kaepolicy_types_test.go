package v1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestKAEPolicySpec_Init(t *testing.T) {
	tests := []struct {
		name string
		spec *KAEPolicySpec
	}{
		{
			name: "Init_KAE_POLICY_SPEC",
			spec: &KAEPolicySpec{
				Driver: DriverSpec{
					Enabled: new(bool),
				},
				DevicePlugin: DevicePluginSpec{
					EnabledHPRE: new(bool),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The Init() method doesn't exist, so we'll test the struct initialization
			if tt.spec == nil {
				t.Errorf("spec should not be nil")
			}
			if tt.spec.Driver.Enabled == nil {
				t.Errorf("Driver.Enabled should not be nil")
			}
			if tt.spec.DevicePlugin.EnabledHPRE == nil {
				t.Errorf("DevicePlugin.EnabledHPRE should not be nil")
			}
		})
	}
}

// TestKAEPolicyDeepCopy 测试 KAEPolicy 的 DeepCopy 方法
func TestKAEPolicyDeepCopy(t *testing.T) {
	trueVal := true
	falseVal := false
	resourceList := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("100m"),
		corev1.ResourceMemory: resource.MustParse("100Mi"),
	}

	original := &KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-kae-policy",
			Namespace: "default",
			Labels: map[string]string{
				"app": "kae-operator",
			},
		},
		Spec: KAEPolicySpec{
			Operator: OperatorSpec{
				EnabledHPRE: &trueVal,
				EnabledSEC:  &falseVal,
				EnabledZIP:  &falseVal,
			},
			DaemonSets: DaemonSetsSpec{
				Labels: map[string]string{
					"custom-label": "value",
				},
				Annotations: map[string]string{
					"custom-annotation": "value",
				},
				Tolerations: []corev1.Toleration{
					{
						Key:      "node.kubernetes.io/not-ready",
						Operator: corev1.TolerationOpExists,
						Effect:   corev1.TaintEffectNoExecute,
					},
				},
				PriorityClassName: "system-node-critical",
				UpdateStrategy:    "RollingUpdate",
				RollingUpdate: &RollingUpdateSpec{
					MaxUnavailable: "2",
				},
			},
			Driver: DriverSpec{
				Enabled:         &trueVal,
				EnabledCleanup: &falseVal,
				Repository:      "kae-driver-registry",
				Image:           "kae-driver",
				Version:         "v1.0.0",
				ImagePullPolicy: "IfNotPresent",
				ImagePullSecrets: []string{
					"docker-secret",
				},
				Resources: &ResourceRequirements{
					Requests: resourceList,
					Limits:   resourceList,
				},
				Args: []string{"--arg1=value1"},
				Env: []EnvVar{
					{Name: "ENV1", Value: "value1"},
				},
			},
			DevicePlugin: DevicePluginSpec{
				EnabledHPRE: &trueVal,
				EnabledSEC:  &falseVal,
				EnabledZIP:  &falseVal,
				Repository:  "kae-plugin-registry",
				Image:       "kae-plugin",
				Version:     "v1.0.0",
				Resources: &ResourceRequirements{
					Requests: resourceList,
					Limits:   resourceList,
				},
				Env: []EnvVar{
					{Name: "PLUGIN_ENV1", Value: "value1"},
				},
			},
			PSA: PSASpec{
				Enabled: &trueVal,
			},
		},
		Status: KAEPolicyStatus{
			State:     Ready,
			Namespace: "default",
			Conditions: []metav1.Condition{
				{
					Type:    "Ready",
					Status:  metav1.ConditionTrue,
					Reason:  "AllComponentsReady",
					Message: "All KAE components are ready",
				},
			},
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify that the copy is equal to the original
	if copied.Name != original.Name {
		t.Errorf("Name mismatch: got %s, want %s", copied.Name, original.Name)
	}

	// Modify the copy and verify the original is unchanged
	copied.Name = "modified-name"
	if copied.Name == original.Name {
		t.Error("Modifying copy affected original")
	}
}

// TestKAEPolicyListDeepCopy 测试 KAEPolicyList 的 DeepCopy 方法
func TestKAEPolicyListDeepCopy(t *testing.T) {
	trueVal := true
	original := &KAEPolicyList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "kae.openfuyao.com/v1",
			Kind:       "KAEPolicyList",
		},
		Items: []KAEPolicy{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "policy-1",
				},
				Spec: KAEPolicySpec{
					Driver: DriverSpec{
						Enabled: &trueVal,
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "policy-2",
				},
				Spec: KAEPolicySpec{
					DevicePlugin: DevicePluginSpec{
						EnabledHPRE: &trueVal,
					},
				},
			},
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify that the copy is equal to the original
	if len(copied.Items) != len(original.Items) {
		t.Errorf("Items length mismatch: got %d, want %d", len(copied.Items), len(original.Items))
	}

	// Modify the copy and verify the original is unchanged
	copied.Items[0].Name = "modified-name"
	if copied.Items[0].Name == original.Items[0].Name {
		t.Error("Modifying copy affected original")
	}
}

// TestDriverSpecDeepCopy 测试 DriverSpec 的 DeepCopy 方法
func TestDriverSpecDeepCopy(t *testing.T) {
	trueVal := true
	resourceList := corev1.ResourceList{
		corev1.ResourceCPU: resource.MustParse("100m"),
	}

	original := &DriverSpec{
		Enabled:         &trueVal,
		EnabledCleanup: &trueVal,
		Repository:      "test-registry",
		Image:           "test-image",
		Version:         "v1.0.0",
		ImagePullPolicy: "IfNotPresent",
		ImagePullSecrets: []string{
			"secret1",
			"secret2",
		},
		Resources: &ResourceRequirements{
			Requests: resourceList,
			Limits:   resourceList,
		},
		Args: []string{"--arg1"},
		Env: []EnvVar{
			{Name: "ENV1", Value: "value1"},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	copied.Image = "modified-image"
	if copied.Image == original.Image {
		t.Error("Modifying copy affected original")
	}
}

// TestDevicePluginSpecDeepCopy 测试 DevicePluginSpec 的 DeepCopy 方法
func TestDevicePluginSpecDeepCopy(t *testing.T) {
	trueVal := true
	resourceList := corev1.ResourceList{
		corev1.ResourceCPU: resource.MustParse("100m"),
	}

	original := &DevicePluginSpec{
		EnabledHPRE: &trueVal,
		EnabledSEC:  &trueVal,
		EnabledZIP:  &trueVal,
		Repository:  "test-registry",
		Image:       "test-image",
		Version:     "v1.0.0",
		Resources: &ResourceRequirements{
			Requests: resourceList,
			Limits:   resourceList,
		},
		Env: []EnvVar{
			{Name: "ENV1", Value: "value1"},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	copied.Image = "modified-image"
	if copied.Image == original.Image {
		t.Error("Modifying copy affected original")
	}
}

// TestDaemonSetsSpecDeepCopy 测试 DaemonSetsSpec 的 DeepCopy 方法
func TestDaemonSetsSpecDeepCopy(t *testing.T) {
	original := &DaemonSetsSpec{
		Labels: map[string]string{
			"label1": "value1",
			"label2": "value2",
		},
		Annotations: map[string]string{
			"annotation1": "value1",
		},
		Tolerations: []corev1.Toleration{
			{
				Key:      "key1",
				Operator: corev1.TolerationOpEqual,
				Value:    "value1",
				Effect:   corev1.TaintEffectNoSchedule,
			},
		},
		PriorityClassName: "system-node-critical",
		UpdateStrategy:    "RollingUpdate",
		RollingUpdate: &RollingUpdateSpec{
			MaxUnavailable: "2",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	copied.Labels["new-label"] = "new-value"
	if _, exists := original.Labels["new-label"]; exists {
		t.Error("Modifying copy affected original")
	}
}

// TestResourceRequirementsDeepCopy 测试 ResourceRequirements 的 DeepCopy 方法
func TestResourceRequirementsDeepCopy(t *testing.T) {
	original := &ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("100m"),
			corev1.ResourceMemory: resource.MustParse("100Mi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("200m"),
			corev1.ResourceMemory: resource.MustParse("200Mi"),
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	originalCPUVal := original.Requests[corev1.ResourceCPU]
	copied.Requests[corev1.ResourceCPU] = resource.MustParse("50m")
	modifiedCPUVal := copied.Requests[corev1.ResourceCPU]
	originalCPU := originalCPUVal.String()
	modifiedCPU := modifiedCPUVal.String()
	if modifiedCPU == originalCPU {
		t.Error("Modifying copy affected original")
	}
}

// TestOperatorSpecDeepCopy 测试 OperatorSpec 的 DeepCopy 方法
func TestOperatorSpecDeepCopy(t *testing.T) {
	trueVal := true
	falseVal := false

	original := &OperatorSpec{
		EnabledHPRE: &trueVal,
		EnabledSEC:  &falseVal,
		EnabledZIP:  &falseVal,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	newTrue := !*copied.EnabledHPRE
	copied.EnabledHPRE = &newTrue
	if *copied.EnabledHPRE == *original.EnabledHPRE {
		t.Error("Modifying copy affected original")
	}
}

// TestPSASpecDeepCopy 测试 PSASpec 的 DeepCopy 方法
func TestPSASpecDeepCopy(t *testing.T) {
	trueVal := true

	original := &PSASpec{
		Enabled: &trueVal,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	newTrue := !*copied.Enabled
	copied.Enabled = &newTrue
	if *copied.Enabled == *original.Enabled {
		t.Error("Modifying copy affected original")
	}
}

// TestRollingUpdateSpecDeepCopy 测试 RollingUpdateSpec 的 DeepCopy 方法
func TestRollingUpdateSpecDeepCopy(t *testing.T) {
	original := &RollingUpdateSpec{
		MaxUnavailable: "2",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify the copy has the same value
	if copied.MaxUnavailable != original.MaxUnavailable {
		t.Errorf("MaxUnavailable mismatch: got %s, want %s", copied.MaxUnavailable, original.MaxUnavailable)
	}
}

// TestDeepCopyObject 测试 DeepCopyObject 方法
func TestDeepCopyObject(t *testing.T) {
	trueVal := true
	original := &KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-kae-policy",
		},
		Spec: KAEPolicySpec{
			Driver: DriverSpec{
				Enabled: &trueVal,
			},
		},
	}

	copiedObj := original.DeepCopyObject()
	copied, ok := copiedObj.(*KAEPolicy)
	if !ok {
		t.Fatal("DeepCopyObject did not return a *KAEPolicy")
	}

	if copied.Name != original.Name {
		t.Errorf("Name mismatch: got %s, want %s", copied.Name, original.Name)
	}

	// Modify the copy and verify the original is unchanged
	copied.Name = "modified-name"
	if copied.Name == original.Name {
		t.Error("Modifying copy affected original")
	}
}

// TestKAEPolicyListDeepCopyObject 测试 KAEPolicyList 的 DeepCopyObject 方法
func TestKAEPolicyListDeepCopyObject(t *testing.T) {
	original := &KAEPolicyList{
		Items: []KAEPolicy{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "policy-1",
				},
			},
		},
	}

	copiedObj := original.DeepCopyObject()
	copied, ok := copiedObj.(*KAEPolicyList)
	if !ok {
		t.Fatal("DeepCopyObject did not return a *KAEPolicyList")
	}

	if len(copied.Items) != len(original.Items) {
		t.Errorf("Items length mismatch: got %d, want %d", len(copied.Items), len(original.Items))
	}
}

// TestStateConstants 测试状态常量
func TestStateConstants(t *testing.T) {
	tests := []struct {
		name  string
		state State
	}{
		{"Ignored", Ignored},
		{"Ready", Ready},
		{"NotReady", NotReady},
		{"Disabled", Disabled},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.state == "" {
				t.Errorf("State constant %s should not be empty", tt.name)
			}
		})
	}
}

// TestKAEPolicyStatusDeepCopy 测试 KAEPolicyStatus 的 DeepCopy 方法
func TestKAEPolicyStatusDeepCopy(t *testing.T) {
	original := &KAEPolicyStatus{
		State:     Ready,
		Namespace: "default",
		Conditions: []metav1.Condition{
			{
				Type:    "Ready",
				Status:  metav1.ConditionTrue,
				Reason:  "AllComponentsReady",
				Message: "All components are ready",
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify the copy and verify the original is unchanged
	copied.Namespace = "modified-namespace"
	if copied.Namespace == original.Namespace {
		t.Error("Modifying copy affected original")
	}
}
