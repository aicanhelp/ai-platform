/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package pod

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/common"
)

func TestNewResourceNameList(t *testing.T) {
	tests := []struct {
		name     string
		extender bool
		wantLen  int
		wantType []corev1.ResourceName
	}{
		{
			name:     "extender true",
			extender: true,
			wantLen:  2,
			wantType: []corev1.ResourceName{common.ExtenderResourceCPU, common.ExtenderResourceMemory},
		},
		{
			name:     "extender false",
			extender: false,
			wantLen:  2,
			wantType: []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewResourceNameList(tt.extender)
			if len(result.ResourceNameList) != tt.wantLen {
				t.Errorf("NewResourceNameList() length = %d, want %d", len(result.ResourceNameList), tt.wantLen)
			}
			if !reflect.DeepEqual(result.ResourceNameList, tt.wantType) {
				t.Errorf("NewResourceNameList() = %v, want %v", result.ResourceNameList, tt.wantType)
			}
		})
	}
}

func TestValidateResource(t *testing.T) {
	tests := []struct {
		name         string
		resourceName corev1.ResourceName
		value        string
		expected     bool
	}{
		{"CPU valid integer", common.ExtenderResourceCPU, "1000", true},
		{"CPU valid with k", common.ExtenderResourceCPU, "1000k", true},
		{"CPU invalid decimal", common.ExtenderResourceCPU, "100.5", false},
		{"CPU invalid unit", common.ExtenderResourceCPU, "100m", false},

		{"Memory valid integer", common.ExtenderResourceMemory, "1000", true},
		{"Memory valid KiB", common.ExtenderResourceMemory, "1000Ki", true},
		{"Memory valid Mi", common.ExtenderResourceMemory, "1000Mi", true},
		{"Memory valid Gi", common.ExtenderResourceMemory, "1000Gi", true},
		{"Memory invalid decimal", common.ExtenderResourceMemory, "100.5", false},
		{"Memory invalid unit", common.ExtenderResourceMemory, "100KB", false},

		{"Unknown resource", corev1.ResourceStorage, "1000", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := validateResource(tt.resourceName, tt.value)
			if result != tt.expected {
				t.Errorf("validateResource(%s, %s) = %v, want %v", tt.resourceName, tt.value, result, tt.expected)
			}
		})
	}
}

func TestValidateExtenderItem(t *testing.T) {
	tests := []struct {
		name     string
		item     *core.ExtendedResourceItem
		expected error
	}{
		{
			name: "valid CPU and Memory",
			item: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU:    resource.MustParse("1000"),
					common.ExtenderResourceMemory: resource.MustParse("1Gi"),
				},
				Limits: corev1.ResourceList{
					common.ExtenderResourceCPU:    resource.MustParse("2000"),
					common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				},
			},
			expected: nil,
		},
		{
			name: "invalid CPU format",
			item: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("100.5"),
				},
			},
			expected: fmt.Errorf("container requests ExtenderCPU validate error"),
		},
		{
			name: "empty item",
			item: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{},
				Limits:   corev1.ResourceList{},
			},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateExtenderItem(tt.item)
			if (err != nil) != (tt.expected != nil) {
				t.Errorf("validateExtenderItem() error = %v, expected error = %v", err, tt.expected)
			}
			if err != nil && tt.expected != nil && err.Error() != tt.expected.Error() {
				t.Errorf("validateExtenderItem() error = %v, expected error = %v", err, tt.expected)
			}
		})
	}
}

func TestGetExtenderResources(t *testing.T) {
	tests := []struct {
		name        string
		container   *corev1.Container
		expected    *core.ExtendedResourceItem
		expectError bool
	}{
		{
			name: "valid extender resources",
			container: &corev1.Container{
				Name: "test-container",
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						common.ExtenderResourceCPU:    resource.MustParse("1000"),
						common.ExtenderResourceMemory: resource.MustParse("1Gi"),
					},
					Limits: corev1.ResourceList{
						common.ExtenderResourceCPU:    resource.MustParse("2000"),
						common.ExtenderResourceMemory: resource.MustParse("2Gi"),
					},
				},
			},
			expected: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU:    resource.MustParse("1000"),
					common.ExtenderResourceMemory: resource.MustParse("1Gi"),
				},
				Limits: corev1.ResourceList{
					common.ExtenderResourceCPU:    resource.MustParse("2000"),
					common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				},
			},
			expectError: false,
		},
		{
			name: "mixed native and extender resources - should error",
			container: &corev1.Container{
				Name: "test-container",
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						common.ExtenderResourceCPU: resource.MustParse("1000"),
						corev1.ResourceCPU:         resource.MustParse("500m"),
					},
				},
			},
			expectError: true,
		},
		{
			name: "only limits - should auto-fill requests",
			container: &corev1.Container{
				Name: "test-container",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						common.ExtenderResourceCPU: resource.MustParse("2000"),
					},
				},
			},
			expected: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("2000"),
				},
				Limits: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("2000"),
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := getExtenderResoruces(tt.container)

			if tt.expectError {
				if err == nil {
					t.Errorf("getExtenderResoruces() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("getExtenderResoruces() unexpected error: %v", err)
				return
			}

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("getExtenderResoruces() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestSetExtenderResourceAnnotation(t *testing.T) {
	tests := []struct {
		name          string
		annotations   map[string]string
		configuration *core.ExtendedResourceConfiguration
		expectError   bool
	}{
		{
			name:        "valid configuration",
			annotations: map[string]string{},
			configuration: &core.ExtendedResourceConfiguration{
				Containers: map[string]core.ExtendedResourceItem{
					"test-container": {
						Requests: corev1.ResourceList{
							common.ExtenderResourceCPU: resource.MustParse("1000"),
						},
						Limits: corev1.ResourceList{
							common.ExtenderResourceCPU: resource.MustParse("2000"),
						},
					},
				},
			},
			expectError: false,
		},
		{
			name:          "nil configuration - should error",
			annotations:   map[string]string{},
			configuration: nil,
			expectError:   true,
		},
		{
			name:          "nil annotations - should error",
			annotations:   nil,
			configuration: &core.ExtendedResourceConfiguration{},
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := setExtenderResourceAnnotation(tt.annotations, tt.configuration)

			if tt.expectError {
				if err == nil {
					t.Errorf("setExtenderResourceAnnotation() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("setExtenderResourceAnnotation() unexpected error: %v", err)
				return
			}

			if _, exists := result[common.ExtenderResourceConfigurationAnnotation]; !exists {
				t.Errorf("ExtenderResourceConfigurationAnnotation not set")
			}

			if _, exists := result[common.PriorityAnnotationKey]; !exists || result[common.PriorityAnnotationKey] != "true" {
				t.Errorf("PriorityAnnotationKey not set correctly")
			}

			var config core.ExtendedResourceConfiguration
			if err := json.Unmarshal([]byte(result[common.ExtenderResourceConfigurationAnnotation]), &config); err != nil {
				t.Errorf("Failed to unmarshal annotation: %v", err)
			}
		})
	}
}

func TestSetExtenderResourceAnnotationList(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		configuration *core.ExtendedResourceConfiguration
		expectError   bool
	}{
		{
			name: "new annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			configuration: &core.ExtendedResourceConfiguration{
				Containers: map[string]core.ExtendedResourceItem{},
			},
			expectError: false,
		},
		{
			name: "existing same annotation - no change",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.ExtenderResourceConfigurationAnnotation: `{"containers":{},"initContainers":{}}`,
					},
				},
			},
			configuration: &core.ExtendedResourceConfiguration{
				Containers: map[string]core.ExtendedResourceItem{},
			},
			expectError: false,
		},
		{
			name: "existing different annotation - should update",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.ExtenderResourceConfigurationAnnotation: `{"containers":{"old-container":{}},"initContainers":{}}`,
					},
				},
			},
			configuration: &core.ExtendedResourceConfiguration{
				Containers: map[string]core.ExtendedResourceItem{
					"new-container": {},
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := setExtenderResourceAnnotationlist(tt.pod, tt.configuration)

			if tt.expectError {
				if err == nil {
					t.Errorf("setExtenderResourceAnnotationlist() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("setExtenderResourceAnnotationlist() unexpected error: %v", err)
			}
		})
	}
}

// TestPodResourceMutatePod tests the PodResourceMutator.MutatePod method
func TestPodResourceMutatePod(t *testing.T) {
	mutator := &PodResourceMutator{}

	tests := []struct {
		name        string
		pod         *corev1.Pod
		expected    bool
		expectError bool
	}{
		{
			name: "valid pod with extender resources",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
								Limits: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("2000"),
								},
							},
						},
					},
				},
			},
			expected:    true,
			expectError: false,
		},
		{
			name: "pod with no containers - should error",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{},
				},
			},
			expected:    false,
			expectError: true,
		},
		{
			name: "pod with init containers",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name: "init-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("500"),
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
							},
						},
					},
				},
			},
			expected:    true,
			expectError: false,
		},
		{
			name: "pod with mixed resources - should error",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
									corev1.ResourceCPU:         resource.MustParse("500m"),
								},
							},
						},
					},
				},
			},
			expected:    false,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allowed, err := mutator.MutatePod(tt.pod)

			if tt.expectError {
				if err == nil {
					t.Errorf("MutatePod() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("MutatePod() unexpected error: %v", err)
				return
			}

			if allowed != tt.expected {
				t.Errorf("MutatePod() = %v, want %v", allowed, tt.expected)
			}

			if allowed {
				annotations := tt.pod.GetAnnotations()
				if annotations == nil {
					t.Errorf("Annotations should be set after successful mutation")
					return
				}

				if _, exists := annotations[common.ExtenderResourceConfigurationAnnotation]; !exists {
					t.Errorf("ExtenderResourceConfigurationAnnotation should be set")
				}

				_, exists := annotations[common.PriorityAnnotationKey]
				if !exists || annotations[common.PriorityAnnotationKey] != "true" {
					t.Errorf("PriorityAnnotationKey should be set to 'true'")
				}
			}
		})
	}
}

func TestCheckExtenderResource(t *testing.T) {
	tests := []struct {
		name        string
		container   *corev1.Container
		item        *core.ExtendedResourceItem
		expectError bool
	}{
		{
			name: "valid extender resource only",
			container: &corev1.Container{
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{},
					Limits:   corev1.ResourceList{},
				},
			},
			item: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("1000"),
				},
				Limits: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("2000"),
				},
			},
			expectError: false,
		},
		{
			name: "container with native resources - should error",
			container: &corev1.Container{
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU: resource.MustParse("500m"),
					},
				},
			},
			item: &core.ExtendedResourceItem{
				Requests: corev1.ResourceList{
					common.ExtenderResourceCPU: resource.MustParse("1000"),
				},
			},
			expectError: true,
		},
		{
			name: "no resources at all - should error",
			container: &corev1.Container{
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{},
					Limits:   corev1.ResourceList{},
				},
			},
			item:        &core.ExtendedResourceItem{},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := checkExtenderResource(tt.container, tt.item)

			if tt.expectError {
				if err == nil {
					t.Errorf("checkExtenderResource() expected error but got nil")
				}
			} else {
				if err != nil {
					t.Errorf("checkExtenderResource() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestVerifyCPUEqual(t *testing.T) {
	tests := []struct {
		name      string
		requests  corev1.ResourceList
		limits    corev1.ResourceList
		expected  bool
		expectErr bool
		errMsg    string
	}{
		{
			name: "CPU equal",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1000m"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "CPU not equal",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1000m"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("2"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must be equal",
		},
		{
			name: "CPU requests missing",
			requests: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must have both requests and limits",
		},
		{
			name: "CPU limits missing",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must have both requests and limits",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := verifyCPUEqual("test-container", tt.requests, tt.limits)

			if tt.expectErr {
				if err == nil {
					t.Errorf("verifyCPUEqual() expected error but got nil")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("verifyCPUEqual() error = %v, expected to contain %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("verifyCPUEqual() unexpected error: %v", err)
					return
				}
			}

			if result != tt.expected {
				t.Errorf("verifyCPUEqual() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestVerifyMemoryEqual(t *testing.T) {
	tests := []struct {
		name      string
		requests  corev1.ResourceList
		limits    corev1.ResourceList
		expected  bool
		expectErr bool
		errMsg    string
	}{
		{
			name: "Memory equal",
			requests: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "Memory not equal",
			requests: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("2Gi"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must be equal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := verifyMemoryEqual("test-container", tt.requests, tt.limits)

			if tt.expectErr {
				if err == nil {
					t.Errorf("verifyMemoryEqual() expected error but got nil")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("verifyMemoryEqual() error = %v, expected to contain %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("verifyMemoryEqual() unexpected error: %v", err)
					return
				}
			}

			if result != tt.expected {
				t.Errorf("verifyMemoryEqual() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestVerifyCPUInteger(t *testing.T) {
	tests := []struct {
		name      string
		requests  corev1.ResourceList
		expected  bool
		expectErr bool
		errMsg    string
	}{
		{
			name: "CPU integer - whole number",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "CPU integer - millicore multiple",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1000m"),
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "CPU not integer - decimal",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1.5"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must be an integer",
		},
		{
			name: "CPU not integer - fractional millicore",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1500m"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "must be an integer",
		},
		{
			name: "CPU requests missing",
			requests: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected:  false,
			expectErr: true,
			errMsg:    "CPU requests not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := verifyCPUInteger("test-container", tt.requests)

			if tt.expectErr {
				if err == nil {
					t.Errorf("verifyCPUInteger() expected error but got nil")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("verifyCPUInteger() error = %v, expected to contain %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("verifyCPUInteger() unexpected error: %v", err)
					return
				}
			}

			if result != tt.expected {
				t.Errorf("verifyCPUInteger() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestVerifyResource(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		resourceType string
		expected     bool
		expectErr    bool
	}{
		{
			name: "has extend resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
							},
						},
					},
				},
			},
			resourceType: "extend",
			expected:     true,
			expectErr:    false,
		},
		{
			name: "has raw resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1000m"),
								},
							},
						},
					},
				},
			},
			resourceType: "raw",
			expected:     true,
			expectErr:    false,
		},
		{
			name: "no extend resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1000m"),
								},
							},
						},
					},
				},
			},
			resourceType: "extend",
			expected:     false,
			expectErr:    false,
		},
		{
			name: "invalid resource type",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{},
			},
			resourceType: "invalid",
			expected:     false,
			expectErr:    true,
		},
		{
			name: "has extend resources in init container",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									common.ExtenderResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
					Containers: []corev1.Container{},
				},
			},
			resourceType: "extend",
			expected:     true,
			expectErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := VerifyResource(tt.pod, tt.resourceType)

			if tt.expectErr {
				if err == nil {
					t.Errorf("VerifyResource() expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("VerifyResource() unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("VerifyResource() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestVerifyPodResources(t *testing.T) {
	tests := []struct {
		name      string
		pod       *corev1.Pod
		podType   string
		expectErr bool
		errMsg    string
	}{
		{
			name: "BE pod with extend resources only - valid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
								Limits: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
							},
						},
					},
				},
			},
			podType:   "be",
			expectErr: false,
		},
		{
			name: "BE pod with raw resources - invalid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1000m"),
								},
							},
						},
					},
				},
			},
			podType:   "be",
			expectErr: true,
			errMsg:    "BE pod must use extend resources",
		},
		{
			name: "LS pod with raw resources only - valid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1000m"),
								},
							},
						},
					},
				},
			},
			podType:   "ls",
			expectErr: false,
		},
		{
			name: "LS pod with extend resources - invalid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU: resource.MustParse("1000"),
								},
							},
						},
					},
				},
			},
			podType:   "ls",
			expectErr: true,
			errMsg:    "LS pod cannot use extend resources",
		},
		{
			name: "HLS pod with raw resources only - valid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1000m"),
								},
							},
						},
					},
				},
			},
			podType:   "hls",
			expectErr: false,
		},
		{
			name: "invalid pod type",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{},
			},
			podType:   "invalid",
			expectErr: false, // 其他类型应该放行
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := VerifyPodResources(tt.pod, tt.podType)

			if tt.expectErr {
				if err == nil {
					t.Errorf("VerifyPodResources() expected error but got nil")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("VerifyPodResources() error = %v, expected to contain %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("VerifyPodResources() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestVerifyResourceEquality(t *testing.T) {
	tests := []struct {
		name      string
		pod       *corev1.Pod
		podType   string
		expected  bool
		expectErr bool
		errMsg    string
	}{
		{
			name: "HLS pod with all conditions met - valid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
				},
			},
			podType:   "hls",
			expected:  true,
			expectErr: false,
		},
		{
			name: "HLS pod with CPU not equal - invalid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("2"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
				},
			},
			podType:   "hls",
			expected:  false,
			expectErr: true,
			errMsg:    "HLS pod validation failed",
		},
		{
			name: "LS pod with all conditions met - invalid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
				},
			},
			podType:   "ls",
			expected:  false,
			expectErr: true,
			errMsg:    "cannot satisfy all three conditions simultaneously",
		},
		{
			name: "LS pod with only two conditions met - valid",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1.5"), // 非整数CPU
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1.5"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
				},
			},
			podType:   "ls",
			expected:  true,
			expectErr: false,
		},
		{
			name: "invalid pod type",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{},
						},
					},
				},
			},
			podType:   "invalid",
			expected:  false,
			expectErr: true,
			errMsg:    "invalid pod type",
		},
		{
			name: "pod with init containers",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name: "init-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name: "main-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("2"),
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("2"),
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			podType:   "hls",
			expected:  true,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := VerifyResourceEquality(tt.pod, tt.podType)

			if tt.expectErr {
				if err == nil {
					t.Errorf("VerifyResourceEquality() expected error but got nil")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("VerifyResourceEquality() error = %v, expected to contain %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("VerifyResourceEquality() unexpected error: %v", err)
					return
				}
			}

			if result != tt.expected {
				t.Errorf("VerifyResourceEquality() = %v, want %v", result, tt.expected)
			}
		})
	}
}
