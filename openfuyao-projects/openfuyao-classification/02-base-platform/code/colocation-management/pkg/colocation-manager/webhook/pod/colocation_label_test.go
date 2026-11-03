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
* Date: 2025-08-18
 */

package pod

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

// TestColocationLabelMutator 测试ColocationLabelMutator
func TestColocationLabelMutator(t *testing.T) {
	tests := []struct {
		name               string
		pod                *v1.Pod
		expectedAnnotation string
		expectedValue      string
		expectMutated      bool
		expectError        bool
	}{
		{
			name: "pod with nil annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod-1",
					Annotations: nil,
				},
			},
			expectedAnnotation: common.OversubNodeLabel,
			expectedValue:      "true",
			expectMutated:      true,
			expectError:        false,
		},
		{
			name: "pod with empty annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod-2",
					Annotations: map[string]string{},
				},
			},
			expectedAnnotation: common.OversubNodeLabel,
			expectedValue:      "true",
			expectMutated:      true,
			expectError:        false,
		},
		{
			name: "pod with existing annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod-3",
					Annotations: map[string]string{
						"existing-annotation": "existing-value",
						"another-annotation":  "another-value",
					},
				},
			},
			expectedAnnotation: common.OversubNodeLabel,
			expectedValue:      "true",
			expectMutated:      true,
			expectError:        false,
		},
		{
			name: "pod with existing oversub annotation",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod-4",
					Annotations: map[string]string{
						common.OversubNodeLabel: "false",
						"another-annotation":    "another-value",
					},
				},
			},
			expectedAnnotation: common.OversubNodeLabel,
			expectedValue:      "true",
			expectMutated:      true,
			expectError:        false,
		},
		{
			name: "pod with correct oversub annotation already set",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod-5",
					Annotations: map[string]string{
						common.OversubNodeLabel: "true",
					},
				},
			},
			expectedAnnotation: common.OversubNodeLabel,
			expectedValue:      "true",
			expectMutated:      true,
			expectError:        false,
		},
	}

	mutator := &ColocationLabelMutator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalAnnotations := make(map[string]string)
			if tt.pod.Annotations != nil {
				for k, v := range tt.pod.Annotations {
					originalAnnotations[k] = v
				}
			}

			mutated, err := mutator.MutatePod(tt.pod)

			if tt.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if mutated != tt.expectMutated {
				t.Errorf("Expected mutated=%v, got %v", tt.expectMutated, mutated)
			}

			if tt.pod.Annotations == nil {
				t.Error("Annotations map should not be nil after mutation")
				return
			}

			value, exists := tt.pod.Annotations[tt.expectedAnnotation]
			if !exists {
				t.Errorf("Expected annotation '%s' not found", tt.expectedAnnotation)
				return
			}

			if value != tt.expectedValue {
				t.Errorf("Expected annotation value '%s', got '%s'", tt.expectedValue, value)
			}

			for k, v := range originalAnnotations {
				if k != tt.expectedAnnotation {
					if currentValue, exists := tt.pod.Annotations[k]; !exists || currentValue != v {
						t.Errorf("Existing annotation '%s' was modified or removed", k)
					}
				}
			}

			t.Logf("Test passed")
		})
	}
}

// TestColocationEdgeCases 测试边界情况
func TestColocationEdgeCases(t *testing.T) {
	mutator := &ColocationLabelMutator{}

	t.Run("pod with nil object meta", func(t *testing.T) {
		pod := &v1.Pod{}
		mutated, err := mutator.MutatePod(pod)

		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !mutated {
			t.Error("Expected pod to be mutated")
		}
		if pod.Annotations == nil {
			t.Error("Annotations map should be created")
		}
		if pod.Annotations[common.OversubNodeLabel] != "true" {
			t.Error("Oversub annotation should be set to true")
		}
	})

	t.Run("multiple mutations on same pod", func(t *testing.T) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pod",
			},
		}

		mutated1, err1 := mutator.MutatePod(pod)
		if err1 != nil || !mutated1 {
			t.Error("First mutation should succeed")
		}

		mutated2, err2 := mutator.MutatePod(pod)
		if err2 != nil {
			t.Errorf("Second mutation should not fail: %v", err2)
		}
		if !mutated2 {
			t.Error("Second mutation should return true (operation was performed)")
		}

		if pod.Annotations[common.OversubNodeLabel] != "true" {
			t.Error("Annotation value should remain 'true' after multiple mutations")
		}
	})
}
