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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

// TestBaseQoSMutator 测试基础QoS mutator
func TestBaseQoSMutator(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		expectedError string
		expectedClass string
	}{
		{
			name:          "nil annotations",
			pod:           &v1.Pod{},
			expectedError: "pod annotations are nil",
		},
		{
			name: "missing qos annotation",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			expectedError: "qoS level annotation not found",
		},
		{
			name: "successful priority class setting - HLS",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
			},
			expectedClass: "priority-hls",
		},
		{
			name: "successful priority class setting - LS",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "LS",
					},
				},
			},
			expectedClass: "priority-ls",
		},
		{
			name: "existing matching priority class",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
				Spec: v1.PodSpec{
					PriorityClassName: "priority-hls",
				},
			},
			expectedClass: "priority-hls",
		},
		{
			name: "conflicting priority class",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
				Spec: v1.PodSpec{
					PriorityClassName: "priority-ls",
				},
			},
			expectedError: "existing PriorityClassName conflicts with QoS level",
		},
	}

	mutator := &baseQoSMutator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := mutator.ensurePriorityClass(tt.pod)

			if tt.expectedError != "" {
				if err == nil {
					t.Errorf("Expected error '%s', but got nil", tt.expectedError)
				} else if !strings.Contains(err.Error(), tt.expectedError) {
					t.Errorf("Expected error containing '%s', but got '%s'", tt.expectedError, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.pod.Spec.PriorityClassName != tt.expectedClass {
				t.Errorf("Expected PriorityClassName '%s', but got '%s'", tt.expectedClass, tt.pod.Spec.PriorityClassName)
			}
		})
	}
}

// TestBaseQoSMutatorClean 测试清理priority字段
func TestBaseQoSMutatorClean(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		shouldCleanup bool
		expectedErr   bool
	}{
		{
			name: "pod without priority field",
			pod:  &v1.Pod{},
		},
	}

	mutator := &baseQoSMutator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := mutator.cleanupPriorityField(tt.pod)

			if tt.expectedErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectedErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestHLSPodQoSMutator 测试HLS mutator
func TestHLSPodQoSMutator(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		expectMutated bool
		expectError   bool
	}{
		{
			name: "HLS pod successful mutation",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
			},
			expectMutated: true,
			expectError:   false,
		},
		{
			name: "HLS pod with error",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Annotations: map[string]string{},
				},
			},
			expectMutated: false,
			expectError:   true,
		},
	}

	mutator := NewHLSPodQoSMutator()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mutated, err := mutator.MutatePod(tt.pod)

			if tt.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if mutated != tt.expectMutated {
				t.Errorf("Expected mutated=%v, got %v", tt.expectMutated, mutated)
			}
		})
	}
}

// TestLSPodQoSMutator 测试LS mutator
func TestLSPodQoSMutator(t *testing.T) {
	mutator := NewLSPodQoSMutator()

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pod",
			Annotations: map[string]string{
				common.QosClassAnnotationKey: "LS",
			},
		},
	}

	mutated, err := mutator.MutatePod(pod)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !mutated {
		t.Error("Expected pod to be mutated")
	}
	if pod.Spec.PriorityClassName != "priority-ls" {
		t.Errorf("Expected priority-ls, got %s", pod.Spec.PriorityClassName)
	}
}

// TestBEPodQoSMutator 测试BE mutator
func TestBEPodQoSMutator(t *testing.T) {
	mutator := NewBEPodQoSMutator()

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pod",
			Annotations: map[string]string{
				common.QosClassAnnotationKey: "BE",
			},
		},
	}

	mutated, err := mutator.MutatePod(pod)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !mutated {
		t.Error("Expected pod to be mutated")
	}
	if pod.Spec.PriorityClassName != "priority-be" {
		t.Errorf("Expected priority-be, got %s", pod.Spec.PriorityClassName)
	}
}

// TestEdgeCases 测试边界情况
func TestEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		qosLevel string
		expected string
	}{
		{"mixed case", "HlS", "priority-hls"},
		{"lower case", "be", "priority-be"},
		{"upper case", "LS", "priority-ls"},
	}

	mutator := &baseQoSMutator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: tt.qosLevel,
					},
				},
			}

			err := mutator.ensurePriorityClass(pod)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if pod.Spec.PriorityClassName != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, pod.Spec.PriorityClassName)
			}
		})
	}
}

// TestErrorMessages 测试错误消息内容
func TestErrorMessages(t *testing.T) {
	mutator := &baseQoSMutator{}

	pod := &v1.Pod{}
	err := mutator.ensurePriorityClass(pod)
	if err == nil || !strings.Contains(err.Error(), "pod annotations are nil") {
		t.Errorf("Expected nil annotations error, got: %v", err)
	}

	pod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
	}
	err = mutator.ensurePriorityClass(pod)
	if err == nil || !strings.Contains(err.Error(), "qoS level annotation not found") {
		t.Errorf("Expected missing annotation error, got: %v", err)
	}
}
