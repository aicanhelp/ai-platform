/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsXPUName(t *testing.T) {
	assert.True(t, IsXPUName(VNPUName))
	assert.False(t, IsXPUName("aaa"))
}

func TestGetXPUType(t *testing.T) {
	container := &v1.Container{
		Resources: v1.ResourceRequirements{
			Limits: v1.ResourceList{},
		},
	}
	if got := GetXPUType(container, "xpu."); got != "" {
		t.Errorf("GetXPUType() = %v, want %v", got, "")
	}

	container = &v1.Container{
		Resources: v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceName("xpu.310P"): resource.MustParse("1"),
			},
		},
	}
	if got := GetXPUType(container, "xpu."); got != "310P" {
		t.Errorf("GetXPUType() = %v, want %v", got, "310P")
	}

	container = &v1.Container{
		Resources: v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceName("cpu"): resource.MustParse("1"),
			},
		},
	}
	if got := GetXPUType(container, "xpu."); got != "" {
		t.Errorf("GetXPUType() = %v, want %v", got, "")
	}
}

func TestSafePrint(t *testing.T) {
	const digValue = 123
	if SafePrint() != "" {
		t.Error("SafePrint with no arguments should return an empty string")
	}

	if SafePrint("test") != "test" {
		t.Error("SafePrint with single argument should return the argument")
	}

	if SafePrint("test", digValue, true) != "test123 true" {
		t.Error("SafePrint with multiple arguments should concatenate the arguments")
	}

	if SafePrint("test\n", "\r123") != "test  123" {
		t.Error("SafePrint with newline and carriage return should replace them with space")
	}
}

func TestGetVXPUResource(t *testing.T) {
	const expectedVXPUValue = 10
	t.Run("ResourceExists", func(t *testing.T) {
		container := &v1.Container{
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName("vxpu"): *resource.NewQuantity(expectedVXPUValue, resource.BinarySI),
				},
			},
		}

		result := GetVXPUResource(container, "vxpu")

		if result != expectedVXPUValue {
			t.Errorf("Expected 10, but got %d", result)
		}
	})

	t.Run("ResourceNotExists", func(t *testing.T) {
		container := &v1.Container{
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{},
			},
		}

		result := GetVXPUResource(container, "vxpu")

		if result != -1 {
			t.Errorf("Expected -1, but got %d", result)
		}
	})
}

func TestGetNodeSchedulerPolicy(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected string
	}{
		{
			name:     "nil pod returns binpack",
			pod:      nil,
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "nil annotations returns binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "empty annotation returns binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "explicit binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						NodeSchedulerPolicyAnnotation: SchedulerPolicyBinpack,
					},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "explicit spread",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						NodeSchedulerPolicyAnnotation: SchedulerPolicySpread,
					},
				},
			},
			expected: SchedulerPolicySpread,
		},
		{
			name: "invalid value returns binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						NodeSchedulerPolicyAnnotation: "invalid",
					},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "empty string value returns binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						NodeSchedulerPolicyAnnotation: "",
					},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetNodeSchedulerPolicy(tt.pod)
			if result != tt.expected {
				t.Errorf("GetNodeSchedulerPolicy() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetDeviceSchedulerPolicy(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected string
	}{
		{
			name:     "nil pod returns binpack",
			pod:      nil,
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "default binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
		{
			name: "explicit spread",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						DeviceSchedulerPolicyAnnotation: SchedulerPolicySpread,
					},
				},
			},
			expected: SchedulerPolicySpread,
		},
		{
			name: "explicit binpack",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						DeviceSchedulerPolicyAnnotation: SchedulerPolicyBinpack,
					},
				},
			},
			expected: SchedulerPolicyBinpack,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetDeviceSchedulerPolicy(tt.pod)
			if result != tt.expected {
				t.Errorf("GetDeviceSchedulerPolicy() = %v, want %v", result, tt.expected)
			}
		})
	}
}
