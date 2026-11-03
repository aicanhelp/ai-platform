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
)

// TestSchedulerMutator 测试SchedulerMutator功能
func TestSchedulerMutator(t *testing.T) {
	tests := []struct {
		name              string
		pod               *v1.Pod
		expectedScheduler string
		expectMutated     bool
		expectError       bool
		errorMessage      string
		description       string
	}{
		{
			name:          "nil pod",
			pod:           nil,
			expectMutated: false,
			expectError:   true,
			errorMessage:  "pod cannot be nil",
			description:   "应该返回错误当pod为nil时",
		},
		{
			name: "pod with empty scheduler name",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-1",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					SchedulerName: "", // 空调度器名称
				},
			},
			expectedScheduler: "volcano",
			expectMutated:     true,
			expectError:       false,
			description:       "应该将空调度器名称设置为volcano",
		},
		{
			name: "pod with default scheduler",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-2",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					SchedulerName: "default-scheduler", // 默认调度器
				},
			},
			expectedScheduler: "volcano",
			expectMutated:     true,
			expectError:       false,
			description:       "应该将默认调度器替换为volcano",
		},
		{
			name: "pod with custom scheduler",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-3",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					SchedulerName: "custom-scheduler", // 自定义调度器
				},
			},
			expectedScheduler: "volcano",
			expectMutated:     true,
			expectError:       false,
			description:       "应该将自定义调度器替换为volcano",
		},
		{
			name: "pod already using volcano scheduler",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-4",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					SchedulerName: "volcano", // 已经是volcano
				},
			},
			expectedScheduler: "volcano",
			expectMutated:     false, // 不需要修改，返回false
			expectError:       false,
			description:       "应该检测到已经是volcano调度器并返回false",
		},
		{
			name: "pod with volcano scheduler case insensitive",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-5",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					SchedulerName: "Volcano", // 大小写不同
				},
			},
			expectedScheduler: "volcano", // 应该设置为小写
			expectMutated:     true,      // 因为大小写不同，需要修改
			expectError:       false,
			description:       "应该将大小写不同的volcano统一为小写",
		},
	}

	mutator := &SchedulerMutator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			mutated, err := mutator.MutatePod(tt.pod)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				} else if err.Error() != tt.errorMessage {
					t.Errorf("Expected error message '%s', got '%s'", tt.errorMessage, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if mutated != tt.expectMutated {
				t.Errorf("Expected mutated=%v, got %v", tt.expectMutated, mutated)
			}

			if tt.pod != nil && tt.pod.Spec.SchedulerName != tt.expectedScheduler {
				t.Errorf("Expected scheduler name '%s', got '%s'", tt.expectedScheduler, tt.pod.Spec.SchedulerName)
			}

			t.Logf("Test passed: %s", tt.description)
		})
	}
}

// TestScheulerEdgeCases 测试边界情况
func TestScheulerEdgeCases(t *testing.T) {
	mutator := &SchedulerMutator{}

	t.Run("pod with minimal fields", func(t *testing.T) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "minimal-pod",
				Namespace: "test",
			},
		}

		mutated, err := mutator.MutatePod(pod)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !mutated {
			t.Error("Expected pod to be mutated")
		}
		if pod.Spec.SchedulerName != "volcano" {
			t.Errorf("Expected scheduler 'volcano', got '%s'", pod.Spec.SchedulerName)
		}
	})

	t.Run("multiple mutations on same pod", func(t *testing.T) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "multi-mutation-pod",
				Namespace: "default",
			},
			Spec: v1.PodSpec{
				SchedulerName: "default-scheduler",
			},
		}

		mutated1, err1 := mutator.MutatePod(pod)
		if err1 != nil || !mutated1 {
			t.Error("First mutation should succeed and return true")
		}
		if pod.Spec.SchedulerName != "volcano" {
			t.Error("Scheduler should be set to volcano after first mutation")
		}

		mutated2, err2 := mutator.MutatePod(pod)
		if err2 != nil {
			t.Errorf("Second mutation should not fail: %v", err2)
		}
		if mutated2 {
			t.Error("Second mutation should return false (no change needed)")
		}
		if pod.Spec.SchedulerName != "volcano" {
			t.Error("Scheduler should remain volcano after second mutation")
		}
	})
}
