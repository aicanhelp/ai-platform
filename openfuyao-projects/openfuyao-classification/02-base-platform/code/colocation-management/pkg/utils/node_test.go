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

package utils

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestIsOversubscriptionNode(t *testing.T) {
	tests := []struct {
		name     string
		node     *corev1.Node
		expected bool
	}{
		{
			name: "oversubscription node - both labels present and true",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
						common.OversubNodeLabel:    "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "not oversubscription node - colocation true but oversub false",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
						common.OversubNodeLabel:    "false",
					},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - colocation true but oversub missing",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
					},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - colocation false but oversub true",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "false",
						common.OversubNodeLabel:    "true",
					},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - colocation missing but oversub true",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.OversubNodeLabel: "true",
					},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - both labels missing",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - colocation true but oversub empty",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
						common.OversubNodeLabel:    "",
					},
				},
			},
			expected: false,
		},
		{
			name: "not oversubscription node - colocation true but oversub random value",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
						common.OversubNodeLabel:    "random",
					},
				},
			},
			expected: false,
		},
		{
			name:     "not oversubscription node - nil node",
			node:     nil,
			expected: false,
		},
		{
			name: "not oversubscription node - nil labels",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: nil,
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsOversubscriptionNode(tt.node)
			if result != tt.expected {
				t.Errorf("IsOversubscriptionNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsColocationNode(t *testing.T) {
	tests := []struct {
		name     string
		node     *corev1.Node
		expected bool
	}{
		{
			name: "colocation node - label present and true",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "not colocation node - label false",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "false",
					},
				},
			},
			expected: false,
		},
		{
			name: "not colocation node - label missing",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
			},
			expected: false,
		},
		{
			name: "not colocation node - label empty",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "",
					},
				},
			},
			expected: false,
		},
		{
			name: "not colocation node - label random value",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "random",
					},
				},
			},
			expected: false,
		},
		{
			name:     "not colocation node - nil node",
			node:     nil,
			expected: false,
		},
		{
			name: "not colocation node - nil labels",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: nil,
				},
			},
			expected: false,
		},
		{
			name: "colocation node - case insensitive true",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.ColocationNodeLabel: "True",
					},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsColocationNode(tt.node)
			if result != tt.expected {
				t.Errorf("IsColocationNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}
