// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package serverusage

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

func TestNewGetServer(t *testing.T) {
	tests := []struct {
		name     string
		node     *corev1.Node
		cardName string
		expected common.Labels
	}{
		{
			name: "Atlas 800I A3",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"kubernetes.io/arch": "amd64",
					},
				},
			},
			cardName: "Atlas 800I A3",
			expected: common.Labels{"server-usage": "infer"},
		},
		{
			name: "Atlas 800I A2",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"kubernetes.io/arch": "arm64",
					},
				},
			},
			cardName: "Atlas 800I A2",
			expected: common.Labels{"server-usage": "infer"},
		},
		{
			name: "Unknown Device",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"kubernetes.io/arch": "arm64",
					},
				},
			},
			cardName: "Unknown Device",
			expected: common.Labels{},
		},
		{
			name: "No Devices",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"kubernetes.io/arch": "arm64",
					},
				},
			},
			cardName: "UNKNOWN",
			expected: common.Labels{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(utils.GetCardInfo, func(*corev1.Node) (string, error) {
				return tt.cardName, nil
			})
			defer patch.Reset()

			result, _ := NewServerLabeler(tt.node)

			if !assert.Equal(t, result, tt.expected) {
				t.Errorf("Test %s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}
