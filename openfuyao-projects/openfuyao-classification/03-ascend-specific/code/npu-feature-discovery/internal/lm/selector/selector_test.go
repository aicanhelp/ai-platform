// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package selector

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestHasControlPlaneLabel(t *testing.T) {
	type args struct {
		labels map[string]string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "has control plane label",
			args: args{
				labels: map[string]string{controlPlaneRoleLabelKey: "control-plane"},
			},
			want: true,
		},
		{
			name: "not has control plane label",
			args: args{
				labels: map[string]string{},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hasControlPlaneLabel(tt.args.labels); got != tt.want {
				t.Errorf("hasControlPlaneLabel() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewSelectorLabeler(t *testing.T) {
	type args struct {
		node *corev1.Node
	}
	tests := []struct {
		name string
		args args
		want common.Labeler
	}{
		{
			name: "is control panel but has no npu",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							controlPlaneRoleLabelKey: "",
						},
					},
				},
			},
			want: common.Labels{
				MasterLabelKey: masterLabelValue,
			},
		},
		{
			name: "is not control panel",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{},
					},
				},
			},
			want: common.Labels{
				WorkerRoleLabelKey: workerRoleLabelValue,
				WorkerLabelKey:     workerLabelValue,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewSelectorLabeler(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newSelectorLabeler() = %v, want %v", got, tt.want)
			}
		})
	}
}
