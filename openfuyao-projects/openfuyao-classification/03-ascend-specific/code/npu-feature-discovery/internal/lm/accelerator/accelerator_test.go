// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package accelerator

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestNewAcceleratorLabeler(t *testing.T) {
	type args struct {
		node *corev1.Node
	}
	tests := []struct {
		name string
		args args
		want common.Labeler
	}{
		{
			name: "test unknown type",
			args: args{
				&corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							"placeholder": "no accelerator",
						},
					},
				},
			},
			want: common.Labels{AcceleratorLabelKey: "unknown"},
		},
		{
			name: "Ascend A5 d806",
			args: args{node: &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				common.NpuDeviceLabelPrefix + "d806.present": "true",
			}}}},
			want: common.Labels{AcceleratorLabelKey: "huawei-AscendA5"},
		},
		{
			name: "test single accelerator",
			args: args{
				&corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							common.NpuDeviceLabelPrefix + "d100.present": "huawei-Ascend310",
						},
					},
				},
			},
			want: common.Labels{AcceleratorLabelKey: "huawei-Ascend310"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewAcceleratorLabeler(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newAcceleratorLabeler() = %v, want %v", got, tt.want)
			}
		})
	}
}
