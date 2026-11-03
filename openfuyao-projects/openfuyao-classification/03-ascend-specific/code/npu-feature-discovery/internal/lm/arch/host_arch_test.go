// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package arch

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestNewHostArchLabeler(t *testing.T) {
	type args struct {
		node *corev1.Node
	}
	tests := []struct {
		name string
		args args
		want common.Labeler
	}{
		{
			name: "unknown arch",
			args: args{
				// no label here so node.Labels["kubernetes.io/arch"] will raise err
				node: &corev1.Node{},
			},
			want: common.Labels{HostArchLabelKey: "unknown"},
		},
		{
			name: "x86",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							"kubernetes.io/arch": "amd64",
						},
					},
				},
			},
			want: common.Labels{HostArchLabelKey: "huawei-x86"},
		},
		{
			name: "arm",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							"kubernetes.io/arch": "arm",
						},
					},
				},
			},
			want: common.Labels{HostArchLabelKey: "arm"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewHostArchLabeler(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newHostArchLabeler() = %v, want %v", got, tt.want)
			}
		})
	}
}
