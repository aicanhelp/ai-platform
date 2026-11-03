// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package ascend

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestNewAscendPresenceLabeler(t *testing.T) {
	type args struct {
		node *corev1.Node
	}
	tests := []struct {
		name  string
		args  args
		want  common.Labeler
		want1 bool
	}{
		{
			name: "not has npu",
			args: args{
				// no relative labels here
				node: &corev1.Node{},
			},
			want:  common.Labels{},
			want1: false,
		},
		{
			name: "has npu",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							common.NpuDeviceLabelPrefix + "d100.present": common.NpuDeviceLabelValue,
						},
					},
				},
			},
			want:  common.Labels{NPUPresentLabelKey: ""},
			want1: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := NewAscendPresenceLabeler(tt.args.node)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newAscendPresenceLabeler() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("newAscendPresenceLabeler() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
