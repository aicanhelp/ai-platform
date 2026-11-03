// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package runtime

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

type argsNewRuntimeLabeler struct {
	node *corev1.Node
}

func GenArgsNewRuntimeLabeler(containerRuntimeVersion string) argsNewRuntimeLabeler {
	return argsNewRuntimeLabeler{
		node: &corev1.Node{
			Status: corev1.NodeStatus{
				NodeInfo: corev1.NodeSystemInfo{
					ContainerRuntimeVersion: containerRuntimeVersion,
				},
			},
		},
	}
}
func TestNewRuntimeLabeler(t *testing.T) {
	tests := []struct {
		name    string
		args    argsNewRuntimeLabeler
		want    common.Labeler
		wantErr bool
	}{
		{
			name:    "parse error",
			args:    GenArgsNewRuntimeLabeler("://wrong/url"),
			want:    common.Labels{},
			wantErr: true,
		},
		{
			name:    "unsupported runtime",
			args:    GenArgsNewRuntimeLabeler("unsupported://1.0.0"),
			want:    common.Labels{},
			wantErr: true,
		},
		{
			name:    "positive path",
			args:    GenArgsNewRuntimeLabeler("containerd://2.0.4"),
			want:    common.Labels{RuntimeLabelKey: "containerd"},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewRuntimeLabeler(tt.args.node)
			if (err != nil) != tt.wantErr {
				t.Errorf("newRuntimeLabeler() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newRuntimeLabeler() got = %v, want %v", got, tt.want)
			}
		})
	}
}
