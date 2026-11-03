// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//
//	http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package lm

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/urfave/cli/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	"openfuyao.com/npu-feature-discovery/internal/lm/ascend"
	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/lm/runtime"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

func TestNewAscendLabelerCannotGetNode(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return nil, nil, fmt.Errorf("failed to get Kubernetes client")
	})
	defer patch.Reset()
	type args struct {
		c *cli.Context
	}
	tests := []struct {
		name                 string
		argsCannotGetNode    args
		wantCannotGetNode    common.Labeler
		wantErrCannotGetNode bool
	}{
		{
			name: "get node failed",
			argsCannotGetNode: args{
				c: nil,
			},
			wantCannotGetNode:    nil,
			wantErrCannotGetNode: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewAscendLabeler(tt.argsCannotGetNode.c)
			if (err != nil) != tt.wantErrCannotGetNode {
				t.Errorf("newAscendLabeler() error = %v, wantErr %v", err, tt.wantErrCannotGetNode)
				return
			}
			if !reflect.DeepEqual(got, tt.wantCannotGetNode) {
				t.Errorf("newAscendLabeler() = %v, want %v", got, tt.wantCannotGetNode)
			}
		})
	}
}

func TestNewAscendLabelerCannotGetRuntimeLabeler(t *testing.T) {
	node := &corev1.Node{}
	patch0 := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	patch1 := gomonkey.ApplyFunc(runtime.NewRuntimeLabeler, func(node *corev1.Node) (common.Labeler, error) {
		return nil, fmt.Errorf("unsupported CRI runtime")
	})
	defer patch0.Reset()
	defer patch1.Reset()
	type args struct {
		c *cli.Context
	}
	tests := []struct {
		name                           string
		argsCannotGetRuntimeLabeler    args
		wantCannotGetRuntimeLabeler    common.Labeler
		wantErrCannotGetRuntimeLabeler bool
	}{
		{
			name: "get runtime failed",
			argsCannotGetRuntimeLabeler: args{
				c: nil,
			},
			wantCannotGetRuntimeLabeler:    nil,
			wantErrCannotGetRuntimeLabeler: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewAscendLabeler(tt.argsCannotGetRuntimeLabeler.c)
			if (err != nil) != tt.wantErrCannotGetRuntimeLabeler {
				t.Errorf("newAscendLabeler() error = %v, wantErr %v", err, tt.wantErrCannotGetRuntimeLabeler)
				return
			}
			if !reflect.DeepEqual(got, tt.wantCannotGetRuntimeLabeler) {
				t.Errorf("newAscendLabeler() = %v, want %v", got, tt.wantCannotGetRuntimeLabeler)
			}
		})
	}
}

func TestNewAscendLabelerSuccessNoAscend(t *testing.T) {
	node := &corev1.Node{}
	patchGetNode := gomonkey.ApplyFunc(utils.GetNode, func(*cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	patchRuntime := gomonkey.ApplyFunc(runtime.NewRuntimeLabeler, func(*corev1.Node) (common.Labeler, error) {
		return common.Labels{"runtime": "containerd"}, nil
	})
	patchAscend := gomonkey.ApplyFunc(ascend.NewAscendPresenceLabeler, func(*corev1.Node) (common.Labeler, bool) {
		return common.Labels{}, false
	})
	defer patchGetNode.Reset()
	defer patchRuntime.Reset()
	defer patchAscend.Reset()

	type args struct{ c *cli.Context }
	tests := []struct {
		name             string
		args             args
		wantErr          bool
		wantLabelsSubset map[string]string
	}{
		{
			name:             "success no ascend returns merged runtime and selector",
			args:             args{c: nil},
			wantErr:          false,
			wantLabelsSubset: map[string]string{"runtime": "containerd"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewAscendLabeler(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewAscendLabeler() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if got != nil {
					t.Errorf("NewAscendLabeler() = %v, want nil on error", got)
				}
				return
			}
			if got == nil {
				t.Fatal("NewAscendLabeler() got nil labeler")
			}
			for k, v := range tt.wantLabelsSubset {
				if got.Labels()[k] != v {
					t.Errorf("NewAscendLabeler() labels[%q] = %q, want %q", k, got.Labels()[k], v)
				}
			}
		})
	}
}

func TestNewAscendLabelerSuccessAscendNoDevicePlugin(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			Capacity: corev1.ResourceList{"cpu": resource.MustParse("4")},
		},
	}
	patchGetNode := gomonkey.ApplyFunc(utils.GetNode, func(*cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	patchRuntime := gomonkey.ApplyFunc(runtime.NewRuntimeLabeler, func(*corev1.Node) (common.Labeler, error) {
		return common.Labels{"runtime": "containerd"}, nil
	})
	patchAscend := gomonkey.ApplyFunc(ascend.NewAscendPresenceLabeler, func(*corev1.Node) (common.Labeler, bool) {
		return common.Labels{ascend.NPUPresentLabelKey: ""}, true
	})
	defer patchGetNode.Reset()
	defer patchRuntime.Reset()
	defer patchAscend.Reset()

	type args struct{ c *cli.Context }
	tests := []struct {
		name             string
		args             args
		wantErr          bool
		wantLabelsSubset map[string]string
	}{
		{
			name:    "success ascend present but no device plugin",
			args:    args{c: nil},
			wantErr: false,
			wantLabelsSubset: map[string]string{
				"runtime":                 "containerd",
				ascend.NPUPresentLabelKey: "",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewAscendLabeler(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewAscendLabeler() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if got != nil {
					t.Errorf("NewAscendLabeler() = %v, want nil on error", got)
				}
				return
			}
			if got == nil {
				t.Fatal("NewAscendLabeler() got nil labeler")
			}
			for k, v := range tt.wantLabelsSubset {
				if got.Labels()[k] != v {
					t.Errorf("NewAscendLabeler() labels[%q] = %q, want %q", k, got.Labels()[k], v)
				}
			}
		})
	}
}

func TestNewAscendLabelerSuccessAscendWithDevicePlugin(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			Capacity: corev1.ResourceList{
				"huawei.com/ascend-910": resource.MustParse("1"),
			},
		},
	}
	patchGetNode := gomonkey.ApplyFunc(utils.GetNode, func(*cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	patchRuntime := gomonkey.ApplyFunc(runtime.NewRuntimeLabeler, func(*corev1.Node) (common.Labeler, error) {
		return common.Labels{"runtime": "containerd"}, nil
	})
	patchAscend := gomonkey.ApplyFunc(ascend.NewAscendPresenceLabeler, func(*corev1.Node) (common.Labeler, bool) {
		return common.Labels{ascend.NPUPresentLabelKey: ""}, true
	})
	defer patchGetNode.Reset()
	defer patchRuntime.Reset()
	defer patchAscend.Reset()

	type args struct{ c *cli.Context }
	tests := []struct {
		name             string
		args             args
		wantErr          bool
		wantLabelsSubset map[string]string
	}{
		{
			name:    "success ascend with device plugin returns merged labeler",
			args:    args{c: nil},
			wantErr: false,
			wantLabelsSubset: map[string]string{
				"runtime":                 "containerd",
				ascend.NPUPresentLabelKey: "",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewAscendLabeler(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewAscendLabeler() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if got != nil {
					t.Errorf("NewAscendLabeler() = %v, want nil on error", got)
				}
				return
			}
			if got == nil {
				t.Fatal("NewAscendLabeler() got nil labeler")
			}
			for k, v := range tt.wantLabelsSubset {
				if got.Labels()[k] != v {
					t.Errorf("NewAscendLabeler() labels[%q] = %q, want %q", k, got.Labels()[k], v)
				}
			}
		})
	}
}

func TestCheckDevicePlugin(t *testing.T) {
	tests := []struct {
		name string
		node *corev1.Node
		want bool
	}{
		{
			name: "device plugin resource with quantity > 0 returns true",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"huawei.com/ascend-910": resource.MustParse("1"),
					},
				},
			},
			want: true,
		},
		{
			name: "device plugin resource with quantity = 0 returns false",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"huawei.com/ascend-910": resource.MustParse("0"),
					},
				},
			},
			want: false,
		},
		{
			name: "no device plugin resources returns false",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"cpu":    resource.MustParse("4"),
						"memory": resource.MustParse("8Gi"),
					},
				},
			},
			want: false,
		},
		{
			name: "empty Capacity returns false",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{},
				},
			},
			want: false,
		},
		{
			name: "multiple device plugin resources, one with quantity > 0 returns true",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"huawei.com/ascend-910": resource.MustParse("0"),
						"huawei.com/ascend-310": resource.MustParse("1"),
					},
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := checkDevicePlugin(tt.node); got != tt.want {
				t.Errorf("checkDevicePlugin() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendLabelerSuccess(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	existingLabeler := common.Labels{"existing": "label"}
	newLabeler := common.Labels{"new": "label"}

	labelerFunc := func(n *corev1.Node) (common.Labeler, error) {
		if n != node {
			t.Errorf("appendLabeler passed node %v, want %v", n, node)
		}
		return newLabeler, nil
	}

	initial := common.Labelers{existingLabeler}
	got := appendLabeler(initial, labelerFunc, node)

	if len(got) != 2 {
		t.Fatalf("appendLabeler() len = %d, want 2", len(got))
	}
	if !reflect.DeepEqual(got[0].Labels(), existingLabeler.Labels()) {
		t.Errorf("appendLabeler() first element changed")
	}
	if !reflect.DeepEqual(got[1].Labels(), newLabeler.Labels()) {
		t.Errorf("appendLabeler() second element = %v, want %v", got[1].Labels(), newLabeler.Labels())
	}
}

func TestAppendLabelerFuncReturnsError(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	existingLabeler := common.Labels{"existing": "label"}

	labelerFunc := func(*corev1.Node) (common.Labeler, error) {
		return common.Labels{}, fmt.Errorf("labeler creation failed")
	}

	initial := common.Labelers{existingLabeler}
	got := appendLabeler(initial, labelerFunc, node)

	// On error, labelers should be unchanged
	if len(got) != 1 {
		t.Fatalf("appendLabeler() len = %d, want 1 (unchanged)", len(got))
	}
	if !reflect.DeepEqual(got[0].Labels(), existingLabeler.Labels()) {
		t.Errorf("appendLabeler() element changed on error")
	}
}
