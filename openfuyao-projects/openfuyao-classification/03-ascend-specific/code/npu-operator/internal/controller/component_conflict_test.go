// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func TestSelectorsDefinitelyDisjoint(t *testing.T) {
	tests := []struct {
		name string
		a    map[string]string
		b    map[string]string
		want bool
	}{
		{
			name: "same key different value is disjoint",
			a:    map[string]string{"npu.openfuyao.com/stack": "mindcluster"},
			b:    map[string]string{"npu.openfuyao.com/stack": "vnpu"},
			want: true,
		},
		{
			name: "different keys are not statically disjoint",
			a:    map[string]string{"npu.openfuyao.com/stack": "vnpu"},
			b:    map[string]string{"huawei.com/vnpu": "ready"},
			want: false,
		},
		{
			name: "empty selector is not statically disjoint",
			a:    map[string]string{},
			b:    map[string]string{"huawei.com/vnpu": "ready"},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, selectorsDefinitelyDisjoint(tt.a, tt.b))
		})
	}
}

func TestFindOverlappingNodes(t *testing.T) {
	ctx := context.Background()
	nodes := []client.Object{
		testNode("mind", map[string]string{"npu.openfuyao.com/stack": "mindcluster"}),
		testNode("vnpu", map[string]string{"npu.openfuyao.com/stack": "vnpu", "huawei.com/vnpu": "ready"}),
		testNode("both", map[string]string{"huawei.com/vnpu": "ready", "npu.openfuyao.com/dra": "ready"}),
	}
	c := fakeNodeClient(t, nodes...)

	tests := []struct {
		name string
		a    map[string]string
		b    map[string]string
		want []string
	}{
		{
			name: "static disjoint skips dynamic list",
			a:    map[string]string{"npu.openfuyao.com/stack": "mindcluster"},
			b:    map[string]string{"npu.openfuyao.com/stack": "vnpu"},
		},
		{
			name: "both empty selectors conflict without listing all nodes",
			a:    map[string]string{},
			b:    map[string]string{},
			want: []string{allNodesSelectorName},
		},
		{
			name: "a empty lists b and conflicts when b selects nodes",
			a:    map[string]string{},
			b:    map[string]string{"huawei.com/vnpu": "ready"},
			want: []string{"both", "vnpu"},
		},
		{
			name: "b empty lists a and conflicts when a selects nodes",
			a:    map[string]string{"npu.openfuyao.com/dra": "ready"},
			b:    map[string]string{},
			want: []string{"both"},
		},
		{
			name: "both non-empty lists more specific selector and filters other selector",
			a:    map[string]string{"huawei.com/vnpu": "ready"},
			b:    map[string]string{"npu.openfuyao.com/dra": "ready"},
			want: []string{"both"},
		},
		{
			name: "no dynamic overlap",
			a:    map[string]string{"npu.openfuyao.com/stack": "vnpu"},
			b:    map[string]string{"npu.openfuyao.com/dra": "ready"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := findOverlappingNodes(ctx, c, tt.a, tt.b)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDetectComponentConflicts(t *testing.T) {
	ctx := context.Background()
	c := fakeNodeClient(t,
		testNode("m0", map[string]string{"huawei.com/vnpu": "ready", "npu.openfuyao.com/dra": "ready"}),
	)

	r := &NPUClusterPolicyReconciler{
		Client: c,
		instance: &v1.NPUClusterPolicy{Spec: v1.NPUClusterPolicySpec{
			DevicePlugin: v1.DevicePluginSpec{
				Managed: true,
				PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
					"npu.openfuyao.com/stack": "mindcluster",
				}},
			},
			VNPU: v1.VNPUSpec{
				Managed: true,
				PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
					"huawei.com/vnpu": "ready",
				}},
				DevicePlugin: v1.VNPUDevicePluginSpec{Managed: true},
			},
			DRA: v1.DRASpec{
				Managed: true,
				KubeletPlugin: v1.DRAComponentSpec{
					Managed: true,
					PlacementSpec: v1.PlacementSpec{NodeSelector: map[string]string{
						"npu.openfuyao.com/dra": "ready",
					}},
				},
			},
		}},
	}

	conflicts, err := r.detectComponentConflicts(ctx)
	require.NoError(t, err)
	require.Len(t, conflicts, 1)
	assert.Equal(t, "vnpu-device-plugin", conflicts[0].Left)
	assert.Equal(t, "dra-kubelet-plugin", conflicts[0].Right)
	assert.Equal(t, []string{"m0"}, conflicts[0].Nodes)
}

func TestDetectComponentConflictsIgnoresDisabledComponents(t *testing.T) {
	ctx := context.Background()
	r := &NPUClusterPolicyReconciler{
		Client: fakeNodeClient(t),
		instance: &v1.NPUClusterPolicy{Spec: v1.NPUClusterPolicySpec{
			DevicePlugin: v1.DevicePluginSpec{Managed: true},
			VNPU: v1.VNPUSpec{
				Managed:      true,
				DevicePlugin: v1.VNPUDevicePluginSpec{Managed: false},
			},
			DRA: v1.DRASpec{
				Managed:       false,
				KubeletPlugin: v1.DRAComponentSpec{Managed: true},
			},
		}},
	}

	conflicts, err := r.detectComponentConflicts(ctx)
	require.NoError(t, err)
	assert.Empty(t, conflicts)
}

func testNode(name string, labels map[string]string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
	}
}

func fakeNodeClient(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}
