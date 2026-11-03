// Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
package controller

import (
	"context"
	"github.com/agiledragon/gomonkey/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	v1 "openfuyao.com/npu-operator/api/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const testStr = "test"

func getLabelTestReonciler() *NPUClusterPolicyReconciler {
	reconciler := &NPUClusterPolicyReconciler{
		instance: &v1.NPUClusterPolicy{
			Spec: v1.NPUClusterPolicySpec{
				NodeD: v1.NodeDSpec{
					Managed: true,
				},
			},
		},
	}
	return reconciler
}

func newNodeWithLabels(labels map[string]string) *corev1.Node {
	return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: labels}}
}

func Test_labelNPUNodes(t *testing.T) {

	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{
				//ContainerRuntimeVersion: "containerd://1.7.2",
				ContainerRuntimeVersion: "containerd",
			},
		},
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node2",
			Labels: map[string]string{
				nfdLabelPrefix: "true",
			},
		},
	}

	// 使用 fake client，并包含预置的 Node 对象
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(node1, node2). // 将 Node 对象添加到 fake client 中
		Build()

	reconciler := &NPUClusterPolicyReconciler{
		Client: fakeClient,
		instance: &v1.NPUClusterPolicy{
			Spec: v1.NPUClusterPolicySpec{
				NodeD: v1.NodeDSpec{
					Managed: false,
				},
			},
		},
	}
	reconciler.labelNPUNodes(context.TODO())
}

func Test_addAnnotation(t *testing.T) {
	tests := []struct {
		name        string
		anno        map[string]string
		key         string
		value       string
		expectedRes bool
	}{
		{
			name:        "Test_addAnnotation1",
			anno:        nil,
			expectedRes: true,
		},
		{
			name: "Test_addAnnotation2",
			anno: map[string]string{
				testStr: testStr,
			},
			key:         testStr,
			value:       testStr,
			expectedRes: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := addAnnotation(tt.anno, tt.key, tt.value)
			assert.Equal(t, res, tt.expectedRes)
		})
	}
}

func Test_hasNPUDeviceLabels(t *testing.T) {
	type args struct {
		labels map[string]string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "hasNPUDeviceLabels",
			args: args{
				labels: map[string]string{
					npuDeviceLabelPrefix + "d100.present": "true",
				},
			},
			want: true,
		},
		{
			name: "A5 d806 is present",
			args: args{labels: map[string]string{
				"feature.node.kubernetes.io/pci-1200_19e5_d806.present": "true",
			}},
			want: true,
		},
		{
			name: "A5 d806 is not present",
			args: args{labels: map[string]string{
				"feature.node.kubernetes.io/pci-1200_19e5_d806.present": "false",
			}},
			want: false,
		},
		{
			name: "don't have NPU device labels",
			args: args{},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, hasNPUDeviceLabels(tt.args.labels), "hasNPUDeviceLabels(%v)", tt.args.labels)
		})
	}
}

func Test_hasNPUPresentLabel(t *testing.T) {
	type args struct {
		labels map[string]string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "has NPUPresentLabel",
			args: args{
				labels: map[string]string{
					npuPresentLabelKey: "true",
				},
			},
			want: true,
		},
		{
			name: "don't have NPUPresentLabel",
			args: args{},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, hasNPUPresentLabel(tt.args.labels), "hasNPUPresentLabel(%v)", tt.args.labels)
		})
	}
}

func Test_removeLabelIfExists(t *testing.T) {
	type args struct {
		labels   map[string]string
		labelKey string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "remove label",
			args: args{
				labels: map[string]string{
					testStr: testStr,
				},
				labelKey: testStr,
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, removeLabelIfExists(tt.args.labels, tt.args.labelKey), "removeLabelIfExists(%v, %v)", tt.args.labels, tt.args.labelKey)
		})
	}
}

func TestNPUClusterPolicyReconciler_updateNodeDEnableLabel(t *testing.T) {

	type args struct {
		node    *corev1.Node
		managed bool
	}
	tests := []struct {
		name       string
		args       args
		want       bool
		wantLabels map[string]string
	}{
		{
			name: "add label when noded is managed",
			args: args{managed: true, node: newNodeWithLabels(map[string]string{testStr: testStr})},
			want: true,
			wantLabels: map[string]string{
				testStr:             testStr,
				nodeDEnableLabelKey: nodeDEnableLabelValue,
			},
		},
		{
			name: "keep existing label when noded is managed",
			args: args{managed: true, node: newNodeWithLabels(map[string]string{nodeDEnableLabelKey: nodeDEnableLabelValue})},
			want: false,
			wantLabels: map[string]string{
				nodeDEnableLabelKey: nodeDEnableLabelValue,
			},
		},
		{
			name:       "remove label key when noded is unmanaged",
			args:       args{managed: false, node: newNodeWithLabels(map[string]string{nodeDEnableLabelKey: nodeDEnableLabelValue})},
			want:       true,
			wantLabels: map[string]string{},
		},
		{
			name: "initialize labels when noded is managed",
			args: args{managed: true, node: &corev1.Node{}},
			want: true,
			wantLabels: map[string]string{
				nodeDEnableLabelKey: nodeDEnableLabelValue,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := getLabelTestReonciler()
			r.instance.Spec.NodeD.Managed = tt.args.managed
			assert.Equalf(t, tt.want, r.updateNodeDEnableLabel(tt.args.node), "updateNodeDEnableLabel(%v)", tt.args.node)
			assert.Equal(t, tt.wantLabels, tt.args.node.Labels)
		})
	}
}

func TestNPUClusterPolicyReconciler_updateVNPUReadyLabel(t *testing.T) {
	tests := []struct {
		name      string
		managed   bool
		npuNode   bool
		labels    map[string]string
		want      bool
		wantLabel bool
	}{
		{
			name:      "add label when vnpu is managed on npu node",
			managed:   true,
			npuNode:   true,
			labels:    map[string]string{},
			want:      true,
			wantLabel: true,
		},
		{
			name:      "keep label when vnpu is managed on npu node",
			managed:   true,
			npuNode:   true,
			labels:    map[string]string{vnpuReadyLabelKey: vnpuReadyLabelValue},
			want:      false,
			wantLabel: true,
		},
		{
			name:      "remove label when vnpu is unmanaged",
			managed:   false,
			npuNode:   true,
			labels:    map[string]string{vnpuReadyLabelKey: vnpuReadyLabelValue},
			want:      true,
			wantLabel: false,
		},
		{
			name:      "remove label from non npu node",
			managed:   true,
			npuNode:   false,
			labels:    map[string]string{vnpuReadyLabelKey: vnpuReadyLabelValue},
			want:      true,
			wantLabel: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := getLabelTestReonciler()
			r.instance.Spec.VNPU.Managed = tt.managed
			node := newNodeWithLabels(tt.labels)
			assert.Equal(t, tt.want, r.updateVNPUReadyLabel(node, tt.npuNode))
			_, exists := node.Labels[vnpuReadyLabelKey]
			assert.Equal(t, tt.wantLabel, exists)
			if exists {
				assert.Equal(t, vnpuReadyLabelValue, node.Labels[vnpuReadyLabelKey])
			}
		})
	}
}

func Test_addAnnotation1(t *testing.T) {
	type args struct {
		anno  map[string]string
		key   string
		value string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "add annotation",
			args: args{
				key:   testStr,
				value: testStr,
			},
			want: true,
		},
		{
			name: "no add annotation",
			args: args{
				anno: map[string]string{
					testStr: testStr,
				},
				key:   testStr,
				value: testStr,
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, addAnnotation(tt.args.anno, tt.args.key, tt.args.value), "addAnnotation(%v, %v, %v)", tt.args.anno, tt.args.key, tt.args.value)
		})
	}
}

func TestNPUClusterPolicyReconciler_removeRuntimeAnnotations(t *testing.T) {
	type args struct {
		annotations map[string]string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "remove annotation",
			args: args{
				annotations: map[string]string{
					runtimeVersionAnnotationKey: testStr,
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := getLabelTestReonciler()
			assert.Equalf(t, tt.want, r.removeRuntimeAnnotations(tt.args.annotations), "removeRuntimeAnnotations(%v)", tt.args.annotations)
		})
	}
}

func TestParseRuntime(t *testing.T) {
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{
				ContainerRuntimeVersion: "containerd://1.7.2",
			},
		},
	}
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(node1).
		Build()
	r := getLabelTestReonciler()
	r.Client = fakeClient
	r.parseRuntimeInfo(context.TODO(), node1)

}

func TestNPUClusterPolicyReconciler_checkRuntime(t *testing.T) {
	type args struct {
		ctx  context.Context
		node *corev1.Node
	}
	tests := []struct {
		name            string
		args            args
		want            bool
		want1           bool
		wantRuntimeInfo *runtimeInfo
		wantErr         assert.ErrorAssertionFunc
	}{
		{
			name: "checkRuntime",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
					},
					Status: corev1.NodeStatus{
						NodeInfo: corev1.NodeSystemInfo{
							ContainerRuntimeVersion: "containerd://1.7.2",
						},
					},
				},
			},
			want:    false,
			want1:   false,
			wantErr: nil,
		},
		{
			name: "add runtime annotation",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
						Annotations: map[string]string{
							runtimeVersionAnnotationKey: testStr,
						},
					},
				},
			},
			want:            true,
			want1:           true,
			wantErr:         nil,
			wantRuntimeInfo: &runtimeInfo{name: "containerd"},
		},
		{
			name: "remove runtime annotation",
			args: args{
				node: &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
						Annotations: map[string]string{
							runtimeVersionAnnotationKey: testStr,
						},
					},
				},
			},
			want:  false,
			want1: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := getLabelTestReonciler()
			r.nodeRuntimeEndpoints = map[string]string{}
			r.runtimes = map[string]struct{}{}
			g := gomonkey.ApplyPrivateMethod(r, "getRuntime", func(_ *NPUClusterPolicyReconciler, _ context.Context, _ *corev1.Node) (runtime *runtimeInfo, err error) {
				return tt.wantRuntimeInfo, nil
			})
			defer g.Reset()
			got, got1, err := r.checkRuntime(tt.args.ctx, tt.args.node)
			assert.NoError(t, err)
			assert.Equalf(t, tt.want, got, "checkRuntime(%v, %v)", tt.args.ctx, tt.args.node)
			assert.Equalf(t, tt.want1, got1, "checkRuntime(%v, %v)", tt.args.ctx, tt.args.node)
		})
	}
}
