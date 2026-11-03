// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package lm

import (
	"context"
	"fmt"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	"openfuyao.com/npu-feature-discovery/internal/lm/ascend"
	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/lm/runtime"
	"openfuyao.com/npu-feature-discovery/internal/lm/selector"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

func TestA5NodeLabelLifecycle(t *testing.T) {
	ctx := context.Background()
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "a5-node", Labels: map[string]string{
			"node-role.kubernetes.io/control-plane": "",
			"user-label":                            "preserved",
		}},
		Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{ContainerRuntimeVersion: "containerd://1.7.2"}},
	}
	client := fake.NewSimpleClientset(node)
	patch := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		n, err := client.CoreV1().Nodes().Get(c.Context, node.Name, metav1.GetOptions{})
		return client, n, err
	})
	defer patch.Reset()
	cliContext := &cli.Context{Context: ctx}
	const pciLabel = "feature.node.kubernetes.io/pci-1200_19e5_d806.present"
	for _, value := range []string{"true", "false", "true", "", "true"} {
		n, err := client.CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
		require.NoError(t, err)
		if value == "" {
			delete(n.Labels, pciLabel)
		} else {
			n.Labels[pciLabel] = value
		}
		_, err = client.CoreV1().Nodes().Update(ctx, n, metav1.UpdateOptions{})
		require.NoError(t, err)
		labeler, err := NewAscendLabeler(cliContext)
		require.NoError(t, err)
		require.NoError(t, Output(cliContext, labeler.Labels()))
		updated, err := client.CoreV1().Nodes().Get(ctx, node.Name, metav1.GetOptions{})
		require.NoError(t, err)
		if value == "true" {
			require.Contains(t, updated.Labels, ascend.NPUPresentLabelKey)
			require.Empty(t, updated.Labels[ascend.NPUPresentLabelKey])
			require.Equal(t, "dls-worker-node", updated.Labels[selector.WorkerLabelKey])
		} else {
			require.NotContains(t, updated.Labels, ascend.NPUPresentLabelKey)
			require.NotContains(t, updated.Labels, selector.WorkerLabelKey)
		}
		require.Equal(t, "dls-master-node", updated.Labels[selector.MasterLabelKey])
		require.Equal(t, "preserved", updated.Labels["user-label"])
	}
}

func TestLabelsOutputCannotGetNode(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return nil, nil, fmt.Errorf("failed to get Kubernetes client")
	})
	defer patch.Reset()
	type args struct {
		c *cli.Context
	}
	tests := []struct {
		name              string
		labels            common.Labels
		argsCannotGetNode args
		wantErr           bool
	}{
		{
			name:   "getNode func failed",
			labels: common.Labels{"key0": "value0", "key1": "value1", "key2": "value2"},
			argsCannotGetNode: args{
				c: nil,
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Output(tt.argsCannotGetNode.c, tt.labels); (err != nil) != tt.wantErr {
				t.Errorf("Output() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
func TestLabelsOutputNoUpdate(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			Capacity: corev1.ResourceList{
				"huawei.com/ascend-910": resource.MustParse("1"),
			},
		},
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				ascend.NPUPresentLabelKey: "1",
				runtime.RuntimeLabelKey:   "1",
			},
		},
	}
	patch := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	defer patch.Reset()
	type args struct {
		ctxOutputNoUpdate *cli.Context
	}
	tests := []struct {
		name               string
		labels             common.Labels
		argsOutputNoUpdate args
		wantErr            bool
	}{
		{
			name: "update always equals to false",
			labels: map[string]string{
				ascend.NPUPresentLabelKey: "1",
				runtime.RuntimeLabelKey:   "1",
			},
			argsOutputNoUpdate: args{
				ctxOutputNoUpdate: nil,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Output(tt.argsOutputNoUpdate.ctxOutputNoUpdate, tt.labels); (err != nil) != tt.wantErr {
				t.Errorf("Output() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLabelsOutputNoAccelerator(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			Capacity: corev1.ResourceList{
				"someResource": resource.MustParse("1"),
			},
		},
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"placeholder": "no accelerator",
			},
		},
	}
	patch := gomonkey.ApplyFunc(utils.GetNode, func(c *cli.Context) (kubernetes.Interface, *corev1.Node, error) {
		return fake.NewSimpleClientset(node), node, nil
	})
	defer patch.Reset()
	type args struct {
		ctxOutputNoAccelerator *cli.Context
	}
	tests := []struct {
		name                    string
		labels                  common.Labels
		argsOutputNoAccelerator args
		wantErr                 bool
	}{
		{
			name:   "no accelerator on node",
			labels: map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
			argsOutputNoAccelerator: args{
				ctxOutputNoAccelerator: cli.NewContext(cli.NewApp(), nil, nil),
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Output(tt.argsOutputNoAccelerator.ctxOutputNoAccelerator, tt.labels); (err != nil) != tt.wantErr {
				t.Errorf("Output() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestAddLabel(t *testing.T) {
	type args struct {
		labels map[string]string
		key    string
		value  string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "key already exists and values is the same in labels",
			args: args{
				labels: map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				key:    "key0",
				value:  "value0",
			},
			want: false,
		},
		{
			// after test labels should be {"key0": "value3", "key1": "value1", "key2": "value2"}
			name: "key already exists but value is different",
			args: args{
				labels: map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				key:    "key0",
				value:  "value3",
			},
			want: true,
		},
		{
			name: "key not exists",
			args: args{
				labels: map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				key:    "key3",
				value:  "value3",
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := addLabel(tt.args.labels, tt.args.key, tt.args.value); got != tt.want {
				t.Errorf("addLabel() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoveLabels(t *testing.T) {
	type args struct {
		labels          map[string]string
		removeLabelKeys []string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "all removeLabelKeys are in labels",
			args: args{
				labels:          map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				removeLabelKeys: []string{"key0", "key1"},
			},
			want: true,
		},
		{
			name: "part of removeLabelKeys are not in labels",
			args: args{
				labels:          map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				removeLabelKeys: []string{"key2", "key3", "key4"},
			},
			want: true,
		},
		{
			name: "none of removeLabelKeys are in labels",
			args: args{
				labels:          map[string]string{"key0": "value0", "key1": "value1", "key2": "value2"},
				removeLabelKeys: []string{"key3", "key4"},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := removeLabels(tt.args.labels, tt.args.removeLabelKeys); got != tt.want {
				t.Errorf("removeLabels() = %v, want %v", got, tt.want)
			}
		})
	}
}
