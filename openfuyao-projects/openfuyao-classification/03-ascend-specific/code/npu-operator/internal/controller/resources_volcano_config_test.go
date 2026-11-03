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
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func TestVolcanoPluginNamingAcrossVersions(t *testing.T) {
	tests := []struct {
		name, tag, configured, expected, arch string
		withoutNode                           bool
	}{
		{"new version without node", "v1.12.0-v26.1.1", "volcano-npu_v7.3.0_linux-aarch64", "volcano-npu_v26.1.1", "", true},
		{"new version with distro suffix", "v1.12.0-v26.1.1-oeneuler", "volcano-npu_v7.3.0_linux-aarch64", "volcano-npu_v26.1.1", "", true},
		{"old version with build suffix", "v1.9.0-v7.3.0-openeuler-arm64", "volcano-npu_v26.1.1", "volcano-npu_v7.3.0_linux-aarch64", "arm64", false},
		{"unrecognized tag retains configured plugin", "latest", "volcano-npu_v7.3.0_linux-aarch64", "volcano-npu_v7.3.0_linux-aarch64", "", true},
		{"ambiguous tag retains configured plugin", "v1.12.0-v26.1.1-build-v27.0.0", "volcano-npu_v26.1.1", "volcano-npu_v26.1.1", "", true},
		{"new default without tag", "", "volcano-npu_v26.1.1", "volcano-npu_v26.1.1", "", true},
		{"future release", "v1.12.0-v26.10.0", "volcano-npu_v26.1.1", "volcano-npu_v26.10.0", "", true},
		{"old release amd64", "v1.9.0-v7.3.0", "volcano-npu_v26.1.1", "volcano-npu_v7.3.0_linux-x86_64", "amd64", false},
		{"old release arm64", "v1.9.0-v7.3.0", "volcano-npu_v26.1.1", "volcano-npu_v7.3.0_linux-aarch64", "arm64", false},
		{"previous patch", "v1.12.0-v26.1.0", "volcano-npu_v26.1.1", "volcano-npu_v26.1.0_linux-aarch64", "arm64", false},
		{"legacy RC", "v1.9.0-7.1.RC1", "volcano-npu_v26.1.1", "volcano-npu_7.1.RC1_linux-x86_64", "amd64", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const prefix = "tiers:\n- plugins:\n    - name: \""
			const suffix = "\" # NPU plugin\n      enablePreemptable: true\n"
			cm, r := newVolcanoSchedulerConfigMapArchTestReconciler(t, prefix+tt.configured+suffix)
			r.instance.Spec.VCScheduler.Image.Tag = tt.tag
			require.NoError(t, r.Update(context.Background(), r.instance))
			node := &corev1.Node{}
			require.NoError(t, r.Get(context.Background(), client.ObjectKey{Name: "node01"}, node))
			if tt.withoutNode {
				require.NoError(t, r.Delete(context.Background(), node))
				t.Setenv("OPERATOR_NODE_NAME", "")
			} else {
				node.Status.NodeInfo.Architecture = tt.arch
				require.NoError(t, r.Status().Update(context.Background(), node))
			}
			require.NoError(t, updateVolcanoSchedulerConfigMap(context.Background(), r, cm))
			assert.Equal(t, prefix+tt.expected+suffix, cm.Data["volcano-scheduler.conf"])
		})
	}
}

func TestVolcanoPluginVersionFromImageTag(t *testing.T) {
	for _, tt := range []struct{ tag, expected string }{
		{"", ""},
		{"v1.12.0-v26.1.1", "v26.1.1"},
		{"v1.12.0-v26.1.1-oeneuler", "v26.1.1"},
		{"v1.12.0-v26.1.1-openeuler24.03-arm64-build20261009", "v26.1.1"},
		{"v26.1.1-openeuler", "v26.1.1"},
		{"custom_openeuler_v26.1.1_arm64", "v26.1.1"},
		{"v1.12.0_26.1.1_custom", "26.1.1"},
		{"v1.9.0-v7.3.0-linux-aarch64", "v7.3.0"},
		{"volcano-7.1.RC1", "7.1.RC1"},
		{"v1.9.0-v7.1.RC1-openeuler", "v7.1.RC1"},
	} {
		t.Run(tt.tag, func(t *testing.T) {
			assert.Equal(t, tt.expected, volcanoPluginVersion(tt.tag))
		})
	}
}

func TestVolcanoPluginVersionRejectsUnrecognizedTags(t *testing.T) {
	for _, tag := range []string{
		"latest", "custom-openeuler24.03", "prefixv26.1.1suffix", "v26.1.1.2",
		"v1.12.0-v26.1.1-build-v27.0.0", "custom-v26.1.1-v27.0.0",
	} {
		t.Run(tag, func(t *testing.T) {
			assert.Empty(t, volcanoPluginVersion(tag))
		})
	}
}

func TestVolcanoPluginVersionSwitch(t *testing.T) {
	cm, r := newVolcanoSchedulerConfigMapArchTestReconciler(t, "tiers:\n- plugins:\n  - name: volcano-npu_v26.1.1")
	for _, step := range []struct{ tag, plugin string }{
		{"v1.12.0-v26.1.1", "volcano-npu_v26.1.1"},
		{"v1.9.0-v7.3.0", "volcano-npu_v7.3.0_linux-x86_64"},
		{"v1.12.0-v26.1.1", "volcano-npu_v26.1.1"},
	} {
		r.instance.Spec.VCScheduler.Image.Tag = step.tag
		require.NoError(t, r.Update(context.Background(), r.instance))
		require.NoError(t, updateVolcanoSchedulerConfigMap(context.Background(), r, cm))
		assert.Equal(t, "tiers:\n- plugins:\n  - name: "+step.plugin, cm.Data["volcano-scheduler.conf"])
	}
}

func TestVolcanoPluginVNPUConfigUnchanged(t *testing.T) {
	const config = "tiers:\n- plugins:\n  - name: volcano-npu_custom"
	cm, r := newVolcanoSchedulerConfigMapArchTestReconciler(t, config)
	r.instance.Spec.Volcano.Flavor = v1.VolcanoFlavorVNPU
	t.Setenv("OPERATOR_NODE_NAME", "")
	require.NoError(t, updateVolcanoSchedulerConfigMap(context.Background(), r, cm))
	assert.Equal(t, config, cm.Data["volcano-scheduler.conf"])
}
