/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package versions

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const validManifest = `apiVersion: bke.bocloud.com/v1
kind: VersionManifest
spec:
  manifestVersion: "1"
  componentVersions:
    kubernetes: "v1.34.3-of.1"
    etcd: "v3.6.7-of.1"
    containerd: "v2.1.1"
    openFuyao: "latest"
  imageTags:
    etcd: "3.6.7-of.1"
    pause: "3.9"
`

func TestLoad(t *testing.T) {
	manifest, err := Load()
	require.NoError(t, err)
	assert.Equal(t, "v1.34.3-of.1", manifest.Spec.ComponentVersions.Kubernetes)
	assert.Equal(t, "3.6.7-of.1", manifest.Spec.ImageTags.Etcd)
	assert.Equal(t, "v1.34.3-of.1", KubernetesVersion())
	assert.Equal(t, "v3.6.7-of.1", EtcdVersion())
	assert.Equal(t, "v2.1.1", ContainerdVersion())
	assert.Equal(t, "latest", OpenFuyaoVersion())
	assert.Equal(t, "3.6.7-of.1", EtcdImageTag())
	assert.Equal(t, "3.9", PauseImageTag())
}

func TestParseRequiredFields(t *testing.T) {
	tests := []struct {
		name string
		old  string
		new  string
		path string
	}{
		{"api version", "apiVersion: bke.bocloud.com/v1", "apiVersion: ''", "apiVersion"},
		{"kind", "kind: VersionManifest", "kind: ''", "kind"},
		{"manifest version", "manifestVersion: \"1\"", "manifestVersion: ''", "spec.manifestVersion"},
		{"kubernetes", "kubernetes: \"v1.34.3-of.1\"", "kubernetes: ''", "spec.componentVersions.kubernetes"},
		{"etcd component", "etcd: \"v3.6.7-of.1\"", "etcd: ''", "spec.componentVersions.etcd"},
		{"containerd", "containerd: \"v2.1.1\"", "containerd: ''", "spec.componentVersions.containerd"},
		{"openFuyao", "openFuyao: \"latest\"", "openFuyao: ''", "spec.componentVersions.openFuyao"},
		{"etcd image", "    etcd: \"3.6.7-of.1\"", "    etcd: ''", "spec.imageTags.etcd"},
		{"pause image", "pause: \"3.9\"", "pause: ''", "spec.imageTags.pause"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := strings.Replace(validManifest, tt.old, tt.new, 1)
			_, err := parse([]byte(data))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.path+" is required")
		})
	}
}

func TestParseRejectsMalformedYAML(t *testing.T) {
	_, err := parse([]byte("spec: ["))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse embedded versions.yaml")
}

func TestValidateSupportedSchema(t *testing.T) {
	manifest, err := parse([]byte(validManifest))
	require.NoError(t, err)

	tests := []struct {
		name   string
		mutate func(*Manifest)
		want   string
	}{
		{"api version", func(m *Manifest) { m.APIVersion = "other/v1" }, "unsupported apiVersion"},
		{"kind", func(m *Manifest) { m.Kind = "Other" }, "unsupported kind"},
		{"manifest version", func(m *Manifest) { m.Spec.ManifestVersion = "2" }, "unsupported spec.manifestVersion"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := manifest
			tt.mutate(&candidate)
			err := Validate(&candidate)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}
