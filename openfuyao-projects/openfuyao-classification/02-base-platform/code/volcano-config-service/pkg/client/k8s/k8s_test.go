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

package k8s

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient_NilConfig(t *testing.T) {
	cfg := &KubernetesCfg{}
	client, err := NewKubernetesClient(cfg)
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewKubernetesClient_ValidConfig(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{Host: "https://example.com"},
		QPS:        100,
		Burst:      200,
	}
	client, err := NewKubernetesClient(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.NotNil(t, client.KubernetesClient())
	assert.NotNil(t, client.SnapshotClient())
	assert.NotNil(t, client.ApiExtensionsClient())
	assert.NotNil(t, client.ConfigClient())
	assert.Equal(t, cfg.KubeConfig, client.ConfigClient())
}

func TestNewKubernetesClient_QPSBurstSet(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{Host: "https://example.com"},
		QPS:        50,
		Burst:      100,
	}
	client, err := NewKubernetesClient(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.Equal(t, float32(50), cfg.KubeConfig.QPS)
	assert.Equal(t, 100, cfg.KubeConfig.Burst)
}

func TestKubernetesCfg_Validate_NilConfig(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: nil,
	}
	errs := cfg.Validate()
	assert.NotEmpty(t, errs)
}

func TestKubernetesCfg_Validate_NonExistentFile(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfigFile: "/nonexistent/path/kubeconfig",
		KubeConfig:     &rest.Config{Host: "https://example.com"},
	}
	errs := cfg.Validate()
	assert.NotEmpty(t, errs)
}

func TestKubernetesCfg_Validate_EmptyFile(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{Host: "https://example.com"},
	}
	errs := cfg.Validate()
	assert.Empty(t, errs)
}

func TestGetKubeConfigFile(t *testing.T) {
	result := getKubeConfigFile()
	_ = result
}

func TestGetKubeConfig_FromFile(t *testing.T) {
	t.Setenv("KUBERNETES_SERVICE_HOST", "")
	t.Setenv("KUBERNETES_SERVICE_PORT", "")

	tempDir := t.TempDir()
	kubeDir := filepath.Join(tempDir, ".kube")
	os.MkdirAll(kubeDir, 0755)
	kubeconfigPath := filepath.Join(kubeDir, "config")
	kubeconfigContent := `apiVersion: v1
kind: Config
clusters:
- name: test
  cluster:
    server: https://example.com
contexts:
- name: test
  context:
    cluster: test
    user: test
users:
- name: test
current-context: test`
	os.WriteFile(kubeconfigPath, []byte(kubeconfigContent), 0644)

	t.Setenv("HOME", tempDir)

	config := GetKubeConfig()
	assert.NotNil(t, config)
	assert.Equal(t, "https://example.com", config.Host)
}

func TestNewKubernetesCfg_WithFile(t *testing.T) {
	t.Setenv("KUBERNETES_SERVICE_HOST", "")
	t.Setenv("KUBERNETES_SERVICE_PORT", "")

	tempDir := t.TempDir()
	kubeDir := filepath.Join(tempDir, ".kube")
	os.MkdirAll(kubeDir, 0755)
	kubeconfigPath := filepath.Join(kubeDir, "config")
	kubeconfigContent := `apiVersion: v1
kind: Config
clusters:
- name: test
  cluster:
    server: https://example.com
contexts:
- name: test
  context:
    cluster: test
    user: test
users:
- name: test
current-context: test`
	os.WriteFile(kubeconfigPath, []byte(kubeconfigContent), 0644)

	t.Setenv("HOME", tempDir)

	cfg := NewKubernetesCfg()
	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg.KubeConfig)
	assert.NotEmpty(t, cfg.KubeConfigFile)
	assert.Equal(t, float32(1000000), cfg.QPS)
	assert.Equal(t, 1000000, cfg.Burst)
}
