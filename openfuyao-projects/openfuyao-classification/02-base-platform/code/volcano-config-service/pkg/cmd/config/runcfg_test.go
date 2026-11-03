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

package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/client/k8s"
	"volcano-config-service/pkg/server/runtime"
)

func TestNewRunConfig_Type(t *testing.T) {
	cfg := &RunConfig{
		Server: runtime.NewServerConfig(),
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
		},
	}
	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg.Server)
	assert.NotNil(t, cfg.KubernetesCfg)
}

func TestRunConfig_Validate_NoErrors(t *testing.T) {
	cfg := &RunConfig{
		Server: runtime.NewServerConfig(),
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
		},
	}
	errs := cfg.Validate()
	assert.Empty(t, errs)
}

func TestRunConfig_Validate_WithErrors(t *testing.T) {
	cfg := &RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 0,
			SecurePort:   0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: nil,
		},
	}
	errs := cfg.Validate()
	assert.NotEmpty(t, errs)
}

func TestNewRunConfig_WithKubeconfig(t *testing.T) {
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

	cfg := NewRunConfig()
	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg.Server)
	assert.NotNil(t, cfg.KubernetesCfg)
	assert.NotNil(t, cfg.KubernetesCfg.KubeConfig)

	errs := cfg.Validate()
	assert.Empty(t, errs)
}
