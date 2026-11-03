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

	"k8s.io/client-go/rest"

	"application-management-service/pkg/client/k8s"
	"application-management-service/pkg/server/runtime"
)

func TestNewRunConfig(t *testing.T) {
	prepareKubeconfigForTest(t)

	cfg := NewRunConfig()
	if cfg == nil {
		t.Fatalf("Expected non-nil config")
	}
	if cfg.Server == nil {
		t.Fatalf("Expected ServerConfig non-nil")
	}
	if cfg.KubernetesCfg == nil {
		t.Fatalf("Expected KubernetesCfg non-nil")
	}
}

func TestRunConfigValidate_NoErrors(t *testing.T) {
	cfg := &RunConfig{
		Server:        &runtime.ServerConfig{InsecurePort: 18080},
		KubernetesCfg: &k8s.KubernetesCfg{KubeConfig: &rest.Config{}},
	}

	errs := cfg.Validate()
	if len(errs) != 0 {
		t.Fatalf("Expected no errors, got %v", errs)
	}
}

func TestRunConfigValidate_HasErrors(t *testing.T) {
	cfg := &RunConfig{
		Server:        &runtime.ServerConfig{},
		KubernetesCfg: &k8s.KubernetesCfg{},
	}

	errs := cfg.Validate()
	if len(errs) == 0 {
		t.Fatalf("Expected errors, got none")
	}
}

// prepareKubeconfigForTest creates a minimal kubeconfig in a temporary home
// directory so that k8s.GetKubeConfig can fall back to the file path instead
// of terminating the process via zlog.Fatalf in environments without a cluster.
func prepareKubeconfigForTest(t *testing.T) {
	t.Helper()

	tmpHome := t.TempDir()
	t.Setenv("USERPROFILE", tmpHome)
	t.Setenv("HOME", tmpHome)

	kubeDir := filepath.Join(tmpHome, ".kube")
	if err := os.MkdirAll(kubeDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}

	// Minimal kubeconfig; BuildConfigFromFlags should be able to parse it.
	kubeconfig := `apiVersion: v1
clusters:
- cluster:
    server: https://example
  name: example
contexts:
- context:
    cluster: example
    user: example
  name: example
current-context: example
users:
- name: example
  user:
    token: dummy
`
	if err := os.WriteFile(filepath.Join(kubeDir, "config"), []byte(kubeconfig), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}
