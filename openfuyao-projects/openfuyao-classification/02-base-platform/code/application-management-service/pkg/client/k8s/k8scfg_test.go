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

	"k8s.io/client-go/rest"
)

func TestNewKubernetesCfg_Defaults(t *testing.T) {
	prepareKubeconfigForTest(t)
	cfg := NewKubernetesCfg()
	if cfg == nil {
		t.Fatalf("Expected non-nil cfg")
	}
	if cfg.QPS != 1e6 {
		t.Fatalf("Expected QPS=1e6, got %v", cfg.QPS)
	}
	if cfg.Burst != 1e6 {
		t.Fatalf("Expected Burst=1e6, got %v", cfg.Burst)
	}
}

func TestKubernetesCfg_Validate(t *testing.T) {
	cfg := &KubernetesCfg{KubeConfig: &rest.Config{}}
	if errs := cfg.Validate(); len(errs) != 0 {
		t.Fatalf("Expected no errors, got %v", errs)
	}
}

func TestKubernetesCfg_Validate_FileNotExist(t *testing.T) {
	cfg := &KubernetesCfg{KubeConfigFile: "/path/does/not/exist", KubeConfig: &rest.Config{}}
	if errs := cfg.Validate(); len(errs) == 0 {
		t.Fatalf("Expected errors, got none")
	}
}

func TestKubernetesCfg_Validate_NilConfig(t *testing.T) {
	cfg := &KubernetesCfg{KubeConfigFile: ""}
	errs := cfg.Validate()
	if len(errs) != 1 {
		t.Fatalf("Expected 1 error, got %v", errs)
	}
}

func TestGetKubeConfig_OutOfCluster_UseFile(t *testing.T) {
	prepareKubeconfigForTest(t)

	cfg := GetKubeConfig()
	if cfg == nil {
		t.Fatalf("Expected non-nil rest.Config")
	}
	if cfg.Host != "https://example" {
		t.Fatalf("Expected Host=https://example, got %s", cfg.Host)
	}
}

func prepareKubeconfigForTest(t *testing.T) {
	t.Helper()

	tmpHome := t.TempDir()
	userProfileOrig := os.Getenv("USERPROFILE")
	homeOrig := os.Getenv("HOME")

	t.Cleanup(func() {
		_ = os.Setenv("USERPROFILE", userProfileOrig)
		_ = os.Setenv("HOME", homeOrig)
		_ = os.Unsetenv("KUBERNETES_SERVICE_HOST")
		_ = os.Unsetenv("KUBERNETES_SERVICE_PORT")
	})

	_ = os.Setenv("USERPROFILE", tmpHome)
	_ = os.Setenv("HOME", tmpHome)
	_ = os.Unsetenv("KUBERNETES_SERVICE_HOST")
	_ = os.Unsetenv("KUBERNETES_SERVICE_PORT")

	kubeDir := filepath.Join(tmpHome, ".kube")
	if err := os.MkdirAll(kubeDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	kubeconfigPath := filepath.Join(kubeDir, "config")

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
	if err := os.WriteFile(kubeconfigPath, []byte(kubeconfig), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}
