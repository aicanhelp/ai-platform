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
)

// writeFakeKubeconfig writes a syntactically valid kubeconfig into a temp
// HOME so that k8s.GetKubeConfig() succeeds on machines without in-cluster
// credentials or ~/.kube/config (e.g. CI runners).
func writeFakeKubeconfig(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if err := os.MkdirAll(filepath.Join(home, ".kube"), 0o755); err != nil {
		t.Fatalf("create .kube dir: %v", err)
	}
	content := `apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: dummy
contexts:
- context:
    cluster: dummy
    user: dummy
  name: dummy
current-context: dummy
users:
- name: dummy
  user:
    token: dummy
`
	path := filepath.Join(home, ".kube", "config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write kubeconfig: %v", err)
	}
}

func TestNewRunConfig(t *testing.T) {
	writeFakeKubeconfig(t)
	cfg := NewRunConfig()
	if cfg == nil {
		t.Fatalf("NewRunConfig() returned nil")
	}
	if cfg.Server == nil {
		t.Fatalf("NewRunConfig() Server is nil")
	}
	if cfg.KubernetesCfg == nil {
		t.Fatalf("NewRunConfig() KubernetesCfg is nil")
	}
}

func TestRunConfigValidate(t *testing.T) {
	writeFakeKubeconfig(t)
	cfg := NewRunConfig()
	_ = cfg.Validate()
}
