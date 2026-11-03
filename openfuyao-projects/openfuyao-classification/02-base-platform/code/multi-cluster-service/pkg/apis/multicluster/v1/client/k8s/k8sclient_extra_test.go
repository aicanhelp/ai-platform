/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"k8s.io/client-go/util/homedir"
)

func TestNewKubernetesClientExtra(t *testing.T) {
	t.Run("nil kubeconfig returns error", func(t *testing.T) {
		if _, err := NewKubernetesClient(&KubernetesCfg{KubeConfig: nil}); err == nil {
			t.Error("NewKubernetesClient() expected error for nil kubeconfig")
		}
	})

	t.Run("accessors return constructed client", func(t *testing.T) {
		cfg := &KubernetesCfg{
			KubeConfig: &rest.Config{Host: "127.0.0.1:1"},
			QPS:        100,
			Burst:      200,
		}
		client, err := NewKubernetesClient(cfg)
		if err != nil {
			t.Fatalf("NewKubernetesClient() error = %v", err)
		}
		if client.Kubernetes() == nil {
			t.Error("Kubernetes() = nil, want non-nil")
		}
		if client.Cfg() == nil || client.Cfg().QPS != 100 || client.Cfg().Burst != 200 {
			t.Errorf("Cfg() = %+v, want QPS/Burst applied", client.Cfg())
		}
	})
}

func TestNewKubernetesCfgExtra(t *testing.T) {
	// GetKubeConfig falls back to $HOME/.kube/config when not in-cluster; the
	// fixture must exist before calling NewKubernetesCfg (else LogFatalf exits).
	if err := ensureHomeKubeconfig(t); err != nil {
		t.Skipf("cannot prepare home kubeconfig: %v", err)
	}

	cfg := NewKubernetesCfg()
	if cfg == nil {
		t.Fatal("NewKubernetesCfg() = nil")
	}
	if cfg.KubeConfig == nil {
		t.Error("KubeConfig = nil, want non-nil")
	}
	if cfg.QPS == 0 || cfg.Burst == 0 {
		t.Errorf("QPS/Burst = %v/%v, want non-zero", cfg.QPS, cfg.Burst)
	}
}

// ensureHomeKubeconfig writes a minimal kubeconfig to $HOME/.kube/config when
// absent, mirroring the environment GetKubeConfig expects. A file created here
// is removed again when the test finishes (t.Cleanup).
func ensureHomeKubeconfig(t *testing.T) error {
	home := homedir.HomeDir()
	if home == "" {
		home = "/root"
	}
	dir := filepath.Join(home, ".kube")
	path := filepath.Join(dir, "config")
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	content := "apiVersion: v1\nkind: Config\ncurrent-context: ctx\nclusters:\n- name: c\n  cluster:\n    server: http://127.0.0.1:1\ncontexts:\n- name: ctx\n  context:\n    cluster: c\n    user: u\nusers:\n- name: u\n  user: {}\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		return err
	}
	// the fixture file was created by this test; remove it again on test exit
	t.Cleanup(func() { _ = os.Remove(path) })
	return nil
}
