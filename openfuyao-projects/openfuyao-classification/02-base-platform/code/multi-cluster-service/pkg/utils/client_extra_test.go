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

package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestCreateClientExtra(t *testing.T) {
	t.Run("success from kubeconfig file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "kubeconfig")
		content := `apiVersion: v1
kind: Config
current-context: test-context
clusters:
- name: test-cluster
  cluster:
    server: http://127.0.0.1:1
contexts:
- name: test-context
  context:
    cluster: test-cluster
    user: test-user
users:
- name: test-user
  user: {}
`
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatalf("write kubeconfig: %v", err)
		}

		client, cfg, err := CreateClient(path)
		if err != nil {
			t.Fatalf("CreateClient() error = %v", err)
		}
		if client == nil {
			t.Error("client = nil, want non-nil")
		}
		if cfg.CurrentContext != "test-context" {
			t.Errorf("CurrentContext = %q, want test-context", cfg.CurrentContext)
		}
	})

	t.Run("missing kubeconfig file", func(t *testing.T) {
		if _, _, err := CreateClient(filepath.Join(t.TempDir(), "not-exist")); err == nil {
			t.Error("CreateClient() expected error for missing file")
		}
	})
}

func TestCreateKarmadaClusterClientExtra(t *testing.T) {
	// KarmadaConfigPath points outside the test environment -> RestConfig fails.
	if _, _, err := CreateKarmadaClusterClient(); err == nil {
		t.Error("CreateKarmadaClusterClient() expected error without karmada kubeconfig")
	}
}

func TestRetriveKubeconfigClientExtra(t *testing.T) {
	t.Run("success from cluster object", func(t *testing.T) {
		mgr := newTestMgrClient(t)
		clusterObj := newTestMgrCluster("member1")
		k8sAPI := newFakeK8sAPI(nil)
		clusterObj.Spec.KubeConfig.Raw = k8sAPI.kubeconfigMapForm(t)
		if err := mgr.Create(testCtx, clusterObj); err != nil {
			t.Fatalf("seed cluster: %v", err)
		}

		client, err := RetriveKubeconfigClient(mgr, "member1")
		if err != nil {
			t.Fatalf("RetriveKubeconfigClient() error = %v", err)
		}
		if client == nil {
			t.Error("client = nil, want non-nil")
		}
	})

	t.Run("host cluster falls back to in-cluster config", func(t *testing.T) {
		// no in-cluster environment available -> CreateHostClient fails
		if _, err := RetriveKubeconfigClient(newTestMgrClient(t), "host"); err == nil {
			t.Error("RetriveKubeconfigClient() expected error for host without in-cluster config")
		}
	})

	t.Run("cluster object missing", func(t *testing.T) {
		if _, err := RetriveKubeconfigClient(newTestMgrClient(t), "ghost"); err == nil {
			t.Error("RetriveKubeconfigClient() expected error for missing cluster object")
		}
	})

	t.Run("invalid kubeconfig data", func(t *testing.T) {
		mgr := newTestMgrClient(t)
		clusterObj := newTestMgrCluster("member1")
		clusterObj.Spec.KubeConfig.Raw = []byte("{invalid")
		if err := mgr.Create(testCtx, clusterObj); err != nil {
			t.Fatalf("seed cluster: %v", err)
		}

		if _, err := RetriveKubeconfigClient(mgr, "member1"); err == nil {
			t.Error("RetriveKubeconfigClient() expected error for invalid kubeconfig data")
		}
	})
}

func TestHostConfToJsonExtra(t *testing.T) {
	const kubeDir = "/root/.kube"
	const kubePath = kubeDir + "/config"

	backup, hadFile, err := readIfExists(kubePath)
	if err != nil {
		t.Skipf("cannot inspect %s: %v", kubePath, err)
	}

	restore := func() {
		if hadFile {
			_ = os.WriteFile(kubePath, backup, 0o600)
			return
		}
		_ = os.Remove(kubePath)
	}
	// restore the pre-test state of /root/.kube/config even when a subtest
	// fails or the test is aborted
	t.Cleanup(restore)

	validYAML := []byte("apiVersion: v1\nkind: Config\ncurrent-context: ctx\n")
	if err := os.MkdirAll(kubeDir, 0o700); err != nil {
		t.Skipf("cannot create %s: %v", kubeDir, err)
	}
	if err := os.WriteFile(kubePath, validYAML, 0o600); err != nil {
		t.Skipf("cannot write %s: %v", kubePath, err)
	}

	t.Run("success", func(t *testing.T) {
		data, err := HostConfToJson()
		if err != nil {
			t.Fatalf("HostConfToJson() error = %v", err)
		}
		if len(data) == 0 {
			t.Error("HostConfToJson() returned empty data")
		}
	})

	t.Run("invalid yaml", func(t *testing.T) {
		if err := os.WriteFile(kubePath, []byte(":\t:not-yaml"), 0o600); err != nil {
			t.Fatalf("write kubeconfig: %v", err)
		}
		if _, err := HostConfToJson(); err == nil {
			t.Error("HostConfToJson() expected error for invalid yaml")
		}
	})
}

func readIfExists(path string) ([]byte, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("read %s: %w", path, err)
	}
	return data, true, nil
}
