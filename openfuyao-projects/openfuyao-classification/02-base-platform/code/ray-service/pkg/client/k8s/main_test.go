/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package k8s create kubernetes config
package k8s

import (
	"os"
	"path"
	"path/filepath"
	"testing"
)

// fakeKubeConfigYaml is a valid-format kubeconfig used for tests. It is never
// used to connect to a real cluster; clientcmd only parses the file.
const fakeKubeConfigYaml = `apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: fake-cluster
contexts:
- context:
    cluster: fake-cluster
    user: fake-user
  name: fake-context
current-context: fake-context
users:
- name: fake-user
  user:
    token: fake-token
`

// kubeConfigFilePath returns the kubeconfig path inside the given home
// directory, built the same way (path.Join, forward slashes) as
// getKubeConfigFile does.
func kubeConfigFilePath(homeDir string) string {
	return path.Join(homeDir, ".kube/config")
}

// setupFakeHome creates a temporary home directory containing a valid fake
// kubeconfig and points the HOME environment variable at it, so that
// GetKubeConfig can build a rest.Config without a real cluster.
func setupFakeHome(t *testing.T) string {
	t.Helper()
	homeDir := t.TempDir()
	kubeDir := filepath.Join(homeDir, ".kube")
	if err := os.MkdirAll(kubeDir, 0750); err != nil {
		t.Fatalf("failed to create fake kube dir: %v", err)
	}
	kubeConfigFile := filepath.Join(kubeDir, "config")
	if err := os.WriteFile(kubeConfigFile, []byte(fakeKubeConfigYaml), 0600); err != nil {
		t.Fatalf("failed to write fake kubeconfig: %v", err)
	}

	origHome := os.Getenv("HOME")
	t.Setenv("HOME", homeDir)
	if origHome == "" {
		t.Cleanup(func() { _ = os.Unsetenv("HOME") })
	} else {
		t.Cleanup(func() { _ = os.Setenv("HOME", origHome) })
	}
	return kubeConfigFilePath(homeDir)
}
