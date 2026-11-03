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

// Package config provides configuration management for the server.
package config

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/client/k8s"
	dbruntime "openfuyao-rayservice/pkg/database/runtime"
	"openfuyao-rayservice/pkg/server/runtime"
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

// setupFakeHome points HOME at a temp directory containing a valid fake
// kubeconfig so that k8s.NewKubernetesCfg does not fatal in tests.
func setupFakeHome(t *testing.T) {
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
}

func TestNewRunConfig(t *testing.T) {
	setupFakeHome(t)

	got := NewRunConfig()
	if got == nil {
		t.Fatalf("NewRunConfig() = nil, want non-nil")
	}
	if got.Server == nil || got.DbConfig == nil || got.KubernetesCfg == nil {
		t.Errorf("NewRunConfig() should initialize Server, DbConfig and KubernetesCfg, got %+v", got)
	}
	if got.KubernetesCfg.KubeConfig == nil {
		t.Errorf("NewRunConfig() KubernetesCfg.KubeConfig = nil, want non-nil")
	}
	if got.KubernetesCfg.KubeConfig.Host != "https://127.0.0.1:6443" {
		t.Errorf("NewRunConfig() KubeConfig.Host = %q, want https://127.0.0.1:6443",
			got.KubernetesCfg.KubeConfig.Host)
	}
}

func TestRunConfigValidate(t *testing.T) {
	setupFakeHome(t)

	tests := []struct {
		name       string
		server     *runtime.ServerConfig
		kubernetes *k8s.KubernetesCfg
		wantErrNum int
	}{
		{
			name:       "default config is valid",
			server:     runtime.NewServerConfig(),
			kubernetes: k8s.NewKubernetesCfg(),
			wantErrNum: 0,
		},
		{
			name: "both server ports disabled",
			server: &runtime.ServerConfig{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   0,
			},
			kubernetes: k8s.NewKubernetesCfg(),
			wantErrNum: 1,
		},
		{
			name:   "nil kubernetes kubeconfig",
			server: runtime.NewServerConfig(),
			kubernetes: &k8s.KubernetesCfg{
				KubeConfig: nil,
			},
			wantErrNum: 1,
		},
		{
			name: "invalid server and kubernetes config",
			server: &runtime.ServerConfig{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   0,
			},
			kubernetes: &k8s.KubernetesCfg{
				KubeConfig: nil,
			},
			wantErrNum: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RunConfig{
				Server:        tt.server,
				DbConfig:      dbruntime.NewDbConfig(),
				KubernetesCfg: tt.kubernetes,
			}
			if got := cfg.Validate(); len(got) != tt.wantErrNum {
				t.Errorf("RunConfig.Validate() = %v, want %d errors", got, tt.wantErrNum)
			}
		})
	}
}

func TestRunConfigValidateDefaultNoKubeConfigFile(t *testing.T) {
	// KubernetesCfg built manually: a rest.Config is present, so validation passes
	// even without a kubeconfig file on disk.
	cfg := &RunConfig{
		Server:   runtime.NewServerConfig(),
		DbConfig: dbruntime.NewDbConfig(),
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://127.0.0.1:6443"},
		},
	}
	if got := cfg.Validate(); len(got) != 0 {
		t.Errorf("RunConfig.Validate() = %v, want no errors", got)
	}
}
