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
	"testing"

	"k8s.io/client-go/rest"
)

func TestGetKubeConfigFile(t *testing.T) {
	kubeConfigFile := setupFakeHome(t)

	tests := []struct {
		name string
		want string
	}{
		{
			name: "kubeconfig file exists in fake home",
			want: kubeConfigFile,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getKubeConfigFile(); got != tt.want {
				t.Errorf("getKubeConfigFile() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetKubeConfig(t *testing.T) {
	setupFakeHome(t)

	config := GetKubeConfig()
	if config == nil {
		t.Fatalf("GetKubeConfig() = nil, want non-nil config from fake kubeconfig")
	}
	if config.Host != "https://127.0.0.1:6443" {
		t.Errorf("GetKubeConfig() Host = %q, want https://127.0.0.1:6443", config.Host)
	}
}

func TestNewKubernetesCfg(t *testing.T) {
	kubeConfigFile := setupFakeHome(t)

	got := NewKubernetesCfg()
	if got.KubeConfigFile != kubeConfigFile {
		t.Errorf("NewKubernetesCfg() KubeConfigFile = %q, want %q", got.KubeConfigFile, kubeConfigFile)
	}
	if got.KubeConfig == nil {
		t.Fatalf("NewKubernetesCfg() KubeConfig = nil, want non-nil")
	}
	if got.KubeConfig.Host != "https://127.0.0.1:6443" {
		t.Errorf("NewKubernetesCfg() KubeConfig.Host = %q, want https://127.0.0.1:6443", got.KubeConfig.Host)
	}
	if got.QPS != 1e6 || got.Burst != 1e6 {
		t.Errorf("NewKubernetesCfg() QPS/Burst = %v/%v, want 1e6/1e6", got.QPS, got.Burst)
	}
}

func TestKubernetesCfgValidate(t *testing.T) {
	kubeConfigFile := setupFakeHome(t)

	tests := []struct {
		name       string
		configFile string
		kubeConfig *rest.Config
		wantErrNum int
	}{
		{
			name:       "valid config",
			configFile: kubeConfigFile,
			kubeConfig: GetKubeConfig(),
			wantErrNum: 0,
		},
		{
			name:       "nil kubeconfig",
			configFile: kubeConfigFile,
			kubeConfig: nil,
			wantErrNum: 1,
		},
		{
			name:       "not exist kubeconfig file",
			configFile: "/not/exist/kubeconfig",
			kubeConfig: GetKubeConfig(),
			wantErrNum: 1,
		},
		{
			name:       "not exist file and nil kubeconfig",
			configFile: "/not/exist/kubeconfig",
			kubeConfig: nil,
			wantErrNum: 2,
		},
		{
			name:       "empty file path with nil kubeconfig",
			configFile: "",
			kubeConfig: nil,
			wantErrNum: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &KubernetesCfg{
				KubeConfigFile: tt.configFile,
				KubeConfig:     tt.kubeConfig,
			}
			if got := k.Validate(); len(got) != tt.wantErrNum {
				t.Errorf("KubernetesCfg.Validate() = %v, want %d errors", got, tt.wantErrNum)
			}
		})
	}
}

func TestGetKubeConfigFileNotSet(t *testing.T) {
	// Point HOME at an empty directory without kubeconfig
	emptyHome := t.TempDir()
	origHome := os.Getenv("HOME")
	t.Setenv("HOME", emptyHome)
	if origHome == "" {
		t.Cleanup(func() { _ = os.Unsetenv("HOME") })
	} else {
		t.Cleanup(func() { _ = os.Setenv("HOME", origHome) })
	}

	if got := getKubeConfigFile(); got != "" {
		t.Errorf("getKubeConfigFile() = %q, want empty when no kubeconfig exists", got)
	}
}
