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

	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient_NilKubeConfig(t *testing.T) {
	cfg := &KubernetesCfg{KubeConfig: nil}
	client, err := NewKubernetesClient(cfg)
	if err == nil {
		t.Fatal("Expected error for nil kubeconfig, got nil")
	}
	if client != nil {
		t.Fatal("Expected nil client for nil kubeconfig")
	}
}

func TestKubernetesClient_Getters(t *testing.T) {
	cfg := &rest.Config{Host: "http://example"}

	k8sClient := fake.NewSimpleClientset()
	var snapshot snapshotclient.Interface = nil
	var apiExt apiextensionsclient.Interface = nil

	c := &kubernetesClient{
		k8s:           k8sClient,
		snapshot:      snapshot,
		apiExtensions: apiExt,
		config:        cfg,
	}

	if c.KubernetesClient() == nil {
		t.Fatal("KubernetesClient() returned nil")
	}
	if c.SnapshotClient() != nil {
		t.Fatal("SnapshotClient() expected nil")
	}
	if c.ApiExtensionsClient() != nil {
		t.Fatal("ApiExtensionsClient() expected nil")
	}
	if c.ConfigClient() == nil {
		t.Fatal("ConfigClient() returned nil")
	}
	if c.ConfigClient().Host != "http://example" {
		t.Fatalf("ConfigClient().Host = %q", c.ConfigClient().Host)
	}
}

func TestValidate(t *testing.T) {
	validFile := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(validFile, []byte("test"), 0o644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	tests := []struct {
		name string
		cfg  *KubernetesCfg
		want int
	}{
		{
			name: "valid config",
			cfg: &KubernetesCfg{
				KubeConfigFile: validFile,
				KubeConfig:     &rest.Config{},
			},
			want: 0,
		},
		{
			name: "missing file and nil kubeconfig",
			cfg: &KubernetesCfg{
				KubeConfigFile: filepath.Join(t.TempDir(), "missing"),
				KubeConfig:     nil,
			},
			want: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := len(tt.cfg.Validate()); got != tt.want {
				t.Fatalf("Validate() got %d errors, want %d", got, tt.want)
			}
		})
	}
}

func TestGetKubeConfigFile(t *testing.T) {
	if got := getKubeConfigFile(); got == "" {
		t.Log("getKubeConfigFile returned empty path, which is acceptable in CI")
	}
}
