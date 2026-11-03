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
)

// TestNewKubernetesClientNilConfig 空配置返回错误
func TestNewKubernetesClientNilConfig(t *testing.T) {
	_, err := NewKubernetesClient(&KubernetesCfg{})
	if err == nil {
		t.Error("expected error for nil kube config, got nil")
	}
}

// TestNewKubernetesClientValid 正常配置创建client并校验访问器
func TestNewKubernetesClientValid(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{Host: "http://127.0.0.1:6443"},
		QPS:        100,
		Burst:      200,
	}
	client, err := NewKubernetesClient(cfg)
	if err != nil {
		t.Fatalf("NewKubernetesClient failed: %v", err)
	}
	if client.Kubernetes() == nil {
		t.Error("expected non-nil kubernetes client")
	}
	if client.Snapshot() == nil {
		t.Error("expected non-nil snapshot client")
	}
	if client.ApiExtensions() == nil {
		t.Error("expected non-nil apiextensions client")
	}
	if client.Config() == nil {
		t.Error("expected non-nil rest config")
	}
	if client.Config().QPS != 100 || client.Config().Burst != 200 {
		t.Errorf("QPS/Burst not propagated: QPS=%v Burst=%v",
			client.Config().QPS, client.Config().Burst)
	}
}

// TestNewKubernetesCfg 校验默认k8s配置生成
func TestNewKubernetesCfg(t *testing.T) {
	cfg := NewKubernetesCfg()
	if cfg == nil {
		t.Fatal("NewKubernetesCfg returned nil")
	}
	if cfg.QPS != 1e6 || cfg.Burst != 1e6 {
		t.Errorf("unexpected QPS/Burst: %v/%v", cfg.QPS, cfg.Burst)
	}
}

// TestKubernetesCfgValidate 配置校验各分支
func TestKubernetesCfgValidate(t *testing.T) {
	// 文件不存在 + 配置为空 → 两个错误
	cfg := &KubernetesCfg{KubeConfigFile: filepath.Join(t.TempDir(), "not-exist-config")}
	if errs := cfg.Validate(); len(errs) != 2 {
		t.Errorf("expected 2 errors, got %d: %v", len(errs), errs)
	}

	// 仅配置为空 → 一个错误
	cfg = &KubernetesCfg{}
	if errs := cfg.Validate(); len(errs) != 1 {
		t.Errorf("expected 1 error, got %d: %v", len(errs), errs)
	}

	// 文件存在 + 配置非空 → 无错误
	kubeFile := filepath.Join(t.TempDir(), "kube-config")
	if err := os.WriteFile(kubeFile, []byte("fake"), 0600); err != nil {
		t.Fatalf("write temp kubeconfig failed: %v", err)
	}
	cfg = &KubernetesCfg{
		KubeConfigFile: kubeFile,
		KubeConfig:     &rest.Config{Host: "http://127.0.0.1:6443"},
	}
	if errs := cfg.Validate(); len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}
}

// TestGetKubeConfig 非集群环境下获取配置(回退到文件/空)
func TestGetKubeConfig(t *testing.T) {
	config := GetKubeConfig()
	// 非集群且无可用kubeconfig文件时返回nil；存在~/.kube/config时返回非nil。
	// 两种结果均为合法行为，此处仅保证函数不panic且行为确定。
	if config != nil && config.Host == "" {
		t.Errorf("unexpected config with empty host: %+v", config)
	}
}
