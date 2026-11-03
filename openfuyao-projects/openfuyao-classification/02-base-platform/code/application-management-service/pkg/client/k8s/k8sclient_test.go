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
	"testing"

	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient_NilKubeConfig(t *testing.T) {
	_, err := NewKubernetesClient(&KubernetesCfg{KubeConfig: nil})
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}

func TestNewKubernetesClient_SetsQPSAndBurst(t *testing.T) {
	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{Host: "https://example"},
		QPS:        123,
		Burst:      456,
	}

	client, err := NewKubernetesClient(cfg)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if cfg.KubeConfig.QPS != 123 {
		t.Fatalf("Expected QPS set to 123, got %v", cfg.KubeConfig.QPS)
	}
	if cfg.KubeConfig.Burst != 456 {
		t.Fatalf("Expected Burst set to 456, got %v", cfg.KubeConfig.Burst)
	}

	if client == nil {
		t.Fatalf("Expected non-nil client")
	}
	if client.ConfigClient() != cfg.KubeConfig {
		t.Fatalf("Expected ConfigClient to return input kubeconfig")
	}
	if client.KubernetesClient() == nil {
		t.Fatalf("Expected KubernetesClient non-nil")
	}
	if client.SnapshotClient() == nil {
		t.Fatalf("Expected SnapshotClient non-nil")
	}
	if client.ApiExtensionsClient() == nil {
		t.Fatalf("Expected ApiExtensionsClient non-nil")
	}
}

