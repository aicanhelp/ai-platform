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
	"testing"
)

func TestNewKubernetesClient(t *testing.T) {
	setupFakeHome(t)

	t.Run("nil kubeconfig returns error", func(t *testing.T) {
		cfg := &KubernetesCfg{
			KubeConfigFile: getKubeConfigFile(),
			KubeConfig:     nil,
			QPS:            1e6,
			Burst:          1e6,
		}
		if _, err := NewKubernetesClient(cfg); err == nil {
			t.Errorf("NewKubernetesClient() error = nil, want error for nil kubeconfig")
		}
	})

	t.Run("valid kubeconfig returns client", func(t *testing.T) {
		cfg := &KubernetesCfg{
			KubeConfigFile: getKubeConfigFile(),
			KubeConfig:     GetKubeConfig(),
			QPS:            100,
			Burst:          200,
		}
		client, err := NewKubernetesClient(cfg)
		if err != nil {
			t.Fatalf("NewKubernetesClient() error = %v, want nil", err)
		}
		if client.K8s() == nil {
			t.Errorf("client.K8s() = nil, want non-nil")
		}
		if client.Snapshot() == nil {
			t.Errorf("client.Snapshot() = nil, want non-nil")
		}
		if client.Api() == nil {
			t.Errorf("client.Api() = nil, want non-nil")
		}
		if client.Cfg() == nil {
			t.Errorf("client.Cfg() = nil, want non-nil")
		}
		if client.Cfg().QPS != 100 || client.Cfg().Burst != 200 {
			t.Errorf("client.Cfg() QPS/Burst = %v/%v, want 100/200", client.Cfg().QPS, client.Cfg().Burst)
		}
	})
}
