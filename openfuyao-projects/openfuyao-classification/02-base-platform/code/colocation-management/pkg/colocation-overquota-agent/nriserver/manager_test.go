/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-15
 */

package nriserver

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/colocation-management/cmd/colocation-overquota-agent/apps"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/config"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
)

func TestNriManagerValidConfig(t *testing.T) {
	tests := []struct {
		name         string
		cgroupDriver string
		expectSet    bool
	}{
		{
			name:         "valid systemd driver",
			cgroupDriver: "systemd",
			expectSet:    true,
		},
		{
			name:         "valid cgroupfs driver",
			cgroupDriver: "cgroupfs",
			expectSet:    true,
		},
		{
			name:         "invalid driver",
			cgroupDriver: "invalid",
			expectSet:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalDriver := core.GetCgroupDriver()
			defer core.SetCgroupDriver(originalDriver)

			manager := NriManager{
				config: config.Config{
					CgroupDriver: tt.cgroupDriver,
				},
				kubeClient: &kubernetes.Clientset{}, // 非nil
			}

			manager.ValidConfig()

			if tt.expectSet {
				assert.Equal(t, tt.cgroupDriver, core.GetCgroupDriver())
			} else {
				assert.Equal(t, originalDriver, core.GetCgroupDriver())
			}
		})
	}
}

func TestNriManagerValidConfigNoKubeClient(t *testing.T) {
	originalDriver := core.GetCgroupDriver()
	defer core.SetCgroupDriver(originalDriver)

	manager := NriManager{
		config: config.Config{
			CgroupDriver: "systemd",
		},
		kubeClient: nil,
	}

	manager.ValidConfig()

	assert.Equal(t, originalDriver, core.GetCgroupDriver())
}

// TestNewNriManagerSocketMissing exercises the error path of NewNriManager:
// without a real NRI socket at /host-run/nri/nri.sock, server.NewNriServer
// returns an error and NewNriManager propagates it.
func TestNewNriManagerSocketMissing(t *testing.T) {
	cfg := &apps.Configuration{
		CgroupRoot:   "/sys/fs/cgroup",
		CgroupDriver: "systemd",
	}
	mgr, err := NewNriManager(cfg)
	assert.Error(t, err)
	// On error a zero-value NriManager is returned.
	assert.Equal(t, NriManager{}, mgr)
}

func TestNriManagerShutdown(t *testing.T) {
	mgr := NriManager{}
	// Shutdown only logs; assert it does not panic.
	assert.NotPanics(t, func() {
		mgr.Shutdown()
	})
}

// TestNriManagerValidConfigTable exercises both driver branches and the
// default (no-op) branch of ValidConfig.
func TestNriManagerValidConfigTable(t *testing.T) {
	tests := []struct {
		name         string
		cgroupDriver string
		kubeClient   *kubernetes.Clientset
		expectDriver string
	}{
		{name: "systemd", cgroupDriver: "systemd", kubeClient: &kubernetes.Clientset{}, expectDriver: "systemd"},
		{name: "cgroupfs", cgroupDriver: "cgroupfs", kubeClient: &kubernetes.Clientset{}, expectDriver: "cgroupfs"},
		{name: "invalid driver keeps default", cgroupDriver: "garbage", kubeClient: &kubernetes.Clientset{}, expectDriver: "systemd"},
		{name: "nil kubeclient returns early", cgroupDriver: "cgroupfs", kubeClient: nil, expectDriver: "systemd"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := core.GetCgroupDriver()
			defer core.SetCgroupDriver(original)
			core.SetCgroupDriver("systemd")

			mgr := NriManager{
				config: config.Config{
					CgroupDriver: tt.cgroupDriver,
				},
				kubeClient: tt.kubeClient,
			}
			mgr.ValidConfig()
			assert.Equal(t, tt.expectDriver, core.GetCgroupDriver())
		})
	}
}

// TestNriManagerImplementsManager verifies the Manager interface contract.
func TestNriManagerImplementsManager(t *testing.T) {
	var _ Manager = (*NriManager)(nil)
}
