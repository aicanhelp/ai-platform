/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightconfig

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateRuntimeCompletenessAcceptsValueAndCustomChanges(t *testing.T) {
	release := loadReleasePolicy(t)
	runtime := loadReleasePolicy(t)
	runtime.Operations.Init.Resources[0].Minimum++
	runtime.Operations.Init.CustomChecks = []CustomCheck{{Key: "extra"}}
	assert.NoError(t, ValidateRuntimeCompleteness(release, runtime))
}

func TestValidateRuntimeCompletenessRejectsRemovedFixedPolicy(t *testing.T) {
	release := loadReleasePolicy(t)
	tests := []struct {
		name   string
		mutate func(*Config)
		part   string
	}{
		{"init port", func(config *Config) { config.Operations.Init.Ports = nil }, "operations.init.ports"},
		{"init platform", func(config *Config) {
			config.Operations.Init.Platform.Distributions = nil
		}, "platform.distributions"},
		{"path requirement", func(config *Config) {
			config.Operations.Init.Paths[0].Require = []string{"create"}
		}, "notReadOnly"},
		{"cluster resource", func(config *Config) {
			config.Operations.ClusterCreate.Resources = nil
		}, "operations.clusterCreate.resources"},
		{"cluster unique", func(config *Config) {
			config.Operations.ClusterCreate.Topology.Unique = []string{"ip"}
		}, "hostname"},
		{"cluster odd", func(config *Config) {
			config.Operations.ClusterCreate.Topology.Masters.Odd = false
		}, "masters.odd"},
		{"management api", func(config *Config) {
			config.Operations.ClusterCreate.ManagementCluster.API = false
		}, "managementCluster.api"},
		{"management crd", func(config *Config) {
			config.Operations.ClusterCreate.ManagementCluster.CRDs = []string{"bkeclusters"}
		}, "bkenodes"},
		{"management kubeconfig", func(config *Config) {
			config.Operations.ClusterCreate.ManagementCluster.RequiredKubeconfig = ""
		}, "requiredKubeconfig"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime := loadReleasePolicy(t)
			test.mutate(&runtime)
			assert.ErrorContains(t, ValidateRuntimeCompleteness(release, runtime), test.part)
		})
	}
}

func loadReleasePolicy(t *testing.T) Config {
	t.Helper()
	config, err := LoadFile(filepath.Join("..", "..", "configs", "preflight", "preflight.yaml"))
	require.NoError(t, err)
	return config
}
