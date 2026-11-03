/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package command

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

const (
	testClusterFrom = "bke"
)

func createTestUpgrade() *Upgrade {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	return &Upgrade{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Node: &confv1beta1.Node{
			IP:       testNodeIP,
			Hostname: testHost,
		},
		BKEConfig:   testBKEConfig,
		Phase:       confv1beta1.BKEClusterPhase("Running"),
		ClusterFrom: testClusterFrom,
		BackUpEtcd:  false,
	}
}

func TestUpgradeValidate(t *testing.T) {
	tests := []struct {
		name    string
		upgrade *Upgrade
		wantErr bool
	}{
		{
			name:    "Valid upgrade",
			upgrade: createTestUpgrade(),
			wantErr: false,
		},
		{
			name: "Invalid - empty BKEConfig",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.BKEConfig = ""
				return u
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil node",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.Node = nil
				return u
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.Client = nil
				return u
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.Scheme = nil
				return u
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.NameSpace = ""
				return u
			}(),
			wantErr: true,
		},
		{
			name: "Valid - empty ClusterFrom (uses default)",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.ClusterFrom = ""
				return u
			}(),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.upgrade.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUpgradeNew(t *testing.T) {
	tests := []struct {
		name    string
		upgrade *Upgrade
		wantErr bool
	}{
		{
			name:    "Valid upgrade new",
			upgrade: createTestUpgrade(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			upgrade: func() *Upgrade {
				u := createTestUpgrade()
				u.BKEConfig = ""
				return u
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.upgrade.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.upgrade.commandName)
			}
		})
	}
}

func TestUpgradeCommandName(t *testing.T) {
	upgrade := createTestUpgrade()
	err := upgrade.New()
	assert.NoError(t, err)

	assert.Contains(t, upgrade.commandName, UpgradeNodeCommandNamePrefix)
	assert.Contains(t, upgrade.commandName, testNodeIP)
}

func TestUpgradeWithBackUpEtcd(t *testing.T) {
	upgrade := createTestUpgrade()
	upgrade.BackUpEtcd = true
	err := upgrade.New()
	assert.NoError(t, err)
	assert.NotNil(t, upgrade)
}

func TestUpgradeWithEtcdVersion(t *testing.T) {
	upgrade := createTestUpgrade()
	upgrade.EtcdVersion = "v3.5.21-of.1"
	err := upgrade.New()
	assert.NoError(t, err)
	assert.NotNil(t, upgrade)
}

func TestUpgradeWithDefaultClusterFrom(t *testing.T) {
	upgrade := createTestUpgrade()
	upgrade.ClusterFrom = ""
	err := upgrade.New()
	assert.NoError(t, err)
	assert.NotNil(t, upgrade)
}

func TestUpgradeValidateBKEConfig(t *testing.T) {
	tests := []struct {
		name      string
		bkeConfig string
		wantErr   bool
	}{
		{
			name:      "Valid BKEConfig",
			bkeConfig: testBKEConfig,
			wantErr:   false,
		},
		{
			name:      "Empty BKEConfig",
			bkeConfig: "",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrade := createTestUpgrade()
			upgrade.BKEConfig = tt.bkeConfig
			err := upgrade.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUpgradeValidateNode(t *testing.T) {
	tests := []struct {
		name    string
		node    *confv1beta1.Node
		wantErr bool
	}{
		{
			name: "Valid node",
			node: &confv1beta1.Node{
				IP:       testNodeIP,
				Hostname: testHost,
			},
			wantErr: false,
		},
		{
			name:    "Nil node",
			node:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrade := createTestUpgrade()
			upgrade.Node = tt.node
			err := upgrade.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUpgradeValidateClusterFrom(t *testing.T) {
	tests := []struct {
		name        string
		clusterFrom string
		expectFrom  string
	}{
		{
			name:        "Valid ClusterFrom",
			clusterFrom: testClusterFrom,
			expectFrom:  testClusterFrom,
		},
		{
			name:        "Empty ClusterFrom (uses default)",
			clusterFrom: "",
			expectFrom:  "bke",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrade := createTestUpgrade()
			upgrade.ClusterFrom = tt.clusterFrom
			err := upgrade.Validate()
			assert.NoError(t, err)
			if tt.clusterFrom == "" {
				assert.Equal(t, tt.expectFrom, upgrade.ClusterFrom)
			}
		})
	}
}

func TestUpgradeBaseCommandValidation(t *testing.T) {
	tests := []struct {
		name        string
		baseCommand BaseCommand
		wantErr     bool
	}{
		{
			name: "Valid base command",
			baseCommand: BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: false,
		},
		{
			name: "Nil client",
			baseCommand: BaseCommand{
				Client:    nil,
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
		{
			name: "Nil scheme",
			baseCommand: BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    nil,
			},
			wantErr: true,
		},
		{
			name: "Empty namespace",
			baseCommand: BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: "",
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrade := &Upgrade{
				BaseCommand: tt.baseCommand,
				Node: &confv1beta1.Node{
					IP:       testNodeIP,
					Hostname: testHost,
				},
				BKEConfig: testBKEConfig,
				Phase:     confv1beta1.BKEClusterPhase("Running"),
			}
			err := upgrade.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUpgradeNewCreatesValidCommand(t *testing.T) {
	upgrade := createTestUpgrade()
	err := upgrade.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, upgrade.commandName)
	assert.Contains(t, upgrade.commandName, UpgradeNodeCommandNamePrefix)
	assert.Contains(t, upgrade.commandName, testNodeIP)
}

func TestUpgradeWithDifferentPhases(t *testing.T) {
	phases := []confv1beta1.BKEClusterPhase{
		confv1beta1.BKEClusterPhase("InitControlPlane"),
		confv1beta1.BKEClusterPhase("JoinControlPlane"),
		confv1beta1.BKEClusterPhase("JoinWorker"),
	}

	for _, phase := range phases {
		t.Run(string(phase), func(t *testing.T) {
			upgrade := createTestUpgrade()
			upgrade.Phase = phase
			err := upgrade.New()
			assert.NoError(t, err)
			assert.NotNil(t, upgrade)
		})
	}
}
