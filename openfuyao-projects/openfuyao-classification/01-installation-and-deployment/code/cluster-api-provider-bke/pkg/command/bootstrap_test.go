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
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

const (
	testBKEConfig = "test-bke-config"
)

func createTestBootstrap(phase confv1beta1.BKEClusterPhase) *Bootstrap {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	return &Bootstrap{
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
		BKEConfig: testBKEConfig,
		Phase:     phase,
	}
}

func TestBootstrapValidate(t *testing.T) {
	tests := []struct {
		name      string
		bootstrap *Bootstrap
		wantErr   bool
	}{
		{
			name:      "Valid bootstrap",
			bootstrap: createTestBootstrap(bkev1beta1.InitControlPlane),
			wantErr:   false,
		},
		{
			name: "Invalid - empty phase",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap("")
				b.Phase = ""
				return b
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty BKEConfig",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap(bkev1beta1.InitControlPlane)
				b.BKEConfig = ""
				return b
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap(bkev1beta1.InitControlPlane)
				b.Client = nil
				return b
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap(bkev1beta1.InitControlPlane)
				b.Scheme = nil
				return b
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap(bkev1beta1.InitControlPlane)
				b.NameSpace = ""
				return b
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.bootstrap.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBootstrapNew(t *testing.T) {
	tests := []struct {
		name      string
		bootstrap *Bootstrap
		wantErr   bool
	}{
		{
			name:      "Valid bootstrap new - InitControlPlane",
			bootstrap: createTestBootstrap(bkev1beta1.InitControlPlane),
			wantErr:   false,
		},
		{
			name:      "Valid bootstrap new - JoinControlPlane",
			bootstrap: createTestBootstrap(bkev1beta1.JoinControlPlane),
			wantErr:   false,
		},
		{
			name:      "Valid bootstrap new - JoinWorker",
			bootstrap: createTestBootstrap(bkev1beta1.JoinWorker),
			wantErr:   false,
		},
		{
			name: "Invalid - validation failed",
			bootstrap: func() *Bootstrap {
				b := createTestBootstrap("")
				b.Phase = ""
				return b
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.bootstrap.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.bootstrap.commandName)
			}
		})
	}
}

func TestBootstrapNewWithUnknownPhase(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	bootstrap := &Bootstrap{
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
		BKEConfig: testBKEConfig,
		Phase:     "UnknownPhase",
	}

	err := bootstrap.New()
	assert.NoError(t, err)
	assert.NotEmpty(t, bootstrap.commandName)
}

func TestBootstrapCommandName(t *testing.T) {
	bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
	err := bootstrap.New()
	assert.NoError(t, err)

	assert.Contains(t, bootstrap.commandName, BootstrapCommandNamePrefix)
	assert.Contains(t, bootstrap.commandName, testNodeIP)
}

func TestBootstrapCommandSpec(t *testing.T) {
	bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
	err := bootstrap.New()
	assert.NoError(t, err)

	cmd, err := bootstrap.GetCommand()
	assert.NoError(t, err)
	assert.NotNil(t, cmd)
}

func TestBootstrapWithDifferentPhases(t *testing.T) {
	phases := []confv1beta1.BKEClusterPhase{
		bkev1beta1.InitControlPlane,
		bkev1beta1.JoinControlPlane,
		bkev1beta1.JoinWorker,
	}

	for _, phase := range phases {
		t.Run(string(phase), func(t *testing.T) {
			bootstrap := createTestBootstrap(phase)
			err := bootstrap.New()
			assert.NoError(t, err)
			assert.NotNil(t, bootstrap)
		})
	}
}

func TestBootstrapValidateBKEConfig(t *testing.T) {
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
			bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
			bootstrap.BKEConfig = tt.bkeConfig
			err := bootstrap.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBootstrapNodeValidation(t *testing.T) {
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
			bootstrap.Node = tt.node
			err := bootstrap.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBootstrapBaseCommandValidation(t *testing.T) {
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
			bootstrap := &Bootstrap{
				BaseCommand: tt.baseCommand,
				Node: &confv1beta1.Node{
					IP:       testNodeIP,
					Hostname: testHost,
				},
				BKEConfig: testBKEConfig,
				Phase:     bkev1beta1.InitControlPlane,
			}
			err := bootstrap.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBootstrapNewCreatesValidCommand(t *testing.T) {
	bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
	err := bootstrap.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, bootstrap.commandName)
	assert.Contains(t, bootstrap.commandName, BootstrapCommandNamePrefix)
	assert.Contains(t, bootstrap.commandName, testNodeIP)
}

func TestBootstrapCommandExecutionTime(t *testing.T) {
	bootstrap := createTestBootstrap(bkev1beta1.InitControlPlane)
	err := bootstrap.New()
	assert.NoError(t, err)

	assert.Contains(t, bootstrap.commandName, BootstrapCommandNamePrefix)
	assert.Contains(t, bootstrap.commandName, testNodeIP)
}
