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

func createTestReset() *Reset {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	return &Reset{
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
		Extra:     []string{"extra1", "extra2"},
	}
}

func TestResetValidate(t *testing.T) {
	tests := []struct {
		name    string
		reset   *Reset
		wantErr bool
	}{
		{
			name:    "Valid reset",
			reset:   createTestReset(),
			wantErr: false,
		},
		{
			name: "Invalid - empty BKEConfig",
			reset: func() *Reset {
				r := createTestReset()
				r.BKEConfig = ""
				return r
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil node",
			reset: func() *Reset {
				r := createTestReset()
				r.Node = nil
				return r
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			reset: func() *Reset {
				r := createTestReset()
				r.Client = nil
				return r
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			reset: func() *Reset {
				r := createTestReset()
				r.Scheme = nil
				return r
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			reset: func() *Reset {
				r := createTestReset()
				r.NameSpace = ""
				return r
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.reset.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestResetNew(t *testing.T) {
	tests := []struct {
		name    string
		reset   *Reset
		wantErr bool
	}{
		{
			name:    "Valid reset new",
			reset:   createTestReset(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			reset: func() *Reset {
				r := createTestReset()
				r.BKEConfig = ""
				return r
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.reset.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.reset.commandName)
			}
		})
	}
}

func TestResetNewIncludesShutdownScope(t *testing.T) {
	reset := createTestReset()
	assert.NoError(t, reset.New())

	list := &agentv1beta1.CommandList{}
	assert.NoError(t, reset.Client.List(reset.Ctx, list))
	assert.NotEmpty(t, list.Items)
	assert.Contains(t, list.Items[0].Spec.Commands[0].Command, "scope=cert,manifests,kubelet,container,containerRuntime,source,extra,global-cert,shutdown")
}

func TestResetCommandName(t *testing.T) {
	reset := createTestReset()
	err := reset.New()
	assert.NoError(t, err)

	assert.Contains(t, reset.commandName, ResetNodeCommandNamePrefix)
	assert.Contains(t, reset.commandName, testNodeIP)
}

func TestResetWithDeepRestore(t *testing.T) {
	reset := createTestReset()
	reset.DeepRestore = true
	err := reset.New()
	assert.NoError(t, err)
	assert.NotNil(t, reset)
}

func TestResetWithEmptyExtra(t *testing.T) {
	reset := createTestReset()
	reset.Extra = []string{}
	err := reset.New()
	assert.NoError(t, err)
	assert.NotNil(t, reset)
}

func TestResetValidateBKEConfig(t *testing.T) {
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
			reset := createTestReset()
			reset.BKEConfig = tt.bkeConfig
			err := reset.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestResetNodeValidation(t *testing.T) {
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
			reset := createTestReset()
			reset.Node = tt.node
			err := reset.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestResetBaseCommandValidation(t *testing.T) {
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
			reset := &Reset{
				BaseCommand: tt.baseCommand,
				Node: &confv1beta1.Node{
					IP:       testNodeIP,
					Hostname: testHost,
				},
				BKEConfig: testBKEConfig,
			}
			err := reset.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestResetNewCreatesValidCommand(t *testing.T) {
	reset := createTestReset()
	err := reset.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, reset.commandName)
	assert.Contains(t, reset.commandName, ResetNodeCommandNamePrefix)
	assert.Contains(t, reset.commandName, testNodeIP)
}

func TestResetCommandScope(t *testing.T) {
	reset := createTestReset()
	err := reset.New()
	assert.NoError(t, err)
	assert.NotNil(t, reset)
}
