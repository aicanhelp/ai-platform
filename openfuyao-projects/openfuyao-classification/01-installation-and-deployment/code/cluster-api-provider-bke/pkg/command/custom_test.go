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
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func createTestCustom() *Custom {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
	}

	return &Custom{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Nodes:        nodes,
		CommandName:  testCmdName,
		CommandSpec:  GenerateDefaultCommandSpec(),
		CommandLabel: BKEClusterLabel,
	}
}

func TestCustomValidate(t *testing.T) {
	tests := []struct {
		name    string
		custom  *Custom
		wantErr bool
	}{
		{
			name:    "Valid custom",
			custom:  createTestCustom(),
			wantErr: false,
		},
		{
			name: "Invalid - empty nodes",
			custom: func() *Custom {
				c := createTestCustom()
				c.Nodes = bkenode.Nodes{}
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty command name",
			custom: func() *Custom {
				c := createTestCustom()
				c.CommandName = ""
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil command spec",
			custom: func() *Custom {
				c := createTestCustom()
				c.CommandSpec = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			custom: func() *Custom {
				c := createTestCustom()
				c.Client = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			custom: func() *Custom {
				c := createTestCustom()
				c.Scheme = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			custom: func() *Custom {
				c := createTestCustom()
				c.NameSpace = ""
				return c
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.custom.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomNew(t *testing.T) {
	tests := []struct {
		name    string
		custom  *Custom
		wantErr bool
	}{
		{
			name:    "Valid custom new",
			custom:  createTestCustom(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			custom: func() *Custom {
				c := createTestCustom()
				c.CommandName = ""
				return c
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.custom.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.custom.commandName)
			}
		})
	}
}

func TestCustomValidateNodes(t *testing.T) {
	tests := []struct {
		name    string
		nodes   bkenode.Nodes
		wantErr bool
	}{
		{
			name: "Valid nodes",
			nodes: bkenode.Nodes{
				{IP: testNodeIP, Hostname: testHost},
			},
			wantErr: false,
		},
		{
			name:    "Empty nodes",
			nodes:   bkenode.Nodes{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			custom := createTestCustom()
			custom.Nodes = tt.nodes
			err := custom.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomValidateCommandName(t *testing.T) {
	tests := []struct {
		name        string
		commandName string
		wantErr     bool
	}{
		{
			name:        "Valid command name",
			commandName: testCmdName,
			wantErr:     false,
		},
		{
			name:        "Empty command name",
			commandName: "",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			custom := createTestCustom()
			custom.CommandName = tt.commandName
			err := custom.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomValidateCommandSpec(t *testing.T) {
	tests := []struct {
		name        string
		commandSpec *agentv1beta1.CommandSpec
		wantErr     bool
	}{
		{
			name:        "Valid command spec",
			commandSpec: GenerateDefaultCommandSpec(),
			wantErr:     false,
		},
		{
			name:        "Nil command spec",
			commandSpec: nil,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			custom := createTestCustom()
			custom.CommandSpec = tt.commandSpec
			err := custom.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomBaseCommandValidation(t *testing.T) {
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
			custom := &Custom{
				BaseCommand: tt.baseCommand,
				Nodes: bkenode.Nodes{
					{IP: testNodeIP, Hostname: testHost},
				},
				CommandName: testCmdName,
				CommandSpec: GenerateDefaultCommandSpec(),
			}
			err := custom.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCustomNewCreatesValidCommand(t *testing.T) {
	custom := createTestCustom()
	err := custom.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, custom.commandName)
	assert.Equal(t, testCmdName, custom.commandName)
}

func TestCustomWithMultipleNodes(t *testing.T) {
	custom := createTestCustom()
	custom.Nodes = bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}
	err := custom.New()
	assert.NoError(t, err)
	assert.NotNil(t, custom)
}
