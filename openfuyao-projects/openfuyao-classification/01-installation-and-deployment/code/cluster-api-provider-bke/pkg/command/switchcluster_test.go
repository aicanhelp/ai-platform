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

func createTestSwitch() *Switch {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
	}

	return &Switch{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Nodes:       nodes,
		ClusterName: testClName,
	}
}

func TestSwitchValidate(t *testing.T) {
	tests := []struct {
		name    string
		switchC *Switch
		wantErr bool
	}{
		{
			name:    "Valid switch",
			switchC: createTestSwitch(),
			wantErr: false,
		},
		{
			name: "Invalid - empty cluster name",
			switchC: func() *Switch {
				s := createTestSwitch()
				s.ClusterName = ""
				return s
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			switchC: func() *Switch {
				s := createTestSwitch()
				s.Client = nil
				return s
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			switchC: func() *Switch {
				s := createTestSwitch()
				s.Scheme = nil
				return s
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			switchC: func() *Switch {
				s := createTestSwitch()
				s.NameSpace = ""
				return s
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.switchC.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSwitchNew(t *testing.T) {
	tests := []struct {
		name    string
		switchC *Switch
		wantErr bool
	}{
		{
			name:    "Valid switch new",
			switchC: createTestSwitch(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			switchC: func() *Switch {
				s := createTestSwitch()
				s.ClusterName = ""
				return s
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.switchC.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.switchC.commandName)
			}
		})
	}
}

func TestSwitchCommandName(t *testing.T) {
	switchC := createTestSwitch()
	err := switchC.New()
	assert.NoError(t, err)

	assert.Contains(t, switchC.commandName, SwitchClusterCommandNamePrefix)
	assert.Contains(t, switchC.commandName, testClName)
}

func TestSwitchWithMultipleNodes(t *testing.T) {
	switchC := createTestSwitch()
	switchC.Nodes = bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}
	err := switchC.New()
	assert.NoError(t, err)
	assert.NotNil(t, switchC)
}

func TestSwitchValidateClusterName(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
		wantErr     bool
	}{
		{
			name:        "Valid cluster name",
			clusterName: testClName,
			wantErr:     false,
		},
		{
			name:        "Empty cluster name",
			clusterName: "",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switchC := createTestSwitch()
			switchC.ClusterName = tt.clusterName
			err := switchC.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSwitchBaseCommandValidation(t *testing.T) {
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
			switchC := &Switch{
				BaseCommand: tt.baseCommand,
				Nodes: bkenode.Nodes{
					{IP: testNodeIP, Hostname: testHost},
				},
				ClusterName: testClName,
			}
			err := switchC.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSwitchNewCreatesValidCommand(t *testing.T) {
	switchC := createTestSwitch()
	err := switchC.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, switchC.commandName)
	assert.Contains(t, switchC.commandName, SwitchClusterCommandNamePrefix)
	assert.Contains(t, switchC.commandName, testClName)
}
