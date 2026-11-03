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

func createTestHosts() *Hosts {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
	}

	return &Hosts{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Nodes:         nodes,
		BkeConfigName: testBKEConfig,
	}
}

func TestHostsValidate(t *testing.T) {
	tests := []struct {
		name    string
		hosts   *Hosts
		wantErr bool
	}{
		{
			name:    "Valid hosts",
			hosts:   createTestHosts(),
			wantErr: false,
		},
		{
			name: "Invalid - empty nodes",
			hosts: func() *Hosts {
				h := createTestHosts()
				h.Nodes = bkenode.Nodes{}
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty BkeConfigName",
			hosts: func() *Hosts {
				h := createTestHosts()
				h.BkeConfigName = ""
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			hosts: func() *Hosts {
				h := createTestHosts()
				h.Client = nil
				return h
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.hosts.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHostsNew(t *testing.T) {
	tests := []struct {
		name    string
		hosts   *Hosts
		wantErr bool
	}{
		{
			name:    "Valid hosts new",
			hosts:   createTestHosts(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			hosts: func() *Hosts {
				h := createTestHosts()
				h.BkeConfigName = ""
				return h
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.hosts.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.hosts.commandName)
			}
		})
	}
}

func TestHostsCommandName(t *testing.T) {
	hosts := createTestHosts()
	err := hosts.New()
	assert.NoError(t, err)

	assert.Contains(t, hosts.commandName, K8sHostsCommandName)
}

func TestHostsWithMultipleNodes(t *testing.T) {
	hosts := createTestHosts()
	hosts.Nodes = bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}
	err := hosts.New()
	assert.NoError(t, err)
	assert.NotNil(t, hosts)
}

func TestHostsValidateNodes(t *testing.T) {
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
			hosts := createTestHosts()
			hosts.Nodes = tt.nodes
			err := hosts.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHostsValidateBkeConfigName(t *testing.T) {
	tests := []struct {
		name          string
		bkeConfigName string
		wantErr       bool
	}{
		{
			name:          "Valid BkeConfigName",
			bkeConfigName: testBKEConfig,
			wantErr:       false,
		},
		{
			name:          "Empty BkeConfigName",
			bkeConfigName: "",
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hosts := createTestHosts()
			hosts.BkeConfigName = tt.bkeConfigName
			err := hosts.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHostsBaseCommandValidation(t *testing.T) {
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
			hosts := &Hosts{
				BaseCommand: tt.baseCommand,
				Nodes: bkenode.Nodes{
					{IP: testNodeIP, Hostname: testHost},
				},
				BkeConfigName: testBKEConfig,
			}
			err := hosts.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHostsNewCreatesValidCommand(t *testing.T) {
	hosts := createTestHosts()
	err := hosts.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, hosts.commandName)
	assert.Contains(t, hosts.commandName, K8sHostsCommandName)
}

func TestHostsCommandExecutionTime(t *testing.T) {
	hosts := createTestHosts()
	err := hosts.New()
	assert.NoError(t, err)

	assert.Contains(t, hosts.commandName, K8sHostsCommandName)
}
