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

func createTestPing() *Ping {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
	}

	return &Ping{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Nodes: nodes,
	}
}

func TestPingValidate(t *testing.T) {
	tests := []struct {
		name    string
		ping    *Ping
		wantErr bool
	}{
		{
			name:    "Valid ping",
			ping:    createTestPing(),
			wantErr: false,
		},
		{
			name: "Invalid - nil client",
			ping: func() *Ping {
				p := createTestPing()
				p.Client = nil
				return p
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			ping: func() *Ping {
				p := createTestPing()
				p.Scheme = nil
				return p
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			ping: func() *Ping {
				p := createTestPing()
				p.NameSpace = ""
				return p
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.ping.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPingNew(t *testing.T) {
	tests := []struct {
		name    string
		ping    *Ping
		wantErr bool
	}{
		{
			name:    "Valid ping new",
			ping:    createTestPing(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			ping: func() *Ping {
				p := createTestPing()
				p.Client = nil
				return p
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.ping.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.ping.commandName)
			}
		})
	}
}

func TestPingCommandName(t *testing.T) {
	ping := createTestPing()
	err := ping.New()
	assert.NoError(t, err)

	assert.Contains(t, ping.commandName, PingCommandNamePrefix)
}

func TestPingWithMultipleNodes(t *testing.T) {
	ping := createTestPing()
	ping.Nodes = bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}
	err := ping.New()
	assert.NoError(t, err)
	assert.NotNil(t, ping)
}

func TestPingBaseCommandValidation(t *testing.T) {
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
			ping := &Ping{
				BaseCommand: tt.baseCommand,
				Nodes: bkenode.Nodes{
					{IP: testNodeIP, Hostname: testHost},
				},
			}
			err := ping.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPingNewCreatesValidCommand(t *testing.T) {
	ping := createTestPing()
	err := ping.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, ping.commandName)
	assert.Contains(t, ping.commandName, PingCommandNamePrefix)
}

func TestPingWait(t *testing.T) {
	t.Skip("Wait() requires running cluster/agent to query command status - skipping for unit tests")
}

func TestPingWaitWithMultipleNodes(t *testing.T) {
	t.Skip("Wait() requires running cluster/agent to query command status - skipping for unit tests")
}
