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

func createTestENV() *ENV {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost, Role: []string{bkenode.MasterNodeRole}},
		{IP: testNodeIP2, Hostname: "node2"},
	}

	return &ENV{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		Nodes:         nodes,
		BkeConfigName: testBKEConfig,
		Extra:         []string{"extra1"},
		ExtraHosts:    []string{"host1"},
	}
}

func TestENVValidate(t *testing.T) {
	tests := []struct {
		name    string
		env     *ENV
		wantErr bool
	}{
		{
			name:    "Valid ENV",
			env:     createTestENV(),
			wantErr: false,
		},
		{
			name: "Invalid - empty nodes",
			env: func() *ENV {
				e := createTestENV()
				e.Nodes = bkenode.Nodes{}
				return e
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty BkeConfigName",
			env: func() *ENV {
				e := createTestENV()
				e.BkeConfigName = ""
				return e
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			env: func() *ENV {
				e := createTestENV()
				e.Client = nil
				return e
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.env.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestENVNewConatinerdReset(t *testing.T) {
	tests := []struct {
		name    string
		env     *ENV
		wantErr bool
	}{
		{
			name:    "Valid containerd reset",
			env:     createTestENV(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			env: func() *ENV {
				e := createTestENV()
				e.BkeConfigName = ""
				return e
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.env.NewConatinerdReset()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.env.commandName)
			}
		})
	}
}

func TestENVNewConatinerdRedeploy(t *testing.T) {
	tests := []struct {
		name    string
		env     *ENV
		wantErr bool
	}{
		{
			name:    "Valid containerd redeploy",
			env:     createTestENV(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			env: func() *ENV {
				e := createTestENV()
				e.BkeConfigName = ""
				return e
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.env.NewConatinerdRedeploy()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.env.commandName)
			}
		})
	}
}

func TestENVNewConatinerdRedeploy_WithContainerdVersion(t *testing.T) {
	env := createTestENV()
	env.ContainerdVersion = "1.7.0"

	err := env.NewConatinerdRedeploy()
	assert.NoError(t, err)

	cmd, getErr := env.GetCommand()
	assert.NoError(t, getErr)
	assert.Len(t, cmd.Spec.Commands, 1)
	assert.Contains(t, cmd.Spec.Commands[0].Command, "containerdVersion=1.7.0")
}

func TestENVNew(t *testing.T) {
	tests := []struct {
		name    string
		env     *ENV
		wantErr bool
	}{
		{
			name:    "Valid ENV new - normal mode",
			env:     createTestENV(),
			wantErr: false,
		},
		{
			name: "Valid ENV new - dry run mode",
			env: func() *ENV {
				e := createTestENV()
				e.DryRun = true
				return e
			}(),
			wantErr: false,
		},
		{
			name: "Valid ENV new - deep restore mode",
			env: func() *ENV {
				e := createTestENV()
				e.DeepRestore = true
				return e
			}(),
			wantErr: false,
		},
		{
			name: "Valid ENV new - pre pull image mode",
			env: func() *ENV {
				e := createTestENV()
				e.PrePullImage = true
				return e
			}(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			env: func() *ENV {
				e := createTestENV()
				e.BkeConfigName = ""
				return e
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.env.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.env.commandName)
			}
		})
	}
}

func TestENVGetCommandName(t *testing.T) {
	tests := []struct {
		name           string
		dryRun         bool
		expectedPrefix string
	}{
		{
			name:           "Normal mode",
			dryRun:         false,
			expectedPrefix: K8sEnvCommandName,
		},
		{
			name:           "Dry run mode",
			dryRun:         true,
			expectedPrefix: K8sEnvDryRunCommandName,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := createTestENV()
			env.DryRun = tt.dryRun
			commandName := env.getCommandName()
			assert.Equal(t, tt.expectedPrefix, commandName)
		})
	}
}

func TestENVGetScope(t *testing.T) {
	tests := []struct {
		name          string
		deepRestore   bool
		isImmutableOS bool
		expectedScope string
	}{
		{
			name:          "Normal scope",
			deepRestore:   false,
			isImmutableOS: false,
			expectedScope: "scope=cert,manifests,container,kubelet,extra",
		},
		{
			name:          "Deep restore scope",
			deepRestore:   true,
			isImmutableOS: false,
			expectedScope: "scope=cert,manifests,container,kubelet,containerRuntime,extra",
		},
		{
			name:          "Immutable OS scope",
			deepRestore:   false,
			isImmutableOS: true,
			expectedScope: "scope=cert,manifests,container,kubelet",
		},
		{
			name:          "Immutable OS scope with deep restore (immutable takes priority)",
			deepRestore:   true,
			isImmutableOS: true,
			expectedScope: "scope=cert,manifests,container,kubelet",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := createTestENV()
			env.DeepRestore = tt.deepRestore
			env.IsImmutableOS = tt.isImmutableOS
			scope := env.getScope()
			assert.Equal(t, tt.expectedScope, scope)
		})
	}
}

func TestENVGetInitScope(t *testing.T) {
	tests := []struct {
		name          string
		isImmutableOS bool
		expectedScope string
	}{
		{
			name:          "Normal init scope",
			isImmutableOS: false,
			expectedScope: "scope=time,hosts,dns,kernel,firewall,selinux,swap,httpRepo,runtime,iptables,registry,image,extra",
		},
		{
			name:          "Immutable OS init scope",
			isImmutableOS: true,
			expectedScope: "scope=time,hosts,dns,kernel,runtime,iptables,registry,image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := createTestENV()
			env.IsImmutableOS = tt.isImmutableOS
			scope := env.getInitScope()
			assert.Equal(t, tt.expectedScope, scope)
		})
	}
}

func TestENVBuildCommandSpec(t *testing.T) {
	env := createTestENV()
	bkeConfigStr := GenerateBkeConfigStr(testNS, testBKEConfig)
	extra := "extra=extra1"
	extraHosts := "extraHosts=host1"
	scope := env.getScope()

	commandSpec := env.buildCommandSpec(bkeConfigStr, extra, extraHosts, scope)

	assert.NotNil(t, commandSpec)
	assert.Equal(t, numC3, len(commandSpec.Commands))
	assert.Equal(t, numC0, commandSpec.TTLSecondsAfterFinished)
}

func TestENVValidateNodes(t *testing.T) {
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
			env := createTestENV()
			env.Nodes = tt.nodes
			err := env.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestENVValidateBkeConfigName(t *testing.T) {
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
			env := createTestENV()
			env.BkeConfigName = tt.bkeConfigName
			err := env.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestENVWithMasterNode(t *testing.T) {
	env := createTestENV()
	env.PrePullImage = true
	err := env.New()
	assert.NoError(t, err)
	assert.NotNil(t, env)
}

func TestENVCreatePrePullImageCommand(t *testing.T) {
	env := createTestENV()
	bkeConfigStr := GenerateBkeConfigStr(testNS, testBKEConfig)
	env.createPrePullImageCommand(bkeConfigStr)
	assert.NotNil(t, env)
}

func TestENVNewCreatesValidCommand(t *testing.T) {
	env := createTestENV()
	err := env.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, env.commandName)
	assert.Contains(t, env.commandName, K8sEnvCommandName)
}

func TestENVCommandExecutionTime(t *testing.T) {
	env := createTestENV()
	err := env.New()
	assert.NoError(t, err)

	assert.Contains(t, env.commandName, K8sEnvCommandName)
}

func TestENVWithEmptyExtra(t *testing.T) {
	env := createTestENV()
	env.Extra = []string{}
	env.ExtraHosts = []string{}
	err := env.New()
	assert.NoError(t, err)
	assert.NotNil(t, env)
}

func TestENVBaseCommandValidation(t *testing.T) {
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
			env := &ENV{
				BaseCommand: tt.baseCommand,
				Nodes: bkenode.Nodes{
					{IP: testNodeIP, Hostname: testHost},
				},
				BkeConfigName: testBKEConfig,
			}
			err := env.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestENVBuildCommandSpecIncludesImageScope(t *testing.T) {
	env := createTestENV()
	commandSpec := env.buildCommandSpec(
		GenerateBkeConfigStr(testNS, testBKEConfig),
		"extra=extra1",
		"extraHosts=host1",
		env.getScope(),
	)

	assert.Contains(t, commandSpec.Commands[2].Command, "scope=time,hosts,dns,kernel,firewall,selinux,swap,httpRepo,runtime,iptables,registry,image,extra")
}
