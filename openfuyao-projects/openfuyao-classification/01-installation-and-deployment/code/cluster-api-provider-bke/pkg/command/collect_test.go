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
	testEtcdCertDir = "/etc/etcd/pki"
	testK8sCertDir  = "/etc/kubernetes/pki"
)

func createTestCollect() *Collect {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	return &Collect{
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
		EtcdCertificatesDir: testEtcdCertDir,
		K8sCertificatesDir:  testK8sCertDir,
	}
}

func TestCollectValidate(t *testing.T) {
	tests := []struct {
		name    string
		collect *Collect
		wantErr bool
	}{
		{
			name:    "Valid collect",
			collect: createTestCollect(),
			wantErr: false,
		},
		{
			name: "Invalid - empty EtcdCertificatesDir",
			collect: func() *Collect {
				c := createTestCollect()
				c.EtcdCertificatesDir = ""
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty K8sCertificatesDir",
			collect: func() *Collect {
				c := createTestCollect()
				c.K8sCertificatesDir = ""
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil node",
			collect: func() *Collect {
				c := createTestCollect()
				c.Node = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty cluster name",
			collect: func() *Collect {
				c := createTestCollect()
				c.ClusterName = ""
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil client",
			collect: func() *Collect {
				c := createTestCollect()
				c.Client = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - nil scheme",
			collect: func() *Collect {
				c := createTestCollect()
				c.Scheme = nil
				return c
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty namespace",
			collect: func() *Collect {
				c := createTestCollect()
				c.NameSpace = ""
				return c
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectNew(t *testing.T) {
	tests := []struct {
		name    string
		collect *Collect
		wantErr bool
	}{
		{
			name:    "Valid collect new",
			collect: createTestCollect(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			collect: func() *Collect {
				c := createTestCollect()
				c.EtcdCertificatesDir = ""
				return c
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.collect.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.collect.commandName)
			}
		})
	}
}

func TestCollectCommandName(t *testing.T) {
	collect := createTestCollect()
	err := collect.New()
	assert.NoError(t, err)

	assert.Contains(t, collect.commandName, CollectCertCommandNamePrefix)
	assert.Contains(t, collect.commandName, testClName)
}

func TestCollectValidateEtcdCertificatesDir(t *testing.T) {
	tests := []struct {
		name        string
		etcdCertDir string
		wantErr     bool
	}{
		{
			name:        "Valid EtcdCertificatesDir",
			etcdCertDir: testEtcdCertDir,
			wantErr:     false,
		},
		{
			name:        "Empty EtcdCertificatesDir",
			etcdCertDir: "",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collect := createTestCollect()
			collect.EtcdCertificatesDir = tt.etcdCertDir
			err := collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectValidateK8sCertificatesDir(t *testing.T) {
	tests := []struct {
		name       string
		k8sCertDir string
		wantErr    bool
	}{
		{
			name:       "Valid K8sCertificatesDir",
			k8sCertDir: testK8sCertDir,
			wantErr:    false,
		},
		{
			name:       "Empty K8sCertificatesDir",
			k8sCertDir: "",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collect := createTestCollect()
			collect.K8sCertificatesDir = tt.k8sCertDir
			err := collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectValidateClusterName(t *testing.T) {
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
			collect := createTestCollect()
			collect.ClusterName = tt.clusterName
			err := collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectNodeValidation(t *testing.T) {
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
			collect := createTestCollect()
			collect.Node = tt.node
			err := collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectBaseCommandValidation(t *testing.T) {
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
			collect := &Collect{
				BaseCommand: tt.baseCommand,
				Node: &confv1beta1.Node{
					IP:       testNodeIP,
					Hostname: testHost,
				},
				EtcdCertificatesDir: testEtcdCertDir,
				K8sCertificatesDir:  testK8sCertDir,
			}
			collect.ClusterName = testClName
			err := collect.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCollectNewCreatesValidCommand(t *testing.T) {
	collect := createTestCollect()
	err := collect.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, collect.commandName)
	assert.Contains(t, collect.commandName, CollectCertCommandNamePrefix)
	assert.Contains(t, collect.commandName, testClName)
}
