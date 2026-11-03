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

const (
	testImageRepo        = "registry.example.com"
	testManifestsDir     = "/etc/kubernetes/manifests"
	testIngressVIP       = "192.168.1.100"
	testControlPlaneVIP  = "192.168.1.101"
	testControlPlanePort = 6443
	testVirtualRouterId  = "50"
)

func createTestHA() *HA {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	masterNodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}

	return &HA{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		MasterNodes:              masterNodes,
		IngressNodes:             bkenode.Nodes{},
		IngressVIP:               "",
		ControlPlaneEndpointPort: testControlPlanePort,
		ControlPlaneEndpointVIP:  testControlPlaneVIP,
		ThirdImageRepo:           testImageRepo,
		FuyaoImageRepo:           testImageRepo,
		ManifestsDir:             testManifestsDir,
		VirtualRouterId:          testVirtualRouterId,
		WaitVIP:                  false,
	}
}

func createTestIngressHA() *HA {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	ingressNodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
	}

	return &HA{
		BaseCommand: BaseCommand{
			Ctx:         context.Background(),
			Client:      fakeClient,
			NameSpace:   testNS,
			Scheme:      scheme,
			ClusterName: testClName,
		},
		MasterNodes:              bkenode.Nodes{},
		IngressNodes:             ingressNodes,
		IngressVIP:               testIngressVIP,
		ControlPlaneEndpointPort: 0,
		ControlPlaneEndpointVIP:  "",
		ThirdImageRepo:           testImageRepo,
		FuyaoImageRepo:           testImageRepo,
		ManifestsDir:             testManifestsDir,
		VirtualRouterId:          testVirtualRouterId,
		WaitVIP:                  false,
	}
}

func TestHAValidate(t *testing.T) {
	tests := []struct {
		name    string
		ha      *HA
		wantErr bool
	}{
		{
			name:    "Valid Master HA",
			ha:      createTestHA(),
			wantErr: false,
		},
		{
			name:    "Valid Ingress HA",
			ha:      createTestIngressHA(),
			wantErr: false,
		},
		{
			name: "Invalid - empty ImageRepo",
			ha: func() *HA {
				h := createTestHA()
				h.ThirdImageRepo = ""
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - empty ManifestsDir",
			ha: func() *HA {
				h := createTestHA()
				h.ManifestsDir = ""
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - both Master and Ingress HA configured",
			ha: func() *HA {
				h := createTestHA()
				h.IngressNodes = bkenode.Nodes{{IP: testNodeIP2, Hostname: "node2"}}
				h.IngressVIP = testIngressVIP
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - Master HA without port",
			ha: func() *HA {
				h := createTestHA()
				h.ControlPlaneEndpointPort = 0
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - Master HA without VIP",
			ha: func() *HA {
				h := createTestHA()
				h.ControlPlaneEndpointVIP = ""
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - Ingress HA without VIP",
			ha: func() *HA {
				h := createTestIngressHA()
				h.IngressVIP = ""
				return h
			}(),
			wantErr: true,
		},
		{
			name: "Invalid - no nodes configured",
			ha: func() *HA {
				h := createTestHA()
				h.MasterNodes = bkenode.Nodes{}
				return h
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHANew(t *testing.T) {
	tests := []struct {
		name    string
		ha      *HA
		wantErr bool
	}{
		{
			name:    "Valid Master HA new",
			ha:      createTestHA(),
			wantErr: false,
		},
		{
			name:    "Valid Ingress HA new",
			ha:      createTestIngressHA(),
			wantErr: false,
		},
		{
			name: "Invalid - validation failed",
			ha: func() *HA {
				h := createTestHA()
				h.ThirdImageRepo = ""
				return h
			}(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.ha.New()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, tt.ha.commandName)
			}
		})
	}
}

func TestHAIsMasterHa(t *testing.T) {
	masterHA := createTestHA()
	err := masterHA.Validate()
	assert.NoError(t, err)
	assert.True(t, masterHA.isMasterHa)

	ingressHA := createTestIngressHA()
	err = ingressHA.Validate()
	assert.NoError(t, err)
	assert.False(t, ingressHA.isMasterHa)
}

func TestHAValidateImageRepo(t *testing.T) {
	tests := []struct {
		name      string
		imageRepo string
		wantErr   bool
	}{
		{
			name:      "Valid ImageRepo",
			imageRepo: testImageRepo,
			wantErr:   false,
		},
		{
			name:      "Empty ImageRepo",
			imageRepo: "",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha := createTestHA()
			ha.ThirdImageRepo = tt.imageRepo
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHAValidateManifestsDir(t *testing.T) {
	tests := []struct {
		name         string
		manifestsDir string
		wantErr      bool
	}{
		{
			name:         "Valid ManifestsDir",
			manifestsDir: testManifestsDir,
			wantErr:      false,
		},
		{
			name:         "Empty ManifestsDir",
			manifestsDir: "",
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha := createTestHA()
			ha.ManifestsDir = tt.manifestsDir
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHAValidateControlPlaneEndpointPort(t *testing.T) {
	tests := []struct {
		name    string
		port    int32
		wantErr bool
	}{
		{
			name:    "Valid port",
			port:    testControlPlanePort,
			wantErr: false,
		},
		{
			name:    "Empty port",
			port:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha := createTestHA()
			ha.ControlPlaneEndpointPort = tt.port
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHAValidateControlPlaneEndpointVIP(t *testing.T) {
	tests := []struct {
		name    string
		vip     string
		wantErr bool
	}{
		{
			name:    "Valid VIP",
			vip:     testControlPlaneVIP,
			wantErr: false,
		},
		{
			name:    "Empty VIP",
			vip:     "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha := createTestHA()
			ha.ControlPlaneEndpointVIP = tt.vip
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHAValidateIngressVIP(t *testing.T) {
	tests := []struct {
		name    string
		vip     string
		wantErr bool
	}{
		{
			name:    "Valid Ingress VIP",
			vip:     testIngressVIP,
			wantErr: false,
		},
		{
			name:    "Empty Ingress VIP",
			vip:     "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha := createTestIngressHA()
			ha.IngressVIP = tt.vip
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHABaseCommandValidation(t *testing.T) {
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
			ha := &HA{
				BaseCommand:              tt.baseCommand,
				MasterNodes:              bkenode.Nodes{{IP: testNodeIP, Hostname: testHost}},
				ThirdImageRepo:           testImageRepo,
				FuyaoImageRepo:           testImageRepo,
				ManifestsDir:             testManifestsDir,
				ControlPlaneEndpointPort: testControlPlanePort,
				ControlPlaneEndpointVIP:  testControlPlaneVIP,
			}
			err := ha.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestHANewCreatesValidCommand(t *testing.T) {
	ha := createTestHA()
	err := ha.New()
	assert.NoError(t, err)

	assert.NotEmpty(t, ha.commandName)
	assert.Contains(t, ha.commandName, HACommandName)
}

func TestHAWithDifferentVirtualRouterId(t *testing.T) {
	ha := createTestHA()
	ha.VirtualRouterId = "100"
	err := ha.New()
	assert.NoError(t, err)
	assert.NotNil(t, ha)
}

func TestHAGetHaNodesParam(t *testing.T) {
	ha := createTestHA()
	nodesParam := ha.getHaNodesParam(ha.MasterNodes)

	assert.NotEmpty(t, nodesParam)
	assert.Contains(t, nodesParam, "haNodes=")
	assert.Contains(t, nodesParam, testHost)
	assert.Contains(t, nodesParam, testNodeIP)
}
