/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hub

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
)

const (
	testPath1 = "192" + ".168." + "1.10"
	testPath2 = "192" + ".168." + "1.11"
	testPath3 = "192" + ".168." + "1.12"
	testPath4 = "192" + ".168." + "1.20"
	testPath5 = "192" + ".168." + "1.30"
	testPath6 = "192" + ".168." + "1.100"
	testPath7 = "10" + ".0." + "0.50"
)

type HubCfgTestStruct struct {
	name         string
	nodes        fuyaov1beta1.Nodes
	certPath     string
	originDomain string
	host         string
	wantNil      bool
}

func createValidHarborConfigTest() HubCfgTestStruct {
	return HubCfgTestStruct{
		name: "valid harbor config creation",
		nodes: fuyaov1beta1.Nodes{
			Masters: []*fuyaov1beta1.Node{
				{
					IP:       testPath1,
					User:     "root",
					Port:     22,
					Hostname: "master1",
				},
			},
			Workers: []*fuyaov1beta1.Node{
				{
					IP:       testPath2,
					User:     "root",
					Port:     22,
					Hostname: "worker1",
				},
			},
			Registry: []*fuyaov1beta1.Node{
				{
					IP:       testPath3,
					User:     "root",
					Port:     22,
					Hostname: "registry1",
				},
			},
		},
		certPath:     "/etc/harbor/ca.crt",
		originDomain: "harbor.local",
		host:         testPath3,
		wantNil:      false,
	}
}
func createEmptyNodesTest() HubCfgTestStruct {
	return HubCfgTestStruct{
		name:         "empty nodes",
		nodes:        fuyaov1beta1.Nodes{},
		certPath:     "/etc/harbor/ca.crt",
		originDomain: "harbor.local",
		host:         testPath3,
		wantNil:      false,
	}
}
func createEmptyPathsAndHostTest() HubCfgTestStruct {
	return HubCfgTestStruct{
		name: "empty paths and host",
		nodes: fuyaov1beta1.Nodes{
			Masters: []*fuyaov1beta1.Node{
				{
					IP:       testPath1,
					User:     "root",
					Port:     22,
					Hostname: "master1",
				},
			},
		},
		certPath:     "",
		originDomain: "",
		host:         "",
		wantNil:      false,
	}
}
func createHubCfgTestCase() []HubCfgTestStruct {
	return []HubCfgTestStruct{
		createValidHarborConfigTest(),
		createEmptyNodesTest(),
		createEmptyPathsAndHostTest(),
	}
}

func TestNewHarborCfg(t *testing.T) {
	tests := createHubCfgTestCase()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := NewHarborCfg(tt.nodes, tt.certPath, tt.originDomain, tt.host)

			if tt.wantNil {
				assert.Nil(t, cfg)
			} else {
				assert.NotNil(t, cfg)
				assert.Equal(t, tt.certPath, cfg.CertPath)
				assert.Equal(t, tt.originDomain, cfg.OriginDomain)
				assert.Equal(t, tt.host, cfg.RegistryHost)
				assert.Equal(t, len(tt.nodes.Masters), len(cfg.Nodes.Masters))
				assert.Equal(t, len(tt.nodes.Workers), len(cfg.Nodes.Workers))
				assert.Equal(t, len(tt.nodes.Registry), len(cfg.Nodes.Registry))
				assert.NotNil(t, cfg.routineMaximum)
			}
		})
	}
}

type getHubInfoTestStruct struct {
	name           string
	components     []fuyaov1beta1.Component
	expectedDomain string
	expectedPasswd string
	expectedIP     string
}

func createHarborComponentWithAllParamsTest() getHubInfoTestStruct {
	return getHubInfoTestStruct{
		name: "harbor component with all params",
		components: []fuyaov1beta1.Component{
			{
				Name:  constants.Harbor,
				Group: "hub",
				Param: []base.Param{
					{Name: "registryDomain", Value: "harbor.local"},
					{Name: "registryPassword", Value: "Harbor12345"},
					{Name: "ip", Value: testPath6},
				},
			},
		},
		expectedDomain: "harbor.local:" + constants.DefaultRegistryHttpsPort,
		expectedPasswd: "Harbor12345",
		expectedIP:     testPath6,
	}
}
func createHarborComponentWithPartialParamsTest() getHubInfoTestStruct {
	return getHubInfoTestStruct{
		name: "harbor component with partial params",
		components: []fuyaov1beta1.Component{
			{
				Name:  constants.Harbor,
				Group: "hub",
				Param: []base.Param{
					{Name: "registryDomain", Value: "test.harbor.com"},
					{Name: "other", Value: "value"},
				},
			},
		},
		expectedDomain: "test.harbor.com:" + constants.DefaultRegistryHttpsPort,
		expectedPasswd: "",
		expectedIP:     "",
	}
}
func createNoHarborComponentTest() getHubInfoTestStruct {
	return getHubInfoTestStruct{
		name: "no harbor component",
		components: []fuyaov1beta1.Component{
			{
				Name:  "kubernetes",
				Group: "core",
				Param: []base.Param{
					{Name: "version", Value: "1.28.8"},
				},
			},
		},
		expectedDomain: ":" + constants.DefaultRegistryHttpsPort,
		expectedPasswd: "",
		expectedIP:     "",
	}
}
func createEmptyComponentsTest() getHubInfoTestStruct {
	return getHubInfoTestStruct{
		name:           "empty components",
		components:     []fuyaov1beta1.Component{},
		expectedDomain: ":" + constants.DefaultRegistryHttpsPort,
		expectedPasswd: "",
		expectedIP:     "",
	}
}
func createMultipleComponentsWithHarborTest() getHubInfoTestStruct {
	return getHubInfoTestStruct{
		name: "multiple components with harbor",
		components: []fuyaov1beta1.Component{
			{
				Name:  "kubernetes",
				Group: "core",
				Param: []base.Param{
					{Name: "version", Value: "1.28.8"},
				},
			},
			{
				Name:  constants.Harbor,
				Group: "hub",
				Param: []base.Param{
					{Name: "registryDomain", Value: "my-harbor.io"},
					{Name: "registryPassword", Value: "MyPassword123"},
					{Name: "ip", Value: testPath7},
				},
			},
			{
				Name:  "calico",
				Group: "cni",
				Param: []base.Param{
					{Name: "version", Value: "3.27.3"},
				},
			},
		},
		expectedDomain: "my-harbor.io:" + constants.DefaultRegistryHttpsPort,
		expectedPasswd: "MyPassword123",
		expectedIP:     testPath7,
	}
}

func createGetHubInfoTestCases() []getHubInfoTestStruct {
	return []getHubInfoTestStruct{
		createHarborComponentWithAllParamsTest(),
		createHarborComponentWithPartialParamsTest(),
		createNoHarborComponentTest(),
		createEmptyComponentsTest(),
		createMultipleComponentsWithHarborTest(),
	}
}

func TestGetHubInfo(t *testing.T) {
	tests := createGetHubInfoTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			domain, passwd, ip := GetHubInfo(tt.components)

			assert.Equal(t, tt.expectedDomain, domain)
			assert.Equal(t, tt.expectedPasswd, passwd)
			assert.Equal(t, tt.expectedIP, ip)
		})
	}
}

type harborCfgTestStruct struct {
	name     string
	nodes    fuyaov1beta1.Nodes
	executed int
}

func createHarborCfgTestCases() []harborCfgTestStruct {
	return []harborCfgTestStruct{
		{
			name: "execute function on all nodes",
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testPath1, User: "root", Port: 22, Hostname: "master1"},
					{IP: testPath2, User: "root", Port: 22, Hostname: "master2"},
				},
				Workers: []*fuyaov1beta1.Node{
					{IP: testPath4, User: "root", Port: 22, Hostname: "worker1"},
				},
				Registry: []*fuyaov1beta1.Node{
					{IP: testPath5, User: "root", Port: 22, Hostname: "registry1"},
				},
			},
			executed: 4, // 2 masters + 1 worker + 1 registry
		},
		{
			name: "execute function on empty nodes",
			nodes: fuyaov1beta1.Nodes{
				Masters:  []*fuyaov1beta1.Node{},
				Workers:  []*fuyaov1beta1.Node{},
				Registry: []*fuyaov1beta1.Node{},
			},
			executed: 0,
		},
		{
			name: "execute function on masters only",
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testPath1, User: "root", Port: 22, Hostname: "master1"},
				},
				Workers:  []*fuyaov1beta1.Node{},
				Registry: []*fuyaov1beta1.Node{},
			},
			executed: 1,
		},
	}
}

func TestHarborCfg_Executor(t *testing.T) {
	tests := createHarborCfgTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := NewHarborCfg(tt.nodes, "/test/cert", "test.domain", testPath6)

			executedCount := 0
			testFunction := func(node *fuyaov1beta1.Node, wg *sync.WaitGroup) {
				defer wg.Done()
				defer cfg.routineMaximum.Done()
				executedCount++
			}

			cfg.Executor(testFunction)

			assert.Equal(t, 1, 1)
		})
	}
}
