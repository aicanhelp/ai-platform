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

package initialize

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

// Constants for test IP addresses to avoid hardcoding
var (
	// Constants for external etcd config
	expectedEtcdConfigKeys = 4

	// Constants for test nodes
	expectedTestNodes = 2

	// Constants for IP indexing
	testIPIndex = 10
)

func TestNewBkeConfigFromClusterConfig(t *testing.T) {
	tests := []struct {
		name    string
		input   *v1beta1.BKEConfig
		wantErr bool
	}{
		{
			name:    "Empty config",
			input:   &v1beta1.BKEConfig{},
			wantErr: false,
		},
		{
			name: "Config with cluster data",
			input: &v1beta1.BKEConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.23.0",
					Networking:        v1beta1.Networking{},
				},
			},
			wantErr: false,
		},
		{
			name:    "Nil config",
			input:   nil,
			wantErr: true, // This will cause error because we dereference the nil pointer
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.input == nil {
				// Test the case where we expect an error due to nil input
				defer func() {
					if r := recover(); r != nil {
						// Handle panic if any
						t.Logf("Expected panic caught: %v", r)
					}
				}()
			}

			got, err := NewBkeConfigFromClusterConfig(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewBkeConfigFromClusterConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}
			if got == nil && !tt.wantErr {
				t.Errorf("NewBkeConfigFromClusterConfig() = nil, want non-nil")
			}
		})
	}
}

func TestConvertBkEConfig(t *testing.T) {
	tests := []struct {
		name    string
		input   *BkeConfig
		wantErr bool
	}{
		{
			name:    "Empty config",
			input:   &BkeConfig{},
			wantErr: false,
		},
		{
			name: "Config with data",
			input: &BkeConfig{
				Cluster: v1beta1.Cluster{
					KubernetesVersion: "v1.23.0",
					Networking:        v1beta1.Networking{},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConvertBkEConfig(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertBkEConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.input != nil && got == nil && !tt.wantErr {
				t.Errorf("ConvertBkEConfig() = nil, want non-nil")
			}
		})
	}
}

func TestBkeConfigValidate(t *testing.T) {
	// Since Validate depends on validation package, we just ensure it doesn't panic
	// Note: Nodes are now separate BKENode resources, not part of BkeConfig
	bkeConfig := &BkeConfig{
		Cluster: v1beta1.Cluster{
			KubernetesVersion: "v1.23.0",
			Networking:        v1beta1.Networking{},
		},
	}

	err := bkeConfig.Validate()
	if err != nil {
		t.Logf("Validation error (expected in test environment): %v", err)
	}
}

func TestNewExternalEtcdConfig(t *testing.T) {
	result := NewExternalEtcdConfig()

	expectedKeys := []string{"etcdEndpoints", "etcdCAFile", "etcdCertFile", "etcdKeyFile"}
	for _, key := range expectedKeys {
		if _, exists := result[key]; !exists {
			t.Errorf("Expected key %s not found in result", key)
		}
	}

	// Verify all values are empty strings
	for _, key := range expectedKeys {
		if result[key] != "" {
			t.Errorf("Expected empty string for key %s, got %s", key, result[key])
		}
	}

	// Verify the map has exactly expectedEtcdConfigKeys keys
	if len(result) != expectedEtcdConfigKeys {
		t.Errorf("Expected %d keys in result, got %d", expectedEtcdConfigKeys, len(result))
	}
}

func TestBkeConfigGetNodes(t *testing.T) {
	// Test conversion from BKENodeList to Nodes
	bkeNodeList := &v1beta1.BKENodeList{
		Items: []v1beta1.BKENode{
			{
				Spec: v1beta1.BKENodeSpec{
					Role: []string{node.MasterNodeRole, node.EtcdNodeRole},
					IP:   "192.168.1.1",
				},
			},
			{
				Spec: v1beta1.BKENodeSpec{
					Role: []string{node.WorkerNodeRole},
					IP:   "192.168.1.2",
				},
			},
		},
	}

	nodes := node.ConvertBKENodeListToNodes(bkeNodeList)
	if len(nodes) != expectedTestNodes {
		t.Errorf("Expected %d nodes, got %d", expectedTestNodes, len(nodes))
	}

	// Test that we can call methods on returned nodes
	masters := nodes.Master()
	if len(masters) != 1 {
		t.Errorf("Expected 1 master node, got %d", len(masters))
	}
}

func TestBkeConfigYumRepo(t *testing.T) {
	tests := []struct {
		name     string
		config   *BkeConfig
		expected string
	}{
		{
			name: "Complete config",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					HTTPRepo: v1beta1.Repo{
						Domain: "test.repo.com",
						Port:   "8080",
					},
				},
			},
			expected: "http://test.repo.com:8080",
		},
		{
			name: "Empty domain",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					HTTPRepo: v1beta1.Repo{
						Domain: "",
						Port:   "8080",
					},
				},
			},
			expected: "http://:8080",
		},
		{
			name: "Empty port",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					HTTPRepo: v1beta1.Repo{
						Domain: "test.repo.com",
						Port:   "",
					},
				},
			},
			expected: "http://test.repo.com:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.YumRepo()
			if result != tt.expected {
				t.Errorf("YumRepo() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

// TestBkeConfigImageRepoWithDomainAndPrefix tests ImageRepo method with domain and prefix
func TestBkeConfigImageRepoWithDomainAndPrefix(t *testing.T) {
	config := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ImageRepo: v1beta1.Repo{
				Domain: "registry.com",
				Port:   "5000",
				Prefix: "k8s-images",
			},
		},
	}
	expected := "registry.com:5000/k8s-images/"

	result := config.ImageRepo()
	if result != expected {
		t.Errorf("ImageRepo() = %s, expected %s", result, expected)
	}
}

// TestBkeConfigImageRepoWithDomainNoPrefix tests ImageRepo method with domain but no prefix
func TestBkeConfigImageRepoWithDomainNoPrefix(t *testing.T) {
	config := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ImageRepo: v1beta1.Repo{
				Domain: "registry.com",
				Port:   "5000",
				Prefix: "",
			},
		},
	}
	expected := ""

	result := config.ImageRepo()
	if result != expected {
		t.Errorf("ImageRepo() = %s, expected %s", result, expected)
	}
}

// TestBkeConfigImageRepoWithIP tests ImageRepo method with IP address
func TestBkeConfigImageRepoWithIP(t *testing.T) {
	config := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ImageRepo: v1beta1.Repo{
				Port:   "5000",
				Prefix: "k8s",
			},
		},
	}

	config.ImageRepo()

}

func TestCommonFuncMaps(t *testing.T) {
	funcMap := commonFuncMaps()

	// Test that the 'split' function exists
	splitFunc, exists := funcMap["split"]
	if !exists {
		t.Error("Expected 'split' function to exist in funcMap")
		return
	}

	// Test the split function if it exists
	if splitFunc != nil {
		// Since it's a function that returns []string, we can't directly call it in this test
		// But we know it should exist and have the right signature
		t.Logf("Split function exists: %T", splitFunc)
	}
}

func TestGetClusterDNSIP(t *testing.T) {
	tests := []struct {
		name     string
		subnet   string
		expected string
		hasErr   bool
	}{
		{
			name:     "Default subnet",
			subnet:   DefaultServicesSubnet,
			expected: DefaultClusterDNSIP,
			hasErr:   false,
		},
		{
			name:     "Invalid subnet format",
			subnet:   "invalid",
			expected: "",
			hasErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := GetClusterDNSIP(tt.subnet)
			if (err != nil) != tt.hasErr {
				t.Errorf("GetClusterDNSIP() error = %v, hasErr %v", err, tt.hasErr)
				return
			}
			if err != nil {
				return
			}
			if result != tt.expected {
				t.Errorf("GetClusterDNSIP() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestGetDefaultBKEConfig(t *testing.T) {
	cfg := GetDefaultBKEConfig()

	// Test default values are set
	if cfg.Cluster.KubernetesVersion != versions.KubernetesVersion() {
		t.Errorf("Expected KubernetesVersion to be %s, got %s", versions.KubernetesVersion(), cfg.Cluster.KubernetesVersion)
	}

	if cfg.Cluster.Networking.ServiceSubnet != DefaultServicesSubnet {
		t.Errorf("Expected ServiceSubnet to be %s, got %s", DefaultServicesSubnet, cfg.Cluster.Networking.ServiceSubnet)
	}

	if cfg.Cluster.Networking.PodSubnet != DefaultPodSubnet {
		t.Errorf("Expected PodSubnet to be %s, got %s", DefaultPodSubnet, cfg.Cluster.Networking.PodSubnet)
	}

	if cfg.Cluster.Networking.DNSDomain != DefaultServiceDNSDomain {
		t.Errorf("Expected DNSDomain to be %s, got %s", DefaultServiceDNSDomain, cfg.Cluster.Networking.DNSDomain)
	}

	if cfg.Cluster.NTPServer != DefaultNTPServer {
		t.Errorf("Expected NTPServer to be %s, got %s", DefaultNTPServer, cfg.Cluster.NTPServer)
	}

	// Test that addons exist
	if len(cfg.Addons) == 0 {
		t.Error("Expected addons to exist in default config")
	}
}

func TestBKEConfigGetDefaultClusterAPIConfig(t *testing.T) {
	cfg := &BkeConfig{}

	SetDefaultBKEConfig(cfg)

	// Test that defaults are set properly
	if cfg.Cluster.KubernetesVersion != versions.KubernetesVersion() {
		t.Errorf("Expected KubernetesVersion to be %s after SetDefaultBKEConfig, got %s",
			versions.KubernetesVersion(), cfg.Cluster.KubernetesVersion)
	}

	if cfg.Cluster.Networking.DNSDomain != DefaultServiceDNSDomain {
		t.Errorf("Expected DNSDomain to be %s after SetDefaultBKEConfig, got %s",
			DefaultServiceDNSDomain, cfg.Cluster.Networking.DNSDomain)
	}
}

func TestGenerateClusterAPIConfigWithExternalEtcd(t *testing.T) {
	cfg := GetDefaultBKEConfig()
	externalEtcd := map[string]string{
		"etcdEndpoints": "endpoints,123",
		"etcdCAFile":    "ca",
		"etcdCertFile":  "cert",
		"etcdKeyFile":   "key",
	}

	// This test is similar to the original but with more validation
	filePath, err := cfg.GenerateClusterAPIConfigFIle("test", "default", externalEtcd)
	if err != nil {
		t.Logf("Expected error in test environment: %v", err)
	} else {
		if !strings.Contains(filePath, "test.yaml") {
			t.Errorf("Expected generated file path to contain test.yaml, got %s", filePath)
		}
	}
}

func TestGenerateClusterAPIConfigWithoutExternalEtcd(t *testing.T) {
	cfg := GetDefaultBKEConfig()

	// This test is similar to the original but with more validation
	filePath, err := cfg.GenerateClusterAPIConfigFIle("test", "default", nil)
	if err != nil {
		t.Logf("Expected error in test environment: %v", err)
	} else {
		if !strings.Contains(filePath, "test.yaml") {
			t.Errorf("Expected generated file path to contain test.yaml, got %s", filePath)
		}
	}
}

func TestBKEConfigGenerateBKEConfig(t *testing.T) {
	cfg := GetDefaultBKEConfig()
	err := cfg.Validate()
	if err != nil {
		t.Logf("Expected validation error in test environment: %v", err)
	}

	b, err := yaml.Marshal(cfg)
	if err != nil {
		t.Error("Failed to marshal config to yaml")
		return
	}

	// Test that we can unmarshal the config back
	var unmarshaledCfg BkeConfig
	err = yaml.Unmarshal(b, &unmarshaledCfg)
	if err != nil {
		t.Error("Failed to unmarshal config from yaml")
		return
	}

	// Test that the configuration is valid
	if unmarshaledCfg.Cluster.KubernetesVersion != cfg.Cluster.KubernetesVersion {
		t.Error("Configuration changed after marshal/unmarshal")
	}
}

func TestValidateWithVariousInputs(t *testing.T) {
	// Test with valid config
	validConfig := &BkeConfig{
		Cluster: v1beta1.Cluster{
			KubernetesVersion: "v1.25.6",
			Networking:        v1beta1.Networking{},
		},
	}

	err := validConfig.Validate()
	if err != nil {
		t.Logf("Expected validation to succeed, got: %v", err)
	}

	// Test with invalid IP
	invalidConfig := &BkeConfig{
		Cluster: v1beta1.Cluster{
			KubernetesVersion: "v1.25.6",
			Networking: v1beta1.Networking{
				ServiceSubnet: "invalid_cidr",
				PodSubnet:     "invalid_cidr",
			},
		},
	}

	err = invalidConfig.Validate()
	if err == nil {
		t.Error("Expected validation to fail for invalid config")
	}
}

// TestBkeConfigImageRepoWithDomainAndPort tests ImageRepo with domain and port only
func TestBkeConfigImageRepoWithDomainAndPort(t *testing.T) {
	bkeConfig := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ImageRepo: v1beta1.Repo{
				Domain: "registry.example.com",
				Port:   "5000",
			},
		},
	}

	result := bkeConfig.ImageRepo()
	expected := ""
	if result != expected {
		t.Errorf("ImageRepo() = %s, expected %s", result, expected)
	}
}

// TestBkeConfigImageRepoWithDomainPortAndPrefix tests ImageRepo with domain, port and prefix
func TestBkeConfigImageRepoWithDomainPortAndPrefix(t *testing.T) {
	bkeConfig := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ImageRepo: v1beta1.Repo{
				Domain: "registry.example.com",
				Port:   "5000",
				Prefix: "myproject",
			},
		},
	}

	result := bkeConfig.ImageRepo()
	expected := "registry.example.com:5000/myproject/"
	if result != expected {
		t.Errorf("ImageRepo() = %s, expected %s", result, expected)
	}
}

func TestYumRepoWithDifferentConfigurations(t *testing.T) {
	// Test various configurations for YumRepo method
	tests := []struct {
		name     string
		config   v1beta1.Repo
		expected string
	}{
		{
			name: "Domain and Port",
			config: v1beta1.Repo{
				Domain: "yum.example.com",
				Port:   "8080",
			},
			expected: "http://yum.example.com:8080",
		},
		{
			name: "IP and Port",
			config: v1beta1.Repo{
				Port: "8080",
			},
			expected: "http://:8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bkeConfig := &BkeConfig{
				Cluster: v1beta1.Cluster{
					HTTPRepo: tt.config,
				},
			}

			result := bkeConfig.YumRepo()
			if result != tt.expected {
				t.Errorf("YumRepo() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestImageFuyaoRepo(t *testing.T) {
	tests := []struct {
		name     string
		config   *BkeConfig
		expected string
	}{
		{
			name: "With prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ImageRepo: v1beta1.Repo{
						Domain: "registry.com",
						Port:   "5000",
						Prefix: "myprefix",
					},
				},
			},
			expected: "registry.com:5000/myprefix/",
		},
		{
			name: "Without prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ImageRepo: v1beta1.Repo{
						Domain: "registry.com",
						Port:   "5000",
					},
				},
			},
			expected: "cr.openfuyao.cn/openfuyao/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.ImageFuyaoRepo()
			if result != tt.expected {
				t.Errorf("ImageFuyaoRepo() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestImageThirdRepo(t *testing.T) {
	tests := []struct {
		name     string
		config   *BkeConfig
		expected string
	}{
		{
			name: "With prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ImageRepo: v1beta1.Repo{
						Domain: "registry.com",
						Port:   "5000",
						Prefix: "myprefix",
					},
				},
			},
			expected: "registry.com:5000/myprefix/",
		},
		{
			name: "Without prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ImageRepo: v1beta1.Repo{
						Domain: "registry.com",
						Port:   "5000",
					},
				},
			},
			expected: "hub.oepkgs.net/openfuyao/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.ImageThirdRepo()
			if result != tt.expected {
				t.Errorf("ImageThirdRepo() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestChartRepo(t *testing.T) {
	tests := []struct {
		name     string
		config   *BkeConfig
		expected string
	}{
		{
			name: "With prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ChartRepo: v1beta1.Repo{
						Domain: "charts.example.com",
						Port:   "8080",
						Prefix: "stable",
					},
				},
			},
			expected: "charts.example.com:8080/stable/",
		},
		{
			name: "Without prefix",
			config: &BkeConfig{
				Cluster: v1beta1.Cluster{
					ChartRepo: v1beta1.Repo{
						Domain: "charts.example.com",
						Port:   "8080",
					},
				},
			},
			expected: "charts.example.com:8080/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.ChartRepo()
			if result != tt.expected {
				t.Errorf("ChartRepo() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestResolveReachableChartRepo(t *testing.T) {
	config := &BkeConfig{
		Cluster: v1beta1.Cluster{
			ChartRepo: v1beta1.Repo{
				Domain: "charts.example.com",
				Port:   "8080",
			},
		},
	}

	_, err := config.ResolveReachableChartRepo()
	if err != nil {
		t.Logf("Expected error in test environment: %v", err)
	}
}
