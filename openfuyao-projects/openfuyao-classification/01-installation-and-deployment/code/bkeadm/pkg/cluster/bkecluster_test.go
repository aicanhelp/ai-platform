/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package cluster

import (
	"fmt"
	"net"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	yaml2 "sigs.k8s.io/yaml"
)

const (
	testFileModeReadOnly = os.FileMode(0644)

	testIPv4SegmentA = 192
	testIPv4SegmentB = 168
	testIPv4SegmentC = 1
	testIPv4SegmentD = 100
)

var testIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
).String()

func TestNewBKEClusterFromFile(t *testing.T) {
	// Create a temporary YAML file for testing
	tempFile := t.TempDir() + "/test-cluster.yaml"

	// Valid cluster configuration content
	validConfig := fmt.Sprintf(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: k8s
      httpRepo:
        domain: %s
        ip: %s
        port: %s
      chartRepo:
        domain: %s
        ip: %s
        port: %s
        prefix: charts
      ntpServer: pool.ntp.org
`, testRegistryDomain, testRegistryIP, testRegistryPort, testYumDomain, testYumIP, testYumPort, testChartDomain, testChartIP, testChartPort)

	err := os.WriteFile(tempFile, []byte(validConfig), testFileModeReadOnly)
	assert.NoError(t, err)

	tests := []struct {
		name          string
		file          string
		mockReadFile  func(string) ([]byte, error)
		mockUnmarshal func([]byte, interface{}) error
		expectError   bool
	}{
		{
			name: "successful cluster creation from valid file",
			file: tempFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(validConfig), nil
			},
			mockUnmarshal: func(data []byte, v interface{}) error {
				conf := v.(*configv1beta1.BKECluster)
				conf.APIVersion = "cluster.bocloud.com/v1beta1"
				conf.Kind = "BKECluster"
				conf.Name = "test-cluster"
				conf.Spec.ClusterConfig = &configv1beta1.BKEConfig{
					Cluster: configv1beta1.Cluster{
						ImageRepo: configv1beta1.Repo{
							Domain: "registry.example.com",
							Ip:     testIP,
							Port:   "5000",
							Prefix: "k8s",
						},
					},
				}
				return nil
			},
			expectError: false,
		},
		{
			name: "file read error",
			file: "nonexistent-file.yaml",
			mockReadFile: func(filename string) ([]byte, error) {
				return nil, fmt.Errorf("file not found")
			},
			expectError: true,
		},
		{
			name: "unmarshal error",
			file: tempFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte("invalid yaml"), nil
			},
			mockUnmarshal: func(data []byte, v interface{}) error {
				return fmt.Errorf("unmarshal error")
			},
			expectError: true,
		},
		{
			name: "empty cluster config error",
			file: tempFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
spec:
  clusterConfig: ~
`), nil
			},
			mockUnmarshal: func(data []byte, v interface{}) error {
				conf := v.(*configv1beta1.BKECluster)
				conf.Spec.ClusterConfig = nil
				return nil
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(os.ReadFile, tt.mockReadFile)

			if tt.mockUnmarshal != nil {
				patches.ApplyFunc(yaml2.Unmarshal, tt.mockUnmarshal)
			}

			cluster, err := NewBKEClusterFromFile(tt.file)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, cluster)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cluster)
			}
		})
	}
}

func TestNewBKEClusterFromFileEmptyClusterConfig(t *testing.T) {
	// Test with empty cluster config
	tempDir := t.TempDir()
	tempFile := tempDir + "/empty-config.yaml"

	emptyConfig := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: empty-test-cluster
spec:
  clusterConfig:
`

	err := os.WriteFile(tempFile, []byte(emptyConfig), testFileModeReadOnly)
	assert.NoError(t, err)

	cluster, err := NewBKEClusterFromFile(tempFile)

	assert.Error(t, err)
	assert.Nil(t, cluster)
	assert.Contains(t, err.Error(), "the cluster configuration cannot be empty")
}

func TestNewBKEClusterFromFileInvalidYAML(t *testing.T) {
	// Test with invalid YAML
	tempDir := t.TempDir()
	tempFile := tempDir + "/invalid.yaml"

	invalidYAML := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: invalid-test-cluster
spec:
  clusterConfig:
    cluster:
      # Invalid YAML syntax here
      imageRepo
        domain: registry.example.com
`

	err := os.WriteFile(tempFile, []byte(invalidYAML), testFileModeReadOnly)
	assert.NoError(t, err)

	cluster, err := NewBKEClusterFromFile(tempFile)

	assert.Error(t, err)
	assert.Nil(t, cluster)
}

func TestNewBKEClusterFromFileNonExistentFile(t *testing.T) {
	cluster, err := NewBKEClusterFromFile("non-existent-file.yaml")

	assert.Error(t, err)
	assert.Nil(t, cluster)
}
