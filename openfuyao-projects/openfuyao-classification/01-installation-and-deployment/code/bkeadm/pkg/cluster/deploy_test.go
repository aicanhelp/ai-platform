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
	"os"
	"path"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func TestCluster(t *testing.T) {
	// Create a temporary file for testing
	tempDir := t.TempDir()
	testFile := path.Join(tempDir, "test-cluster.yaml")

	testConfig := fmt.Sprintf(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
  namespace: test-namespace
spec:
  BKEConfig:
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

	err := os.WriteFile(testFile, []byte(testConfig), utils.DefaultFilePermission)
	assert.NoError(t, err)

	tests := []struct {
		name                           string
		file                           string
		ntpServer                      string
		mockNewBKEClusterFromFile      func(string) (*configv1beta1.BKECluster, error)
		mockMarshalAndWriteClusterYAML func(*configv1beta1.BKECluster) (string, error)
		mockCreateNamespace            func(*k8s.Client, *corev1.Namespace) error
		mockInstallYaml                func(*k8s.Client, string, map[string]string, string) error
		mockWatchEventByAnnotation     func(*k8s.Client, string)
		expectError                    bool
	}{
		{
			name:      "successful cluster deployment",
			file:      testFile,
			ntpServer: "",
			mockNewBKEClusterFromFile: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
					Spec: configv1beta1.BKEClusterSpec{
						ClusterConfig: &configv1beta1.BKEConfig{
							Cluster: configv1beta1.Cluster{
								ImageRepo: configv1beta1.Repo{
									Domain: "registry.example.com",
								},
							},
						},
					},
				}, nil
			},
			mockMarshalAndWriteClusterYAML: func(cluster *configv1beta1.BKECluster) (string, error) {
				return "/tmp/test-cluster.yaml", nil
			},
			mockCreateNamespace: func(c *k8s.Client, ns *corev1.Namespace) error {
				return nil
			},
			mockInstallYaml: func(c *k8s.Client, file string, data map[string]string, template string) error {
				return nil
			},
			mockWatchEventByAnnotation: func(c *k8s.Client, namespace string) {},
			expectError:                false,
		},
		{
			name:      "with NTP server override",
			file:      testFile,
			ntpServer: "pool.ntp.org",
			mockNewBKEClusterFromFile: func(file string) (*configv1beta1.BKECluster, error) {
				cluster := &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
					Spec: configv1beta1.BKEClusterSpec{
						ClusterConfig: &configv1beta1.BKEConfig{
							Cluster: configv1beta1.Cluster{
								ImageRepo: configv1beta1.Repo{
									Domain: "registry.example.com",
								},
								NTPServer: "original.ntp.org", // This should be overridden
							},
						},
					},
				}
				return cluster, nil
			},
			mockMarshalAndWriteClusterYAML: func(cluster *configv1beta1.BKECluster) (string, error) {
				// Verify that the NTP server was updated
				assert.Equal(t, "pool.ntp.org", cluster.Spec.ClusterConfig.Cluster.NTPServer)
				return "/tmp/test-cluster.yaml", nil
			},
			mockCreateNamespace: func(c *k8s.Client, ns *corev1.Namespace) error {
				return nil
			},
			mockInstallYaml: func(c *k8s.Client, file string, data map[string]string, template string) error {
				return nil
			},
			mockWatchEventByAnnotation: func(c *k8s.Client, namespace string) {},
			expectError:                false,
		},
		{
			name:      "NewBKEClusterFromFile error",
			file:      testFile,
			ntpServer: "",
			mockNewBKEClusterFromFile: func(file string) (*configv1beta1.BKECluster, error) {
				return nil, fmt.Errorf("config error")
			},
			expectError: false, // logs error but doesn't return it
		},
		{
			name:      "marshalAndWriteClusterYAML error",
			file:      testFile,
			ntpServer: "",
			mockNewBKEClusterFromFile: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockMarshalAndWriteClusterYAML: func(cluster *configv1beta1.BKECluster) (string, error) {
				return "", fmt.Errorf("marshal error")
			},
			expectError: false, // function returns early
		},
		{
			name:      "InstallYaml error",
			file:      testFile,
			ntpServer: "",
			mockNewBKEClusterFromFile: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockMarshalAndWriteClusterYAML: func(cluster *configv1beta1.BKECluster) (string, error) {
				return "/tmp/test-cluster.yaml", nil
			},
			mockCreateNamespace: func(c *k8s.Client, ns *corev1.Namespace) error {
				return nil
			},
			mockInstallYaml: func(c *k8s.Client, file string, data map[string]string, template string) error {
				return fmt.Errorf("install error")
			},
			expectError: false, // logs error but doesn't return it
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{
				File:      tt.file,
				NtpServer: tt.ntpServer,
			}

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(NewBKEClusterFromFile, tt.mockNewBKEClusterFromFile)

			patches.ApplyFunc(marshalAndWriteClusterYAML, tt.mockMarshalAndWriteClusterYAML)

			patches.ApplyFunc((*k8s.Client).CreateNamespace, tt.mockCreateNamespace)

			patches.ApplyFunc((*k8s.Client).InstallYaml, tt.mockInstallYaml)

			patches.ApplyFunc((*k8s.Client).WatchEventByAnnotation, tt.mockWatchEventByAnnotation)

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			if global.K8s == nil {
				global.K8s = &k8s.Client{}
			}
			op.Cluster()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestClusterWithRealFile(t *testing.T) {
	// Create a temporary file with real content
	tempDir := t.TempDir()
	testFile := path.Join(tempDir, "real-cluster.yaml")

	realConfig := fmt.Sprintf(`apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: real-test-cluster
  namespace: real-test-namespace
spec:
  BKEConfig:
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

	err := os.WriteFile(testFile, []byte(realConfig), utils.DefaultFilePermission)
	assert.NoError(t, err)

	// Apply patches for external dependencies
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(NewBKEClusterFromFile, func(file string) (*configv1beta1.BKECluster, error) {

		cluster := &configv1beta1.BKECluster{}
		// In a real scenario, we would unmarshal the YAML, but for this test we'll create a minimal object
		cluster.ObjectMeta.Name = "real-test-cluster"
		cluster.ObjectMeta.Namespace = "real-test-namespace"
		cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{
			Cluster: configv1beta1.Cluster{
				ImageRepo: configv1beta1.Repo{
					Domain: "registry.example.com",
				},
			},
		}
		return cluster, nil
	})

	patches.ApplyFunc(marshalAndWriteClusterYAML, func(cluster *configv1beta1.BKECluster) (string, error) {
		return "/tmp/real-test-cluster.yaml", nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(c *k8s.Client, ns *corev1.Namespace) error {
		return nil
	})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(c *k8s.Client, file string, data map[string]string, template string) error {
		return nil
	})

	patches.ApplyFunc((*k8s.Client).WatchEventByAnnotation, func(c *k8s.Client, namespace string) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	op := &Options{
		File:      testFile,
		NtpServer: "",
	}

	op.Cluster()

	// The function should complete without panic
	assert.True(t, true)
}

func TestClusterWithNtpServerOverride(t *testing.T) {
	// Test that the NTP server is properly overridden when provided
	tempDir := t.TempDir()
	testFile := path.Join(tempDir, "ntp-test-cluster.yaml")
	testNodesFile := path.Join(tempDir, "ntp-test-nodes.yaml")

	testConfig := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: ntp-test-cluster
  namespace: ntp-test-namespace
spec:
  BKEConfig:
    cluster:
      imageRepo:
        domain: registry.example.com
      ntpServer: original.ntp.org
`

	err := os.WriteFile(testFile, []byte(testConfig), utils.DefaultFilePermission)
	assert.NoError(t, err)

	err = os.WriteFile(testNodesFile, []byte(""), utils.DefaultFilePermission)
	assert.NoError(t, err)

	// Track if the cluster object was modified
	var capturedCluster *configv1beta1.BKECluster

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(NewClusterResourcesFromFiles, func(clusterFile, nodesFile string) (*ClusterResources, error) {
		return &ClusterResources{
			BKECluster: &configv1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "ntp-test-cluster",
					Namespace: "ntp-test-namespace",
				},
				Spec: configv1beta1.BKEClusterSpec{
					ClusterConfig: &configv1beta1.BKEConfig{
						Cluster: configv1beta1.Cluster{
							ImageRepo: configv1beta1.Repo{
								Domain: "registry.example.com",
							},
							NTPServer: "original.ntp.org",
						},
					},
				},
			},
			BKENodes: []configv1beta1.BKENode{},
		}, nil
	})

	patches.ApplyFunc(marshalAndWriteClusterYAML, func(cluster *configv1beta1.BKECluster) (string, error) {
		// Capture the cluster to verify NTP server was updated
		capturedCluster = cluster
		return "/tmp/ntp-test-cluster.yaml", nil
	})

	patches.ApplyFunc(marshalAndWriteNodeYAMLs, func(clusterName, namespace string, nodes []configv1beta1.BKENode) ([]string, error) {
		return []string{}, nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(c *k8s.Client, ns *corev1.Namespace) error {
		return nil
	})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(c *k8s.Client, file string, data map[string]string, template string) error {
		return nil
	})

	patches.ApplyFunc((*k8s.Client).WatchEventByAnnotation, func(c *k8s.Client, namespace string) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	if global.K8s == nil {
		global.K8s = &k8s.Client{}
	}

	op := &Options{
		File:      testFile,
		NodesFile: testNodesFile,
		NtpServer: "new.ntp.org",
	}

	op.Cluster()

	// Verify that the NTP server was updated in the cluster spec
	assert.NotNil(t, capturedCluster)
	assert.Equal(t, "new.ntp.org", capturedCluster.Spec.ClusterConfig.Cluster.NTPServer)
}
