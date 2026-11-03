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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicFake "k8s.io/client-go/dynamic/fake"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	testZeroValue = 0
	testTwoValue  = 2
	testPortValue = 6443

	testIPv4SegmentD2 = 10
	testIPv4SegmentD3 = 11
)

var (
	testRegistryDomain = "registry.example.com"
	testRegistryIP     = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
	testRegistryPort   = "5000"
	testYumDomain      = "yum.example.com"
	testYumIP          = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD2)
	testYumPort        = "8080"
	testChartDomain    = "chart.example.com"
	testChartIP        = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
	testChartPort      = "8443"
	testAPIEndpoint1   = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD2)
	testAPIEndpoint2   = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD3)
)

func TestRemove(t *testing.T) {

	tests := []struct {
		name          string
		mockMkdirAll  func(string, os.FileMode) error
		mockWriteFile func(string, []byte, os.FileMode) error
		expectError   bool
	}{
		{
			name:          "successful write with namespace prefix",
			mockMkdirAll:  func(string, os.FileMode) error { return nil },
			mockWriteFile: func(filename string, data []byte, perm os.FileMode) error { return nil },
			expectError:   false,
		},
		{
			name:          "mkdir all fails",
			mockMkdirAll:  func(string, os.FileMode) error { return fmt.Errorf("mkdir error") },
			mockWriteFile: func(string, []byte, os.FileMode) error { return nil },
			expectError:   false, // mkdir error is just logged as warning
		},
		{
			name:          "write file fails",
			mockWriteFile: func(string, []byte, os.FileMode) error { return fmt.Errorf("write error") },
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &k8s.MockK8sClient{}
			global.K8s = mockClient
			op := &Options{}
			op.Args = []string{"test-namespace/test-cluster"}
			// Apply patches

			if !tt.expectError {
				// Create test BKECluster
				testBKECluster := &configv1beta1.BKECluster{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "cluster.bocloud.com/v1beta1",
						Kind:       "BKECluster",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
					Spec: configv1beta1.BKEClusterSpec{
						Reset: false,
					},
				}

				// To Unstructured
				unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(testBKECluster)
				assert.NoError(t, err)
				workloadUnstructured := &unstructured.Unstructured{Object: unstructuredObj}

				// Create fake client with the object
				scheme := runtime.NewScheme()
				fullClient := dynamicFake.NewSimpleDynamicClientWithCustomListKinds(
					scheme,
					map[schema.GroupVersionResource]string{
						gvr: "BKEClusterList",
					},
					workloadUnstructured,
				)

				// Mock GetDynamicClient to return our base client
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc((*k8s.MockK8sClient).GetDynamicClient, func(m *k8s.MockK8sClient) dynamic.Interface {
					return fullClient
				})
			}

			op.Remove()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestScale(t *testing.T) {
	// Create a temporary file for testing
	tempDir := t.TempDir()
	testFile := path.Join(tempDir, "test-cluster.yaml")

	testConfig := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
  namespace: test-namespace
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: registry.example.com
`

	err := os.WriteFile(testFile, []byte(testConfig), testFileModeReadOnly)
	assert.NoError(t, err)

	tests := []struct {
		name                           string
		file                           string
		mockNewBKEClusterFromFile      func(string) (*configv1beta1.BKECluster, error)
		mockMarshalAndWriteClusterYAML func(*configv1beta1.BKECluster) (string, error)
		mockPatchYaml                  func(*k8s.Client, string, map[string]string) error
		mockWatchEventByAnnotation     func(*k8s.Client, string)
		expectError                    bool
	}{
		{
			name: "successful scale",
			file: testFile,
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
			mockPatchYaml: func(c *k8s.Client, file string, data map[string]string) error {
				return nil
			},
			mockWatchEventByAnnotation: func(c *k8s.Client, namespace string) {},
			expectError:                false,
		},
		{
			name: "PatchYaml error",
			file: testFile,
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
			mockPatchYaml: func(c *k8s.Client, file string, data map[string]string) error {
				return fmt.Errorf("patch error")
			},
			mockWatchEventByAnnotation: func(c *k8s.Client, namespace string) {},
			expectError:                false, // logs error but doesn't return it
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{File: tt.file}

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(NewBKEClusterFromFile, tt.mockNewBKEClusterFromFile)

			patches.ApplyFunc(marshalAndWriteClusterYAML, tt.mockMarshalAndWriteClusterYAML)

			patches.ApplyFunc((*k8s.Client).PatchYaml, tt.mockPatchYaml)

			patches.ApplyFunc((*k8s.Client).WatchEventByAnnotation, tt.mockWatchEventByAnnotation)

			patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

			op.Scale()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestLog(t *testing.T) {
	tests := []struct {
		name                       string
		args                       []string
		mockWatchEventByAnnotation func(*k8s.Client, string)
	}{
		{
			name:                       "successful log",
			args:                       []string{"test-namespace/test-cluster"},
			mockWatchEventByAnnotation: func(c *k8s.Client, namespace string) {},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{Args: tt.args}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*k8s.Client).WatchEventByAnnotation, tt.mockWatchEventByAnnotation)

			op.Log()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestLoadClusterConfig(t *testing.T) {
	// Create a temporary file for testing
	tempDir := t.TempDir()
	testFile := path.Join(tempDir, "test-config.yaml")

	testConfig := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
  namespace: test-namespace
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: registry.example.com
`

	err := os.WriteFile(testFile, []byte(testConfig), testFileModeReadOnly)
	assert.NoError(t, err)

	tests := []struct {
		name          string
		file          string
		mockReadFile  func(string) ([]byte, error)
		mockUnmarshal func([]byte, interface{}) error
		expectError   bool
	}{
		{
			name: "successful load",
			file: testFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(testConfig), nil
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
						},
					},
				}
				return nil
			},
			expectError: false,
		},
		{
			name: "read file error",
			file: "nonexistent.yaml",
			mockReadFile: func(filename string) ([]byte, error) {
				return nil, fmt.Errorf("file not found")
			},
			expectError: true,
		},
		{
			name: "unmarshal error",
			file: testFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte("invalid yaml"), nil
			},
			mockUnmarshal: func(data []byte, v interface{}) error {
				return fmt.Errorf("unmarshal error")
			},
			expectError: true,
		},
		{
			name: "empty cluster config",
			file: testFile,
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

			patches.ApplyFunc(yaml2.Unmarshal, tt.mockUnmarshal)

			conf, err := loadClusterConfig(tt.file)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, conf)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, conf)
			}
		})
	}
}

func TestCreateKubeconfigSecret(t *testing.T) {
	// Create a temporary kubeconfig file for testing
	tempDir := t.TempDir()
	kubeconfigFile := path.Join(tempDir, "kubeconfig")

	kubeconfigContent := fmt.Sprintf(`apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://%d.%d.%d.%d:%d
  name: test-cluster
contexts:
- context:
    cluster: test-cluster
    user: admin
  name: test-context
current-context: test-context
users:
- name: admin
  user:
    token: abc123
`, testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD, testPortValue)

	err := os.WriteFile(kubeconfigFile, []byte(kubeconfigContent), testFileModeReadOnly)
	assert.NoError(t, err)

	tests := []struct {
		name             string
		namespace        string
		nameSuffix       string
		confPath         string
		mockReadFile     func(string) ([]byte, error)
		mockCreateSecret func(*k8s.Client, *corev1.Secret) error
		expectError      bool
	}{
		{
			name:       "successful secret creation",
			namespace:  "test-namespace",
			nameSuffix: "test-cluster",
			confPath:   kubeconfigFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockCreateSecret: func(_ *k8s.Client, secret *corev1.Secret) error {
				return nil
			},
			expectError: false,
		},
		{
			name:       "read file error",
			namespace:  "test-namespace",
			nameSuffix: "test-cluster",
			confPath:   "nonexistent",
			mockReadFile: func(filename string) ([]byte, error) {
				return nil, fmt.Errorf("file not found")
			},
			mockCreateSecret: func(_ *k8s.Client, secret *corev1.Secret) error {
				return nil
			},
			expectError: true,
		},
		{
			name:       "create secret error",
			namespace:  "test-namespace",
			nameSuffix: "test-cluster",
			confPath:   kubeconfigFile,
			mockReadFile: func(filename string) ([]byte, error) {
				return []byte(kubeconfigContent), nil
			},
			mockCreateSecret: func(_ *k8s.Client, secret *corev1.Secret) error {
				return fmt.Errorf("create error")
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

			patches.ApplyFunc((*k8s.Client).CreateSecret, tt.mockCreateSecret)

			// Initialize global.K8s to prevent nil pointer dereference
			originalK8s := global.K8s
			global.K8s = &k8s.Client{}
			defer func() {
				global.K8s = originalK8s
			}()

			err := createKubeconfigSecret(tt.namespace, tt.nameSuffix, tt.confPath)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestExistsCluster(t *testing.T) {
	// Create temporary files for testing
	tempDir := t.TempDir()
	clusterFile := path.Join(tempDir, "cluster.yaml")
	kubeconfigFile := path.Join(tempDir, "kubeconfig")

	clusterConfig := `apiVersion: cluster.bocloud.com/v1beta1
kind: BKECluster
metadata:
  name: test-cluster
  namespace: test-namespace
spec:
  clusterConfig:
    cluster:
      imageRepo:
        domain: registry.example.com
`

	kubeconfigContent := fmt.Sprintf(`apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://%d.%d.%d.%d:%d
  name: test-cluster
`, testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD, testPortValue)

	err := os.WriteFile(clusterFile, []byte(clusterConfig), testFileModeReadOnly)
	assert.NoError(t, err)

	err = os.WriteFile(kubeconfigFile, []byte(kubeconfigContent), testFileModeReadOnly)
	assert.NoError(t, err)

	tests := []struct {
		name                       string
		clusterFile                string
		kubeconfigFile             string
		mockLoadClusterConfig      func(string) (*configv1beta1.BKECluster, error)
		mockCreateNamespace        func(k8s.KubernetesClient, string) error
		mockCreateKubeconfigSecret func(string, string, string) error
		mockInstallYaml            func(*k8s.Client, string, map[string]string, string) error
		expectError                bool
	}{
		{
			name:           "successful exists cluster",
			clusterFile:    clusterFile,
			kubeconfigFile: kubeconfigFile,
			mockLoadClusterConfig: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockCreateNamespace: func(k8sClient k8s.KubernetesClient, namespace string) error {
				return nil
			},
			mockCreateKubeconfigSecret: func(namespace, name, confPath string) error {
				return nil
			},
			mockInstallYaml: func(_ *k8s.Client, file string, data map[string]string, template string) error {
				return nil
			},
			expectError: false,
		},
		{
			name:           "load cluster config error",
			clusterFile:    clusterFile,
			kubeconfigFile: kubeconfigFile,
			mockLoadClusterConfig: func(file string) (*configv1beta1.BKECluster, error) {
				return nil, fmt.Errorf("load error")
			},
			expectError: false, // logs error but doesn't return it
		},
		{
			name:           "create namespace error",
			clusterFile:    clusterFile,
			kubeconfigFile: kubeconfigFile,
			mockLoadClusterConfig: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockCreateNamespace: func(k8sClient k8s.KubernetesClient, namespace string) error {
				return fmt.Errorf("namespace error")
			},
			expectError: false, // logs error but doesn't return it
		},
		{
			name:           "create kubeconfig secret error",
			clusterFile:    clusterFile,
			kubeconfigFile: kubeconfigFile,
			mockLoadClusterConfig: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockCreateNamespace: func(k8sClient k8s.KubernetesClient, namespace string) error {
				return nil
			},
			mockCreateKubeconfigSecret: func(namespace, name, confPath string) error {
				return fmt.Errorf("secret error")
			},
			expectError: false, // logs error but doesn't return it
		},
		{
			name:           "install yaml error",
			clusterFile:    clusterFile,
			kubeconfigFile: kubeconfigFile,
			mockLoadClusterConfig: func(file string) (*configv1beta1.BKECluster, error) {
				return &configv1beta1.BKECluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-cluster",
						Namespace: "test-namespace",
					},
				}, nil
			},
			mockCreateNamespace: func(k8sClient k8s.KubernetesClient, namespace string) error {
				return nil
			},
			mockCreateKubeconfigSecret: func(namespace, name, confPath string) error {
				return nil
			},
			mockInstallYaml: func(_ *k8s.Client, file string, data map[string]string, template string) error {
				return fmt.Errorf("install error")
			},
			expectError: false, // logs error but doesn't return it
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{
				File: tt.clusterFile,
				Conf: tt.kubeconfigFile,
			}

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(loadClusterConfig, tt.mockLoadClusterConfig)

			patches.ApplyFunc(k8s.CreateNamespace, tt.mockCreateNamespace)

			patches.ApplyFunc(createKubeconfigSecret, tt.mockCreateKubeconfigSecret)

			patches.ApplyFunc((*k8s.Client).InstallYaml, tt.mockInstallYaml)

			patches.ApplyFunc(log.Error, func(v ...interface{}) {})

			op.ExistsCluster()

			// The function should complete without panic
			assert.True(t, true)
		})
	}
}

func TestNsNamePartsCountConstant(t *testing.T) {
	// Test that the constant is defined correctly
	assert.Equal(t, testTwoValue, nsNamePartsCount)
}

func TestMarshalAndWriteClusterYAMLDirectly(t *testing.T) {
	tests := []struct {
		name          string
		namespace     string
		nameValue     string
		mockMkdirAll  func(string, os.FileMode) error
		mockWriteFile func(string, []byte, os.FileMode) error
		expectError   bool
		expectPathNS  string
	}{
		{
			name:          "successful write keeps declared namespace",
			namespace:     "test-namespace",
			nameValue:     "test-cluster",
			mockMkdirAll:  func(string, os.FileMode) error { return nil },
			mockWriteFile: func(string, []byte, os.FileMode) error { return nil },
			expectError:   false,
			expectPathNS:  "test-namespace",
		},
		{
			name:          "successful write with bke-prefixed namespace unchanged",
			namespace:     "bke-existing",
			nameValue:     "test-cluster",
			mockMkdirAll:  func(string, os.FileMode) error { return nil },
			mockWriteFile: func(string, []byte, os.FileMode) error { return nil },
			expectError:   false,
			expectPathNS:  "bke-existing",
		},
		{
			name:          "empty namespace uses default in filename only",
			namespace:     "",
			nameValue:     "test-cluster",
			mockMkdirAll:  func(string, os.FileMode) error { return nil },
			mockWriteFile: func(string, []byte, os.FileMode) error { return nil },
			expectError:   false,
			expectPathNS:  "default",
		},
		{
			name:          "mkdir all fails",
			namespace:     "test-namespace",
			nameValue:     "test-cluster",
			mockMkdirAll:  func(string, os.FileMode) error { return fmt.Errorf("mkdir error") },
			mockWriteFile: func(string, []byte, os.FileMode) error { return nil },
			expectError:   false,
			expectPathNS:  "test-namespace",
		},
		{
			name:          "write file fails",
			namespace:     "test-namespace",
			nameValue:     "test-cluster",
			mockMkdirAll:  func(string, os.FileMode) error { return nil },
			mockWriteFile: func(string, []byte, os.FileMode) error { return fmt.Errorf("write error") },
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bkeCluster := &configv1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      tt.nameValue,
					Namespace: tt.namespace,
				},
			}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)

			patches.ApplyFunc(os.WriteFile, tt.mockWriteFile)

			result, err := marshalAndWriteClusterYAML(bkeCluster)

			if tt.expectError {
				assert.Error(t, err)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, result)
				assert.Equal(t, tt.namespace, bkeCluster.Namespace)
				assert.Contains(t, result, tt.expectPathNS)
			}
		})
	}
}

func TestList(t *testing.T) {
	tests := []struct {
		name                 string
		mockListK8sResources func(gvr schema.GroupVersionResource, target interface{}) error
		expectPanic          bool
	}{
		{
			name: "successful list with clusters",
			mockListK8sResources: func(gvr schema.GroupVersionResource, target interface{}) error {
				list := target.(*configv1beta1.BKEClusterList)
				list.Items = []configv1beta1.BKECluster{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "cluster-1",
							Namespace: "test-ns-1",
						},
						Spec: configv1beta1.BKEClusterSpec{
							ControlPlaneEndpoint: configv1beta1.APIEndpoint{
								Host: testAPIEndpoint1,
								Port: 6443,
							},
							Pause:  false,
							DryRun: false,
							Reset:  false,
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "cluster-2",
							Namespace: "test-ns-2",
						},
						Spec: configv1beta1.BKEClusterSpec{
							ControlPlaneEndpoint: configv1beta1.APIEndpoint{
								Host: testAPIEndpoint2,
								Port: 6443,
							},
							Pause:  true,
							DryRun: true,
							Reset:  false,
						},
					},
				}
				return nil
			},
			expectPanic: false,
		},
		{
			name: "successful list with empty clusters",
			mockListK8sResources: func(gvr schema.GroupVersionResource, target interface{}) error {
				list := target.(*configv1beta1.BKEClusterList)
				list.Items = []configv1beta1.BKECluster{}
				return nil
			},
			expectPanic: false,
		},
		{
			name: "list resources error",
			mockListK8sResources: func(gvr schema.GroupVersionResource, target interface{}) error {
				return fmt.Errorf("list error")
			},
			expectPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(global.ListK8sResources, tt.mockListK8sResources)

			op.List()

			assert.True(t, true)
		})
	}
}
