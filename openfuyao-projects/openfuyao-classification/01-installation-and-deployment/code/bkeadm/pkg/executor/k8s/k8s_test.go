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

package k8s

import (
	"errors"
	"net"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testNumericZero         = 0
	testNumericOne          = 1
	testNumericTwo          = 2
	testNumericThree        = 3
	testNamespaceLen        = 8
	testKubeconfigLen       = 20
	testNamespaceNameLen    = 10
	testNumericRestartCount = 0

	testIPv4SegmentA = 127
	testIPv4SegmentB = 0
	testIPv4SegmentC = 0
	testIPv4SegmentD = 1
)

var (
	testLoopbackIP = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentD,
	).String()
)

func TestNewKubernetesClient(t *testing.T) {
	tests := []struct {
		name           string
		kubeConfig     string
		fileExists     bool
		buildConfigErr error
		newClientErr   error
		expectedError  bool
	}{
		{
			name:           "kubeconfig does not exist and no home dir",
			kubeConfig:     "",
			fileExists:     false,
			buildConfigErr: nil,
			newClientErr:   nil,
			expectedError:  true,
		},
		{
			name:           "kubeconfig exists but build config fails",
			kubeConfig:     "/tmp/test-kubeconfig",
			fileExists:     true,
			buildConfigErr: errors.New("build config error"),
			newClientErr:   nil,
			expectedError:  true,
		},
		{
			name:           "kubeconfig path provided but file does not exist",
			kubeConfig:     "/nonexistent/path/config",
			fileExists:     false,
			buildConfigErr: nil,
			newClientErr:   nil,
			expectedError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				if tt.kubeConfig != "" && path == tt.kubeConfig {
					return tt.fileExists
				}
				return false
			})

			patches.ApplyFunc(clientcmd.BuildConfigFromFlags, func(masterUrl string, kubeconfigPath string) (interface{}, error) {
				return nil, tt.buildConfigErr
			})

			patches.ApplyFunc(kubernetes.NewForConfig, func(config interface{}) (*kubernetes.Clientset, error) {
				return nil, tt.newClientErr
			})

			client, err := NewKubernetesClient(tt.kubeConfig)
			if tt.expectedError {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}

func TestGetClient(t *testing.T) {
	t.Run("get client returns the underlying clientset", func(t *testing.T) {
		fakeClient := fake.NewSimpleClientset()
		mockClient := &MockK8sClient{}

		patches := gomonkey.NewPatches()

		defer patches.Reset()

		patches.ApplyFunc((*MockK8sClient).GetClient,
			func(_ *MockK8sClient) kubernetes.Interface {
				return fakeClient
			})

		result := mockClient.GetClient()
		assert.NotNil(t, result)
	})
}

func TestGetDynamicClient(t *testing.T) {
	t.Run("get dynamic client returns the underlying dynamic client", func(t *testing.T) {
		mockClient := &MockK8sClient{}

		result := mockClient.GetDynamicClient()
		assert.NotNil(t, result)
	})
}

func TestClientStruct(t *testing.T) {
	t.Run("client struct fields are properly initialized", func(t *testing.T) {
		mockClient := &MockK8sClient{}

		assert.NotNil(t, mockClient.GetClient())
		assert.NotNil(t, mockClient.GetDynamicClient())
	})
}

func TestDetermineNamespace(t *testing.T) {
	tests := []struct {
		name       string
		namespace  string
		unstructNs string
		expectedNs string
	}{
		{
			name:       "namespace parameter is provided",
			namespace:  "test-namespace",
			unstructNs: "other-namespace",
			expectedNs: "test-namespace",
		},
		{
			name:       "namespace parameter is empty, use unstructured namespace",
			namespace:  "",
			unstructNs: "unstruct-namespace",
			expectedNs: "unstruct-namespace",
		},
		{
			name:       "both are empty",
			namespace:  "",
			unstructNs: "",
			expectedNs: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unstruct := &unstructured.Unstructured{}
			unstruct.SetNamespace(tt.unstructNs)

			result := determineNamespaceMock(unstruct, tt.namespace)
			assert.Equal(t, tt.expectedNs, result)
		})
	}
}

func determineNamespaceMock(unstruct *unstructured.Unstructured, ns string) string {
	if ns != "" {
		return ns
	}
	return unstruct.GetNamespace()
}

func TestCreateNamespace(t *testing.T) {
	tests := []struct {
		name          string
		namespace     *corev1.Namespace
		expectedError bool
	}{
		{
			name: "namespace creation succeeds",
			namespace: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-namespace",
				},
			},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*MockK8sClient).CreateNamespace,
				func(_ *MockK8sClient, _ *corev1.Namespace) error {
					return nil
				})

			err := mockClient.CreateNamespace(tt.namespace)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCreateSecret(t *testing.T) {
	tests := []struct {
		name          string
		secret        *corev1.Secret
		expectedError bool
	}{
		{
			name: "secret creation succeeds",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-secret",
					Namespace: "default",
				},
			},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*MockK8sClient).CreateSecret,
				func(_ *MockK8sClient, _ *corev1.Secret) error {
					return nil
				})

			err := mockClient.CreateSecret(tt.secret)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNamespaceRefStruct(t *testing.T) {
	t.Run("namespace ref struct initialization", func(t *testing.T) {
		namespace := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-namespace",
				Namespace: "",
			},
			Spec:   corev1.NamespaceSpec{},
			Status: corev1.NamespaceStatus{},
		}

		assert.Equal(t, "test-namespace", namespace.Name)
		assert.NotNil(t, namespace.Spec)
		assert.NotNil(t, namespace.Status)
	})
}

func TestSecretRefStruct(t *testing.T) {
	t.Run("secret ref struct initialization", func(t *testing.T) {
		secret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-secret",
				Namespace: "default",
			},
			Type: corev1.SecretTypeOpaque,
			Data: map[string][]byte{
				"key": []byte("value"),
			},
		}

		assert.Equal(t, "test-secret", secret.Name)
		assert.Equal(t, "default", secret.Namespace)
		assert.Equal(t, corev1.SecretTypeOpaque, secret.Type)
	})
}

func TestPackageCreateNamespace(t *testing.T) {
	tests := []struct {
		name          string
		namespaceName string
		createErr     error
		expectedError bool
	}{
		{
			name:          "create namespace successfully",
			namespaceName: "test-namespace",
			createErr:     nil,
			expectedError: false,
		},
		{
			name:          "create namespace fails",
			namespaceName: "fail-namespace",
			createErr:     errors.New("create error"),
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*MockK8sClient).CreateNamespace,
				func(_ *MockK8sClient, _ *corev1.Namespace) error {
					return tt.createErr
				})

			err := CreateNamespace(mockClient, tt.namespaceName)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWatchEventByAnnotation(t *testing.T) {
	tests := []struct {
		name string
		ns   string
	}{
		{
			name: "watch event with default namespace",
			ns:   "default",
		},
		{
			name: "watch event with kube-system namespace",
			ns:   "kube-system",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			assert.NotPanics(t, func() {
				mockClient.WatchEventByAnnotation(tt.ns)
			})
		})
	}
}

func TestKubernetesClientInterface(t *testing.T) {
	t.Run("kubernetes client interface can be used with mock client", func(t *testing.T) {
		var client KubernetesClient
		client = &MockK8sClient{}

		assert.NotNil(t, client.GetClient())
		assert.NotNil(t, client.GetDynamicClient())
	})
}

func TestMockK8sClientMethods(t *testing.T) {
	t.Run("mock k8s client returns nil for all interface methods", func(t *testing.T) {
		mockClient := &MockK8sClient{}

		assert.NotNil(t, mockClient.GetClient())
		assert.NotNil(t, mockClient.GetDynamicClient())

		assert.NoError(t, mockClient.InstallYaml("test.yaml", nil, "test"))
		assert.NoError(t, mockClient.PatchYaml("test.yaml", nil))
		assert.NoError(t, mockClient.UninstallYaml("test.yaml", "test"))
		assert.NoError(t, mockClient.CreateNamespace(&corev1.Namespace{}))
		assert.NoError(t, mockClient.CreateSecret(&corev1.Secret{}))

		ns, _ := mockClient.GetNamespace("test.yaml")
		assert.Equal(t, "", ns)
	})
}

func TestGetNamespace(t *testing.T) {
	tests := []struct {
		name          string
		namespace     string
		expectedError bool
	}{
		{
			name:          "get namespace returns empty string",
			namespace:     "",
			expectedError: false,
		},
		{
			name:          "get namespace returns namespace name",
			namespace:     "test-namespace",
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			ns, err := mockClient.GetNamespace("/tmp/test.yaml")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, ns)
			}
		})
	}
}

func TestInstallYaml(t *testing.T) {
	tests := []struct {
		name          string
		variables     map[string]string
		expectedError bool
	}{
		{
			name: "install yaml with template rendering",
			variables: map[string]string{
				"Name": "test-namespace",
			},
			expectedError: false,
		},
		{
			name:          "install yaml with empty variables",
			variables:     map[string]string{},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			tmpFile, err := os.CreateTemp("", "test-template-*.yaml")
			assert.NoError(t, err)
			defer os.Remove(tmpFile.Name())

			_, err = tmpFile.WriteString(`apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Name }}
`)
			assert.NoError(t, err)
			tmpFile.Close()

			err = mockClient.InstallYaml(tmpFile.Name(), tt.variables, "test-namespace")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPatchYaml(t *testing.T) {
	tests := []struct {
		name          string
		variables     map[string]string
		expectedError bool
	}{
		{
			name: "patch yaml with template rendering",
			variables: map[string]string{
				"Key": "value",
			},
			expectedError: false,
		},
		{
			name:          "patch yaml with empty variables",
			variables:     map[string]string{},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			tmpFile, err := os.CreateTemp("", "test-template-*.yaml")
			assert.NoError(t, err)
			defer os.Remove(tmpFile.Name())

			_, err = tmpFile.WriteString(`apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
`)
			assert.NoError(t, err)
			tmpFile.Close()

			err = mockClient.PatchYaml(tmpFile.Name(), tt.variables)
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUninstallYaml(t *testing.T) {
	tests := []struct {
		name          string
		yamlContent   string
		expectedError bool
	}{
		{
			name: "uninstall yaml processes resources",
			yamlContent: `apiVersion: v1
kind: Namespace
metadata:
  name: test-namespace
`,
			expectedError: false,
		},
		{
			name:          "uninstall yaml with empty content",
			yamlContent:   "",
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockK8sClient{}

			tmpFile, err := os.CreateTemp("", "test-*.yaml")
			assert.NoError(t, err)
			defer os.Remove(tmpFile.Name())

			_, err = tmpFile.WriteString(tt.yamlContent)
			assert.NoError(t, err)
			tmpFile.Close()

			err = mockClient.UninstallYaml(tmpFile.Name(), "")
			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRenderTemplateToTempFile(t *testing.T) {
	tests := []struct {
		name            string
		templateContent string
		variables       map[string]string
		expectedFile    bool
		expectedError   bool
	}{
		{
			name: "render template successfully",
			templateContent: `apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Name }}
`,
			variables: map[string]string{
				"Name": "test-namespace",
			},
			expectedFile:  true,
			expectedError: false,
		},
		{
			name:            "render template with empty variables",
			templateContent: "name: test",
			variables:       map[string]string{},
			expectedFile:    true,
			expectedError:   false,
		},
		{
			name:            "render non-existent file",
			templateContent: "",
			variables:       nil,
			expectedFile:    false,
			expectedError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.templateContent != "" {
				tmpFile, err := os.CreateTemp("", "test-template-*.yaml")
				assert.NoError(t, err)
				defer os.Remove(tmpFile.Name())

				_, err = tmpFile.WriteString(tt.templateContent)
				assert.NoError(t, err)
				tmpFile.Close()

				result, cleanup, err := renderTemplateToTempFile(tmpFile.Name(), tt.variables)
				defer cleanup()

				if tt.expectedError {
					assert.Error(t, err)
					assert.Equal(t, "", result)
				} else {
					assert.NoError(t, err)
					assert.NotNil(t, cleanup)
					if tt.expectedFile {
						assert.NotEmpty(t, result)
						_, err := os.Stat(result)
						assert.NoError(t, err)
					}
				}
			} else {
				result, cleanup, err := renderTemplateToTempFile("/non/existent/file.yaml", tt.variables)
				if cleanup != nil {
					defer cleanup()
				}

				assert.Error(t, err)
				assert.Equal(t, "", result)
			}
		})
	}
}

func TestProcessYamlResources(t *testing.T) {
	t.Run("process yaml resources with valid namespace", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "namespaces",
								Kind:    "Namespace",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-yaml-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: Namespace
metadata:
  name: test-namespace
`)
		assert.NoError(t, err)
		tmpFile.Close()

		err = c.processYamlResources(tmpFile.Name(), func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
			assert.NotNil(t, unstruct)
			assert.NotNil(t, mapping)
			return nil
		})
		assert.NoError(t, err)
	})

	t.Run("process yaml resources with empty file", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-empty-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())
		tmpFile.Close()

		err = c.processYamlResources(tmpFile.Name(), func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
			return nil
		})
		assert.NoError(t, err)
	})

	t.Run("process yaml resources with non-existent file", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return nil, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		err := c.processYamlResources("/non/existent/file.yaml", func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
			return nil
		})
		assert.Error(t, err)
	})
}

func TestCreateResource(t *testing.T) {
	t.Run("create resource with nil dynamic client panics", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "namespaces",
								Kind:    "Namespace",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		unstruct := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Namespace",
				"metadata": map[string]interface{}{
					"name":      "test-namespace",
					"namespace": "",
				},
			},
		}

		mapping := &meta.RESTMapping{
			Resource: schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "namespaces",
			},
		}

		assert.Panics(t, func() {
			_ = c.createResource(unstruct, mapping, "")
		})
	})

	t.Run("create resource with namespace and nil dynamic client panics", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "configmaps",
								Kind:    "ConfigMap",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		unstruct := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ConfigMap",
				"metadata": map[string]interface{}{
					"name":      "test-config",
					"namespace": "default",
				},
			},
		}

		mapping := &meta.RESTMapping{
			Resource: schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "configmaps",
			},
		}

		assert.Panics(t, func() {
			_ = c.createResource(unstruct, mapping, "test-namespace")
		})
	})
}

func TestDeleteResource(t *testing.T) {
	t.Run("delete resource with nil dynamic client panics", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "namespaces",
								Kind:    "Namespace",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		unstruct := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Namespace",
				"metadata": map[string]interface{}{
					"name":      "test-namespace",
					"namespace": "",
				},
			},
		}

		mapping := &meta.RESTMapping{
			Resource: schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "namespaces",
			},
		}

		assert.Panics(t, func() {
			_ = c.deleteResource(unstruct, mapping, "")
		})
	})

	t.Run("delete resource with namespace and nil dynamic client panics", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "configmaps",
								Kind:    "ConfigMap",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		unstruct := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ConfigMap",
				"metadata": map[string]interface{}{
					"name":      "test-config",
					"namespace": "default",
				},
			},
		}

		mapping := &meta.RESTMapping{
			Resource: schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "configmaps",
			},
		}

		assert.Panics(t, func() {
			_ = c.deleteResource(unstruct, mapping, "test-namespace")
		})
	})
}

func TestDetermineNamespaceWithClient(t *testing.T) {
	tests := []struct {
		name       string
		namespace  string
		unstructNs string
		expectedNs string
	}{
		{
			name:       "namespace parameter is provided",
			namespace:  "test-namespace",
			unstructNs: "other-namespace",
			expectedNs: "test-namespace",
		},
		{
			name:       "namespace parameter is empty, use unstructured namespace",
			namespace:  "",
			unstructNs: "unstruct-namespace",
			expectedNs: "unstruct-namespace",
		},
		{
			name:       "both are empty",
			namespace:  "",
			unstructNs: "",
			expectedNs: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
				return &Client{
					ClientSet:     &kubernetes.Clientset{},
					DynamicClient: nil,
				}, nil
			})

			client, _ := NewKubernetesClient("")
			c := client.(*Client)

			unstruct := &unstructured.Unstructured{}
			unstruct.SetNamespace(tt.unstructNs)

			result := c.determineNamespace(unstruct, tt.namespace)
			assert.Equal(t, tt.expectedNs, result)
		})
	}
}

func TestInstallYamlWithClient(t *testing.T) {
	t.Run("install yaml with client panics due to nil dynamic client", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "namespaces",
								Kind:    "Namespace",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-install-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Name }}
`)
		assert.NoError(t, err)
		tmpFile.Close()

		assert.Panics(t, func() {
			_ = c.InstallYaml(tmpFile.Name(), map[string]string{"Name": "test-ns"}, "test-namespace")
		})
	})
}

func TestPatchYamlWithClient(t *testing.T) {
	t.Run("patch yaml with client panics due to nil dynamic client", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "configmaps",
								Kind:    "ConfigMap",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-patch-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
`)
		assert.NoError(t, err)
		tmpFile.Close()

		assert.Panics(t, func() {
			_ = c.PatchYaml(tmpFile.Name(), map[string]string{"Key": "value"})
		})
	})
}

func TestUninstallYamlWithClient(t *testing.T) {
	t.Run("uninstall yaml with client panics due to nil dynamic client", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
			return []*restmapper.APIGroupResources{
				{
					Group: metav1.APIGroup{
						Name: "",
						Versions: []metav1.GroupVersionForDiscovery{
							{Version: "v1"},
						},
					},
					VersionedResources: map[string][]metav1.APIResource{
						"v1": {
							{
								Name:    "namespaces",
								Kind:    "Namespace",
								Version: "v1",
							},
						},
					},
				},
			}, nil
		})

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-uninstall-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: Namespace
metadata:
  name: test-namespace
`)
		assert.NoError(t, err)
		tmpFile.Close()

		assert.Panics(t, func() {
			_ = c.UninstallYaml(tmpFile.Name(), "")
		})
	})
}

func TestGetNamespaceWithClient(t *testing.T) {
	t.Run("get namespace from yaml file", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-getns-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: Namespace
metadata:
  name: my-test-namespace
`)
		assert.NoError(t, err)
		tmpFile.Close()

		ns, err := c.GetNamespace(tmpFile.Name())
		assert.NoError(t, err)
		assert.Equal(t, "my-test-namespace", ns)
	})

	t.Run("get namespace from yaml without namespace resource", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(NewKubernetesClient, func(kubeConfig string) (KubernetesClient, error) {
			return &Client{
				ClientSet:     &kubernetes.Clientset{},
				DynamicClient: nil,
			}, nil
		})

		client, _ := NewKubernetesClient("")
		c := client.(*Client)

		tmpFile, err := os.CreateTemp("", "test-no-ns-*.yaml")
		assert.NoError(t, err)
		defer os.Remove(tmpFile.Name())

		_, err = tmpFile.WriteString(`apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
`)
		assert.NoError(t, err)
		tmpFile.Close()

		ns, err := c.GetNamespace(tmpFile.Name())
		assert.NoError(t, err)
		assert.Equal(t, "", ns)
	})
}

// TestGetRESTMapperCachesDiscovery verifies the core fix for sig-installation#282:
// API discovery must happen once and be reused across processYamlResources calls,
// not rebuilt on every call (which caused 65 full discovery passes -> ~9 min of
// client-side throttling when installing 64 nodes).
func TestGetRESTMapperCachesDiscovery(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	discoveryCalls := 0
	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
		discoveryCalls++
		return []*restmapper.APIGroupResources{
			{
				Group: metav1.APIGroup{
					Name: "",
					Versions: []metav1.GroupVersionForDiscovery{
						{Version: "v1"},
					},
				},
				VersionedResources: map[string][]metav1.APIResource{
					"v1": {
						{
							Name:    "namespaces",
							Kind:    "Namespace",
							Version: "v1",
						},
					},
				},
			},
		}, nil
	})

	c := &Client{ClientSet: &kubernetes.Clientset{}}

	tmpFile, err := os.CreateTemp("", "test-cache-*.yaml")
	assert.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	_, err = tmpFile.WriteString("apiVersion: v1\nkind: Namespace\nmetadata:\n  name: cached-ns\n")
	assert.NoError(t, err)
	tmpFile.Close()

	handler := func(*unstructured.Unstructured, *meta.RESTMapping) error { return nil }

	// Two processYamlResources calls on the same Client must share one discovery.
	// Before the fix this would call GetAPIGroupResources twice (one per call);
	// at 64 nodes it became 65 calls.
	err = c.processYamlResources(tmpFile.Name(), handler)
	assert.NoError(t, err)
	err = c.processYamlResources(tmpFile.Name(), handler)
	assert.NoError(t, err)

	assert.Equal(t, 1, discoveryCalls, "discovery should be cached across processYamlResources calls")
	assert.NotNil(t, c.restMapper, "RESTMapper should be cached on the Client")
}

// TestNewKubernetesClientSetsRateLimit verifies the bkeadm client overrides
// client-go's low default rate limit so large-cluster install/scale does not
// hit client-side throttling. See sig-installation#282.
func TestNewKubernetesClientSetsRateLimit(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(string) bool { return true })

	patches.ApplyFunc(clientcmd.BuildConfigFromFlags, func(string, string) (*rest.Config, error) {
		return &rest.Config{Host: "https://127.0.0.1:6443"}, nil
	})

	var capturedConfig *rest.Config
	patches.ApplyFunc(kubernetes.NewForConfig, func(cfg *rest.Config) (*kubernetes.Clientset, error) {
		capturedConfig = cfg
		return &kubernetes.Clientset{}, nil
	})

	patches.ApplyFunc(dynamic.NewForConfig, func(*rest.Config) (dynamic.Interface, error) {
		return nil, nil
	})

	client, err := NewKubernetesClient("/tmp/any-kubeconfig")
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.NotNil(t, capturedConfig)
	assert.Equal(t, float32(clientQPS), capturedConfig.QPS)
	assert.Equal(t, clientBurst, capturedConfig.Burst)
}

// TestProcessYamlResourcesNoMatchRetry verifies the correctness safety net for
// sig-installation#282: when a CRD is applied earlier in the same run and is not
// yet in the cached discovery, RESTMapping returns a NoMatchError and the code
// must invalidate the cache, re-discover once, and retry — otherwise CRD-based
// installs (e.g. containerd/kubelet CRD + its instance) would fail.
func TestProcessYamlResourcesNoMatchRetry(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// Simulate a CRD appearing mid-run: first discovery pass lacks the target
	// kind, the second (after invalidate+Reset) includes it.
	discoveryCalls := 0
	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(discovery.DiscoveryInterface) ([]*restmapper.APIGroupResources, error) {
		discoveryCalls++
		// Group "example.com" with the Crontab kind only appears from the 2nd call.
		resources := []*restmapper.APIGroupResources{
			{
				Group: metav1.APIGroup{
					Name: "",
					Versions: []metav1.GroupVersionForDiscovery{
						{Version: "v1"},
					},
				},
				VersionedResources: map[string][]metav1.APIResource{
					"v1": {
						{Name: "namespaces", Kind: "Namespace", Version: "v1"},
					},
				},
			},
		}
		if discoveryCalls >= 2 {
			resources = append(resources, &restmapper.APIGroupResources{
				Group: metav1.APIGroup{
					Name: "example.com",
					Versions: []metav1.GroupVersionForDiscovery{
						{GroupVersion: "example.com/v1", Version: "v1"},
					},
				},
				VersionedResources: map[string][]metav1.APIResource{
					"v1": {
						{Name: "crontabs", Kind: "Crontab", Version: "v1", Namespaced: true},
					},
				},
			})
		}
		return resources, nil
	})

	c := &Client{ClientSet: &kubernetes.Clientset{}}

	tmpFile, err := os.CreateTemp("", "test-nomatch-*.yaml")
	assert.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	_, err = tmpFile.WriteString("apiVersion: example.com/v1\nkind: Crontab\nmetadata:\n  name: my-cron\n  namespace: default\n")
	assert.NoError(t, err)
	tmpFile.Close()

	handlerCalled := false
	err = c.processYamlResources(tmpFile.Name(), func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
		handlerCalled = true
		assert.Equal(t, "crontabs", mapping.Resource.Resource)
		return nil
	})

	// The NoMatchError retry path must have fired: discovery ran twice (initial
	// miss + post-invalidate re-discovery) and the handler received the mapping.
	assert.NoError(t, err)
	assert.True(t, handlerCalled, "handler should be called after the NoMatchError retry")
	assert.Equal(t, 2, discoveryCalls, "discovery should run once initially and once after invalidation")
}
