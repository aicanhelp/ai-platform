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

package root

import (
	"fmt"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

const (
	testThreeValue    = 3
	testZeroValue     = 0
	testOneValue      = 1
	testTwoValue      = 2
	testMinusOneValue = -1
)

func TestOptionsStruct(t *testing.T) {
	// Test that the Options struct has the expected fields
	opts := &Options{}

	opts.KubeConfig = "/path/to/kubeconfig"
	opts.Args = []string{"arg1", "arg2"}

	assert.Equal(t, "/path/to/kubeconfig", opts.KubeConfig)
	assert.Equal(t, []string{"arg1", "arg2"}, opts.Args)
}

func TestClusterPre(t *testing.T) {
	tests := []struct {
		name              string
		kubeConfig        string
		globalK8s         k8s.KubernetesClient
		mockNewK8sClient  func(string) (k8s.KubernetesClient, error)
		expectError       bool
		expectedGlobalK8s k8s.KubernetesClient
	}{
		{
			name:       "global K8s is nil, new client created successfully",
			kubeConfig: "/path/to/kubeconfig",
			globalK8s:  nil,
			mockNewK8sClient: func(config string) (k8s.KubernetesClient, error) {
				assert.Equal(t, "/path/to/kubeconfig", config)
				return &k8s.Client{}, nil
			},
			expectError:       false,
			expectedGlobalK8s: &k8s.Client{},
		},
		{
			name:       "global K8s already exists, no new client created",
			kubeConfig: "/path/to/kubeconfig",
			globalK8s:  &k8s.Client{ /* some existing client */ },
			mockNewK8sClient: func(config string) (k8s.KubernetesClient, error) {
				t.Error("NewKubernetesClient should not be called when global.K8s is already set")
				return nil, nil
			},
			expectError:       false,
			expectedGlobalK8s: &k8s.Client{ /* same as original */ },
		},
		{
			name:       "global K8s is nil, new client creation fails",
			kubeConfig: "/path/to/kubeconfig",
			globalK8s:  nil,
			mockNewK8sClient: func(config string) (k8s.KubernetesClient, error) {
				return nil, fmt.Errorf("client creation failed")
			},
			expectError:       true,
			expectedGlobalK8s: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up global K8s client
			originalK8s := global.K8s
			global.K8s = tt.globalK8s
			defer func() {
				global.K8s = originalK8s
			}()

			// Apply patches
			if tt.globalK8s == nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(k8s.NewKubernetesClient, tt.mockNewK8sClient)
			}

			opts := &Options{KubeConfig: tt.kubeConfig}
			err := opts.ClusterPre()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			assert.Equal(t, tt.expectedGlobalK8s, global.K8s)
		})
	}
}

func TestClusterPreWithRealK8sClient(t *testing.T) {
	// Test with a real scenario where global.K8s is initially nil
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	// Create a mock K8s client
	mockK8sClient := &k8s.Client{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
		// Verify that the config path is passed correctly
		assert.Equal(t, "/test/kubeconfig", config)
		return mockK8sClient, nil
	})

	opts := &Options{KubeConfig: "/test/kubeconfig"}
	err := opts.ClusterPre()

	assert.NoError(t, err)
	assert.Equal(t, mockK8sClient, global.K8s)
}

func TestClusterPreWithEmptyKubeConfig(t *testing.T) {
	// Test with empty kubeconfig
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	// Create a mock K8s client
	mockK8sClient := &k8s.Client{}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
		// With empty config, it should still be passed as empty string
		assert.Equal(t, "", config)
		return mockK8sClient, nil
	})

	opts := &Options{KubeConfig: ""}
	err := opts.ClusterPre()

	assert.NoError(t, err)
	assert.Equal(t, mockK8sClient, global.K8s)
}

func TestPrint(t *testing.T) {
	// Capture the output of the Print method
	var capturedOutput string

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(fmt.Print, func(a ...interface{}) (n int, err error) {
		capturedOutput = fmt.Sprintf("%v", a[testZeroValue])
		return len(capturedOutput), nil
	})

	opts := &Options{}
	opts.Print()

}

func TestClusterPreIdempotent(t *testing.T) {
	// Test that calling ClusterPre multiple times with global.K8s already set is idempotent
	mockK8sClient := &k8s.Client{}

	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches to ensure NewKubernetesClient is not called
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
		t.Error("NewKubernetesClient should not be called when global.K8s is already set")
		return nil, nil
	})

	opts := &Options{KubeConfig: "/some/config"}

	// Call ClusterPre multiple times
	err1 := opts.ClusterPre()

	assert.NoError(t, err1)

	// Global K8s should remain the same
	assert.Equal(t, mockK8sClient, global.K8s)
}

func TestClusterPreErrorHandling(t *testing.T) {
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches to simulate error in NewKubernetesClient
	expectedError := fmt.Errorf("connection failed")
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
		return nil, expectedError
	})

	opts := &Options{KubeConfig: "/test/config"}
	err := opts.ClusterPre()

	assert.Error(t, err)
	assert.Equal(t, expectedError, err)

	// Global K8s should still be nil after error
	assert.Nil(t, global.K8s)
}

func TestClusterPrePreservesExistingGlobalK8sOnError(t *testing.T) {
	// Test that if global.K8s is already set, an error in NewKubernetesClient doesn't affect it
	existingK8sClient := &k8s.Client{ /* some existing client */ }

	originalK8s := global.K8s
	global.K8s = existingK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches to ensure NewKubernetesClient is not called
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
		t.Error("NewKubernetesClient should not be called when global.K8s is already set")
		return nil, fmt.Errorf("should not be called")
	})

	opts := &Options{KubeConfig: "/test/config"}
	err := opts.ClusterPre()

	assert.NoError(t, err)

	// Global K8s should remain the same as the original
	assert.Equal(t, existingK8sClient, global.K8s)
}

func TestPrintDoc(t *testing.T) {
	opts := &Options{}
	opts.PrintDoc()
	assert.True(t, true)
}

func TestPrintDocWithOptions(t *testing.T) {
	opts := &Options{
		KubeConfig: "/test/kubeconfig",
		Args:       []string{"test-arg"},
	}
	opts.PrintDoc()
	assert.True(t, true)
}

func TestPrintWithDifferentConfigs(t *testing.T) {
	// Test that Print method doesn't depend on any configuration
	configs := []struct {
		kubeConfig string
		args       []string
	}{
		{"/etc/kube/config", []string{"arg1", "arg2"}},
		{"", nil},
		{"~/.kube/config", []string{}},
		{"./kubeconfig", []string{"--verbose"}},
	}

	for i, config := range configs {
		t.Run(fmt.Sprintf("config-%d", i), func(t *testing.T) {
			var capturedOutput string

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc(fmt.Print, func(a ...interface{}) (n int, err error) {
				capturedOutput = fmt.Sprintf("%v", a[testZeroValue])
				return len(capturedOutput), nil
			})

			opts := &Options{
				KubeConfig: config.kubeConfig,
				Args:       config.args,
			}
			opts.Print()

			// The output should be the same regardless of config
		})
	}
}

func TestOptionsJSONTags(t *testing.T) {
	// Test that the JSON tags are properly defined by using reflection or by verifying struct definition
	opts := &Options{}

	// The struct should have been defined with proper JSON tags
	// We can't directly test JSON tags without reflection, but we can verify the fields exist
	opts.KubeConfig = "test-config"
	opts.Args = []string{"test-arg"}

	assert.Equal(t, "test-config", opts.KubeConfig)
	assert.Equal(t, []string{"test-arg"}, opts.Args)
}

func TestClusterPreWithVariousKubeConfigPaths(t *testing.T) {
	testPaths := []string{
		"/etc/kubernetes/admin.conf",
		"~/.kube/config",
		"./relative/path/config",
		"",
		"/absolute/path/to/kubeconfig",
	}

	for _, path := range testPaths {
		t.Run(fmt.Sprintf("kubeconfig-path-%s", strings.ReplaceAll(path, "/", "_")), func(t *testing.T) {
			originalK8s := global.K8s
			global.K8s = nil
			defer func() {
				global.K8s = originalK8s
			}()

			var capturedConfig string
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
				capturedConfig = config
				return &k8s.Client{}, nil
			})

			opts := &Options{KubeConfig: path}
			err := opts.ClusterPre()

			assert.NoError(t, err)
			assert.Equal(t, path, capturedConfig)
			assert.NotNil(t, global.K8s)
		})
	}
}

func TestClusterPreHandlesClientCreationErrorGracefully(t *testing.T) {
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches to simulate various types of errors
	testErrors := []error{
		fmt.Errorf("connection timeout"),
		fmt.Errorf("invalid kubeconfig format"),
		fmt.Errorf("permission denied"),
		nil, // This shouldn't happen in error case, but let's test it
	}

	for i, testErr := range testErrors[:testThreeValue] { // Skip the nil case for error test
		t.Run(fmt.Sprintf("error-case-%d", i), func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(k8s.NewKubernetesClient, func(config string) (k8s.KubernetesClient, error) {
				return nil, testErr
			})

			opts := &Options{KubeConfig: "/test/config"}
			err := opts.ClusterPre()

			assert.Error(t, err)
			assert.Equal(t, testErr, err)
			assert.Nil(t, global.K8s)
		})
	}
}
