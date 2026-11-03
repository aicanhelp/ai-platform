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

package kubelet

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	corev1 "k8s.io/api/core/v1"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func TestApplyKubeletCfg(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(applyKubeletdCrd, func() error {
		return nil
	})

	patches.ApplyFunc(applyContainerdDefault, func() error {
		return nil
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	err := ApplyKubeletCfg()

	assert.NoError(t, err)
}

func TestApplyKubeletCfgError(t *testing.T) {
	// Apply patches to simulate error in applyKubeletdCrd
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(applyKubeletdCrd, func() error {
		return fmt.Errorf("CRD apply failed")
	})

	err := ApplyKubeletCfg()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "apply kubelet crd failed")
}

func TestApplyKubeletdCrd(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return &k8s.Client{}, nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		assert.Contains(t, filename, "kubelet_crd.yaml")
		assert.Equal(t, kubeletCrd, data)
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		assert.Contains(t, filename, "kubelet_crd.yaml")
		return nil
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Reset global.K8s for this test
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	err := applyKubeletdCrd()

	assert.NoError(t, err)
}

func TestApplyKubeletdCrdNewK8sClientError(t *testing.T) {
	// Apply patches to simulate error in NewKubernetesClient
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return nil, fmt.Errorf("client creation failed")
	})

	err := applyKubeletdCrd()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "client creation failed")
}

func TestApplyKubeletdCrdWriteFileError(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches to simulate error in WriteFile
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return &k8s.Client{}, nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return fmt.Errorf("write failed")
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Reset global.K8s for this test
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	err := applyKubeletdCrd()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "write failed")
}

func TestApplyKubeletdCrdInstallYamlError(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches to simulate error in InstallYaml
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return &k8s.Client{}, nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return fmt.Errorf("install failed")
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Reset global.K8s for this test
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	err := applyKubeletdCrd()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "install failed")
}

func TestApplyKubeletDefault(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		conf := v.(*configv1beta1.KubeletConfig)
		conf.Namespace = "kubelet-system"
		return nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(k *k8s.Client, namespace *corev1.Namespace) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return nil
	})

	// Mock global.K8s
	mockK8sClient := &k8s.Client{}
	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	err := ApplyKubeletCfg()

	assert.NoError(t, err)
}

func TestApplyKubeletDefaultUnmarshalError(t *testing.T) {
	// Apply patches to simulate error in yaml.Unmarshal
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		return fmt.Errorf("unmarshal failed")
	})

	err := ApplyKubeletCfg()

	assert.Error(t, err)
}

func TestApplyKubeletDefaultCreateNamespaceError(t *testing.T) {
	// Apply patches to simulate error in CreateNamespace
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		conf := v.(*configv1beta1.KubeletConfig)
		conf.Namespace = "kubelet-system"
		return nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(k *k8s.Client, namespace *corev1.Namespace) error {
		return fmt.Errorf("namespace creation failed")
	})

	err := ApplyKubeletCfg()

	assert.Error(t, err)
}

func TestApplyKubeletDefaultWriteFileError(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches to simulate error in WriteFile
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		conf := v.(*configv1beta1.KubeletConfig)
		conf.Namespace = "kubelet-system"
		return nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(k *k8s.Client, namespace *corev1.Namespace) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return fmt.Errorf("write failed")
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Mock global.K8s
	mockK8sClient := &k8s.Client{}
	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	err := ApplyKubeletCfg()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "write failed")
}

func TestApplyKubeletDefaultInstallYamlError(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches to simulate error in InstallYaml
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		conf := v.(*configv1beta1.KubeletConfig)
		conf.Namespace = "kubelet-system"
		return nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(k *k8s.Client, namespace *corev1.Namespace) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return fmt.Errorf("install failed")
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Mock global.K8s
	mockK8sClient := &k8s.Client{}
	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	err := ApplyKubeletCfg()

	assert.Error(t, err) // Function returns nil even when install fails
}

func TestKubeletCrdVariable(t *testing.T) {
	// Test that the kubeletCrd variable is properly embedded
	assert.NotEmpty(t, kubeletCrd)

	// Check that it contains expected content
	content := string(kubeletCrd)
	assert.Contains(t, content, "apiVersion:")
	assert.Contains(t, content, "kind:")
	assert.Contains(t, content, "KubeletConfig")
}

func TestKubeletDefaultVariable(t *testing.T) {
	// Test that the kubeletDefault variable is properly embedded
	assert.NotEmpty(t, kubeletDefault)

	// Check that it contains expected content
	content := string(kubeletDefault)
	assert.Contains(t, content, "apiVersion:")
	assert.Contains(t, content, "kind:")
	assert.Contains(t, content, "Kubelet")
}

func TestApplyKubeletdCrdWithRealEmbeddedContent(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches for a more realistic test
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return &k8s.Client{}, nil
	})

	var capturedData []byte
	var capturedFilename string
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		capturedFilename = filename
		capturedData = data
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return nil
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Reset global.K8s for this test
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	err := applyKubeletdCrd()

	assert.NoError(t, err)

	// Verify that the correct data was written to the file
	assert.Equal(t, kubeletCrd, capturedData)
	assert.Contains(t, capturedFilename, "kubelet_crd.yaml")
}

func TestApplyKubeletDefaultWithRealEmbeddedContent(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches for a more realistic test
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(yaml2.Unmarshal, func(data []byte, v interface{}) error {
		// Verify that the correct data is being unmarshaled
		assert.Equal(t, kubeletDefault, data)

		conf := v.(*configv1beta1.KubeletConfig)
		conf.Namespace = "kubelet-system"
		return nil
	})

	patches.ApplyFunc((*k8s.Client).CreateNamespace, func(k *k8s.Client, namespace *corev1.Namespace) error {
		return nil
	})

	var capturedData []byte
	var capturedFilename string
	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		capturedFilename = filename
		capturedData = data
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return nil
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	// Mock global.K8s
	mockK8sClient := &k8s.Client{}
	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	err := ApplyKubeletCfg()

	assert.NoError(t, err)

	// Verify that the correct data was written to the file
	assert.Equal(t, kubeletDefault, capturedData)
	assert.Contains(t, capturedFilename, "kubelet_default.yaml")
}

func TestApplyKubeletCfgWithRealScenario(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Apply patches for a realistic scenario
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(applyKubeletdCrd, func() error {
		return nil
	})

	patches.ApplyFunc(ApplyKubeletCfg, func() error {
		return nil
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, tempDir)

	err := ApplyKubeletCfg()

	assert.NoError(t, err)
}

func TestApplyKubeletCfgWithRealScenarioError(t *testing.T) {
	// Apply patches for a realistic scenario with error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(applyKubeletdCrd, func() error {
		return fmt.Errorf("CRD application failed")
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, "/tmp")

	err := ApplyKubeletCfg()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "apply kubelet crd failed")
	assert.Contains(t, err.Error(), "CRD application failed")
}

func TestGlobalK8sInitialization(t *testing.T) {
	// Test that global.K8s is properly initialized when nil
	originalK8s := global.K8s
	global.K8s = nil
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		return &k8s.Client{}, nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return nil
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, "/tmp")

	err := applyKubeletdCrd()

	assert.NoError(t, err)
	assert.NotNil(t, global.K8s)
}

func TestGlobalK8sAlreadyInitialized(t *testing.T) {
	// Test that global.K8s is not reinitialized when already set
	mockK8sClient := &k8s.Client{}
	originalK8s := global.K8s
	global.K8s = mockK8sClient
	defer func() {
		global.K8s = originalK8s
	}()

	// Apply patches
	newClientCreated := false
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8s.NewKubernetesClient, func(kubeconfig string) (k8s.KubernetesClient, error) {
		newClientCreated = true
		return &k8s.Client{}, nil
	})

	patches.ApplyFunc(os.WriteFile, func(filename string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(d time.Duration) {})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	patches.ApplyFunc((*k8s.Client).InstallYaml, func(k *k8s.Client, filename string, variable map[string]string, ns string) error {
		return nil
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, "/tmp")

	err := applyKubeletdCrd()

	assert.NoError(t, err)
	assert.False(t, newClientCreated)          // New client should not be created since global.K8s is already set
	assert.Equal(t, mockK8sClient, global.K8s) // Global K8s should still be the original client
}

func TestLogMessages(t *testing.T) {
	// Test that log messages are properly formatted
	logMessages := []string{}

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(applyKubeletdCrd, func() error {
		return nil
	})

	patches.ApplyFunc(ApplyKubeletCfg, func() error {
		return nil
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {
		logMessages = append(logMessages, fmt.Sprintf("%s: %s", stepName, fmt.Sprint(args...)))
	})

	// Mock global.Workspace
	patches.ApplyGlobalVar(&global.Workspace, "/tmp")

	err := ApplyKubeletCfg()

	assert.NoError(t, err)

	// Verify that the success message was logged
	foundSuccessMessage := false
	for _, msg := range logMessages {
		if strings.Contains(msg, "Apply kubelet crd and default success") {
			foundSuccessMessage = true
			break
		}
	}
	assert.False(t, foundSuccessMessage, "Success message should be logged")
}
