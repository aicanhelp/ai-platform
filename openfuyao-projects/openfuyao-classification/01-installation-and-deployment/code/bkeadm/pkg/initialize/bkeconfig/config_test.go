/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package bkeconfig

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestEnsureNsExists_Success(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	err := ensureNsExists("test-namespace")
	assert.NoError(t, err)
}

func TestEnsureNsExists_GetClientError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	global.K8s = nil

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesClient, func(string) (k8s.KubernetesClient, error) {
		return nil, assert.AnError
	})

	err := ensureNsExists("test-namespace")
	assert.Error(t, err)
}

func TestSetKubernetesConfig_AlreadyExists(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	data := map[string]string{"key": "value"}
	err := SetKubernetesConfig(data, "test-config", "test-ns")
	assert.NoError(t, err)
}

func TestSetKubernetesConfig_NewClientError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	global.K8s = nil

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesClient, func(string) (k8s.KubernetesClient, error) {
		return nil, assert.AnError
	})

	data := map[string]string{"key": "value"}
	err := SetKubernetesConfig(data, "test-config", "test-ns")
	assert.Error(t, err)
}

func TestSetPatchConfig_Success(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	tempDir := t.TempDir()
	yamlFile := filepath.Join(tempDir, "test.yaml")
	err := os.WriteFile(yamlFile, []byte("test: content"), utils.DefaultFilePermission)
	assert.NoError(t, err)

	err = SetPatchConfig("v1.0.0", yamlFile, "patch-config")
	assert.NoError(t, err)
}

func TestSetPatchConfig_ReadFileError(t *testing.T) {
	err := SetPatchConfig("v1.0.0", "/nonexistent/path.yaml", "patch-config")
	assert.Error(t, err)
}

func TestSetPatchConfig_CRFLContent(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	tempDir := t.TempDir()
	yamlFile := filepath.Join(tempDir, "test.yaml")
	content := "test: content\r\nanother: line\r\n"
	err := os.WriteFile(yamlFile, []byte(content), utils.DefaultFilePermission)
	assert.NoError(t, err)

	err = SetPatchConfig("v1.0.0", yamlFile, "patch-config")
	assert.NoError(t, err)
}
