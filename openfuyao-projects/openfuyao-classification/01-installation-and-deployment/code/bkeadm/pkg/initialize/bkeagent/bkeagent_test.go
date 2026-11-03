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

package bkeagent

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

func TestInstallBKEAgentCRD_Success(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		return nil
	})

	err := InstallBKEAgentCRD()
	assert.NoError(t, err)
}

func TestInstallBKEAgentCRD_WriteFileError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return assert.AnError
	})

	err := InstallBKEAgentCRD()
	assert.Error(t, err)
}

func TestInstallBKEAgentCRD_InstallYamlError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		return assert.AnError
	})

	err := InstallBKEAgentCRD()
	assert.Error(t, err)
}
