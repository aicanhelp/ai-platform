/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/client-go/kubernetes/scheme"
)

// TestAddResourcesHandlersSuccess 获取组件成功
func TestAddResourcesHandlersSuccess(t *testing.T) {
	res, handler, err := getResourcesAndHandlers("../../assets/component-device-plugin-hpre")
	assert.Nil(t, err)
	assert.NotNil(t, res)
	assert.Equal(t, "kae-hpre-device-plugin-daemonset", res.DaemonSet.Name)
	assert.NotNil(t, handler)
}

// TestGetResourcesAndHandlersFilePathWalkDirError walkDir 错误导致解析失败
func TestGetResourcesAndHandlersFilePathWalkDirError(t *testing.T) {
	mockPath := "mock/path"

	// Mock filepath.WalkDir to return an error
	patchWalkDir := gomonkey.ApplyFunc(filepath.WalkDir, func(root string, fn fs.WalkDirFunc) error {
		return fmt.Errorf("walkdir error")
	})
	defer patchWalkDir.Reset()

	// Perform the test
	res, ctrl, err := getResourcesAndHandlers(mockPath)

	// Assert results
	assert.Error(t, err)
	assert.Empty(t, res)
	assert.Empty(t, ctrl)
}

// TestGetAssetsFromFileReadError 读取文件失败
func TestGetAssetsFromFileReadError(t *testing.T) {
	// Mock os.ReadFile to return an error
	patchReadFile := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return nil, fmt.Errorf("readfile error")
	})
	defer patchReadFile.Reset()

	// Perform the test
	manifests, err := getAssetsFrom("../../assets/component-device-plugin-hpre")

	// Assert results
	assert.Error(t, err)
	assert.Empty(t, manifests)
}

// TestExtractKindFromManifest kind 字段解析
func TestExtractKindFromManifest(t *testing.T) {
	manifest := []byte(`apiVersion: v1
kind: ServiceAccount
metadata:
  name: test-sa`)

	kind := extractKindFromManifest(manifest)
	assert.Equal(t, "ServiceAccount", kind)
}

// TestHandleResourceUnknownKind 未知kind
func TestHandleResourceUnknownKind(t *testing.T) {
	res := Resources{}
	ctrl := resourceHandler{}
	s := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme.Scheme, scheme.Scheme,
		json.SerializerOptions{Yaml: true, Pretty: false, Strict: false})

	err := handleResource("UnknownKind", []byte{}, &res, &ctrl, s)
	assert.NoError(t, err)
}
