/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pipelines

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/metadata"
)

const fileModeLow = 0644

// TestShowVersion_Success 测试成功显示版本信息
func TestShowVersion_Success(t *testing.T) {
	// 创建临时文件模拟真实场景
	tmpDir := t.TempDir()
	metadataFile := filepath.Join(tmpDir, constants.MetaDataFile)

	// 创建模拟的 metadata 文件
	mockMetadata := `{
        "openfuyao-system-version": "v1.0.0"
    }`
	err := os.WriteFile(metadataFile, []byte(mockMetadata), fileModeLow)
	assert.NoError(t, err)

	// Mock GetAssetAbsPath 返回临时目录
	patches := gomonkey.ApplyFunc(common.GetAssetAbsPath,
		func() (string, error) {
			return tmpDir, nil
		})
	defer patches.Reset()

	// 捕获标准输出
	old := os.Stdout
	r, w, err := os.Pipe()
	assert.NoError(t, err)
	os.Stdout = w

	// 执行测试
	err = ShowVersion()

	// 恢复标准输出并读取内容
	err = w.Close()
	assert.NoError(t, err)
	os.Stdout = old
	var buf bytes.Buffer
	_, err = io.Copy(&buf, r)
	assert.NoError(t, err)
	output := buf.String()

	// 验证结果
	assert.NoError(t, err, "ShowVersion should not return error")
	assert.Contains(t, output, "openfuyao-system-version: v1.0.0",
		"Output should contain version information")
}

// TestShowVersion_GetAssetAbsPathFailed 测试获取资源路径失败
func TestShowVersion_GetAssetAbsPathFailed(t *testing.T) {
	// Mock GetAssetAbsPath 返回错误
	expectedErr := fmt.Errorf("failed to get asset path")
	patches := gomonkey.ApplyFunc(common.GetAssetAbsPath,
		func() (string, error) {
			return "", expectedErr
		})
	defer patches.Reset()

	// 执行测试
	err := ShowVersion()

	// 验证结果
	assert.Error(t, err, "ShowVersion should return error when GetAssetAbsPath fails")
	assert.Equal(t, expectedErr, err, "Error should match expected error")
}

// TestShowVersion_UnmarshalJSONFailed 测试JSON解析失败
func TestShowVersion_UnmarshalJSONFailed(t *testing.T) {
	// Mock GetAssetAbsPath 返回成功
	patches1 := gomonkey.ApplyFunc(common.GetAssetAbsPath,
		func() (string, error) {
			return "/mock/path", nil
		})
	defer patches1.Reset()

	// Mock UnmarshalJSON 返回错误
	expectedErr := fmt.Errorf("failed to unmarshal JSON")
	patches2 := gomonkey.ApplyFunc(common.UnmarshalJSON[*metadata.Version],
		func(versionMetaData *metadata.Version, metadataFile string) error {
			return expectedErr
		})
	defer patches2.Reset()

	// 执行测试
	err := ShowVersion()

	// 验证结果
	assert.Error(t, err, "ShowVersion should return error when UnmarshalJSON fails")
}
