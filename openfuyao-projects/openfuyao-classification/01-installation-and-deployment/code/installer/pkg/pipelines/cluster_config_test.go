/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pipelines

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/constants"
)

func TestCreateSimplestCfgFile(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("cluster-config-test-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	args := common.Argument{
		FileSaveFolder: tempDir,
	}
	err = CreateClusterConfigFile(args)
	assert.NoError(t, err)
}

func TestCreateSimplestCfgFileWithAdvanced(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("cluster-config-test-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// mock 包路径和包名
	pkgPath, pkgName := "tempDir", "fuyao-openEuler-22.03-amd64"
	patches := gomonkey.ApplyFunc(common.UnCompressOnePackage, func() (string, string, error) {
		return filepath.Join(tempDir, pkgPath), pkgName, nil
	})
	defer patches.Reset()

	// 创建临时高级参数文件
	advancedFileDir := filepath.Join(tempDir, pkgPath, pkgName, constants.Config)
	err = os.MkdirAll(advancedFileDir, os.ModePerm)
	if err != nil {
		t.Fatal("Failed to create advanced parameter file directory")
	}
	advancedFilePath := filepath.Join(advancedFileDir, constants.AdvancedParamsFile)

	advancedFile, err := os.Create(advancedFilePath)
	if err != nil {
		t.Fatalf("Failed to create advanced parameter file: %v", err)
	}
	err = advancedFile.Close()
	if err != nil {
		t.Fatal("Failed to close advanced parameter file")
	}

	args := common.Argument{
		FileSaveFolder:       tempDir,
		EnableAdvancedParams: true,
	}
	err = CreateClusterConfigFile(args)
	assert.NoError(t, err)
}
