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

package common

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"installer/pkg/constants"
)

func TestCopyLocalFile(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-copy-local-file-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 创建临时文件
	_, err = os.Create(filepath.Join(tempDir, "1.txt"))

	// 创建目标目录
	targetDir := filepath.Join(tempDir, constants.Targets)
	err = os.Mkdir(targetDir, 0755)
	assert.NoError(t, err)

	// 复制文件
	err = CopyLocalFile(filepath.Join(tempDir, "1.txt"), filepath.Join(targetDir, "2.txt"))
	assert.NoError(t, err)

	files, err := ListFiles(targetDir)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(files))
}

func TestCompressTarGZ(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-compress-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	srcDir := filepath.Join(tempDir, "src")
	err = os.Mkdir(srcDir, 0755)
	assert.NoError(t, err)
	file, err := os.Create(filepath.Join(srcDir, "1.txt"))
	assert.NoError(t, err)
	file.Close()

	// 压缩文件
	err = CompressTarGZ(srcDir, filepath.Join(tempDir, "src.tar.gz"))
	assert.NoError(t, err)

	files, err := ListFiles(tempDir)
	assert.NoError(t, err)
	fmt.Println(files)
	assert.Equal(t, 2, len(files))

	dstDir := filepath.Join(tempDir, "dst")
	err = os.Mkdir(dstDir, 0755)
	assert.NoError(t, err)
	UnCompressTarGZ(filepath.Join(tempDir, "src.tar.gz"), dstDir)
}
