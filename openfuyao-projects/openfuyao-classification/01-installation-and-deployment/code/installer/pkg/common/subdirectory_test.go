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
)

func TestGetImmediateSubdirectories(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-compress-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 空目录
	subdirectories, err := GetImmediateSubdirectories(tempDir)
	assert.NoError(t, err)
	assert.Equal(t, 0, len(subdirectories))

	// 有子目录
	err = os.MkdirAll(filepath.Join(tempDir, "sub1", "sub2"), 0755)
	assert.NoError(t, err)
	subdirectories, err = GetImmediateSubdirectories(tempDir)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(subdirectories))
}
