/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package fylogs

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestFileWrapWrite(t *testing.T) {
	file, tempDir, err := setUp()
	if err != nil {
		t.Fatalf("unable to create temp file: %v", err)
	}

	fw := &fileWrap{
		target: file,
	}

	num, err := fw.Write([]byte("test"))
	if err != nil {
		t.Errorf("failed to write: %v", err)
	}
	if num != len("test") {
		t.Errorf("num write wrong: %d", num)
	}

	if err = file.Close(); err != nil {
		t.Errorf("failed to close file: %v", err)
	}

	num, err = fw.Write([]byte("test"))
	if err == nil {
		t.Errorf("Write should return an error")
	}

	if err = os.RemoveAll(tempDir); err != nil {
		t.Errorf("failed to remove kubeproxy-config-change: %v", err)
	}
}

func setUp() (*os.File, string, error) {
	tempDir, err := os.MkdirTemp("", "file-wrap-test")
	if err != nil {
		return nil, "", fmt.Errorf("unable to create temporary directory: %v", err)
	}
	fullPath := filepath.Join(tempDir, "test")
	file, err := os.OpenFile(fullPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, "", fmt.Errorf("unexpected error when creating temp file: %v", err)
	}

	return file, tempDir, nil
}
