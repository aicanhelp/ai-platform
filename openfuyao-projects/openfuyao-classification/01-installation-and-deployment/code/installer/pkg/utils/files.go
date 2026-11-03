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

package utils

import (
	"fmt"
	"os"
	"path/filepath"
)

// Mkdir is create dir
func Mkdir(dir string, perm os.FileMode) error {
	_, err := os.Stat(dir)
	if err == nil {
		return nil
	} else if os.IsNotExist(err) {
		if err := os.Mkdir(dir, perm); err != nil {
			return fmt.Errorf("failed to create dir :%v", err)
		}
	} else {
		return err
	}
	return nil
}

// MkdirAll is create all dir
func MkdirAll(dir string, perm os.FileMode) error {
	_, err := os.Stat(dir)
	if err == nil {
		return nil
	} else if os.IsNotExist(err) {
		if err := os.MkdirAll(dir, perm); err != nil {
			return fmt.Errorf("failed to create dir :%v", err)
		}
	} else {
		return err
	}
	return nil
}

// Remove 文件存在则删除，文件不存在则正常返回
func Remove(filePath string) error {
	_, err := os.Stat(filePath)
	if err == nil {
		return os.Remove(filePath)
	} else if os.IsNotExist(err) {
		return nil
	} else {
		return err
	}
}

// GetDirFiles 获取当前文件夹下文件文件名
func GetDirFiles(dir string) ([]string, error) {
	var files []string

	err := filepath.Walk(dir, func(path string, fileInfo os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !fileInfo.IsDir() {
			files = append(files, path)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return files, nil
}
