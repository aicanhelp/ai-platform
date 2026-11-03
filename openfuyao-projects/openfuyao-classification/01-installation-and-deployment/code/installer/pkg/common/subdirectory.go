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

package common

import (
	"os"
	"path/filepath"
)

// SubDirectory is used to obtain the name of a file or folder under a subdirectory
type SubDirectory struct {
	RootDir string
}

// GetSubdirectoryNames 根据路径获取子路径下所有文件夹名
func (s *SubDirectory) GetSubdirectoryNames() ([]string, error) {
	var subdirectoryNames []string

	err := filepath.Walk(s.RootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && path != s.RootDir {
			subdirectoryNames = append(subdirectoryNames, info.Name())
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return subdirectoryNames, nil
}

// GetImmediateSubdirectories 获取文件夹下第一层子文件夹名称
func GetImmediateSubdirectories(rootDir string) ([]string, error) {
	var subdirectories []string

	// 打开根目录
	dir, err := os.Open(rootDir)
	if err != nil {
		return nil, err
	}
	defer dir.Close()

	// 读取根目录下的文件和子文件夹
	entries, err := dir.Readdir(-1)
	if err != nil {
		return nil, err
	}

	// 遍历文件夹
	for _, entry := range entries {
		if entry.IsDir() {
			subdirectories = append(subdirectories, entry.Name())
		}
	}

	return subdirectories, nil
}

// ListFiles 获取当前文件夹下文件文件名
func ListFiles(dir string) ([]string, error) {
	var files []string

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			files = append(files, path)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return files, nil
}
