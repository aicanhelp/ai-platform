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
	"os"
	"testing"
)

func TestNewWrapFileConfig(t *testing.T) {
	wfc := NewWrapFileConfig()
	if wfc == nil {
		t.Errorf("got a nil target from NewWrapFileConfig")
	} else {
		if wfc.rotateFileConfig == nil {
			t.Errorf("expect rotateFileConfig is not nil from target created by NewWrapFileConfig")
		}
	}
}

func TestNewWrapFile(t *testing.T) {
	// 初始化测试对象
	w := WrapFileConfig{
		rotateFileConfig: &rotateFileConfig{},
	}
	tempFile, err := os.CreateTemp("", "TestNewWrapFile")
	if err != nil {
		t.Errorf("fail to create temp file: %v", err)
	}
	defer deferCloseAndRemoveFile(tempFile, t)

	type testCase struct {
		name                  string
		IgnoreEnospc          bool
		EnableRotate          bool
		CheckLogDeletedPeriod int
		file                  *os.File
		hasErr                bool
	}
	testCases := []testCase{
		{
			"skip and return",
			false,
			false,
			0,
			nil,
			false,
		},
		{
			"file not exist error and return",
			true,
			true,
			1,
			nil,
			true,
		},
		{
			"no error and return",
			true,
			true,
			1,
			tempFile,
			false,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			w.IgnoreEnospc = tc.IgnoreEnospc
			w.EnableRotate = tc.EnableRotate
			w.CheckLogDeletedPeriod = tc.CheckLogDeletedPeriod

			_, err := w.NewWrapFile(tc.file)

			if tc.hasErr {
				if err == nil {
					t.Errorf("expect an error when %s, but got nil", tc.name)
				}
			} else {
				if err != nil {
					t.Errorf("expect no error when %s, but got %v", tc.name, err)
				}
			}
		})
	}
}

func TestNewRotateConfig(t *testing.T) {
	rc := newRotateConfig()
	if rc == nil {
		t.Errorf("got a nil target from NewRotateConfig")
	}
}
