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

import "os"

func newFileWrap(outputFile *os.File, filePath string, filePermissions uint32) *fileWrap {
	return &fileWrap{
		target:          outputFile,
		filePath:        filePath,
		filePermissions: filePermissions,
		onErrFunc:       make([]onErrFunc, 0),
		normalFunc:      make([]normalFunc, 0),
	}
}

type onErrFunc func()
type normalFunc func()

type fileWrap struct {
	target          *os.File
	filePath        string
	filePermissions uint32
	onErrFunc       []onErrFunc
	normalFunc      []normalFunc

	writtenBytes int
	writtenErr   error
}

func (f *fileWrap) Write(p []byte) (int, error) {
	f.writtenBytes, f.writtenErr = f.target.Write(p)
	if f.writtenErr == nil {
		for _, f := range f.normalFunc {
			f()
		}
	} else {
		for _, f := range f.onErrFunc {
			f()
		}
		f.writtenBytes = len(p)
	}
	return f.writtenBytes, f.writtenErr
}
