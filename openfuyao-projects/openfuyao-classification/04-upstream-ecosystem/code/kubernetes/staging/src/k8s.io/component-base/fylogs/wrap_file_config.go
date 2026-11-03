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
	"io"
	"os"
)

const (
	notCheck = 0
)

// NewWrapFileConfig returns a new WrapFileConfig object
func NewWrapFileConfig() *WrapFileConfig {
	return &WrapFileConfig{
		rotateFileConfig: newRotateConfig(),
	}
}

// WrapFileConfig describe information of wrap file
type WrapFileConfig struct {
	FilePath              string
	IgnoreEnospc          bool
	CheckLogDeletedPeriod int
	FilePermissions       uint32
	*rotateFileConfig
}

// NewWrapFile create a new wrap file config
func (w WrapFileConfig) NewWrapFile(file *os.File) (io.Writer, error) {
	if !w.IgnoreEnospc && !w.EnableRotate && w.CheckLogDeletedPeriod == 0 {
		return file, nil
	}

	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file size with err: %v", err)
	}

	fileWrap := newFileWrap(file, w.FilePath, w.FilePermissions)
	var rotator *rotator
	if w.EnableRotate {
		rotator = newRotator(fileWrap, &w, stat)
		fileWrap.normalFunc = append(fileWrap.normalFunc, rotator.rotate)
	}

	if w.CheckLogDeletedPeriod > notCheck {
		var checkLogFileHandler *checkLogFileHandler
		if rotator != nil {
			updateTargetFunc := func(file *os.File, writtenBytes int64) {
				rotator.target = file
				rotator.counter = writtenBytes
			}
			checkLogFileHandler = newCheckLogFileHandler(fileWrap, &w, updateTargetFunc)
		} else {
			checkLogFileHandler = newCheckLogFileHandler(fileWrap, &w, nil)
		}
		go checkLogFileHandler.periodicallyCheckIfLogFileIsDeleted()
	}

	if w.IgnoreEnospc {
		var errHandler *errHandler
		if rotator != nil {
			updateCountFunc := func() {
				rotator.rotate()
			}
			errHandler = newErrHandler(fileWrap, updateCountFunc)
		} else {
			errHandler = newErrHandler(fileWrap, nil)
		}

		fileWrap.onErrFunc = append(fileWrap.onErrFunc, errHandler.handleErr)
		fileWrap.normalFunc = append(fileWrap.normalFunc, errHandler.handleRecover)
	}

	return fileWrap, nil
}

type rotateFileConfig struct {
	EnableRotate    bool
	EnableCompress  bool
	EachFileMaxSize int64
	TotalSizeLimit  int64
	TotalCountLimit int
	CompressPerm    uint32
}

func newRotateConfig() *rotateFileConfig {
	return &rotateFileConfig{}
}
