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

// Package fylogs contains the processing logic for log files. For example,
// log rotation, log compression, checking whether logs exist, and disk full processing
package fylogs

import (
	"fmt"
	"os"
	"time"
)

type updateTargetFunc func(file *os.File, writtenBytes int64)

func newCheckLogFileHandler(
	fileWrap *fileWrap,
	wrapFileConfig *WrapFileConfig,
	updateFunc updateTargetFunc,
) *checkLogFileHandler {
	var updateTarget updateTargetFunc
	if updateFunc != nil {
		updateTarget = updateFunc
	} else {
		updateTarget = func(file *os.File, writtenBytes int64) { fileWrap.target = file }
	}
	return &checkLogFileHandler{
		fileWrap:              fileWrap,
		checkLogDeletedPeriod: wrapFileConfig.CheckLogDeletedPeriod,
		updateTarget:          updateTarget,
	}
}

type checkLogFileHandler struct {
	*fileWrap
	checkLogDeletedPeriod int
	updateTarget          updateTargetFunc
}

func (c *checkLogFileHandler) periodicallyCheckIfLogFileIsDeleted() {
	retryPeriod := time.Duration(c.checkLogDeletedPeriod) * time.Second
	timer := time.NewTimer(retryPeriod)
	for {
		select {
		case <-timer.C:
		}
		_, err := os.Stat(c.filePath)
		if err != nil && os.IsNotExist(err) {
			c.installNewLogFile()
		}
		timer.Reset(retryPeriod)
	}
}

func (c *checkLogFileHandler) installNewLogFile() {
	logFile, err := createLogFile(c.filePath, c.filePermissions)
	if err != nil {
		printErrWithIgnoreFailed(fmt.Errorf("create file(%s) failed : %v", c.filePath, err))
		return
	}
	msg := fmt.Sprintf(
		"%s => create new log file (%s) while origin log file not exist.\n",
		time.Now().Format(time.RFC3339),
		c.filePath,
	)
	writtenBytes, err := logFile.Write([]byte(msg))
	if err != nil {
		printErrWithIgnoreFailed(fmt.Errorf("write failed to new log file(%s) with error : %v", c.filePath, err))
		return
	}
	c.updateTarget(logFile, int64(writtenBytes))
}
