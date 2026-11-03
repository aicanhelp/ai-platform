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
	"errors"
	"fmt"
	"syscall"
	"time"
)

type updateCountFunc func()

func newErrHandler(writerWrap *fileWrap, updateFunc updateCountFunc) *errHandler {
	var updateCount updateCountFunc
	if updateFunc != nil {
		updateCount = updateFunc
	} else {
		updateCount = func() {}
	}
	return &errHandler{
		fileWrap:    writerWrap,
		updateCount: updateCount,
	}
}

type errHandler struct {
	*fileWrap
	hasError    bool
	errTime     time.Time
	updateCount updateCountFunc
}

func (e *errHandler) handleErr() {
	if errors.Is(e.writtenErr, syscall.ENOSPC) && !e.hasError {
		e.errTime = time.Now()
		e.hasError = true
		e.writtenErr = nil
	}

	if errors.Is(e.writtenErr, syscall.ENOSPC) {
		e.writtenErr = nil
	}
}

func (e *errHandler) handleRecover() {
	if e.hasError {
		msg := fmt.Sprintf("No space left on device begin from : %s, errHandler time is: %s\n",
			e.errTime.Format(time.RFC3339), time.Now().Format(time.RFC3339))
		if count, writeErr := e.target.Write([]byte(msg)); writeErr == nil {
			e.errTime = time.Time{}
			e.hasError = false
			originWrittenBytes := e.writtenBytes
			e.writtenBytes = count
			e.updateCount()
			e.writtenBytes = originWrittenBytes
		}
	}
}
