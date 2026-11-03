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
	"io/ioutil"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestHandleErr(t *testing.T) {
	testCases := []struct {
		name          string
		writtenErr    error
		hasError      bool
		expectedError bool
	}{
		{
			name:          "ENOSPC error with no previous error",
			writtenErr:    syscall.ENOSPC,
			hasError:      false,
			expectedError: true,
		},
		{
			name:          "ENOSPC error with previous error",
			writtenErr:    syscall.ENOSPC,
			hasError:      true,
			expectedError: true,
		},
		{
			name:          "non-ENOSPC error with previous error",
			writtenErr:    errors.New("some error"),
			hasError:      true,
			expectedError: true,
		},
		{
			name:          "non-ENOSPC error with no previous error",
			writtenErr:    errors.New("some error"),
			hasError:      false,
			expectedError: false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			handler := &errHandler{
				fileWrap: &fileWrap{writtenErr: testCase.writtenErr},
				hasError: testCase.hasError,
			}
			handler.handleErr()

			if handler.hasError != testCase.expectedError {
				t.Errorf("handler.hasError = %v; expected %v", handler.hasError, testCase.expectedError)
			}
		})

	}
}

func TestHandleRecover(t *testing.T) {
	testCases := []struct {
		name          string
		hasError      bool
		errTime       time.Time
		expectedMsg   string
		expectedCount int
		expectedErr   error
	}{
		{
			name:        "no error",
			hasError:    false,
			errTime:     time.Time{},
			expectedMsg: "",
		},
		{
			name:        "has error",
			hasError:    true,
			errTime:     time.Now(),
			expectedMsg: fmt.Sprintf("No space left on device begin from"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			tmpFile, err := ioutil.TempFile("", "example")
			if err != nil {
				t.Fatalf("Failed to create temporary file: %v", err)
			}
			defer os.Remove(tmpFile.Name())
			defer tmpFile.Close()

			e := &errHandler{fileWrap: &fileWrap{
				target:          tmpFile,
				onErrFunc:       make([]onErrFunc, 0),
				normalFunc:      make([]normalFunc, 0),
				filePath:        tmpFile.Name(),
				filePermissions: 0644,
			}, hasError: testCase.hasError, errTime: testCase.errTime, updateCount: func() {}}

			e.handleRecover()

			fileContext, err := os.ReadFile(tmpFile.Name())
			if err != nil {
				t.Errorf("read file fail: %q", err)
			}
			isContainsString := false
			if strings.Contains(string(fileContext), testCase.expectedMsg) {
				isContainsString = true
			}
			if testCase.hasError && !isContainsString {
				t.Errorf("The file does not contain the following strings: %q ", testCase.expectedMsg)
			}

			if e.hasError {
				t.Errorf("expected hasError to be false but got true")
			}

			if e.errTime != (time.Time{}) {
				t.Errorf("expected errTime to be zero value but got %v", e.errTime)
			}
		})
	}
}
