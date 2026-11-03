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
	"bufio"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPeriodicallyCheckIfLogFileIsDeleted(t *testing.T) {
	testCases := []struct {
		checkLogDeletedPeriod int
		fileExists            bool
		expectedInstallCalled bool
	}{
		{1, true, false},
		{1, false, true},
	}

	for i, testCase := range testCases {
		// Create a temporary file and close it
		tmpFile, err := os.CreateTemp("", "example")
		if err != nil {
			t.Fatalf("Test case %d: Failed to create temporary file: %v", i, err)
		}
		err = tmpFile.Close()
		if err != nil {
			t.Fatalf("Test case %d: Failed to close temporary file: %v", i, err)
		}

		// Create a checkLogFileHandler instance
		handler := &checkLogFileHandler{
			fileWrap:              &fileWrap{filePath: tmpFile.Name(), filePermissions: 0644},
			checkLogDeletedPeriod: testCase.checkLogDeletedPeriod,
			updateTarget:          func(file *os.File, writtenBytes int64) {},
		}

		// Set fileExists based on test case
		if !testCase.fileExists {
			if err := os.Remove(tmpFile.Name()); err != nil {
				t.Errorf("Test case %d: Failed to remove temporary file: %v", i, err)
			}
		}

		// Run periodicallyCheckIfLogFileIsDeleted method
		go handler.periodicallyCheckIfLogFileIsDeleted()

		// Wait for a while to allow the method to run
		time.Sleep(time.Duration(testCase.checkLogDeletedPeriod+1) * time.Second)

		file, err := os.Open(tmpFile.Name())
		if err != nil {
			t.Errorf("can not open file err: %s", err)
			return
		}
		defer os.Remove(tmpFile.Name())
		defer file.Close()

		scanner := bufio.NewScanner(file)
		isContainsString := false
		for scanner.Scan() {
			line := scanner.Text()
			isContainsString = strings.Contains(line, "create new log file")
			if isContainsString {
				break
			}
		}

		// Check if installNewLogFile method is called as expected
		if testCase.expectedInstallCalled && !isContainsString {
			t.Errorf("Test case %d: installNewLogFile method is expected to be called, but it is not.", i)
		} else if !testCase.expectedInstallCalled && isContainsString {
			t.Errorf("Test case %d: installNewLogFile method is not expected to be called, but it is.", i)
		}
	}
}
