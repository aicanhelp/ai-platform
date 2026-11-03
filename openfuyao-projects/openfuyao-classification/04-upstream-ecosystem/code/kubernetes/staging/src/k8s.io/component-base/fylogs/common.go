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
	"time"
)

func createLogFile(logFilePath string, logFilePermission uint32) (*os.File, error) {
	return os.OpenFile(logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, os.FileMode(logFilePermission))
}

func printErrWithIgnoreFailed(err error) {
	msg := fmt.Sprintf("%s => %v\n", time.Now().Format(time.RFC3339), err)
	_, writeErr := os.Stderr.Write([]byte(msg))
	fmt.Printf("print err: %v", writeErr)
}
