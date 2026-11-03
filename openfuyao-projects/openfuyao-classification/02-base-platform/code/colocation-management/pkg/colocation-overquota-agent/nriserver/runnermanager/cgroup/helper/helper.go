/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package helper

import (
	"fmt"

	"openfuyao.com/colocation-management/pkg/utils"
)

// CgroupFileRead reads the content of a file at the specified path.
func CgroupFileRead(path string) ([]byte, error) {
	exist := utils.PathExist(path)
	if !exist {
		return []byte{}, fmt.Errorf("cgroup path not exist, path %v", path)
	}
	return utils.ReadFile(path)
}

// CgroupFIleWrite writes the content to a file at the specified path.
func CgroupFIleWrite(path string, content string) error {
	exist := utils.PathExist(path)
	if !exist {
		return fmt.Errorf("cgroup path not exist, path %v", path)
	}
	return utils.WriteFile(path, content)
}
