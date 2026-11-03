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

package utils

import (
	"log" // 建议引入 log 包来记录警告
	"time"
)

func UnixFmtTime(unixSec int64) string {
	// 尝试加载指定的时区
	loc, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		// 如果加载失败（例如，系统没有时区数据），记录一个警告并使用UTC作为后备。
		// 这可以防止程序因为时区问题而崩溃。
		log.Printf("Warning: failed to load location 'Asia/Shanghai', falling back to UTC. Error: %v", err)
		loc = time.UTC
	}
	return time.Unix(unixSec, 0).In(loc).Format("2006-01-02 15:04:05")
}
