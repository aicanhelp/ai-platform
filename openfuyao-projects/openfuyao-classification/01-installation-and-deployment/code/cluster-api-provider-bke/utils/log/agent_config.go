/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package log

import olog "gopkg.openfuyao.cn/common-modules/ologger/v26/log"

func DefaultAgentConfig() olog.Config {
	enableFile, enableConsole := true, false
	compress, sanitize, includeIP := true, true, true
	watchLevel := false
	return olog.Config{
		Path:           "/var/log/openFuyao/bkeagent.log",
		Level:          olog.INFO,
		Format:         "json",
		TimeZone:       "local",
		IncludeIP:      &includeIP,
		EnableFile:     &enableFile,
		EnableConsole:  &enableConsole,
		MaxSizeMB:      100,
		MaxBackups:     30,
		MaxAgeDays:     14,
		Compress:       &compress,
		EnableSanitize: &sanitize,
		WatchLevel:     &watchLevel,
	}
}
