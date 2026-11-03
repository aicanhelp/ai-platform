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

import (
	"strings"

	olog "gopkg.openfuyao.cn/common-modules/ologger/v26/log"
)

// NormalizeAllowedLevel trims and uppercases level, returning the canonical
// ologger name when only allowlisted levels are accepted for agentLogLevel /
// --log-level (whitelist-based validation before values enter systemd units).
// Empty or non-allowlisted input returns ("", false).
func NormalizeAllowedLevel(level string) (string, bool) {
	name := strings.ToUpper(strings.TrimSpace(level))
	switch name {
	case olog.TRACE, olog.DEBUG, olog.INFO, olog.WARNING, olog.ERROR, olog.CRITICAL:
		return name, true
	default:
		return "", false
	}
}
