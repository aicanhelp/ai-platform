/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package utils provides shared test constructors and runtime utilities for EPP plugins.
package utils

import (
	"encoding/json"
	"fmt"
	"time"
)

// Duration accepts either a Go duration string ("100ms") or a numeric
// nanosecond value, mirroring how the GIE config loader serializes durations.
type Duration time.Duration

// UnmarshalJSON parses Duration from either a string or an integer value.
func (d *Duration) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		parsed, parseErr := time.ParseDuration(text)
		if parseErr != nil {
			return fmt.Errorf("invalid duration %q: %w", text, parseErr)
		}
		*d = Duration(parsed)
		return nil
	}

	var nanos int64
	if err := json.Unmarshal(data, &nanos); err != nil {
		return fmt.Errorf("invalid duration value %s: %w", string(data), err)
	}
	*d = Duration(time.Duration(nanos))
	return nil
}

// Std returns the standard library duration value.
func (d Duration) Std() time.Duration {
	return time.Duration(d)
}

// DurationMillisecondsSince provides the exported DurationMillisecondsSince operation.
func DurationMillisecondsSince(start time.Time) int64 {
	elapsed := time.Since(start).Milliseconds()
	if elapsed < 0 {
		return 0
	}
	return elapsed
}
