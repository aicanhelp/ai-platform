/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"fmt"
	"strconv"
	"time"

	"github.com/pkg/errors"
)

// TimeFormat +8
const TimeFormat = "2006-01-02 15:04:05"

// DecimalBase is the decimal number base
const DecimalBase = 10

// DefaultTimeRangeYears is the default time range in years when no start time is provided
const DefaultTimeRangeYears = 10

// Int64BitSize is the bit size for parsing 64-bit integers
const Int64BitSize = 64

// DaysPerYear is the number of days in a year (approximate)
const DaysPerYear = 365

// HoursPerDay is the number of hours in a day
const HoursPerDay = 24

func TimeNow() string {
	return time.Now().Format(TimeFormat)
}

func TimeNowUnix() string {
	return strconv.FormatInt(time.Now().Unix(), DecimalBase)
}

func FormatTime(t time.Time) string {
	return t.Format(TimeFormat)
}

func ParseTimeUnix(timestamp string) (time.Time, error) {

	// 如果时间格式为 值保留到秒即可

	i, err := strconv.ParseInt(timestamp, DecimalBase, Int64BitSize)
	if err != nil {
		return time.Time{}, err
	}
	return time.Unix(i, 0), nil
}

// ParseTimeFromTo
// from, to is time unix string
func ParseTimeFromTo(from, to string) (startTime, endTime time.Time, err error) {
	// 如果开始时间为空则不限制开始时间
	if from == "" {
		//则开始时间 设置为十年前
		startTime = time.Now().Add(-DefaultTimeRangeYears * DaysPerYear * HoursPerDay * time.Hour)
	} else {
		// 解析开始时间
		startTime, err = ParseTimeUnix(from)
		if err != nil {
			err = fmt.Errorf("Error parsing start time: %w", err)
			return
		}
	}
	if to == "" {
		//则结束时间 设置为现在
		endTime = time.Now()
	} else {
		// 解析结束时间
		endTime, err = ParseTimeUnix(to)
		if err != nil {
			err = fmt.Errorf("Error parsing end time: %w", err)
			return
		}
	}
	// 如果开始时间大于结束时间则返回错误
	if startTime.After(endTime) {
		err = errors.New("Start time must be before end time")
		return
	}
	return
}
