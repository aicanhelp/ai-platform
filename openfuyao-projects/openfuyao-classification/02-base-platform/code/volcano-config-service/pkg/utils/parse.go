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

package utils

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	singleBoundPartLength = 1
	doubleBoundPartLength = 2
)

// ParseRanges parses the given string representation of integer ranges and returns an integer slice.
// The string should be comma-separated with each range either being a single number or a range like "1-5".
// Returns an empty slice if rangesStr is an empty string.
func ParseRanges(rangesStr string) ([]int, error) {
	rangesStr = strings.TrimSpace(rangesStr)

	if rangesStr == "" {
		return []int{}, nil
	}

	var result []int
	parts := strings.Split(rangesStr, ",")
	for _, part := range parts {
		bounds := strings.Split(part, "-")
		if len(bounds) == singleBoundPartLength {
			elem, err := strconv.Atoi(bounds[0])
			if err != nil {
				return []int{}, err
			}
			result = append(result, elem)
		} else if len(bounds) == doubleBoundPartLength {
			start, err := strconv.Atoi(bounds[0])
			if err != nil {
				return []int{}, err
			}
			end, err := strconv.Atoi(bounds[1])
			if err != nil {
				return []int{}, err
			}
			for elem := start; elem <= end; elem++ {
				result = append(result, elem)
			}
		} else {
			return []int{}, fmt.Errorf("invalid range format: %s", part)
		}
	}
	return result, nil
}
