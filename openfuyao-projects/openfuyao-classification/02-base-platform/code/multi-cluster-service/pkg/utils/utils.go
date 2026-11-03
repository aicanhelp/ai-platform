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

// Package utils contains a collection of utility functions and helpers that provide
// common functionalities useful across the entire engineering project.
package utils

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	// Kilo represents the binary multiplier for kilobyte (KiB), eqal to 2^10 bytes.
	Kilo = "Ki"
	// Mega represents the binary multiplier for megabyte (MiB), eqal to 2^20 bytes.
	Mega = "Mi"
	// Giga represents the binary multiplier for gigabyte (GiB), eqal to 2^30 bytes.
	Giga = "Gi"
)

// ScaleToGiB examines the suffix to determine if need to convert to GiBy
func ScaleToGiB(str string) string {
	unitLen := len(Kilo)
	suffix := str[len(str)-unitLen:]

	switch suffix {
	case Kilo:
		return convertKiToGi(str)
	case Mega:
		return convertMiToGi(str)
	case Giga:
		return str
	default:
		return ""
	}
}

func convertKiToGi(str string) string {
	numstr := strings.Split(str, Kilo)[0]
	num, err := convertStrToInt(numstr)
	if err != nil {
		return ""
	}
	numgi := float64(num / (MegabytesPerGigabyte * MegabytesPerGigabyte))

	return fmt.Sprintf("%s%s", convertFloatToStr(numgi), Giga)
}

func convertMiToGi(str string) string {
	numstr := strings.Split(str, Mega)[0]
	num, err := convertStrToInt(numstr)
	if err != nil {
		return ""
	}
	numgi := float64(num) / (MegabytesPerGigabyte)

	return fmt.Sprintf("%s%s", convertFloatToStr(numgi), Giga)
}

func convertStrToInt(str string) (int, error) {
	return strconv.Atoi(str)
}

func convertFloatToStr(num float64) string {
	return fmt.Sprintf("%.1f", num)
}
