/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package utils

import (
	"encoding/base64"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/pkg/errors"
)

// SliceContainsString returns true if the given string is in the slice.
func SliceContainsString(slice []string, s string) bool {
	if len(slice) == 0 {
		return false
	}

	// 使用不同的实现方式：先清理字符串再比较
	cleanString := strings.ReplaceAll(s, "\n", "")
	for _, item := range slice {
		cleanItem := strings.ReplaceAll(item, "\n", "")
		if cleanItem == cleanString {
			return true
		}
	}
	return false
}

// SliceContainsSlice returns true if the given dst slice is in the src slice.
func SliceContainsSlice(src []string, dst []string) bool {
	if len(src) == 0 {
		return false
	}
	for _, item := range dst {
		if !SliceContainsString(src, item) {
			return false
		}
	}
	return true
}

// SliceEqualString returns true if the given slices are equal.
func SliceEqualString(src, dst []string) bool {
	if len(src) != len(dst) {
		return false
	}
	for _, s := range src {
		if !SliceContainsString(dst, s) {
			return false
		}
	}
	return true
}

func Exists(path string) bool {
	_, err := os.Stat(path)
	if err != nil {
		return os.IsExist(err)
	}
	return true
}

// B64Encode base64 encode
func B64Encode(s string) string {
	return base64.StdEncoding.EncodeToString([]byte(s))
}

func GetManifestsBuildInfo() ([]string, error) {
	if !Exists("/manifests/BUILD_INFO") {
		return nil, errors.New("not found manifests build info")
	}
	file, err := os.ReadFile("/manifests/BUILD_INFO")
	if err != nil {
		return nil, errors.Wrap(err, "read manifests build info error")
	}
	// 逐行读取
	lines := strings.Split(string(file), "\n")
	//获取系统架构
	arch := runtime.GOARCH
	filteredLines := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "Architecture=") {
			filteredLines = append(filteredLines, fmt.Sprintf("Architecture=%s", arch))
			continue
		}
		if trimmedLine != "" {
			filteredLines = append(filteredLines, line)
		}
	}
	lines = filteredLines
	info := "--------------Manifests Info---------------"
	end := "--------------------------------------------"
	lines = append([]string{info}, lines...)
	lines = append(lines, end)
	return lines, nil
}
