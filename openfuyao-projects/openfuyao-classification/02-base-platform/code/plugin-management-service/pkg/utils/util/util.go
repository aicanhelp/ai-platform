/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

/*
Package util include plugin-management-service level util function
*/
package util

import (
	"io"
	"mime/multipart"
)

// ClearByte clear byte slice by setting every index to zero
func ClearByte(value []byte) {
	for i := range value {
		value[i] = 0
	}
}

// Contains return if string slice contains string
func Contains(s []string, str string) bool {
	for _, a := range s {
		if a == str {
			return true
		}
	}
	return false
}

// ContainsOne return if leftArr slice contains one of the element in rightArr
func ContainsOne(leftArr, rightArr []string) bool {
	set := make(map[string]struct{})
	for _, s := range leftArr {
		set[s] = struct{}{}
	}

	for _, s := range rightArr {
		if _, ok := set[s]; ok {
			return true
		}
	}
	return false
}

// ContainsAll return if outer string slice contains inner string slice
func ContainsAll(outer, inner []string) bool {
	set := make(map[string]struct{})
	for _, s := range outer {
		set[s] = struct{}{}
	}

	for _, s := range inner {
		if _, ok := set[s]; !ok {
			return false
		}
	}
	return true
}

// CheckFileSize reads the file in chunks and checks if its size exceeds the limit
func CheckFileSize(file multipart.File, bufferSize, maxFileSize int64) (bool, error) {
	var size int64
	buffer := make([]byte, bufferSize)
	for {
		n, err := (file).Read(buffer)
		size += int64(n)
		if err != nil {
			if err == io.EOF {
				break
			}
			return false, err
		}
		// If size exceeds the limit, return false
		if size > maxFileSize {
			return false, nil
		}
	}
	// Final check to ensure size limit was not exceeded at EOF
	if size > maxFileSize {
		return false, nil
	}
	return true, nil
}
