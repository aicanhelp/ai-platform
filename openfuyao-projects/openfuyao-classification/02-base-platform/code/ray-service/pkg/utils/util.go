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

func Filter[T any](ori []T, f func(T) bool) (res []T) {
	for _, v := range ori {
		if f(v) {
			res = append(res, v)
		}
	}
	return
}

func Map[T, U any](ori []T, f func(T) U) (res []U) {
	for _, v := range ori {
		res = append(res, f(v))
	}
	return
}

func MapInPlace[T any](ori []T, f func(*T)) {
	for index := range ori {
		f(&ori[index])
	}
}

func Reduce[T, U any](ori []T, f func(U, T) U, init U) (res U) {
	res = init
	for _, v := range ori {
		res = f(res, v)
	}
	return
}
