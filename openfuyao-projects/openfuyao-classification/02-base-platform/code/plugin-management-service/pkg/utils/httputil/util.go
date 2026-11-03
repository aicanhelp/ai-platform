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

// Package httputil offer http related utilities
package httputil

import (
	"plugin-management-service/pkg/constant"
)

// ResponseJson Http Response
type ResponseJson struct {
	Code int32  `json:"code,omitempty"`
	Msg  string `json:"msg,omitempty"`
	Data any    `json:"data,omitempty"`
}

// GetResponseJson get restful response struct
func GetResponseJson(code int32, msg string, data any) *ResponseJson {
	return &ResponseJson{
		Code: code,
		Msg:  msg,
		Data: data,
	}
}

// GetDefaultSuccessResponseJson get default success response json
func GetDefaultSuccessResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: nil,
	}
}

// GetDefaultClientFailureResponseJson get default failure response json
func GetDefaultClientFailureResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ClientError,
		Msg:  "bad request",
		Data: nil,
	}
}

// GetDefaultServerFailureResponseJson get default failure response json
func GetDefaultServerFailureResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ServerError,
		Msg:  "remote server busy",
		Data: nil,
	}
}

// GetParamsEmptyErrorResponseJson get default resource empty response json
func GetParamsEmptyErrorResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ClientError,
		Msg:  "parameters not found",
		Data: nil,
	}
}
