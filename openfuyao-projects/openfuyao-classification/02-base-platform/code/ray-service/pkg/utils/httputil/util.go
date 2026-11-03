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

// Package httputil
package httputil

import (
	"bytes"
	"errors"
	"io"
	"net/http"

	"openfuyao-rayservice/pkg/constant"
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

// GetClientFailureResponseJsonWithData get data failure response json
func GetClientFailureResponseJsonWithData(data any) *ResponseJson {
	return &ResponseJson{
		Code: constant.ClientError,
		Msg:  "bad request",
		Data: data,
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

// GetServerFailureResponseJsonWithData get data failure response json
func GetServerFailureResponseJsonWithData(data any) *ResponseJson {
	return &ResponseJson{
		Code: constant.ServerError,
		Msg:  "operation failed",
		Data: data,
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

func ReceiveFile(r *http.Request, key string) (string, error) {
	if r == nil {
		return "", errors.New("request is nil")
	}
	var buf bytes.Buffer
	file, _, err := r.FormFile(key)
	if err != nil {
		return "", err
	}
	defer file.Close()

	// 修改开始：检查 io.Copy 的错误
	_, err = io.Copy(&buf, file)
	if err != nil {
		// 如果复制过程中出错（例如，上传中断），返回错误
		return "", err
	}
	// 修改结束

	contents := buf.String()
	// buf.Reset() 在这里不是必需的，因为 buf 是局部变量，函数结束后会被垃圾回收。
	// 如果您想显式释放内存，可以保留它，但对性能提升微乎其微。
	buf.Reset()
	return contents, nil
}

func BuildSuccessResponse(data interface{}) *ResponseJson {
	return &ResponseJson{
		Code: int32(constant.Success),
		Msg:  "success",
		Data: data,
	}
}

func BuildErrorResponse(code int32, msg string) *ResponseJson {
	return &ResponseJson{
		Code: code,
		Msg:  msg,
		Data: []any{},
	}
}
