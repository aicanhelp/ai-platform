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

// Package responsehandlers provides utilities for handling HTTP responses.
package responsehandlers

import (
	"net/http"
	"runtime"

	"github.com/emicklei/go-restful/v3"

	"openfuyao.com/web-terminal-service/pkg/zlog"
)

const (
	callerSkipCount = 1
)

func logErrorInfo(err error) {
	if err != nil {
		_, fn, line, _ := runtime.Caller(callerSkipCount)
		zlog.LogErrorf("%s:%d %v", fn, line, err)
	}
	return
}

// SendStatusOk sends http.StatusOK.
func SendStatusOk(resp *restful.Response, message string) {
	resp.WriteHeaderAndEntity(http.StatusOK, message)
}

// SendStatusBadRequest writes http.StatusBadRequest and log error.
func SendStatusBadRequest(resp *restful.Response, message string, err error) {
	logErrorInfo(err)

	resp.WriteHeaderAndEntity(http.StatusBadRequest, restful.ServiceError{
		Code:    http.StatusBadRequest,
		Message: message,
	})
}

// SendStatusServerError writes http.StatusInternalServerError and log error.
func SendStatusServerError(resp *restful.Response, message string, err error) {
	logErrorInfo(err)

	resp.WriteHeaderAndEntity(http.StatusInternalServerError, restful.ServiceError{
		Code:    http.StatusInternalServerError,
		Message: message,
	})
}

// SendStatusForbidden writes http.StatusForbidden.
func SendStatusForbidden(resp *restful.Response, message string) {
	resp.WriteHeaderAndEntity(http.StatusForbidden, restful.ServiceError{
		Code:    http.StatusForbidden,
		Message: message,
	})
}
