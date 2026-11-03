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

// Package handlestatus provides utility functions for handling HTTP responses
// and error logging within a web service environment. This package is designed
// to facilitate the standardized response to client requests, especially in
// situations where errors need to be communicated back to the client securely
// and informatively.
package handlestatus

import (
	"net/http"
	"runtime"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/klog/v2"
)

const (
	callerSkipCount = 2
)

var htmlSanitizer = strings.NewReplacer(`&`, "&amp;", `<`, "&lt;", `>`, "&gt;")

// BadRequest writes 400 status and logs the error with file and line info.
func BadRequest(resp *restful.Response, req *restful.Request, err error) {
	writeError(http.StatusBadRequest, resp, req, err)
}

// writeError logs and writes an HTTP error.
func writeError(code int, resp *restful.Response, req *restful.Request, err error) {
	_, file, line, _ := runtime.Caller(callerSkipCount)
	klog.Errorf("%d:%s:%d %v", code, file, line, err)
	http.Error(resp, htmlSanitizer.Replace(err.Error()), code)
}
