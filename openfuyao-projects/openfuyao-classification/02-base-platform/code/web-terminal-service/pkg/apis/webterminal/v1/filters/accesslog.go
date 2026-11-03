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

// Package filters provides HTTP middleware functions to enhance the functionality
// of web services by applying additional processing to the HTTP requests and responses.
// This package is primarily designed to integrate seamlessly into the HTTP request handling
// pipeline, enabling pre- and post-processing of HTTP traffic.
package filters

import (
	"net/http"
	"time"

	"github.com/emicklei/go-restful/v3"

	"openfuyao.com/web-terminal-service/pkg/zlog"
)

// RecordAccessLogs logs HTTP requests and responses in the server.
// It records the method, URL, protocol, status code, content length, and response time of each request.
// Requests with status codes greater than 400 are logged as warnings to highlight potential errors or issues.
func RecordAccessLogs(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	logRequest(req, resp, chain, resp.StatusCode() > http.StatusBadRequest)
}

func logRequest(req *restful.Request, resp *restful.Response, chain *restful.FilterChain, isError bool) {
	start := time.Now()
	chain.ProcessFilter(req, resp)
	statusCode := resp.StatusCode()
	contentLength := resp.ContentLength()
	elapsedTime := time.Since(start).Milliseconds()

	var logFunc func(string, ...interface{})
	if isError {
		logFunc = zlog.LogWarnf
	} else {
		logFunc = zlog.LogInfof
	}

	logFunc("%s %s %s %d %d %dms",
		req.Request.Method,
		req.Request.URL,
		req.Request.Proto,
		statusCode,
		contentLength,
		elapsedTime,
	)
}
