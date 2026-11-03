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

package filters

import (
	"net/http"
	"time"

	"github.com/emicklei/go-restful/v3"

	"console-service/pkg/utils/iputil"
	"console-service/pkg/zlog"
)

type logFunction func(format string, args ...interface{})

func logResponse(req *restful.Request, resp *restful.Response, start time.Time, logFunc logFunction) {
	logFunc("HTTP request details: method=%s, address=%s, url=%s, proto=%s, status=%d, length=%d, duration=%dms",
		req.Request.Method,
		iputil.RemoteIp(req.Request),
		req.Request.URL,
		req.Request.Proto,
		resp.StatusCode(),
		resp.ContentLength(),
		time.Since(start).Milliseconds(),
	)
}

// RecordAccessLogs log recording function
func RecordAccessLogs(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	start := time.Now()
	chain.ProcessFilter(req, resp)
	// StatusBadRequest错误码是400，大于400的StatusCode都是各种不同的http错误
	if resp.StatusCode() > http.StatusBadRequest {
		logResponse(req, resp, start, zlog.Warnf)
	} else {
		logResponse(req, resp, start, zlog.Infof)
	}
}
