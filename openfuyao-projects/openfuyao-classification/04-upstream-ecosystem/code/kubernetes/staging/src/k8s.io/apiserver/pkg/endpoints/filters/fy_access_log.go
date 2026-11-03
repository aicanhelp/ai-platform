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

package filters

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

const (
	layoutOfTime = "2006-01-02 15:04:05.999999999"
)

type wrapResponseWriter struct {
	http.ResponseWriter
	code         int
	responseSize int64
}

func (w *wrapResponseWriter) CloseNotify() <-chan bool {
	closeNotifier, ok := w.ResponseWriter.(http.CloseNotifier)
	if ok {
		return closeNotifier.CloseNotify()
	}
	return nil
}

func (w *wrapResponseWriter) Flush() {
	flusher, ok := w.ResponseWriter.(http.Flusher)
	if ok {
		flusher.Flush()
	}
}

func (w *wrapResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	requestHijacker, ok := w.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("request connection cannot be hijacked: %T", w.ResponseWriter)
	}
	return requestHijacker.Hijack()
}

func (w *wrapResponseWriter) Write(writtenByte []byte) (int, error) {
	w.responseSize += int64(len(writtenByte))
	return w.ResponseWriter.Write(writtenByte)
}

func (w *wrapResponseWriter) WriteHeader(statusCode int) {
	w.ResponseWriter.WriteHeader(statusCode)
	w.code = statusCode
}

// WithAccessLog is used to recorde access log
func WithAccessLog(handler http.Handler, hwAccessLogWriter io.Writer) http.Handler {
	return getHandlerForAccessLog(handler, hwAccessLogWriter)
}

// WithFailedAuthenticationAccessLog is used to recorde access log when authenticate failed
func WithFailedAuthenticationAccessLog(failedHandler http.Handler, hwAccessLogWriter io.Writer) http.Handler {
	return getHandlerForAccessLog(failedHandler, hwAccessLogWriter)
}

func getHandlerForAccessLog(handler http.Handler, hwAccessLogWriter io.Writer) http.Handler {
	if hwAccessLogWriter == nil {
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		info, err := newAccessRecordFrom(req)
		if err != nil {
			runtime.HandleError(fmt.Errorf("failed to create access log info: %v", err))
			handler.ServeHTTP(w, req)
			return
		}

		wrap := &wrapResponseWriter{ResponseWriter: w}
		defer processAccessInfoAfterProcessingTheRequest(info, hwAccessLogWriter, wrap)
		handler.ServeHTTP(wrap, req)
	})
}

func processAccessInfoAfterProcessingTheRequest(
	info *accessInfo,
	hwAccessLogWriter io.Writer,
	wrap *wrapResponseWriter,
) {
	completedTime := time.Now()
	latency := completedTime.Sub(info.requestReceivedTimestamp)
	info.latencyTime = latency.String()
	info.responseStatusCode = wrap.code
	info.responseSize = wrap.responseSize

	processAccessInfo(info, hwAccessLogWriter)
}

func processAccessInfo(info *accessInfo, hwAccessLogWriter io.Writer) {
	logEntry := fmt.Sprintf("%s, %s, %s, %s, %s, %d, %s, %d\n",
		info.requestReceivedTimestamp.Format(layoutOfTime), info.user, info.sourceIPs, info.verb, info.requestURI,
		info.responseSize, info.latencyTime, info.responseStatusCode)
	_, writingErr := hwAccessLogWriter.Write([]byte(logEntry))
	if writingErr != nil {
		runtime.HandleError(fmt.Errorf("error writing to access log file: %w", writingErr))
	}
}

func newAccessRecordFrom(req *http.Request) (*accessInfo, error) {
	requestReceivedTimestamp, ok := request.ReceivedTimestampFrom(req.Context())
	if !ok {
		requestReceivedTimestamp = time.Now()
	}

	ai := &accessInfo{}
	ctx := req.Context()
	requestInfo, found := request.RequestInfoFrom(ctx)
	if !found {
		return nil, errors.New("no RequestInfo found in the context")
	}

	ai.requestReceivedTimestamp = requestReceivedTimestamp
	ai.verb = requestInfo.Verb
	ai.requestURI = requestInfo.Path
	user, ok := request.UserFrom(ctx)
	if ok {
		ai.user = user.GetName()
	}

	ips := utilnet.SourceIPs(req)
	ai.sourceIPs = make([]string, len(ips))
	for i := range ips {
		ai.sourceIPs[i] = ips[i].String()
	}

	return ai, nil
}

type accessInfo struct {
	requestURI               string
	sourceIPs                []string
	requestReceivedTimestamp time.Time
	verb                     string
	responseStatusCode       int
	latencyTime              string
	user                     string
	responseSize             int64
}
