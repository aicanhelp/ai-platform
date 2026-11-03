/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package limiter

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/common/cache"
	"huawei.com/xpu-exporter/common/utils"
)

const (
	kilo = 1000.0

	DefaultDataLimit      = 1024 * 1024 * 10
	defaultMaxConcurrency = 1024
	httpMaxWaitTime       = 5
	maxStringLen          = 20

	DefaultCacheSize = 1024 * 100
	arrLen           = 2

	IPReqLimitReg = "^[1-9]\\d{0,2}/[1-9]\\d{0,2}$"

	MaxIPRequestsPerMinute = 20
)

// limitHandler implements HTTP request limiting with IP-based rate control
type limitHandler struct {
	concurrency       chan struct{}
	httpHandler       http.Handler
	log               bool
	method            string
	limitBytes        int64
	ipExpiredTime     time.Duration
	ipCache           *cache.ConcurrencyLRUCache
	ipRequestCounters map[string]*ipRequestCounter
	mu                sync.Mutex
}

// ipRequestCounter tracks request count and timer for IP-based rate limiting
type ipRequestCounter struct {
	requests int
	timer    *time.Timer
}

// HandlerConfig defines configuration parameters for the rate limiter
type HandlerConfig struct {
	PrintLog bool

	Method string

	LimitBytes int64

	TotalConCurrency int

	IPConCurrency string

	CacheSize int
}

// StatusResponseWriter extends http.ResponseWriter to capture response status
type StatusResponseWriter struct {
	http.ResponseWriter
	http.Hijacker
	Status int
}

// WriteHeader records the HTTP status code before delegating to the underlying writer
func (w *StatusResponseWriter) WriteHeader(status int) {
	w.ResponseWriter.WriteHeader(status)
	w.Status = status
}

// checkIPRequestLimit enforces per-minute request limit for client IP addresses
func (h *limitHandler) checkIPRequestLimit(clientIP, path string, req *http.Request) bool {
	h.mu.Lock()
	defer h.mu.Unlock()

	counter, exists := h.ipRequestCounters[clientIP]
	if !exists {
		counter = &ipRequestCounter{
			requests: 0,
			timer: time.AfterFunc(time.Minute, func() {
				h.mu.Lock()
				delete(h.ipRequestCounters, clientIP)
				h.mu.Unlock()
			}),
		}
		h.ipRequestCounters[clientIP] = counter
	}
	if counter.requests >= MaxIPRequestsPerMinute {
		log.Warningf("IP request limit exceeded:%s: %s <%3d> |%15s |%s |%d ", req.Method,
			path, http.StatusTooManyRequests, clientIP, req.UserAgent(), syscall.Getuid())
		return false
	}
	counter.requests++
	return true
}

// ServeHTTP implements http.Handler interface with rate limiting and concurrency control
func (h *limitHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	req.Body = http.MaxBytesReader(w, req.Body, h.limitBytes)
	ctx := initContext(req)
	path := req.URL.Path
	clientUserAgent := req.UserAgent()
	clientIP := utils.ClientIP(req)

	if !h.checkIPRequestLimit(clientIP, path, req) {
		http.Error(w, "429 Too Many Requests", http.StatusTooManyRequests)
		return
	}

	if clientIP != "" && h.ipCache != nil {
		if !h.ipCache.SetIfNotExist(fmt.Sprintf("key-%s", clientIP), "v", h.ipExpiredTime) {
			log.Warningf("Single IP request reject:%s: %s <%3d> |%15s |%s |%d ", req.Method,
				path, http.StatusServiceUnavailable, clientIP, clientUserAgent, syscall.Getuid())
			http.Error(w, "503 too busy", http.StatusServiceUnavailable)
			return
		}
	}
	select {
	case _, ok := <-h.concurrency:
		if !ok {
			return
		}
		if h.method != "" && req.Method != h.method {
			http.NotFound(w, req)
			h.concurrency <- struct{}{}
			return
		}
		log.Infof("token count:%d", len(h.concurrency))
		cancelCtx, cancelFunc := context.WithCancel(ctx)
		start := time.Now()
		go returnToken(cancelCtx, h.concurrency)
		statusRes := newResponse(w)
		h.httpHandler.ServeHTTP(statusRes, req)
		stop := time.Since(start)
		cancelFunc()
		if stop < httpMaxWaitTime*time.Second {
			h.concurrency <- struct{}{}
		}
		latency := int(math.Ceil(float64(stop.Nanoseconds()) / kilo / kilo))
		if h.log {
			log.Infof("%s %s: %s <%3d> (%dms) |%15s |%s |%d", req.Proto, req.Method, path,
				statusRes.Status, latency, clientIP, clientUserAgent, syscall.Getuid())
		}
	default:
		log.Warningf("Total reject request:%s: %s <%3d> |%15s |%s |%d", req.Method, path,
			http.StatusServiceUnavailable, clientIP, clientUserAgent, syscall.Getuid())
		http.Error(w, "503 too busy", http.StatusServiceUnavailable)
	}
}

// newResponse creates a response writer that tracks HTTP status codes
func newResponse(w http.ResponseWriter) *StatusResponseWriter {
	jk, ok := w.(http.Hijacker)
	if !ok {
		log.Warningln("hijack not implement")
	}
	statusRes := &StatusResponseWriter{
		ResponseWriter: w,
		Status:         http.StatusOK,
		Hijacker:       jk,
	}
	return statusRes
}

// initContext creates a background context for HTTP request processing
func initContext(req *http.Request) context.Context {
	return context.Background()
}

// returnToken manages concurrency token return with timeout protection
func returnToken(ctx context.Context, concurrency chan struct{}) {
	defer func() {
		if err := recover(); err != nil {
			log.Errorf("go routine failed with %v", err)
		}
	}()

	timeAfterTrigger := time.After(time.Second * httpMaxWaitTime)
	if concurrency == nil || timeAfterTrigger == nil {
		log.Errorln("return token error")
		return
	}
	for {
		select {
		case _, ok := <-timeAfterTrigger:
			if !ok {
				return
			}
			concurrency <- struct{}{}
			log.Infof("recover token num：%d", len(concurrency))
			return
		case _, ok := <-ctx.Done():
			err := ctx.Err()
			if !ok || err != nil {
				log.Infof("%+v:%+v", err, ok)
			}
			return
		}

	}
}

// createHandler initializes a new rate limiting handler with concurrency control
func createHandler(ch chan struct{}, handler http.Handler, printLog bool,
	httpMethod string, bodySizeLimit int64) *limitHandler {
	h := &limitHandler{
		concurrency:       ch,
		httpHandler:       handler,
		log:               printLog,
		method:            httpMethod,
		limitBytes:        bodySizeLimit,
		ipExpiredTime:     time.Duration(-1),
		ipRequestCounters: make(map[string]*ipRequestCounter),
	}
	for i := 0; i < cap(ch); i++ {
		h.concurrency <- struct{}{}
	}
	return h
}

// NewLimitHandler creates a new HTTP handler with rate limiting capabilities
func NewLimitHandler(handler http.Handler, conf *HandlerConfig) (http.Handler, error) {
	if conf == nil {
		return nil, errors.New("parameter error")
	}
	if conf.TotalConCurrency < 1 || conf.TotalConCurrency > defaultMaxConcurrency {
		return nil, errors.New("totalConCurrency parameter error")
	}
	if len(conf.Method) > maxStringLen {
		return nil, errors.New("http method error")
	}
	if conf.CacheSize <= 0 {
		log.Infoln("use default cache size")
		conf.CacheSize = DefaultCacheSize
	}

	reg := regexp.MustCompile(IPReqLimitReg)
	if !reg.Match([]byte(conf.IPConCurrency)) {
		return nil, errors.New("IPConCurrency parameter error")
	}

	conChan := make(chan struct{}, conf.TotalConCurrency)
	h := createHandler(conChan, handler, conf.PrintLog, conf.Method, conf.LimitBytes)

	arr := strings.Split(conf.IPConCurrency, "/")
	if len(arr) != arrLen || arr[0] == "0" {
		return nil, errors.New("IPConCurrency parameter error")
	}

	arr1, err := strconv.ParseInt(arr[1], 0, 0)
	if err != nil {
		return nil, fmt.Errorf("IPConCurrency parameter(%s) error, parse to int failed: %v", arr[1], err)
	}
	arr0, err := strconv.ParseInt(arr[0], 0, 0)
	if err != nil || arr0 == 0 {
		return nil, fmt.Errorf("IPConCurrency parameter(%s) error,parse to int failed: %v", arr[0], err)
	}
	h.ipExpiredTime = time.Duration(arr1 * int64(time.Second) / arr0)
	return h, nil
}
