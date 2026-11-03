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
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"

	"huawei.com/xpu-exporter/common"
	"huawei.com/xpu-exporter/common/cache"
)

const (
	maxConcur2 = 2
)

func initTestData() (*limitHandler, StatusResponseWriter, *http.Request) {
	conChan := make(chan struct{}, 1)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	w := StatusResponseWriter{
		ResponseWriter: nil,
		Status:         0,
	}
	r := &http.Request{
		URL: &url.URL{
			Path: "test.com",
		},
		Header: map[string][]string{"userID": {"1"}, "reqID": {"requestIDxxxx"}},
		Method: "GET",
	}
	return lh, w, r
}

func TestServeHTTP(t *testing.T) {
	h, w, r := initTestData()
	var mock = gomonkey.ApplyMethodFunc(h.httpHandler, "ServeHTTP", func(http.ResponseWriter,
		*http.Request) {
		_ = 0
	})
	defer mock.Reset()
	h.ServeHTTP(w.ResponseWriter, r)
	common.AssertEquals(1, len(h.concurrency), t)

	mock = gomonkey.ApplyFunc(http.Error, func(http.ResponseWriter, string, int) {
		_ = 0
	})
	defer mock.Reset()
	_, ok := <-h.concurrency
	if !ok {
		return
	}
	h.ServeHTTP(w.ResponseWriter, r)
	common.AssertEquals(0, len(h.concurrency), t)
}

func TestReturnToken(t *testing.T) {
	cancelCtx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()
	sc := make(chan struct{}, 1)
	go returnToken(cancelCtx, sc)
	time.Sleep(time.Second * 6)
	common.AssertEquals(1, len(sc), t)
}

func TestNewLimitHandler(t *testing.T) {
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: defaultMaxConcurrency,
		IPConCurrency:    "2/1",
		CacheSize:        DefaultCacheSize,
	}

	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertIsNil(err, t)

	conf.IPConCurrency = "2021/1"
	_, err = NewLimitHandler(http.DefaultServeMux, conf)
	t.Log("", err)
	common.AssertNotNil(err, t)

	conf.CacheSize = 0
	_, err = NewLimitHandler(http.DefaultServeMux, conf)
	t.Log("", err)
	common.AssertNotNil(err, t)

	conf.Method = "20/iajsdkjas2jhjdklsjkldjsdfasd1"
	_, err = NewLimitHandler(http.DefaultServeMux, conf)
	t.Log("", err)
	common.AssertNotNil(err, t)

	conf.TotalConCurrency = 0
	_, err = NewLimitHandler(http.DefaultServeMux, conf)
	t.Log("", err)
	common.AssertNotNil(err, t)
}

func TestWriteHeader(t *testing.T) {
	recorder := httptest.NewRecorder()
	w := &StatusResponseWriter{
		ResponseWriter: recorder,
		Status:         0,
	}
	w.WriteHeader(http.StatusNotFound)
	common.AssertEquals(http.StatusNotFound, w.Status, t)
	common.AssertEquals(http.StatusNotFound, recorder.Code, t)
}

func TestCheckIPRequestLimitBelowLimit(t *testing.T) {
	conChan := make(chan struct{}, 1)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
	}
	result := lh.checkIPRequestLimit("127.0.0.1", "/test", r)
	common.AssertEquals(true, result, t)
	counter, exists := lh.ipRequestCounters["127.0.0.1"]
	if !exists {
		t.Fatal("counter not found")
	}
	common.AssertEquals(1, counter.requests, t)
}

func TestCheckIPRequestLimitExceeded(t *testing.T) {
	conChan := make(chan struct{}, 1)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	lh.ipRequestCounters["127.0.0.1"] = &ipRequestCounter{
		requests: MaxIPRequestsPerMinute,
		timer:    time.NewTimer(time.Minute),
	}
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
	}
	result := lh.checkIPRequestLimit("127.0.0.1", "/test", r)
	common.AssertEquals(false, result, t)
}

func TestServeHTTP429(t *testing.T) {
	conChan := make(chan struct{}, maxConcur2)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	lh.ipRequestCounters["127.0.0.1"] = &ipRequestCounter{
		requests: MaxIPRequestsPerMinute,
		timer:    time.NewTimer(time.Minute),
	}
	recorder := httptest.NewRecorder()
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
		Header: map[string][]string{"X-Forwarded-For": {"127.0.0.1"}},
	}
	lh.ServeHTTP(recorder, r)
	common.AssertEquals(http.StatusTooManyRequests, recorder.Code, t)
}

func TestServeHTTP503IPCache(t *testing.T) {
	conChan := make(chan struct{}, maxConcur2)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	lh.ipCache = cache.New(DefaultCacheSize)
	common.AssertNotNil(lh.ipCache, t)
	lh.ipExpiredTime = time.Minute
	err := lh.ipCache.Set(fmt.Sprintf("key-%s", "127.0.0.1"), "v", time.Minute)
	common.AssertIsNil(err, t)
	recorder := httptest.NewRecorder()
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
		Header: map[string][]string{"X-Forwarded-For": {"127.0.0.1"}},
	}
	lh.ServeHTTP(recorder, r)
	common.AssertEquals(http.StatusServiceUnavailable, recorder.Code, t)
}

func TestServeHTTPMethodMismatch(t *testing.T) {
	conChan := make(chan struct{}, maxConcur2)
	lh := createHandler(conChan, http.DefaultServeMux, false, "POST", DefaultDataLimit)
	recorder := httptest.NewRecorder()
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
		Header: map[string][]string{},
	}
	lh.ServeHTTP(recorder, r)
	common.AssertEquals(http.StatusNotFound, recorder.Code, t)
}

func TestServeHTTPClosedChannel(t *testing.T) {
	conChan := make(chan struct{}, 1)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	<-lh.concurrency
	close(lh.concurrency)
	recorder := httptest.NewRecorder()
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
		Header: map[string][]string{},
	}
	lh.ServeHTTP(recorder, r)
	common.AssertEquals(http.StatusOK, recorder.Code, t)
}

func TestServeHTTPWithLog(t *testing.T) {
	conChan := make(chan struct{}, maxConcur2)
	lh := createHandler(conChan, http.DefaultServeMux, true, "", DefaultDataLimit)
	var mock = gomonkey.ApplyMethodFunc(lh.httpHandler, "ServeHTTP",
		func(http.ResponseWriter, *http.Request) { _ = 0 })
	defer mock.Reset()
	recorder := httptest.NewRecorder()
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
		Header: map[string][]string{},
		Proto:  "HTTP/1.1",
	}
	lh.ServeHTTP(recorder, r)
}

func TestNewLimitHandlerNilConf(t *testing.T) {
	_, err := NewLimitHandler(http.DefaultServeMux, nil)
	common.AssertNotNil(err, t)
}

func TestNewLimitHandlerExceedMaxConcurrency(t *testing.T) {
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: defaultMaxConcurrency + 1,
		IPConCurrency:    "2/1",
		CacheSize:        DefaultCacheSize,
	}
	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertNotNil(err, t)
}

func TestNewLimitHandlerDefaultCacheSize(t *testing.T) {
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: 1,
		IPConCurrency:    "2/1",
		CacheSize:        0,
	}
	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertIsNil(err, t)
	common.AssertEquals(DefaultCacheSize, conf.CacheSize, t)
}

func TestReturnTokenContextCancel(t *testing.T) {
	cancelCtx, cancelFunc := context.WithCancel(context.Background())
	cancelFunc()
	sc := make(chan struct{}, 1)
	go returnToken(cancelCtx, sc)
	time.Sleep(time.Millisecond * 500)
	common.AssertEquals(0, len(sc), t)
}

func TestReturnTokenNilConcurrency(t *testing.T) {
	cancelCtx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()
	returnToken(cancelCtx, nil)
}

func TestReturnTokenRecover(t *testing.T) {
	closed := make(chan struct{})
	close(closed)
	ready := make(chan time.Time, 1)
	ready <- time.Now()
	patchAfter := gomonkey.ApplyFunc(time.After, func(_ time.Duration) <-chan time.Time {
		return ready
	})
	defer patchAfter.Reset()
	returnToken(context.Background(), closed)
}

func TestReturnTokenTimeoutClosed(t *testing.T) {
	closedCh := make(chan time.Time)
	close(closedCh)
	patchAfter := gomonkey.ApplyFunc(time.After, func(_ time.Duration) <-chan time.Time {
		return closedCh
	})
	defer patchAfter.Reset()
	returnToken(context.Background(), make(chan struct{}, 1))
}

func TestNewLimitHandlerZeroIPPrefix(t *testing.T) {
	anyRegex := regexp.MustCompile(`^.*$`)
	patchReg := gomonkey.ApplyFunc(regexp.MustCompile, func(_ string) *regexp.Regexp {
		return anyRegex
	})
	defer patchReg.Reset()
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: 1,
		IPConCurrency:    "0/1",
		CacheSize:        DefaultCacheSize,
	}
	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertNotNil(err, t)
}

func TestNewLimitHandlerParseArr1Error(t *testing.T) {
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: 1,
		IPConCurrency:    "2/1",
		CacheSize:        DefaultCacheSize,
	}
	patchParse := gomonkey.ApplyFunc(strconv.ParseInt, func(_ string, _ int, _ int) (int64, error) {
		return 0, fmt.Errorf("mock parse error")
	})
	defer patchParse.Reset()
	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertNotNil(err, t)
}

func TestNewLimitHandlerParseArr0Error(t *testing.T) {
	conf := &HandlerConfig{
		PrintLog:         false,
		Method:           "",
		LimitBytes:       DefaultDataLimit,
		TotalConCurrency: 1,
		IPConCurrency:    "2/1",
		CacheSize:        DefaultCacheSize,
	}
	calls := 0
	patchParse := gomonkey.ApplyFunc(strconv.ParseInt, func(_ string, _ int, _ int) (int64, error) {
		calls++
		if calls == 1 {
			return 1, nil
		}
		return 0, fmt.Errorf("mock parse error")
	})
	defer patchParse.Reset()
	_, err := NewLimitHandler(http.DefaultServeMux, conf)
	common.AssertNotNil(err, t)
}

func TestCheckIPRequestLimitTimerCallback(t *testing.T) {
	conChan := make(chan struct{}, 1)
	lh := createHandler(conChan, http.DefaultServeMux, false, "", DefaultDataLimit)
	r := &http.Request{
		Method: "GET",
		URL:    &url.URL{Path: "/test"},
	}
	lh.checkIPRequestLimit("127.0.0.1", "/test", r)
	lh.mu.Lock()
	counter, exists := lh.ipRequestCounters["127.0.0.1"]
	lh.mu.Unlock()
	if !exists {
		t.Fatal("counter not found")
	}
	counter.timer.Reset(time.Millisecond)
	time.Sleep(time.Millisecond * 50)
	lh.mu.Lock()
	_, exists = lh.ipRequestCounters["127.0.0.1"]
	lh.mu.Unlock()
	common.AssertEquals(false, exists, t)
}
