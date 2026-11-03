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
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/zlog"
)

// createFilterChain creates a real restful.FilterChain for testing
func createFilterChain(processFilterCalled *bool, delay time.Duration) (*restful.FilterChain, *gomonkey.Patches) {
	// Create a container to get a real FilterChain
	container := restful.NewContainer()
	ws := new(restful.WebService)
	ws.Route(ws.GET("/test").To(func(req *restful.Request, resp *restful.Response) {
		resp.WriteHeader(http.StatusOK)
	}))
	container.Add(ws)

	// Create a filter chain - we'll patch ProcessFilter method
	chain := &restful.FilterChain{}

	// Patch ProcessFilter method to track calls and add delays
	patches := gomonkey.NewPatches()
	patches.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter", func(_ *restful.FilterChain, req *restful.Request, resp *restful.Response) {
		if processFilterCalled != nil {
			*processFilterCalled = true
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	})

	return chain, patches
}

func TestRecordAccessLogsSuccess(t *testing.T) {
	// Create a mock HTTP request
	httpReq := httptest.NewRequest("GET", "/api/v1/metrics?time=2024-01-01%2010:00:00", nil)
	httpReq.Proto = "HTTP/1.1"

	// Create restful request and response
	req := restful.NewRequest(httpReq)
	resp := restful.NewResponse(httptest.NewRecorder())
	resp.WriteHeader(http.StatusOK)
	resp.Write([]byte("test response"))

	var processFilterCalled bool
	chain, chainPatches := createFilterChain(&processFilterCalled, 0)
	defer chainPatches.Reset()

	// Mock the logging functions to capture calls
	var infoCalled bool
	var warnCalled bool
	var infoArgs []interface{}

	infoPatch := gomonkey.ApplyFunc(zlog.FormatInfo, func(template string, args ...interface{}) {
		infoCalled = true
		infoArgs = args
	})
	defer infoPatch.Reset()

	warnPatch := gomonkey.ApplyFunc(zlog.FormatWarn, func(template string, args ...interface{}) {
		warnCalled = true
	})
	defer warnPatch.Reset()

	// Call RecordAccessLogs
	RecordAccessLogs(req, resp, chain)

	// Verify filter chain was called
	assert.True(t, processFilterCalled, "ProcessFilter should be called")

	// Verify info was called (status 200 < 400)
	assert.True(t, infoCalled, "FormatInfo should be called for success status")
	assert.False(t, warnCalled, "FormatWarn should not be called for success status")

	// Verify log message format
	if len(infoArgs) >= 6 {
		assert.Equal(t, "GET", infoArgs[0], "Method should be GET")
		assert.Contains(t, infoArgs[1].(string), "/api/v1/metrics", "URI should contain path")
		assert.Equal(t, "HTTP/1.1", infoArgs[2], "Protocol should be HTTP/1.1")
		assert.Equal(t, http.StatusOK, infoArgs[3], "Status code should be 200")
		assert.GreaterOrEqual(t, infoArgs[4].(int), 0, "Content length should be non-negative")
		assert.GreaterOrEqual(t, infoArgs[5].(int64), int64(0), "Elapsed time should be non-negative")
	}
}

func TestRecordAccessLogsError(t *testing.T) {
	// Create a mock HTTP request
	httpReq := httptest.NewRequest("POST", "/api/v1/invalid", nil)
	httpReq.Proto = "HTTP/1.1"

	// Create restful request and response
	req := restful.NewRequest(httpReq)
	resp := restful.NewResponse(httptest.NewRecorder())
	resp.WriteHeader(http.StatusInternalServerError)
	resp.Write([]byte("error"))

	var processFilterCalled bool
	chain, chainPatches := createFilterChain(&processFilterCalled, 0)
	defer chainPatches.Reset()

	// Mock the logging functions to capture calls
	var infoCalled bool
	var warnCalled bool
	var warnArgs []interface{}

	infoPatch := gomonkey.ApplyFunc(zlog.FormatInfo, func(template string, args ...interface{}) {
		infoCalled = true
	})
	defer infoPatch.Reset()

	warnPatch := gomonkey.ApplyFunc(zlog.FormatWarn, func(template string, args ...interface{}) {
		warnCalled = true
		warnArgs = args
	})
	defer warnPatch.Reset()

	// Call RecordAccessLogs
	RecordAccessLogs(req, resp, chain)

	// Verify filter chain was called
	assert.True(t, processFilterCalled, "ProcessFilter should be called")

	// Verify warn was called (status 500 > 400)
	assert.False(t, infoCalled, "FormatInfo should not be called for error status")
	assert.True(t, warnCalled, "FormatWarn should be called for error status")

	// Verify log message format
	if len(warnArgs) >= 6 {
		assert.Equal(t, "POST", warnArgs[0], "Method should be POST")
		assert.Contains(t, warnArgs[1].(string), "/api/v1/invalid", "URI should contain path")
		assert.Equal(t, "HTTP/1.1", warnArgs[2], "Protocol should be HTTP/1.1")
		assert.Equal(t, http.StatusInternalServerError, warnArgs[3], "Status code should be 500")
		assert.GreaterOrEqual(t, warnArgs[4].(int), 0, "Content length should be non-negative")
		assert.GreaterOrEqual(t, warnArgs[5].(int64), int64(0), "Elapsed time should be non-negative")
	}
}

func TestRecordAccessLogsBadRequest(t *testing.T) {
	// Test with status 400 (exactly at the boundary)
	// StatusBadRequest is 400, condition is > 400, so 400 should use info (not warn)
	httpReq := httptest.NewRequest("GET", "/api/v1/bad", nil)
	httpReq.Proto = "HTTP/1.1"

	req := restful.NewRequest(httpReq)
	resp := restful.NewResponse(httptest.NewRecorder())
	resp.WriteHeader(http.StatusBadRequest)

	var processFilterCalled bool
	chain, chainPatches := createFilterChain(&processFilterCalled, 0)
	defer chainPatches.Reset()

	var infoCalled bool
	var warnCalled bool

	infoPatch := gomonkey.ApplyFunc(zlog.FormatInfo, func(template string, args ...interface{}) {
		infoCalled = true
	})
	defer infoPatch.Reset()

	warnPatch := gomonkey.ApplyFunc(zlog.FormatWarn, func(template string, args ...interface{}) {
		warnCalled = true
	})
	defer warnPatch.Reset()

	RecordAccessLogs(req, resp, chain)

	// Status 400 > 400 is false, so it should use info, not warn
	assert.True(t, infoCalled, "FormatInfo should be called for status 400")
	assert.False(t, warnCalled, "FormatWarn should not be called for status 400")
}
