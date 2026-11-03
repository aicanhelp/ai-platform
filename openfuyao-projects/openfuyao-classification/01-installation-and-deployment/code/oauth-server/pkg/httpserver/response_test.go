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

package httpserver

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"openfuyao/oauth-server/pkg/zlog"
)

const (
	testStatusCodeOK          = 200
	testStatusCodeBadRequest  = 400
	testStatusCodeServerError = 500
	testErrorCodeBadRequest   = 1001
	testErrorCodeOK           = 0
	testErrorCodeServer       = 5001
)

var (
	testMsgOK           = "OK"
	testMsgBadRequest   = "Bad Request"
	testMsgServerError  = "Internal Server Error"
	testContentTypeJSON = "application/json"
	testJsonCodeField   = "Code"
)

func TestHttpResponseStruct(t *testing.T) {
	// Test HttpResponse struct creation and fields
	response := &HttpResponse{
		Code: testErrorCodeOK,
		Msg:  testMsgOK,
		Data: map[string]interface{}{
			"key": "value",
		},
	}

	assert.Equal(t, int32(testErrorCodeOK), response.Code)
	assert.Equal(t, testMsgOK, response.Msg)
	assert.Equal(t, "value", response.Data["key"])

	// Test HttpResponse with nil Data
	responseNilData := &HttpResponse{
		Code: testErrorCodeBadRequest,
		Msg:  testMsgBadRequest,
		Data: nil,
	}

	assert.Equal(t, int32(testErrorCodeBadRequest), responseNilData.Code)
	assert.Equal(t, testMsgBadRequest, responseNilData.Msg)
	assert.Nil(t, responseNilData.Data)
}

func TestRespondWithStatusMsg(t *testing.T) {
	// Create a response recorder
	recorder := httptest.NewRecorder()

	// Test successful response
	RespondWithStatusMsg(recorder, testStatusCodeBadRequest, testErrorCodeBadRequest, testMsgBadRequest)

	// Check status code
	assert.Equal(t, testStatusCodeBadRequest, recorder.Code)

	// Check content type
	assert.Equal(t, testContentTypeJSON, recorder.Header().Get("Content-Type"))

	// Check response body
	var response HttpResponse
	err := json.Unmarshal(recorder.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, int32(testErrorCodeBadRequest), response.Code)
	assert.Equal(t, testMsgBadRequest, response.Msg)
	assert.Nil(t, response.Data)
}

func TestRespondWithStatusMsgWithData(t *testing.T) {
	// Create a response recorder
	recorder := httptest.NewRecorder()

	// Test response with data
	RespondWithStatusMsg(recorder, testStatusCodeOK, testErrorCodeOK, testMsgOK)

	// Check status code
	assert.Equal(t, testStatusCodeOK, recorder.Code)

	// Check content type
	assert.Equal(t, testContentTypeJSON, recorder.Header().Get("Content-Type"))

	// Check response body
	var response HttpResponse
	err := json.Unmarshal(recorder.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, int32(testErrorCodeOK), response.Code)
	assert.Equal(t, testMsgOK, response.Msg)
	assert.Nil(t, response.Data)
}

func TestRespondWithStatusMsgJsonEncodeError(t *testing.T) {
	// Create a response recorder
	recorder := httptest.NewRecorder()

	// Mock zlog.LogFatalf to prevent test from exiting
	patches := gomonkey.ApplyFunc(zlog.LogFatalf, func(format string, args ...interface{}) {
		// Do nothing to prevent test from exiting
	})
	defer patches.Reset()

	// Create a custom ResponseWriter that causes encoding to fail
	badRecorder := &BadResponseWriter{ResponseRecorder: recorder}

	// This should not panic even if json encoding fails
	assert.NotPanics(t, func() {
		RespondWithStatusMsg(badRecorder, testStatusCodeServerError, testErrorCodeServer, testMsgServerError)
	})
}

// BadResponseWriter is a custom ResponseWriter that causes json encoding to fail
type BadResponseWriter struct {
	*httptest.ResponseRecorder
}

// Write overrides the default Write to simulate encoding failure
func (b *BadResponseWriter) Write(data []byte) (int, error) {
	// Simulate a write error when json content is detected
	if strings.Contains(string(data), testJsonCodeField) {
		// Return an error to simulate encoding failure
		return 0, &json.UnsupportedTypeError{}
	}
	return b.ResponseRecorder.Write(data)
}
