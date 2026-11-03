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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"volcano-config-service/pkg/server/request"
)

const (
	expectedStatusOK            = 200
	expectedStatusInternalError = 500
	testPath                    = "/test/path"
	testMethod                  = "GET"
)

// mockRequestInfoResolver implements request.RequestInfoResolver for testing
type mockRequestInfoResolver struct {
	shouldFail   bool
	returnedInfo *request.RequestInfo
}

func (m *mockRequestInfoResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	if m.shouldFail {
		return nil, errors.New("mock resolver error")
	}
	return m.returnedInfo, nil
}

// mockHandler implements http.Handler for testing
type mockHandler struct {
	called      bool
	receivedReq *http.Request
}

func (m *mockHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m.called = true
	m.receivedReq = req
	w.WriteHeader(expectedStatusOK)
}

// TestBuildRequestInfo tests the BuildRequestInfo function
func TestBuildRequestInfo(t *testing.T) {
	mockHandler := &mockHandler{}
	mockResolver := &mockRequestInfoResolver{}

	handler := BuildRequestInfo(mockHandler, mockResolver)

	if handler == nil {
		t.Fatal("BuildRequestInfo returned nil handler")
	}

	builder, ok := handler.(*requestInfoBuilder)
	if !ok {
		t.Fatal("BuildRequestInfo did not return requestInfoBuilder type")
	}

	if builder.nextHandler != mockHandler {
		t.Error("nextHandler not set correctly")
	}

	if builder.resolver != mockResolver {
		t.Error("resolver not set correctly")
	}
}

// TestServeHTTPSuccess tests successful request processing
func TestServeHTTPSuccess(t *testing.T) {
	mockHandler := &mockHandler{}
	mockInfo := &request.RequestInfo{}
	mockResolver := &mockRequestInfoResolver{
		shouldFail:   false,
		returnedInfo: mockInfo,
	}

	handler := BuildRequestInfo(mockHandler, mockResolver)

	req := createTestRequest()
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	validateSuccessfulResponse(t, recorder, mockHandler, mockInfo)
}

// TestServeHTTPResolverError tests error handling when resolver fails
func TestServeHTTPResolverError(t *testing.T) {
	mockHandler := &mockHandler{}
	mockResolver := &mockRequestInfoResolver{
		shouldFail: true,
	}

	handler := BuildRequestInfo(mockHandler, mockResolver)

	req := createTestRequest()
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	validateErrorResponse(t, recorder, mockHandler)
}

// TestServeHTTPContextPropagation tests that context is properly propagated
func TestServeHTTPContextPropagation(t *testing.T) {
	mockHandler := &mockHandler{}
	mockInfo := &request.RequestInfo{}
	mockResolver := &mockRequestInfoResolver{
		shouldFail:   false,
		returnedInfo: mockInfo,
	}

	handler := BuildRequestInfo(mockHandler, mockResolver)

	originalCtx := context.WithValue(context.Background(), "testKey", "testValue")
	req := httptest.NewRequest(testMethod, testPath, nil).WithContext(originalCtx)
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	validateContextPropagation(t, mockHandler, originalCtx)
}

// createTestRequest creates a standard test request
func createTestRequest() *http.Request {
	return httptest.NewRequest(testMethod, testPath, nil)
}

// validateSuccessfulResponse validates successful request processing
func validateSuccessfulResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler, expectedInfo *request.RequestInfo) {

	if recorder.Code != expectedStatusOK {
		t.Errorf("Expected status code %d, got %d", expectedStatusOK, recorder.Code)
	}

	if !mockHandler.called {
		t.Error("Next handler was not called")
	}

	if mockHandler.receivedReq == nil {
		t.Fatal("Received request is nil")
	}

	actualInfo, _ := request.RequestInfoFrom(mockHandler.receivedReq.Context())
	if actualInfo != expectedInfo {
		t.Error("RequestInfo not properly added to context")
	}
}

// validateErrorResponse validates error response handling
func validateErrorResponse(t *testing.T, recorder *httptest.ResponseRecorder, mockHandler *mockHandler) {
	if recorder.Code != expectedStatusInternalError {
		t.Errorf("Expected status code %d, got %d", expectedStatusInternalError, recorder.Code)
	}

	if mockHandler.called {
		t.Error("Next handler should not be called when resolver fails")
	}
}

// validateContextPropagation validates that original context values are preserved
func validateContextPropagation(t *testing.T, mockHandler *mockHandler, originalCtx context.Context) {
	if !mockHandler.called {
		t.Fatal("Handler was not called")
	}

	if mockHandler.receivedReq == nil {
		t.Fatal("Received request is nil")
	}

	receivedCtx := mockHandler.receivedReq.Context()
	originalValue := originalCtx.Value("testKey")
	receivedValue := receivedCtx.Value("testKey")

	if originalValue != receivedValue {
		t.Errorf("Context value not preserved. Expected %v, got %v", originalValue, receivedValue)
	}
}
