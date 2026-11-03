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
	"testing"

	"volcano-config-service/pkg/server/request"
)

const (
	testClusterName = "test-cluster"
)

// TestDispatchCluster tests the DispatchCluster function
func TestDispatchCluster(t *testing.T) {
	mockHandler := &mockHandler{}

	handler := DispatchCluster(mockHandler)

	if handler == nil {
		t.Fatal("DispatchCluster returned nil handler")
	}

	dispatcher, ok := handler.(*clusterDispatcher)
	if !ok {
		t.Fatal("DispatchCluster did not return clusterDispatcher type")
	}

	if dispatcher.nextHandler != mockHandler {
		t.Error("nextHandler not set correctly")
	}
}

// TestClusterDispatcherNoRequestInfo tests handling when RequestInfo is not found in context
func TestClusterDispatcherNoRequestInfo(t *testing.T) {
	mockHandler := &mockHandler{}
	dispatcher := DispatchCluster(mockHandler)

	req := createTestRequest()
	recorder := httptest.NewRecorder()

	dispatcher.ServeHTTP(recorder, req)

	validateClusterDispatcherErrorResponse(t, recorder, mockHandler)
}

// TestClusterDispatcherEmptyCluster tests handling when cluster name is empty
func TestClusterDispatcherEmptyCluster(t *testing.T) {
	mockHandler := &mockHandler{}
	dispatcher := DispatchCluster(mockHandler)

	req := createRequestWithEmptyCluster()
	recorder := httptest.NewRecorder()

	dispatcher.ServeHTTP(recorder, req)

	validateClusterDispatcherSuccessResponse(t, recorder, mockHandler)
}

// TestClusterDispatcherWithCluster tests handling when cluster name is provided
func TestClusterDispatcherWithCluster(t *testing.T) {
	mockHandler := &mockHandler{}
	dispatcher := DispatchCluster(mockHandler)

	req := createRequestWithCluster(testClusterName)
	recorder := httptest.NewRecorder()

	dispatcher.ServeHTTP(recorder, req)

	validateClusterDispatcherSuccessResponse(t, recorder, mockHandler)
}

// TestClusterDispatcherClusterPropagation tests that cluster information is preserved
func TestClusterDispatcherClusterPropagation(t *testing.T) {
	mockHandler := &mockHandler{}
	dispatcher := DispatchCluster(mockHandler)

	req := createRequestWithCluster(testClusterName)
	recorder := httptest.NewRecorder()

	dispatcher.ServeHTTP(recorder, req)

	validateClusterPropagation(t, mockHandler, testClusterName)
}

// createRequestWithEmptyCluster creates a request with empty cluster name
func createRequestWithEmptyCluster() *http.Request {
	req := httptest.NewRequest(testMethod, testPath, nil)
	requestInfo := &request.RequestInfo{
		Cluster: "",
	}
	ctx := request.WithRequestInfo(req.Context(), requestInfo)
	return req.WithContext(ctx)
}

// createRequestWithCluster creates a request with specified cluster name
func createRequestWithCluster(clusterName string) *http.Request {
	req := httptest.NewRequest(testMethod, testPath, nil)
	requestInfo := &request.RequestInfo{
		Cluster: clusterName,
	}
	ctx := request.WithRequestInfo(req.Context(), requestInfo)
	return req.WithContext(ctx)
}

// validateClusterDispatcherSuccessResponse validates successful request processing for cluster dispatcher
func validateClusterDispatcherSuccessResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler) {

	if recorder.Code != expectedStatusOK {
		t.Errorf("Expected status code %d, got %d", expectedStatusOK, recorder.Code)
	}

	if !mockHandler.called {
		t.Error("Next handler was not called")
	}

	if mockHandler.receivedReq == nil {
		t.Error("Received request should not be nil")
	}
}

// validateClusterDispatcherErrorResponse validates error response handling for cluster dispatcher
func validateClusterDispatcherErrorResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler) {

	if recorder.Code != expectedStatusInternalError {
		t.Errorf("Expected status code %d, got %d", expectedStatusInternalError, recorder.Code)
	}

	if mockHandler.called {
		t.Error("Next handler should not be called when RequestInfo is missing")
	}
}

// validateClusterPropagation validates that cluster information is properly propagated
func validateClusterPropagation(t *testing.T, mockHandler *mockHandler, expectedCluster string) {
	if !mockHandler.called {
		t.Fatal("Handler was not called")
	}

	if mockHandler.receivedReq == nil {
		t.Fatal("Received request is nil")
	}

	info, ok := request.RequestInfoFrom(mockHandler.receivedReq.Context())
	if !ok {
		t.Fatal("RequestInfo not found in received request context")
	}

	if info.Cluster != expectedCluster {
		t.Errorf("Expected cluster %s, got %s", expectedCluster, info.Cluster)
	}
}
