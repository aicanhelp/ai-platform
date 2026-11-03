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
	"net/url"
	"testing"

	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/server/request"
)

const (
	validKubeHost   = "https://kubernetes.default.svc"
	invalidKubeHost = ":/invalid-url"
	k8sAPIPrefix    = "/api/kubernetes"
	testK8sPath     = "/api/kubernetes/api/v1/pods"
	testNonK8sPath  = "/api/v1/configs"
	authHeaderKey   = "Authorization"
	authHeaderValue = "Bearer token"
)

// mockRoundTripper implements http.RoundTripper for testing
type mockRoundTripper struct {
	called   bool
	response *http.Response
	err      error
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	m.called = true
	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}

// TestProxyAPIServerSuccess tests successful proxy creation
func TestProxyAPIServerSuccess(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createValidRestConfig()

	handler := ProxyAPIServer(mockHandler, config)

	if handler == nil {
		t.Fatal("ProxyAPIServer returned nil handler")
	}

	proxy, ok := handler.(*apiServerProxy)
	if !ok {
		t.Fatal("ProxyAPIServer did not return apiServerProxy type")
	}

	validateProxyConfiguration(t, proxy, mockHandler, config)
}

// TestProxyAPIServerInvalidHost tests proxy creation with invalid host
func TestProxyAPIServerInvalidHost(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createInvalidHostRestConfig()

	handler := ProxyAPIServer(mockHandler, config)

	if handler != mockHandler {
		t.Error("ProxyAPIServer should return original handler when host parsing fails")
	}
}

// TestAPIServerProxyNoRequestInfo tests handling when RequestInfo is not found
func TestAPIServerProxyNoRequestInfo(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createValidRestConfig()
	proxy := ProxyAPIServer(mockHandler, config)

	req := createTestRequest()
	recorder := httptest.NewRecorder()

	proxy.ServeHTTP(recorder, req)

	validateNoRequestInfoResponse(t, recorder, mockHandler)
}

// TestAPIServerProxyNonK8sRequest tests handling of non-Kubernetes requests
func TestAPIServerProxyNonK8sRequest(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createValidRestConfig()
	proxy := ProxyAPIServer(mockHandler, config)

	req := createNonK8sRequest()
	recorder := httptest.NewRecorder()

	proxy.ServeHTTP(recorder, req)

	validateNonK8sRequestResponse(t, recorder, mockHandler)
}

// TestAPIServerProxyK8sRequest tests handling of Kubernetes requests
func TestAPIServerProxyK8sRequest(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createValidRestConfig()
	proxy := ProxyAPIServer(mockHandler, config)

	req := createK8sRequest()
	recorder := httptest.NewRecorder()

	proxy.ServeHTTP(recorder, req)

	validateK8sRequestResponse(t, recorder, mockHandler)
}

// TestAPIServerProxyAuthHeaderRemoval tests that Authorization header is removed for K8s requests
func TestAPIServerProxyAuthHeaderRemoval(t *testing.T) {
	mockHandler := &mockHandler{}
	config := createValidRestConfig()
	proxy := ProxyAPIServer(mockHandler, config)

	req := createK8sRequestWithAuth()
	recorder := httptest.NewRecorder()

	proxy.ServeHTTP(recorder, req)

	if req.Header.Get(authHeaderKey) != "" {
		t.Error("Authorization header should be removed for K8s requests")
	}
}

// createValidRestConfig creates a valid rest.Config for testing
func createValidRestConfig() *rest.Config {
	return &rest.Config{
		Host: validKubeHost,
	}
}

// createInvalidHostRestConfig creates a rest.Config with invalid host
func createInvalidHostRestConfig() *rest.Config {
	return &rest.Config{
		Host: invalidKubeHost,
	}
}

// createNonK8sRequest creates a request with non-Kubernetes RequestInfo
func createNonK8sRequest() *http.Request {
	req := httptest.NewRequest(testMethod, testNonK8sPath, nil)
	requestInfo := &request.RequestInfo{
		IsK8sRequest: false,
	}
	ctx := request.WithRequestInfo(req.Context(), requestInfo)
	return req.WithContext(ctx)
}

// createK8sRequest creates a request with Kubernetes RequestInfo
func createK8sRequest() *http.Request {
	req := httptest.NewRequest(testMethod, testK8sPath, nil)
	requestInfo := &request.RequestInfo{
		IsK8sRequest: true,
	}
	ctx := request.WithRequestInfo(req.Context(), requestInfo)
	return req.WithContext(ctx)
}

// createK8sRequestWithAuth creates a K8s request with Authorization header
func createK8sRequestWithAuth() *http.Request {
	req := createK8sRequest()
	req.Header.Set(authHeaderKey, authHeaderValue)
	return req
}

// validateProxyConfiguration validates proxy configuration setup
func validateProxyConfiguration(t *testing.T, proxy *apiServerProxy, expectedHandler http.Handler,
	config *rest.Config) {

	if proxy.nextHandler != expectedHandler {
		t.Error("nextHandler not set correctly")
	}

	if proxy.kubeUrl == nil {
		t.Fatal("kubeUrl should not be nil")
	}

	expectedUrl, err := url.Parse(config.Host)
	if err != nil {
		t.Fatal("url parsing error")
	}

	if proxy.kubeUrl.String() != expectedUrl.String() {
		t.Errorf("Expected kubeUrl %s, got %s", expectedUrl.String(), proxy.kubeUrl.String())
	}

	if proxy.roundTripper == nil {
		t.Error("roundTripper should not be nil")
	}
}

// validateNoRequestInfoResponse validates response when RequestInfo is missing
func validateNoRequestInfoResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler) {

	if recorder.Code != expectedStatusInternalError {
		t.Errorf("Expected status code %d, got %d", expectedStatusInternalError, recorder.Code)
	}

	if !mockHandler.called {
		t.Error("Next handler should be called even when RequestInfo is missing")
	}
}

// validateNonK8sRequestResponse validates response for non-Kubernetes requests
func validateNonK8sRequestResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler) {

	if recorder.Code != expectedStatusOK {
		t.Errorf("Expected status code %d, got %d", expectedStatusOK, recorder.Code)
	}

	if !mockHandler.called {
		t.Error("Next handler should be called for non-K8s requests")
	}
}

// validateK8sRequestResponse validates response for Kubernetes requests
func validateK8sRequestResponse(t *testing.T, recorder *httptest.ResponseRecorder,
	mockHandler *mockHandler) {

	if mockHandler.called {
		t.Error("Next handler should not be called for K8s requests")
	}
}
