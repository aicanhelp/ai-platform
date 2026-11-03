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

package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/agiledragon/gomonkey"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestHandlerIsValidUpdateRequest(t *testing.T) {

	tests := []struct {
		name           string
		mountPaths     []string
		request        PromtailUpdateRequest
		getPathsError  error
		expectedResult bool
	}{
		{
			name:           "Valid request",
			mountPaths:     []string{"/var/log"},
			request:        PromtailUpdateRequest{JobName: "valid-job", Path: "/var/log/app.log"},
			expectedResult: true,
		},
		{
			name:          "Error getting mount paths",
			getPathsError: fmt.Errorf("simulated error"),
		},

		{
			name:       "Invalid path prefix",
			mountPaths: []string{"/var/log"},
			request:    PromtailUpdateRequest{JobName: "valid-job", Path: "/invalid/path.log"},
		},

		{
			name:       "Empty job name",
			mountPaths: []string{"/var/log"},
			request:    PromtailUpdateRequest{JobName: "", Path: "/var/log/app.log"},
		},
		{
			name:       "Empty path",
			mountPaths: []string{"/var/log"},
			request:    PromtailUpdateRequest{JobName: "valid-job", Path: ""},
		},
		{
			name:       "Invalid path format",
			mountPaths: []string{"/var/log"},
			request:    PromtailUpdateRequest{JobName: "valid-job", Path: "no-slash-path"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			got := isValidUpdateRequest(tt.request, tt.mountPaths)
			if got != tt.expectedResult {
				t.Errorf("h.isValidUpdateRequest() = %v, want %v", got, tt.expectedResult)
			}
		})
	}
}

func TestSendHTTPRequestResourceOnlyWithMockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte(`{"status": "success", "data": ["file1.log", "file2.log", "file3.log"]}`))
		if err != nil {
			t.Errorf("Write response failed!")
		}
	}))
	defer server.Close()

	req := &HTTPRequest{
		Method:  "GET",
		URL:     server.URL + "/loki/api/v1/label/filename/values",
		Body:    nil,
		Headers: map[string]string{"Content-Type": "application/json"},
	}

	start, end := calculateQueryTimesUnix("500h")
	queryParams := url.Values{}
	queryParams.Add("start", start)
	queryParams.Add("end", end)
	fullURL := fmt.Sprintf("%s?%s", req.URL, queryParams.Encode())

	result, err := sendHTTPRequestResourceOnly(req, fullURL)
	if err != nil {
		t.Fatalf("sendHTTPRequestResourceOnly failed: %v", err)
	}

	if result == nil {
		t.Fatal("Expected non-nil result")
	}

	if status, exists := result["status"]; !exists || status != "success" {
		t.Fatalf("Expected status 'success', got %v", status)
	}
}

func TestGetLokiFileName(t *testing.T) {
	testCases := []struct {
		name        string
		requestsMap map[string]*HTTPRequest
	}{
		{
			"Valid request",
			map[string]*HTTPRequest{
				"getFileNameList": {
					Method:  "GET",
					URL:     "http://test-server/loki/api/v1/label/filename/values",
					Headers: map[string]string{"Content-Type": "application/json"},
				},
			},
		},
		{
			"Request key not found",
			map[string]*HTTPRequest{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if _, err := w.Write([]byte(`{"status":"success","data":["file1.log"]}`)); err != nil {
					t.Errorf("Write response failed!")
				}
			}))
			defer server.Close()
			if req, exists := tc.requestsMap["getFileNameList"]; exists {
				req.URL = server.URL
			}

			handler := &Handler{RequestsMap: tc.requestsMap}
			httpRequest := httptest.NewRequest("GET", "/loki-filenames", nil)
			httpResponse := httptest.NewRecorder()
			restfulRequest := restful.NewRequest(httpRequest)
			restfulResponse := restful.NewResponse(httpResponse)
			restfulResponse.Header().Set("Content-Type", "application/json")

			handler.getLokiFileName(restfulRequest, restfulResponse)

			if tc.name == "Valid request" && restfulResponse.StatusCode() == http.StatusOK {
				var apiResponse ApiResponse
				if err := json.Unmarshal(httpResponse.Body.Bytes(), &apiResponse); err != nil {
					t.Errorf("Unmarshal failed!")
				}
				if apiResponse.Code != http.StatusOK {
					t.Errorf("Expected code 200, got %d", apiResponse.Code)
				}
			}
		})
	}
}

func TestGetAlertingRulesErrorCases(t *testing.T) {
	testCases := []struct {
		name        string
		requestsMap map[string]*HTTPRequest
	}{
		{"Invalid URL", map[string]*HTTPRequest{
			"getAlertingRules": {Method: "GET", URL: "://invalid-url"},
		}},
		{"Invalid domain", map[string]*HTTPRequest{
			"getAlertingRules": {Method: "GET", URL: "http://invalid-domain",
				Headers: map[string]string{"Content-Type": "application/json"}},
		}},
		{"Request key not found", map[string]*HTTPRequest{}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := &Handler{RequestsMap: tc.requestsMap}

			httpRequest := httptest.NewRequest("GET", "/alerting-rules", nil)
			httpResponse := httptest.NewRecorder()
			restfulRequest := restful.NewRequest(httpRequest)
			restfulResponse := restful.NewResponse(httpResponse)

			handler.getAlertingRules(restfulRequest, restfulResponse)

			if restfulResponse.StatusCode() < ResponseStatusErrCode {
				t.Errorf("Expected error status, got %d", restfulResponse.StatusCode())
			}
		})
	}
}

func TestDisplayconfigmap(t *testing.T) {
	patches := gomonkey.ApplyFunc((*Handler).getReadOnlyVolumesByNode,
		func(h *Handler) (map[string][]VolumeMount, error) {
			return map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
				},
			}, nil
		})
	defer patches.Reset()

	h := &Handler{}
	req := &restful.Request{}

	recorder := httptest.NewRecorder()
	resp := &restful.Response{ResponseWriter: recorder}
	resp.SetRequestAccepts("application/json")

	h.displayconfigmap(req, resp)

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), `"code":200`)
	assert.Contains(t, recorder.Body.String(), `"msg":"Operation successful"`)
	assert.Contains(t, recorder.Body.String(), `"mountPath":"/path1"`)
	assert.Contains(t, recorder.Body.String(), `"name":"vol1"`)
	assert.Contains(t, recorder.Body.String(), `"readOnly":true`)
}

type MockResponseWriter struct {
	statusCode int
	body       []byte
	header     http.Header
}

func (m *MockResponseWriter) Header() http.Header {
	if m.header == nil {
		m.header = http.Header{}
	}
	return m.header
}

func (m *MockResponseWriter) Write(data []byte) (int, error) {
	m.body = append(m.body, data...)
	return len(data), nil
}

func (m *MockResponseWriter) WriteHeader(statusCode int) {
	m.statusCode = statusCode
}

func TestGetConfigmapDetails(t *testing.T) {

	client := fake.NewSimpleClientset()

	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail"},
		Data:       map[string]string{"promtail.yaml": `scrape_configs: [{job_name: "test"}]`},
	}
	_, err := client.CoreV1().ConfigMaps("").Create(context.TODO(), cm, metav1.CreateOptions{})
	assert.NoError(t, err)

	h := &Handler{ConfigmapClient: client.CoreV1().ConfigMaps("")}
	req := &restful.Request{}
	mockWriter := &MockResponseWriter{header: http.Header{}}
	resp := &restful.Response{ResponseWriter: mockWriter}
	resp.SetRequestAccepts("application/json")

	h.getConfigmapDetails(req, resp)

	assert.Equal(t, http.StatusOK, mockWriter.statusCode)
	assert.Contains(t, string(mockWriter.body), `"code":200`)
}

func TestGetLogSecretsDebug(t *testing.T) {
	client := fake.NewSimpleClientset()
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "loki"},
		Data: map[string][]byte{
			"loki.yaml": []byte(`test: value`),
		},
	}
	client.CoreV1().Secrets("").Create(context.TODO(), secret, metav1.CreateOptions{})

	h := &Handler{SecretsClient: client.CoreV1().Secrets("")}
	req := &restful.Request{}
	mockWriter := &MockResponseWriter{header: http.Header{}}
	resp := &restful.Response{ResponseWriter: mockWriter}
	resp.SetRequestAccepts("application/json")

	h.getlogSecrets(req, resp)

	t.Logf("Response: %s", string(mockWriter.body))

	assert.Equal(t, http.StatusOK, mockWriter.statusCode)
	assert.Contains(t, string(mockWriter.body), `"code":200`)
}

func TestCheckRequestConfig(t *testing.T) {
	t.Run("Configuration exists - return true", func(t *testing.T) {
		h := &Handler{
			RequestsMap: map[string]*HTTPRequest{
				"filterRequest": {},
			},
		}
		resp := &restful.Response{}

		result := h.checkRequestConfig(resp)
		assert.True(t, result)
	})

	t.Run("Configuration not found - return false", func(t *testing.T) {
		h := &Handler{
			RequestsMap: map[string]*HTTPRequest{},
		}
		mockWriter := &MockResponseWriter{}
		resp := &restful.Response{ResponseWriter: mockWriter}

		result := h.checkRequestConfig(resp)
		assert.False(t, result)
		assert.Equal(t, http.StatusNotFound, mockWriter.statusCode)
	})
}

func TestHandleError(t *testing.T) {
	t.Run("With error", func(t *testing.T) {
		mockWriter := &MockResponseWriter{}
		resp := &restful.Response{ResponseWriter: mockWriter}
		resp.SetRequestAccepts("application/json")

		handleError(resp, "Test error", fmt.Errorf("detailed error"))

		assert.Contains(t, string(mockWriter.body), `"code":400`)
		assert.Contains(t, string(mockWriter.body), `"Unexpected Error occurred"`)
	})

	t.Run("Without error", func(t *testing.T) {
		mockWriter := &MockResponseWriter{}
		resp := &restful.Response{ResponseWriter: mockWriter}
		resp.SetRequestAccepts("application/json")

		handleError(resp, "Test error", nil)

		assert.Contains(t, string(mockWriter.body), `"code":400`)
	})
}
