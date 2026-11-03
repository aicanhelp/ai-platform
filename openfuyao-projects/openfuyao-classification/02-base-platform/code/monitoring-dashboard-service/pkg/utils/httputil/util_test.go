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

package httputil

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"monitoring-dashboard-service/pkg/config/global"
)

// TestGetDefaultServerFailureResponseJson 校验默认失败响应结构
func TestGetDefaultServerFailureResponseJson(t *testing.T) {
	resp := GetDefaultServerFailureResponseJson()
	if resp.Code != global.ServerError {
		t.Errorf("expected code %d, got %d", global.ServerError, resp.Code)
	}
	if resp.Msg != "remote server busy" {
		t.Errorf("unexpected msg: %s", resp.Msg)
	}
	if resp.Data != nil {
		t.Errorf("expected nil data, got %v", resp.Data)
	}
}

// TestGetResponseJSONFromHTTPGetSuccess 正常JSON响应
func TestGetResponseJSONFromHTTPGetSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"success","data":{"result":[]}}`))
	}))
	defer server.Close()

	body, err := GetResponseJSONFromHTTPGet(server.URL)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if body["status"] != "success" {
		t.Errorf("unexpected status in body: %v", body["status"])
	}
}

// TestGetResponseJSONFromHTTPGetRequestError 请求失败场景
func TestGetResponseJSONFromHTTPGetRequestError(t *testing.T) {
	if _, err := GetResponseJSONFromHTTPGet("http://127.0.0.1:1/not-reachable"); err == nil {
		t.Error("expected error for unreachable server, got nil")
	}
}

// TestGetResponseJSONFromHTTPGetStatusNotOK 状态码非200场景
func TestGetResponseJSONFromHTTPGetStatusNotOK(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	_, err := GetResponseJSONFromHTTPGet(server.URL)
	if err == nil || err.Error() != "remote server status not ok" {
		t.Errorf("expected status not ok error, got %v", err)
	}
}

// TestGetResponseJSONFromHTTPGetInvalidJSON 非法JSON响应
func TestGetResponseJSONFromHTTPGetInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{invalid-json`))
	}))
	defer server.Close()

	if _, err := GetResponseJSONFromHTTPGet(server.URL); err == nil {
		t.Error("expected unmarshal error, got nil")
	}
}

// TestGetResponseJSONFromHTTPGetBodyTooLarge 响应体超过大小限制
func TestGetResponseJSONFromHTTPGetBodyTooLarge(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		big := strings.Repeat("a", global.MaxExternalResponseBytes+1024)
		_, _ = w.Write([]byte(big))
	}))
	defer server.Close()

	_, err := GetResponseJSONFromHTTPGet(server.URL)
	if err == nil || err.Error() != "response body too large" {
		t.Errorf("expected body too large error, got %v", err)
	}
}

// TestGetMapFromMapByKey 校验按key获取map值
func TestGetMapFromMapByKey(t *testing.T) {
	cases := []struct {
		name   string
		body   map[string]interface{}
		key    string
		wantNil bool
	}{
		{"key不存在", map[string]interface{}{}, "data", true},
		{"值非map", map[string]interface{}{"data": "string"}, "data", true},
		{"值是nil-map", map[string]interface{}{"data": map[string]interface{}(nil)}, "data", true},
		{"值是map", map[string]interface{}{"data": map[string]interface{}{"a": "b"}}, "data", false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := GetMapFromMapByKey(tc.body, tc.key)
			if tc.wantNil && got != nil {
				t.Errorf("expected nil, got %v", got)
			}
			if !tc.wantNil && got == nil {
				t.Error("expected non-nil map")
			}
		})
	}
}

// TestGetMapListFromMapByKey 校验按key获取map列表值
func TestGetMapListFromMapByKey(t *testing.T) {
	cases := []struct {
		name    string
		body    map[string]interface{}
		key     string
		wantNil bool
	}{
		{"key不存在", map[string]interface{}{}, "result", true},
		{"值非列表", map[string]interface{}{"result": "string"}, "result", true},
		{"列表元素非map", map[string]interface{}{"result": []interface{}{"str"}}, "result", true},
		{"列表元素为nil", map[string]interface{}{"result": []interface{}{nil}}, "result", true},
		{"空列表", map[string]interface{}{"result": []interface{}{}}, "result", true},
		{
			"合法列表",
			map[string]interface{}{"result": []interface{}{map[string]interface{}{"a": "b"}}},
			"result",
			false,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := GetMapListFromMapByKey(tc.body, tc.key)
			if tc.wantNil && got != nil {
				t.Errorf("expected nil, got %v", got)
			}
			if !tc.wantNil && got == nil {
				t.Error("expected non-nil list")
			}
		})
	}
}

// TestGetInterfaceListFromMapByKey 校验按key获取interface列表值
func TestGetInterfaceListFromMapByKey(t *testing.T) {
	cases := []struct {
		name    string
		body    map[string]interface{}
		key     string
		wantNil bool
	}{
		{"key不存在", map[string]interface{}{}, "values", true},
		{"值非列表", map[string]interface{}{"values": 123}, "values", true},
		{"合法列表", map[string]interface{}{"values": []interface{}{"a", "b"}}, "values", false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := GetInterfaceListFromMapByKey(tc.body, tc.key)
			if tc.wantNil && got != nil {
				t.Errorf("expected nil, got %v", got)
			}
			if !tc.wantNil && got == nil {
				t.Error("expected non-nil list")
			}
		})
	}
}
