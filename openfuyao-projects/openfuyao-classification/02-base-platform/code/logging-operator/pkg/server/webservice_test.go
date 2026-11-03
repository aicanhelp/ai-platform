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

package server

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNewWebService(t *testing.T) {
	// 定义测试的 API 版本
	gv := schema.GroupVersion{Group: "testgroup", Version: "v1"}
	expectedPath := "/rest/testgroup/v1" // 假设 ApiRootPath 是 "/api"

	// 调用函数以创建新的 WebService
	webservice := NewWebService(gv)

	// 检查生成的 WebService 的路径是否正确
	if webservice.RootPath() != expectedPath {
		t.Errorf("Expected WebService path to be '%s', got '%s'", expectedPath, webservice.RootPath())
	}

}

func TestGetQueryParamOrDefault(t *testing.T) {
	tests := []struct {
		name         string
		queryParams  map[string]string
		param        string
		defaultValue string
		expected     string
	}{
		{
			name:         "Parameter exists",
			queryParams:  map[string]string{"page": "2", "limit": "10"},
			param:        "page",
			defaultValue: "1",
			expected:     "2",
		},
		{
			name:         "Parameter missing or empty",
			queryParams:  map[string]string{"limit": "10"},
			param:        "page",
			defaultValue: "1",
			expected:     "1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			values := url.Values{}
			for key, value := range tt.queryParams {
				values.Add(key, value)
			}

			req := &http.Request{
				URL: &url.URL{
					RawQuery: values.Encode(),
				},
			}
			restfulRequest := &restful.Request{Request: req}

			result := GetQueryParamOrDefault(restfulRequest, tt.param, tt.defaultValue)
			assert.Equal(t, tt.expected, result)
		})
	}
}
