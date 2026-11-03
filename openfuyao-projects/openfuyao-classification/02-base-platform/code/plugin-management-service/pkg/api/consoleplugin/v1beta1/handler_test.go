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

package v1beta1

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/rest"

	"plugin-management-service/pkg/constant"
	"plugin-management-service/pkg/plugin"
	"plugin-management-service/pkg/server/runtime"
	"plugin-management-service/pkg/utils/httputil"
)

func TestBindPluginRoute(t *testing.T) {
	type args struct {
		webService *restful.WebService
		kubeConfig *rest.Config
	}
	tests := []struct {
		name     string
		args     args
		wangtErr bool
	}{
		{
			"TestBindPluginRoute",
			args{
				webService: runtime.GetPluginWebService(),
				kubeConfig: &rest.Config{},
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			BindPluginRoute(tt.args.webService, tt.args.kubeConfig)
		})
	}
}

var testConsolePluginUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"apiVersion": "console.openfuyao.com/v1beta1",
		"kind":       "ConsolePlugin",
		"metadata": map[string]interface{}{
			"name": "test-consoleplugin",
			"annotations": map[string]interface{}{
				"meta.helm.sh/release-name": "test-release",
			},
		},
		"spec": map[string]interface{}{
			"pluginName":  "test-consoleplugin",
			"displayName": "测试扩展",
			"entrypoint":  "/",
			"backend": map[string]interface{}{
				"type": "Service",
				"service": map[string]interface{}{
					"name":      "test-consoleplugin",
					"namespace": "test-consoleplugin-ns",
				},
			},
			"enabled": true,
		},
	},
}

var testConsolePluginTrimmed = ConsolePluginTrimmed{
	DisplayName: "测试扩展",
	PluginName:  "test-consoleplugin",
	Entrypoint:  "/",
	URL:         "",
	Enabled:     true,
	Release:     "test-release",
}

var dummyConsolePluginUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"apiVersion": "console.openfuyao.com/v1beta1",
		"kind":       "ConsolePlugin",
		"metadata": map[string]interface{}{
			"name": "dummy-consoleplugin",
		},
		"spec": map[string]interface{}{
			"pluginName":  "dummy-consoleplugin",
			"displayName": "Dummy Plugin",
			"entrypoint":  "/container_platform",
			"backend": map[string]interface{}{
				"type": "Service",
				"service": map[string]interface{}{
					"name":      "test-consoleplugin",
					"namespace": "test-consoleplugin-ns",
				},
			},
			"enabled": false,
		},
	},
}

var dummyConsolePluginTrimmed = ConsolePluginTrimmed{
	DisplayName: "Dummy Plugin",
	PluginName:  "dummy-consoleplugin",
	Entrypoint:  "/container_platform",
	URL:         "",
	Enabled:     false,
}

func newFakeDynamicClientSet() *dynamicfake.FakeDynamicClient {
	scheme := k8sruntime.NewScheme()
	return dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		scheme,
		map[schema.GroupVersionResource]string{
			{Group: "console.openfuyao.com", Version: "v1beta1", Resource: "consoleplugins"}: "ConsolePluginList",
		},
		&testConsolePluginUnstructured,
		&dummyConsolePluginUnstructured,
	)
}

func newTestPluginManager() *plugin.ConsolePluginManager {
	return &plugin.ConsolePluginManager{
		Client: newFakeDynamicClientSet(),
	}
}

func newTestHandler() Handler {
	return Handler{
		config:  &rest.Config{},
		manager: newTestPluginManager(),
	}
}

func testBindPluginRoute(webService *restful.WebService) {
	handler := newTestHandler()
	webService.Route(webService.GET("/consoleplugins/").
		To(handler.listConsolePlugins))

	webService.Route(webService.GET("/consoleplugins/{pluginName}").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.getConsolePlugin))

	webService.Route(webService.GET("/consoleplugins/{pluginName}/enabled").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.checkEnablement))

	webService.Route(webService.POST("/consoleplugins/{pluginName}/enabled").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.setEnablement))
}

func initTestContainer() *restful.Container {
	testPluginWebService := restful.WebService{}
	testPluginWebService.Path("/rest/plugin-management/v1beta1").
		Produces(restful.MIME_JSON)
	testBindPluginRoute(&testPluginWebService)
	testContainer := restful.NewContainer()
	testContainer.Add(&testPluginWebService)
	return testContainer
}

func parseResponseJSON(body *bytes.Buffer) (httputil.ResponseJson, error) {
	var respJson httputil.ResponseJson
	resBytes, err := io.ReadAll(body)
	if err != nil {
		return respJson, errors.New("failed to read response body")
	}
	err = json.Unmarshal(resBytes, &respJson)
	if err != nil {
		return respJson, errors.New("failed to parse response json")
	}
	return respJson, nil
}

func parseResponseData(respJson httputil.ResponseJson, resultData any) error {
	resBytes, err := json.Marshal(respJson.Data)
	if err != nil {
		return errors.New("failed to marshal result data")
	}
	err = json.Unmarshal(resBytes, resultData)
	if err != nil {
		return errors.New("failed to unmarshal result data")
	}
	return nil
}

func TestHandlerCheckEnablement(t *testing.T) {
	c := initTestContainer()
	tests := []struct {
		name        string
		pluginName  string
		wantCode    int32
		wantEnabled bool
	}{
		{
			"TestCheckUninstalledPlugin",
			"not-installed",
			constant.ResourceNotFound,
			false, // does not matter
		},
		{
			"TestCheckEnabledPlugin",
			"test-consoleplugin",
			constant.Success,
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(
				"GET",
				fmt.Sprintf("http://example.com/rest/plugin-management/v1beta1/consoleplugins/%s/enabled", tt.pluginName),
				nil,
			)
			resp := httptest.NewRecorder()
			c.Dispatch(resp, req)

			result, err := parseResponseJSON(resp.Body)
			if err != nil {
				t.Error(err.Error())
			}

			if result.Code != tt.wantCode {
				t.Errorf("Checking consoleplugin %s want status code %d, but get %d", tt.pluginName, tt.wantCode, resp.Code)
				return
			}
			if resp.Code == http.StatusNotFound {
				return
			}

			var resultData setEnablementBody
			err = parseResponseData(result, &resultData)
			if err != nil {
				t.Error(err.Error())
			}
			if tt.wantEnabled != resultData.Enabled {
				t.Errorf("Checking consoleplugin %s body not match, want %t, get %t",
					tt.pluginName, tt.wantEnabled, resultData.Enabled)
			}
		})
	}
}

func TestHandlerGetConsolePlugin(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		wantCode   int32
		wantBody   *ConsolePluginTrimmed
	}{
		{
			"GetPluginNotFound",
			"not-installed",
			constant.ResourceNotFound,
			nil,
		},
		{
			"GetPluginFound",
			"test-consoleplugin",
			constant.Success,
			&testConsolePluginTrimmed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := initTestContainer()
			req := httptest.NewRequest(
				"GET",
				fmt.Sprintf("http://example.com/rest/plugin-management/v1beta1/consoleplugins/%s", tt.pluginName),
				nil,
			)
			resp := httptest.NewRecorder()
			c.Dispatch(resp, req)

			result, err := parseResponseJSON(resp.Body)
			if err != nil {
				t.Error(err.Error())
			}

			if result.Code != tt.wantCode {
				t.Errorf("Getting consoleplugin %s want status code %d, but get %d", tt.pluginName, tt.wantCode, result.Code)
			}
			if result.Code == constant.ResourceNotFound {
				return
			}
			var resultData ConsolePluginTrimmed
			err = parseResponseData(result, &resultData)
			if err != nil {
				t.Error(err.Error())
			}
			if !reflect.DeepEqual(&resultData, tt.wantBody) {
				t.Errorf("Wrong consoleplugin get result: %s", tt.pluginName)
			}
		})
	}
}

func TestHandlerListConsolePlugins(t *testing.T) {
	c := initTestContainer()
	t.Run("TestGetAllConsolePlugins", func(t *testing.T) {
		req := httptest.NewRequest(
			"GET",
			"http://example.com/rest/plugin-management/v1beta1/consoleplugins",
			nil,
		)
		resp := httptest.NewRecorder()
		c.Dispatch(resp, req)

		result, err := parseResponseJSON(resp.Body)
		if err != nil {
			t.Error(err.Error())
		}
		if result.Code != constant.Success {
			t.Errorf("Get consoleplugin list failed with status code %d", result.Code)
			return
		}

		var resultData []ConsolePluginTrimmed
		err = parseResponseData(result, &resultData)
		if err != nil {
			t.Error(err.Error())
		}

		wantResList := []*ConsolePluginTrimmed{
			// expected in alphabetical order
			&dummyConsolePluginTrimmed,
			&testConsolePluginTrimmed,
		}
		for i, res := range resultData {
			if !reflect.DeepEqual(&res, wantResList[i]) {
				t.Errorf("Wrong consoleplugin get result: %s", res.PluginName)
			}
		}
	})
}

func TestHandlerSetEnablement(t *testing.T) {
	c := initTestContainer()
	tests := []struct {
		name       string
		pluginName string
		reqBody    []byte
		wantCode   int32
	}{
		{
			"TestInvalidBody",
			"plugin",
			[]byte(""),
			constant.ClientError,
		},
		{
			"TestNonMatchPluginName",
			"plugin1",
			[]byte(`{"pluginName": "plugin2", "enabled": false}`),
			constant.ClientError,
		},
		{
			"TestPluginNotFound",
			"plugin1",
			[]byte(`{"pluginName": "plugin1", "enabled": false}`),
			constant.ServerError,
		},
		{
			"TestSetNonChange",
			"test-consoleplugin",
			[]byte(`{"pluginName": "test-consoleplugin", "enabled": true}`),
			constant.Success,
		},
		{
			"TestSetChange",
			"test-consoleplugin",
			[]byte(`{"pluginName": "test-consoleplugin", "enabled": false}`),
			constant.Success,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(
				"POST",
				fmt.Sprintf("http://example.com/rest/plugin-management/v1beta1/consoleplugins/%s/enabled", tt.pluginName),
				bytes.NewBuffer(tt.reqBody),
			)
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			c.Dispatch(resp, req)

			result, err := parseResponseJSON(resp.Body)
			if err != nil {
				t.Error(err.Error())
			}
			if result.Code != tt.wantCode {
				t.Errorf("Setting consoleplugin %s want status code %d, but get %d", tt.pluginName, tt.wantCode, result.Code)
				return
			}
		})
	}
}

func TestFormatOrder(t *testing.T) {
	testInt := int64(123456)
	testStr := "123456"
	type args struct {
		order *int64
	}
	tests := []struct {
		name string
		args args
		want *string
	}{
		{
			"TestFormatOrderNil",
			args{
				nil,
			},
			nil,
		},
		{
			"TestFormatOrderInt",
			args{
				&testInt,
			},
			&testStr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := formatOrder(tt.args.order); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("formatOrder() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewHandler(t *testing.T) {
	type args struct {
		config *rest.Config
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"TestNewHandler",
			args{
				&rest.Config{},
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newHandler(tt.args.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("newHandler() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestSanitizeLogString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "TestEmptyString",
			input:    "",
			expected: "",
		},
		{
			name:     "TestNormalString",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "TestStringWithTabAndNewLine",
			input:    "hello\tworld\r\n",
			expected: "helloworld",
		},
		{
			name:     "TestStringWithControlCharacters",
			input:    "hello\x7f\x00\x01world",
			expected: "helloworld",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sanitizeLogString(tt.input)
			if result != tt.expected {
				t.Errorf("sanitizeLogString(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
