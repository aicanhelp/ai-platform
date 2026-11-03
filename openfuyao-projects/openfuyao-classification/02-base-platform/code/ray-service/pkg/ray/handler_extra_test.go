/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package ray

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/mock"

	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/utils/httputil"
)

// buildExtraMultipartRequest builds a multipart/form-data POST request with
// an optional file part plus extra form values. The body is written by hand
// (plain MIME parts, no mime/multipart writer) so the wire format stays
// explicit; the handlers parse it through the standard r.FormFile and
// r.Form paths.
func buildExtraMultipartRequest(t *testing.T, path, fileField, fileContent string,
	fields map[string]string) *http.Request {
	t.Helper()
	const boundary = "openfuyao-rayservice-unit-test"

	body := ""
	if fileField != "" {
		body += "--" + boundary + "\r\n" +
			"Content-Disposition: form-data; name=\"" + fileField + "\"; filename=\"config.yaml\"\r\n" +
			"Content-Type: application/octet-stream\r\n" +
			"\r\n" +
			fileContent + "\r\n"
	}
	for key, value := range fields {
		body += "--" + boundary + "\r\n" +
			"Content-Disposition: form-data; name=\"" + key + "\"\r\n" +
			"\r\n" +
			value + "\r\n"
	}
	body += "--" + boundary + "--\r\n"

	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)
	return req
}

// buildExtraFormRequest builds an application/x-www-form-urlencoded POST request.
func buildExtraFormRequest(t *testing.T, path string, form map[string][]string) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path,
		strings.NewReader(url.Values(form).Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return req
}

func decodeExtraResponse(t *testing.T, w *httptest.ResponseRecorder) map[string]interface{} {
	t.Helper()
	var responseBody map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &responseBody); err != nil {
		t.Fatalf("failed to decode response body %q: %v", w.Body.String(), err)
	}
	return responseBody
}

func TestHandlerCrdUpdatePost(t *testing.T) {
	t.Run("update with uploaded file succeeds", func(t *testing.T) {
		mockOp := new(MockOperation)
		mockOp.On("MethodUpdateCrd", TestCrdType, TestNamespace, "file-content", TestCrdName).Return(
			&httputil.ResponseJson{Code: int32(constant.Success), Msg: "success"}, HTTPStatusOK)

		container := createTestContainer(mockOp)
		req := buildExtraMultipartRequest(t,
			fmt.Sprintf("/rest/v1/%s/update/%s/%s", TestCrdType, TestNamespace, TestCrdName),
			"file", "file-content", nil)
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != HTTPStatusOK {
			t.Errorf("crdUpdatePost() status = %d, want %d, body: %s", w.Code, HTTPStatusOK, w.Body.String())
		}
		mockOp.AssertExpectations(t)
	})

	t.Run("update without file returns 400", func(t *testing.T) {
		mockOp := new(MockOperation)
		container := createTestContainer(mockOp)

		req := httptest.NewRequest(http.MethodPost,
			fmt.Sprintf("/rest/v1/%s/update/%s/%s", TestCrdType, TestNamespace, TestCrdName), nil)
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("crdUpdatePost() status = %d, want %d", w.Code, http.StatusBadRequest)
		}
	})

	t.Run("update with unknown crd returns 400", func(t *testing.T) {
		mockOp := new(MockOperation)
		container := createTestContainer(mockOp)

		req := httptest.NewRequest(http.MethodPost,
			fmt.Sprintf("/rest/v1/unknown/update/%s/%s", TestNamespace, TestCrdName), nil)
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("crdUpdatePost() status = %d, want %d for unknown crd", w.Code, http.StatusBadRequest)
		}
		body := decodeExtraResponse(t, w)
		if body["msg"] != "Unknown Resource" {
			t.Errorf("crdUpdatePost() msg = %v, want Unknown Resource", body["msg"])
		}
	})
}

func TestHandlerCrdDeleteDelete(t *testing.T) {
	t.Run("delete existing crd succeeds", func(t *testing.T) {
		mockOp := new(MockOperation)
		mockOp.On("MethodDeleteCrd", TestCrdType, TestNamespace, TestCrdName).Return(
			&httputil.ResponseJson{Code: int32(constant.Success), Msg: "success"}, HTTPStatusOK)

		container := createTestContainer(mockOp)
		req := httptest.NewRequest(http.MethodDelete,
			fmt.Sprintf("/rest/v1/%s/delete/%s/%s", TestCrdType, TestNamespace, TestCrdName), nil)
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != HTTPStatusOK {
			t.Errorf("crdDeleteDelete() status = %d, want %d, body: %s", w.Code, HTTPStatusOK, w.Body.String())
		}
		mockOp.AssertExpectations(t)
	})

	t.Run("delete with unknown crd returns 400", func(t *testing.T) {
		mockOp := new(MockOperation)
		container := createTestContainer(mockOp)

		req := httptest.NewRequest(http.MethodDelete,
			fmt.Sprintf("/rest/v1/unknown/delete/%s/%s", TestNamespace, TestCrdName), nil)
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("crdDeleteDelete() status = %d, want %d for unknown crd", w.Code, http.StatusBadRequest)
		}
		body := decodeExtraResponse(t, w)
		if body["msg"] != "Unknown Resource" {
			t.Errorf("crdDeleteDelete() msg = %v, want Unknown Resource", body["msg"])
		}
	})
}

func TestHandlerCrdCreatePostVariants(t *testing.T) {
	t.Run("create with uploaded file and isStart TRUE", func(t *testing.T) {
		mockOp := new(MockOperation)
		mockOp.On("MethodCreateCrd", TestCrdType, "uploaded-content", TestCreateBy, true).Return(
			&httputil.ResponseJson{Code: int32(constant.Success), Msg: "success"}, HTTPStatusOK)

		container := createTestContainer(mockOp)
		req := buildExtraMultipartRequest(t,
			fmt.Sprintf("/rest/v1/%s/create", TestCrdType), "file", "uploaded-content",
			map[string]string{
				"createBy": TestCreateBy,
				"isStart":  "TRUE",
			})
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != HTTPStatusOK {
			t.Errorf("crdCreatePost() status = %d, want %d, body: %s", w.Code, HTTPStatusOK, w.Body.String())
		}
		mockOp.AssertExpectations(t)
	})

	t.Run("create with form config only", func(t *testing.T) {
		mockOp := new(MockOperation)
		mockOp.On("MethodCreateCrd", TestCrdType, "config-content", TestCreateBy, false).Return(
			&httputil.ResponseJson{Code: int32(constant.Success), Msg: "success"}, HTTPStatusOK)

		container := createTestContainer(mockOp)
		req := buildExtraFormRequest(t, fmt.Sprintf("/rest/v1/%s/create", TestCrdType), map[string][]string{
			"config":   {"config-content"},
			"createBy": {TestCreateBy},
			"isStart":  {"false"},
		})
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != HTTPStatusOK {
			t.Errorf("crdCreatePost() status = %d, want %d, body: %s", w.Code, HTTPStatusOK, w.Body.String())
		}
		mockOp.AssertExpectations(t)
	})

	t.Run("create without config returns 400", func(t *testing.T) {
		mockOp := new(MockOperation)
		container := createTestContainer(mockOp)

		req := buildExtraFormRequest(t, fmt.Sprintf("/rest/v1/%s/create", TestCrdType), map[string][]string{
			"createBy": {TestCreateBy},
		})
		w := httptest.NewRecorder()
		container.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("crdCreatePost() status = %d, want %d for missing config", w.Code, http.StatusBadRequest)
		}
	})
}

func TestHandlerHandleErrorRecoversPanic(t *testing.T) {
	mockOp := new(MockOperation)
	// the operation panics, the deferred handleError must recover it
	mockOp.On("MethodListCrd", TestCrdType, &CrdListRequest{}).
		Return(&httputil.ResponseJson{Code: int32(constant.Success), Msg: "success"}, HTTPStatusOK).
		Run(func(args mock.Arguments) { panic("handler boom") })

	container := createTestContainer(mockOp)
	req := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/rest/v1/%s/list", TestCrdType), nil)
	req.Header.Set("Accept", restful.MIME_JSON)
	w := httptest.NewRecorder()
	container.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("panic recovery status = %d, want %d, body: %s", w.Code, http.StatusBadRequest, w.Body.String())
	}
	body := decodeExtraResponse(t, w)
	if body["msg"] != "500 Internal Server Error" {
		t.Errorf("panic recovery msg = %v, want 500 Internal Server Error", body["msg"])
	}
}
