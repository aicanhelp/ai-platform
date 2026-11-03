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

package httputil

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"openfuyao-rayservice/pkg/constant"
)

func TestGetResponseJsonWithData(t *testing.T) {
	data := map[string]string{"key": "value"}
	got := GetResponseJson(HTTPStatusOK, "success", data)
	want := &ResponseJson{Code: HTTPStatusOK, Msg: "success", Data: data}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetResponseJson() = %v, want %v", got, want)
	}
}

func TestGetClientFailureResponseJsonWithDataPayload(t *testing.T) {
	data := map[string]string{"reason": "invalid"}
	got := GetClientFailureResponseJsonWithData(data)
	want := &ResponseJson{Code: constant.ClientError, Msg: "bad request", Data: data}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetClientFailureResponseJsonWithData() = %v, want %v", got, want)
	}
}

func TestGetServerFailureResponseJsonWithDataPayload(t *testing.T) {
	data := map[string]string{"reason": "internal"}
	got := GetServerFailureResponseJsonWithData(data)
	want := &ResponseJson{Code: constant.ServerError, Msg: "operation failed", Data: data}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetServerFailureResponseJsonWithData() = %v, want %v", got, want)
	}
}

func TestBuildSuccessResponse(t *testing.T) {
	data := map[string]string{"name": "raycluster"}
	got := BuildSuccessResponse(data)
	want := &ResponseJson{Code: int32(constant.Success), Msg: "success", Data: data}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("BuildSuccessResponse() = %v, want %v", got, want)
	}
}

func TestBuildErrorResponse(t *testing.T) {
	got := BuildErrorResponse(int32(constant.ClientError), "invalid request")
	want := &ResponseJson{
		Code: int32(constant.ClientError),
		Msg:  "invalid request",
		Data: []any{},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("BuildErrorResponse() = %v, want %v", got, want)
	}
}

// buildMultipartUploadRequest builds a multipart/form-data request whose
// single file part carries the given content. The body is written by hand
// (plain MIME parts, no mime/multipart writer) so the wire format stays
// explicit; ReceiveFile parses it through the standard r.FormFile path.
func buildMultipartUploadRequest(t *testing.T, fieldName, content string) *http.Request {
	t.Helper()
	const boundary = "openfuyao-rayservice-unit-test"
	body := "--" + boundary + "\r\n" +
		"Content-Disposition: form-data; name=\"" + fieldName + "\"; filename=\"config.yaml\"\r\n" +
		"Content-Type: application/octet-stream\r\n" +
		"\r\n" +
		content + "\r\n" +
		"--" + boundary + "--\r\n"

	req := httptest.NewRequest(http.MethodPost, "/upload", strings.NewReader(body))
	req.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)
	return req
}

func TestReceiveFileSuccess(t *testing.T) {
	content := "apiVersion: ray.io/v1\nkind: RayCluster\n"
	req := buildMultipartUploadRequest(t, "file", content)

	got, err := ReceiveFile(req, "file")
	if err != nil {
		t.Fatalf("ReceiveFile() error = %v, want nil", err)
	}
	if got != content {
		t.Errorf("ReceiveFile() = %q, want %q", got, content)
	}
}

func TestReceiveFileMissingField(t *testing.T) {
	req := buildMultipartUploadRequest(t, "other-field", "content")

	got, err := ReceiveFile(req, "file")
	if err == nil {
		t.Errorf("ReceiveFile() error = nil, want error for missing field")
	}
	if got != "" {
		t.Errorf("ReceiveFile() = %q, want empty string", got)
	}
}
