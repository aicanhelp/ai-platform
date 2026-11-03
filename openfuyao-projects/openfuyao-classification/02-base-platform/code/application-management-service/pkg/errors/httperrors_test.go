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

package errors

import (
	"net/http"
	"testing"
)

func TestTokenExpiredError(t *testing.T) {
	err1 := &TokenExpiredError{Message: "", Response: &http.Response{StatusCode: 401}}
	if err1.Error() != "" {
		t.Errorf("TokenExpiredError.Error() 期望空字符串，实际: %q", err1.Error())
	}
	err2 := &TokenExpiredError{Message: "token失效", Response: nil}
	if err2.Error() != "token has expired" {
		t.Errorf("TokenExpiredError.Error() 期望 'token has expired'，实际: %q", err2.Error())
	}
}

func TestHttpResponseNotOKError(t *testing.T) {
	err := &HttpResponseNotOKError{Message: "not ok"}
	if err.Error() != "not ok" {
		t.Errorf("HttpResponseNotOKError.Error() 期望 'not ok'，实际: %q", err.Error())
	}
}

func TestInvalidJsonHttpBodyError(t *testing.T) {
	err := &InvalidJsonHttpBodyError{Message: "invalid json"}
	if err.Error() != "invalid json" {
		t.Errorf("InvalidJsonHttpBodyError.Error() 期望 'invalid json'，实际: %q", err.Error())
	}
}

func TestFieldNotFoundError(t *testing.T) {
	err := &FieldNotFoundError{Message: "not found", Field: "id"}
	if err.Error() != "not found" {
		t.Errorf("FieldNotFoundError.Error() 期望 'not found'，实际: %q", err.Error())
	}
}
