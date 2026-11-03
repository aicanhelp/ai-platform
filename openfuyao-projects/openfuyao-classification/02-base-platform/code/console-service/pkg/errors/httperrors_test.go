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
	t.Run("empty_message", func(t *testing.T) {
		err := &TokenExpiredError{}
		if got := err.Error(); got != "" {
			t.Errorf("TokenExpiredError.Error() = %q, want empty string", got)
		}
	})

	t.Run("non_empty_message", func(t *testing.T) {
		err := &TokenExpiredError{Message: "expired", Response: &http.Response{}}
		if got := err.Error(); got != "token has expired" {
			t.Errorf("TokenExpiredError.Error() = %q, want token has expired", got)
		}
	})
}

func TestHttpResponseNotOKError(t *testing.T) {
	err := &HttpResponseNotOKError{Message: "status 500"}
	if got := err.Error(); got != "status 500" {
		t.Errorf("HttpResponseNotOKError.Error() = %q, want status 500", got)
	}
}

func TestInvalidJsonHttpBodyError(t *testing.T) {
	err := &InvalidJsonHttpBodyError{Message: "invalid json"}
	if got := err.Error(); got != "invalid json" {
		t.Errorf("InvalidJsonHttpBodyError.Error() = %q, want invalid json", got)
	}
}

func TestFieldNotFoundError(t *testing.T) {
	err := &FieldNotFoundError{Message: "field missing", Field: "access_token"}
	if got := err.Error(); got != "field missing" {
		t.Errorf("FieldNotFoundError.Error() = %q, want field missing", got)
	}
}
