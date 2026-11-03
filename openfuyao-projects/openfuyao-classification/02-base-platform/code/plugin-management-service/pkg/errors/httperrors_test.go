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
	"fmt"
	"net/http"
	"testing"
)

func TestError(t *testing.T) {
	customizedMsg := "Customized message"
	tests := []struct {
		name  string
		error error
		want  string
	}{
		{
			"TestTokenExpiredErrorEmpty",
			&TokenExpiredError{
				"",
				&http.Response{},
			},
			"",
		},
		{
			"TestTokenExpiredError",
			&TokenExpiredError{
				customizedMsg,
				&http.Response{},
			},
			"token has expired",
		},
		{
			"TestHttpResponseNotOKErrorCustomized",
			&HttpResponseNotOKError{
				customizedMsg,
			},
			customizedMsg,
		},
		{
			"TestInvalidJsonHttpBodyErrorCustomized",
			&InvalidJsonHttpBodyError{
				customizedMsg,
			},
			customizedMsg,
		},
		{
			"TestFieldNotFoundErrorCustomized",
			&FieldNotFoundError{
				customizedMsg,
				"",
			},
			customizedMsg,
		},
		{
			"TestRegexErrorCustomized",
			&RegexError{
				customizedMsg,
			},
			fmt.Sprintf("RegexError: %s", customizedMsg),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.error.Error(); got != tt.want {
				t.Errorf("Error() = %v, want %v", got, tt.want)
			}
		})
	}
}
