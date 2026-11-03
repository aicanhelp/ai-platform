/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package errors

import "net/http"

// TokenExpiredError is an error returned on token expiration
type TokenExpiredError struct {
	Message  string
	Response *http.Response
}

func (e *TokenExpiredError) Error() string {
	if e.Message == "" {
		return e.Message
	} else {
		return "token has expired"
	}
}

// HttpResponseNotOKError is an error returned on http response with http status other than 200
type HttpResponseNotOKError struct {
	Message string
}

func (e *HttpResponseNotOKError) Error() string {
	return e.Message
}

// InvalidJsonHttpBodyError is an error returned on http response with invalid json response body
type InvalidJsonHttpBodyError struct {
	Message string
}

func (e *InvalidJsonHttpBodyError) Error() string {
	return e.Message
}

// FieldNotFoundError is an error returned on http response with field that can't be found
type FieldNotFoundError struct {
	Message string
	Field   string
}

func (e *FieldNotFoundError) Error() string {
	return e.Message
}
