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

// Package idp describes the generalized interfaces all idps should implement
package idp

import "net/http"

// LoginInterface is the idp login interface
type LoginInterface interface {
	LoginHandler(w http.ResponseWriter, r *http.Request)
}

// PasswordLoginFormInterface is the idp password form interface
type PasswordLoginFormInterface interface {
	OutputHTML(w http.ResponseWriter, tpl string, name string)
}
