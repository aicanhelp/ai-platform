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

// Package fuyaopassword implements fuyaoidp login interfaces
package fuyaopassword

// PasswordConfirmRequest is the json struct for password confirmation
type PasswordConfirmRequest struct {
	NewPassword []byte `json:"new_password"`
	Then        string `json:"then"`
}

// PasswordResetRequest is the json struct for password reset
type PasswordResetRequest struct {
	Username         string `json:"username"`
	OriginalPassword []byte `json:"original_password"`
	NewPassword      []byte `json:"new_password"`
}

// LoginRequest is the json struct for logging in
type LoginRequest struct {
	Username string `json:"username"`
	Password []byte `json:"password"`
	Then     string `json:"then"`
}
