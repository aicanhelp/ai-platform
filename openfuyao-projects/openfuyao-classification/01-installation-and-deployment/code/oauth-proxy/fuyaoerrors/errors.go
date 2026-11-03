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

// Package fuyaoerrors defines errors occurred in oauth-proxy
package fuyaoerrors

const (
	// ErrStrMissingSessionIDParams is an error message indicating that the session-id parameter is
	// missing in the logout URL.
	ErrStrMissingSessionIDParams = "missing session-id params in logout url"

	// ErrStrSessionIDDeleteFailed is an error message indicating that the session-id deletion failed in the storage.
	ErrStrSessionIDDeleteFailed = "failed to delete session-id in storage"
)
