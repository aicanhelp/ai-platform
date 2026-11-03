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

// Package v1 provides the necessary tools and utilities to interact with mcs api service.
// It is responsible for creating API clients, API servers, registering endpoints and handling
// requests and reponses efficiently.
package v1

const (
	// UserManagerEndpoint is the user-management-operator service adress.
	UserManagerEndpoint = "http://user-management-operator.openfuyao-system.svc.cluster.local:80"

	// UserManagerSignalUrl is the user-management-operator signal url.
	UserManagerSignalUrl = "/rest/user/v1/request-userList"

	// UserManagerBroadcastUrl is the user-management-operator broadcast url.
	UserManagerBroadcastUrl = "/rest/user/v1/broadcast-users?"
)
