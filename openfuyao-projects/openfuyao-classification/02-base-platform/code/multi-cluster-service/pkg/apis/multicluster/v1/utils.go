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

import (
	"fmt"
	"strings"

	"github.com/emicklei/go-restful/v3"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/responsehandlers"
	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

func checkUserAccess(req *restful.Request, resp *restful.Response, client cluster.OperationsInterface,
	operation string) bool {
	username, ok := req.Request.Context().Value("user").(string)
	if !ok {
		responsehandlers.SendStatusBadRequest(resp, "UnAuthorized user, user information no exists.", nil)
		zlog.LogErrorf("Error Retrieving user information")
		return false
	}

	// assert to clusterclient
	clusterClient, ok := client.(*cluster.OperationClient)
	if !ok {
		responsehandlers.SendStatusServerError(resp, "Server error", nil)
		zlog.LogErrorf("Error asserting type from handler-interface to handler clusterclient")
		return false
	}

	pass, err := utils.IsAccessPass(clusterClient.KarmadaClient, username, operation)
	if err != nil {
		responsehandlers.SendStatusServerError(resp, "Server error", err)
		zlog.LogErrorf("Error Checking user permission: %v", err)
		return false
	}

	if !pass {
		responsehandlers.SendStatusForbidden(resp, "Access Forbidden")
		zlog.LogErrorf("Error Operating %s Access Forbidden", operation)
		return false
	}

	return true
}

func sync2UserMgr(c *APIClient, req *restful.Request, body interface{}, cluster string, operation string) []byte {
	token := retrieveToken(req)
	if token == "" {
		zlog.LogErrorf("Error Retrieving token from header")
		return nil
	}

	generateURL := fmt.Sprintf("cluster-name=%s&method=%s", cluster, operation)
	url := fmt.Sprintf("%s%s%s", UserManagerEndpoint, UserManagerBroadcastUrl, generateURL)
	data, err := c.Post(url, body, token)
	if err != nil {
		zlog.LogErrorf("Error Sending cluster info to UserMgr for operation %s: %v", err, operation)
		return nil
	}

	return data
}

func retrieveToken(req *restful.Request) string {
	var token string
	authInfo := req.Request.Header.Get("Authorization")
	if authInfo == "" {
		authInfo = req.Request.Header.Get("X-OpenFuyao-Authorization")
	}
	if authInfo != "" {
		token = strings.TrimPrefix(authInfo, "Bearer ")
	}
	return token
}
