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

package auth

import (
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"console-service/pkg/server/runtime"
	"console-service/pkg/zlog"
)

const (
	subPath = "auth"
)

// AddToContainer define thw webservice, route and forward the request to the handler
func AddToContainer(c *restful.Container, kubeConfig *rest.Config) error {
	webService := runtime.NewRESTWebServiceFromStr(subPath)
	wsWebService := runtime.NewWSWebServiceFromStr(subPath)
	handler, err := NewHandler(kubeConfig)
	if err != nil {
		zlog.Errorf("Fail to create request handler: %v", err)
		return err
	}

	webService.Route(webService.GET("/login").
		Doc("handle login requests").
		To(handler.loginHandler))

	webService.Route(webService.GET("/callback").
		Doc("").
		Param(webService.QueryParameter("code", "authorization code from auth server")).
		Param(webService.QueryParameter("state", "login state")).
		Param(webService.QueryParameter("error", "authorization error type")).
		Param(webService.QueryParameter("error_description", "authorization error description")).
		Param(webService.QueryParameter("error_uri", "authorization error page uri")).
		To(handler.callbackHandler))

	webService.Route(webService.POST("/logout").
		Doc("").
		To(handler.logoutHandler))

	webService.Route(webService.GET("/user").
		Doc("").
		To(handler.getCurrentUserHandler))

	wsWebService.Route(wsWebService.GET("/login-status").
		Doc("").
		To(handler.loginStatusHandler))

	c.Add(webService)
	c.Add(wsWebService)

	return nil
}
