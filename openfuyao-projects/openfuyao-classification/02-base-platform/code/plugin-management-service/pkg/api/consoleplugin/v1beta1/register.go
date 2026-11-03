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

// Package v1beta1 contains all API endpoints for console consoleplugin management
package v1beta1

import (
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"plugin-management-service/pkg/constant"
	"plugin-management-service/pkg/zlog"
)

// BindPluginRoute define the webservice, route of release related function
func BindPluginRoute(webService *restful.WebService, kubeConfig *rest.Config) {
	handler, err := newHandler(kubeConfig)
	if err != nil {
		zlog.Fatalf("consoleplugin handler init failed, err: %v", err)
	}

	webService.Route(webService.GET("/consoleplugins/").
		Doc("List ConsolePlugins").
		To(handler.listConsolePlugins))

	webService.Route(webService.GET("/consoleplugins/{pluginName}").
		Doc("Get ConsolePlugins from name").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.getConsolePlugin))

	webService.Route(webService.GET("/consoleplugins/{pluginName}/enabled").
		Doc("Check if the ConsolePlugin is enabled").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.checkEnablement))

	webService.Route(webService.POST("/consoleplugins/{pluginName}/enabled").
		Doc("Set ConsolePlugin Enablement").
		Param(webService.PathParameter(constant.PluginName, "console consoleplugin name").Required(true)).
		To(handler.setEnablement))
}
