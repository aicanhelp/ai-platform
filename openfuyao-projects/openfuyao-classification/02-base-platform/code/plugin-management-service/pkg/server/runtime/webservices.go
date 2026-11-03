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

package runtime

import (
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"plugin-management-service/pkg/constant"
)

const (
	// ApiRootPath of plugin-management-service
	ApiRootPath = "/rest"
)

var (
	groupVersion = schema.GroupVersion{
		Group:   constant.PluginManagementServiceDefaultHost,
		Version: constant.PluginManagementServiceDefaultAPIVersion,
	}

	webService *restful.WebService
)

func init() {
	initRestfulRegister()
	webService = NewRestfulWebService(groupVersion)
}

// NewRestfulWebService create a webservice with group-version string in root path
func NewRestfulWebService(gv schema.GroupVersion) *restful.WebService {
	return NewWebServiceFromStr(gv.String())
}

// NewWebServiceFromStr create a webservice with specific string in root path
func NewWebServiceFromStr(subPath string) *restful.WebService {
	webservice := restful.WebService{}
	webservice.Path(strings.TrimRight(ApiRootPath+"/"+subPath, "/")).
		Produces(restful.MIME_JSON)
	return &webservice
}

// GetPluginWebService get helm web service
func GetPluginWebService() *restful.WebService {
	return webService
}

func initRestfulRegister() {
	restful.RegisterEntityAccessor("application/merge-patch+json", restful.NewEntityAccessorJSON(restful.MIME_JSON))
	restful.RegisterEntityAccessor("application/json-patch+json", restful.NewEntityAccessorJSON(restful.MIME_JSON))
}
