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

	"console-service/pkg/constant"
)

const (
	// RestRootPath of console-service
	RestRootPath = "/rest"
	// WsRootPath of console-service
	WsRootPath = "/ws"
)

var (
	groupVersion = schema.GroupVersion{
		Group:   constant.ConsoleServiceDefaultHost,
		Version: constant.ConsoleServiceDefaultAPIVersion,
	}

	webService *restful.WebService
)

func init() {
	initRestfulRegister()
	webService = NewRestfulWebService(groupVersion)
}

// NewRestfulWebService create a webservice with group-version string in root path
func NewRestfulWebService(gv schema.GroupVersion) *restful.WebService {
	return NewRESTWebServiceFromStr(gv.String())
}

// NewRESTWebServiceFromStr create a restful webservice with specific string in root path
func NewRESTWebServiceFromStr(subPath string) *restful.WebService {
	webSvc := restful.WebService{}
	webSvc.Path(strings.TrimRight(RestRootPath+"/"+subPath, "/")).
		Produces(restful.MIME_JSON)
	return &webSvc
}

// NewWSWebServiceFromStr create a websocket webservice with specific string in root path
func NewWSWebServiceFromStr(subPath string) *restful.WebService {
	webSvc := restful.WebService{}
	webSvc.Path(strings.TrimRight(WsRootPath+"/"+subPath, "/")).
		Produces(restful.MIME_JSON)
	return &webSvc
}

// GetConsoleWebService get helm web service
func GetConsoleWebService() *restful.WebService {
	return webService
}

func initRestfulRegister() {
	restful.RegisterEntityAccessor("application/merge-patch+json", restful.NewEntityAccessorJSON(restful.MIME_JSON))
	restful.RegisterEntityAccessor("application/json-patch+json", restful.NewEntityAccessorJSON(restful.MIME_JSON))
}
