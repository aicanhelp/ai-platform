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

// Package runtime defines ServerConfig and validates it.
package runtime

import (
	"github.com/emicklei/go-restful/v3"
)

const (
	// ApiRootPath defines the route prefix.
	ApiRootPath = "/rest"

	// WebTerminalBasePath defines the base route.
	WebTerminalBasePath = "/rest/webterminal/v1"
)

// NewWebService creates a new RESTful web service for the specified API group and version.
// The web service is configured to serve JSON content and is rooted at a path derived
// from the provided GroupVersion information and the ApiRootPath.
func NewWebService() *restful.WebService {
	webservice := restful.WebService{}
	webservice.Path(WebTerminalBasePath).
		Consumes(restful.MIME_JSON).
		Produces(restful.MIME_JSON)
	return &webservice
}
