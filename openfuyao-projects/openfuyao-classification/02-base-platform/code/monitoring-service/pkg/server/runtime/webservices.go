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

// Package runtime provides the foundational structures and utilities for configuring
// and managing HTTP servers within an application. It is designed to facilitate the setup
// and validation of server configurations, ensuring robust and secure communication channels
// for web services.
package runtime

import (
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// RestApiRootPath defines the base path for all API endpoints.
const (
	RestApiRootPath = "/rest"
)

// NewRestWebService creates and returns a new web service instance configured with a specific API group and version.
func NewRestWebService(gv schema.GroupVersion) *restful.WebService {
	webservice := restful.WebService{}
	webservice.Path(strings.TrimRight(RestApiRootPath+"/"+gv.String(), "/")).
		Produces(restful.MIME_JSON)

	return &webservice
}
