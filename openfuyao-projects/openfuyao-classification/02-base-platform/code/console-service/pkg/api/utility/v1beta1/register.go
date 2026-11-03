/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package v1beta1

import (
	"github.com/emicklei/go-restful/v3"
)

// BindUtilityWebService bind utility webservice
func BindUtilityWebService(webService *restful.WebService) {
	handler := NewHandler()
	webService.Route(webService.GET("/time-offset").
		Doc("get current time offset").
		To(handler.getCurrentTimeOffset))
}
