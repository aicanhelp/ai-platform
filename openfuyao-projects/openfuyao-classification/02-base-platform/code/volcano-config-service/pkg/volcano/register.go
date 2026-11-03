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

// Package volcano config
package volcano

import (
	"os"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/zlog"
)

const (
	groupName = "/rest/scheduling"
	gversion  = "v1"
)

// GroupVersion 集群版本信息
var groupVersion = schema.GroupVersion{Group: groupName, Version: gversion}

// NewWbeSevice 新建webService
func NewWbeSevice(gv schema.GroupVersion) *restful.WebService {
	webservice := restful.WebService{}

	// TrimRight splits the string using "/"
	webservice.Path(strings.TrimRight(gv.String(), "/")).
		Produces(restful.MIME_JSON)

	return &webservice
}

// ConfigVolcano BindUtilityWebService bind utility webservice
func ConfigVolcano(c *restful.Container, kubeConfig *rest.Config) error {
	if v := os.Getenv("VCS_API_DISABLED"); strings.EqualFold(v, "true") || v == "1" {
		zlog.Warn("volcano api disabled by env VCS_API_DISABLED")
		return nil
	}

	WebService := NewWbeSevice(groupVersion)
	operation, err := NewVolcanoOperation(kubeConfig)

	if err != nil {
		zlog.Error("volcano handler init failed, err: %v", err)
		return err
	}
	h := newHandler(operation)

	getInfoAddToContainer(WebService, h)
	numaAwareController(WebService, h)

	WebService.Route(WebService.GET("/balance").
		Doc("get the state of balance").
		To(h.balanceGet))

	WebService.Route(WebService.PUT("/balance").
		Doc("modify the state of balance").
		To(h.balancePut))

	WebService.Route(WebService.PUT("/default").
		Doc("modify the state of default").
		To(h.defaultPut))

	c.Add(WebService)

	return nil
}

func getInfoAddToContainer(WebService *restful.WebService, h *Handler) {
	WebService.Route(WebService.GET("/nodeList").
		Doc("get node list").
		To(h.nodeListGet))

	WebService.Route(WebService.GET("/nodeList/{name}").
		Param(WebService.PathParameter("name", "node name")).
		Doc("get node list").
		To(h.nodeListGetByName))

	WebService.Route(WebService.GET("/cr/{name}").
		Doc("get cr information").
		Param(WebService.PathParameter("name", "cr name")).
		To(h.crGet))

	WebService.Route(WebService.GET("/uuidToName/{uuid}").
		Doc("trans uuid to name").
		Param(WebService.PathParameter("uuid", "uuid to pod name")).
		To(h.podNameGet))

	WebService.Route(WebService.GET("/scheduler").
		Doc("get the scheduler list").
		To(h.schedulerGet))

	return
}

func numaAwareController(WebService *restful.WebService, h *Handler) {
	WebService.Route(WebService.GET("/numaaware").
		Doc("get the state of numaaware").
		To(h.numaAwareGet))

	WebService.Route(WebService.PUT("/numaaware").
		Doc("modify the state of numaaware").
		To(h.numaAwarePut))

	WebService.Route(WebService.GET("/numaDistance").
		Doc("get the state of numaDistance").
		To(h.numaDistanceGet))

	WebService.Route(WebService.PUT("/numaDistance").
		Doc("modify the state of numaDistance").
		To(h.numaDistancePut))

	WebService.Route(WebService.GET("/numaFast").
		Doc("get the state of numaFast").
		To(h.numaFastGet))

	WebService.Route(WebService.PUT("/numaFast").
		Doc("modify the state of numaFast").
		To(h.numaFastPut))
	return
}
