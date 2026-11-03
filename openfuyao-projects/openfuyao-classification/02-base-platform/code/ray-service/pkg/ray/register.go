/*
 *
 *  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

package ray

import (
	"context"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/zlog"
)

const (
	groupName = "/rest"
	version   = "v1"
)

// GroupVersion 集群版本信息
var groupVersion = schema.GroupVersion{Group: groupName, Version: version}

// NewWebSevice 新建webService
func NewWebSevice(gv schema.GroupVersion) *restful.WebService {
	webservice := restful.WebService{}

	// TrimRight splits the string using "/"
	webservice.Path(strings.TrimRight(gv.String(), "/")).
		Produces(restful.MIME_JSON)

	return &webservice
}

// ConfigRay BindUtilityWebService bind utility webservice
func ConfigRay(c *restful.Container, kubeConfig *rest.Config, ctx context.Context) error {
	WebService := NewWebSevice(groupVersion)
	operation, err := NewRayOperation(kubeConfig, ctx)
	if err != nil {
		zlog.Errorf("RayCluster handler init failed, err: %v", err)
	}

	operation.MethodInitOperation()

	h := newHandler(operation)

	getInfoAddToContainer(WebService, h)

	c.Add(WebService)

	return nil
}

func getInfoAddToContainer(WebService *restful.WebService, h *Handler) {
	WebService.Route(WebService.GET("/overview/all").
		Doc("get overview info").
		To(h.overviewAllGet))

	WebService.Route(WebService.GET("/{crd}/list").
		Doc("get crd list").
		Param(WebService.PathParameter("crd", "crd type")).
		To(h.crdListGet))

	WebService.Route(WebService.POST("/{crd}/create").
		Doc("create crd").
		To(h.crdCreatePost))

	WebService.Route(WebService.DELETE("/{crd}/delete/{namespace}/{name}").
		Doc("delete crd info").
		Param(WebService.PathParameter("crd", "crd type")).
		Param(WebService.PathParameter("name", "crd name")).
		Param(WebService.PathParameter("namespace", "crd namespace")).
		To(h.crdDeleteDelete))

	WebService.Route(WebService.POST("/{crd}/update/{namespace}/{name}").
		Doc("update crd info").
		Param(WebService.PathParameter("crd", "crd type")).
		Param(WebService.PathParameter("name", "crd name")).
		Param(WebService.PathParameter("namespace", "crd namespace")).
		To(h.crdUpdatePost))

	WebService.Route(WebService.POST("/{crd}/restart/{namespace}/{name}").
		Doc("restart crd").
		Param(WebService.PathParameter("crd", "crd type")).
		Param(WebService.PathParameter("name", "crd name")).
		Param(WebService.PathParameter("namespace", "crd namespace")).
		To(h.crdRestartPost))

	WebService.Route(WebService.POST("/{crd}/stop/{namespace}/{name}").
		Doc("stop crd info").
		Param(WebService.PathParameter("crd", "crd type")).
		Param(WebService.PathParameter("name", "crd name")).
		Param(WebService.PathParameter("namespace", "crd namespace")).
		To(h.crdStopPost))

	WebService.Route(WebService.GET("/{crd}/get/{namespace}/{name}").
		Doc("get crd info").
		Param(WebService.PathParameter("crd", "crd type")).
		Param(WebService.PathParameter("name", "crd name")).
		Param(WebService.PathParameter("namespace", "crd namespace")).
		To(h.crdGetGet))

	WebService.Route(WebService.GET("/{crd}/list/meta").
		Doc("get crd list meta info").
		Param(WebService.PathParameter("crd", "crd type")).
		To(h.crdListMetaGet))
}
