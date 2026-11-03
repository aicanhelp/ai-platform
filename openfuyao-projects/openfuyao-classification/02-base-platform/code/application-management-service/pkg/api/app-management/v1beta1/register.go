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

// Package v1beta1
package v1beta1

import (
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/helm"
	"application-management-service/pkg/server/param"
	"application-management-service/pkg/zlog"
)

// BindReleaseRoute define the webservice, route of release related function
func BindReleaseRoute(webService *restful.WebService, kubeConfig *rest.Config) {
	operation, err := helm.NewHelmOperation(kubeConfig)
	if err != nil {
		zlog.Fatalf("helm handler init failed, err: %v", err)
	}
	handler := newHandler(operation)

	getInfoAddToContainer(webService, handler)
	helmReleaseAddToContainer(webService, handler)
}

// getInfoAddToContainer define the webservice, route of  get information function
func getInfoAddToContainer(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.GET("/helm-releases/namespace/{namespace}/release/{release}/getmanifest").
		Doc("get helm manifest").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		Param(webService.PathParameter("action", "get action to do with release")).
		To(handler.getHelmManifest))

	webService.Route(webService.GET("/helm-releases/namespace/{namespace}/release/{release}/getvalues").
		Doc("get helm values").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		To(handler.getHelmValues))

	webService.Route(webService.GET("/helm-releases").
		Doc("list helm release or search specialized release").
		Param(webService.QueryParameter(param.Namespace, "release namespace").Required(false)).
		Param(webService.QueryParameter(param.Release, "release name").Required(false)).
		Param(webService.QueryParameter(param.Status, "release status").Required(false)).
		Param(webService.QueryParameter(param.Page, "page").Required(false).
			DataFormat("page=%d").DefaultValue("page=1")).
		Param(webService.QueryParameter(param.Limit, "limit").Required(false)).
		Param(webService.QueryParameter(param.Ascending, "sort parameters, e.g. descending=true").
			Required(false).DefaultValue("descending=false")).
		Param(webService.QueryParameter(param.Extension, "extension").Required(false)).
		To(handler.getRelease))

	webService.Route(webService.GET("/helm-releases/namespace/{namespace}/release/{release}").
		Doc("get helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace").Required(false)).
		Param(webService.PathParameter(param.Release, "release name").Required(false)).
		To(handler.getReleaseInfo))

	webService.Route(webService.GET("helm-releases/namespace/{namespace}/release/{release}/pods").
		Doc("get all pods and containers which belong with the helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace").Required(false)).
		Param(webService.PathParameter(param.Release, "release name").Required(false)).
		To(handler.getReleasePods))
	webService.Route(webService.GET("helm-releases/releasestatus/{chart}").
		Param(webService.PathParameter(param.Chart, "chart name").Required(true)).
		To(handler.checkReleaseStatus))
	return
}

// helmReleaseAddToContainer define the webservice, route of other helm release information function
func helmReleaseAddToContainer(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.POST("/helm-releases/namespace/{namespace}/release/{release}").
		Doc("create helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		To(handler.createRelease))

	webService.Route(webService.PUT("/helm-releases/namespace/{namespace}/release/{release}").
		Doc("update helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		To(handler.upgradeRelease))

	webService.Route(webService.DELETE("/helm-releases/namespace/{namespace}/release/{release}").
		Doc("delete helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		To(handler.deleteRelease))

	webService.Route(webService.GET("/helm-releases/namespace/{namespace}/release/{release}/history").
		Doc("get helm history").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		Param(webService.QueryParameter("maxHistory", "maximum number of revision to include in history (default 256)")).
		To(handler.getReleaseHistory))

	webService.Route(webService.POST("/helm-releases/namespace/{namespace}/release/{release}/rollback").
		Doc("rollback helm release").
		Param(webService.PathParameter(param.Namespace, "release namespace")).
		Param(webService.PathParameter(param.Release, "release name")).
		Param(webService.QueryParameter("version", "roll back a release to a previous revision").Required(true)).
		Param(webService.QueryParameter("maxHistory", "limit the maximum number of revisions saved per release. "+
			"Use 0 for no limit (default 10)").Required(true)).
		To(handler.rollbackRelease))
}
