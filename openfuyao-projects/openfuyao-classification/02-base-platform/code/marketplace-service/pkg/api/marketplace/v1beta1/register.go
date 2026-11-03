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

/*
Package v1beta1
contains all the endpoint and handler logic
*/
package v1beta1

import (
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"marketplace-service/pkg/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/zlog"
)

// BindMarketPlaceRoute bind webservice route to container
func BindMarketPlaceRoute(webService *restful.WebService, kubeConfig *rest.Config) *Handler {
	operation, err := helm.NewHelmOperation(kubeConfig)
	if err != nil {
		zlog.Fatalf("helm handler init failed, err: %v", err)
	}
	handler := newHandler(operation)
	_, _ = handler.HelmHandler.SyncAllRepos()

	bindHelmChartsRoute(webService, handler)
	bindHelmRepoGetRoutes(webService, handler)
	bindHelmRepoPostRoutes(webService, handler)
	return handler
}

func bindHelmChartsRoute(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.GET("/helm-charts").
		Doc("get latest helm charts").
		Param(webService.QueryParameter(param.Repository, "helm repo name").Required(false)).
		Param(webService.QueryParameter(param.Chart, "helm chart name").Required(false)).
		Param(webService.QueryParameter(param.Scene, "helm chart scene").Required(false)).
		Param(webService.QueryParameter(param.AppType, "app type").Required(false)).
		Param(webService.QueryParameter(param.SortType, "sort type").Required(false)).
		Param(webService.QueryParameter(param.Page, "page").Required(false).
			DataFormat("page=%d").DefaultValue("page=1")).
		Param(webService.QueryParameter(param.Limit, "limit").Required(false)).
		To(handler.getLatestCharts))

	webService.Route(webService.GET("/helm-charts/official-tags").
		Doc("get helm charts with official tags").
		Param(webService.QueryParameter(param.Tag, "harbor tags").Required(true)).
		Param(webService.QueryParameter(param.Page, "page").Required(false).
			DataFormat("page=%d").DefaultValue("page=1")).
		Param(webService.QueryParameter(param.Limit, "limit").Required(false)).
		To(handler.getChartsWithOfficialTags))

	webService.Route(webService.GET("/helm-charts/count").
		Doc("number of helm charts in local registry").
		To(handler.countChart))

	webService.Route(webService.POST("/helm-charts").
		Doc("upload helm chart").
		Param(webService.MultiPartFormParameter(param.Chart, "helm chart").Required(true)).
		Consumes("multipart/form-data").
		To(handler.uploadChart))

	webService.Route(webService.DELETE("/helm-charts/{chart}").
		Doc("delete helm chart versions").
		Param(webService.PathParameter(param.Chart, "deleted chart name").Required(true)).
		To(handler.deleteChart))

	webService.Route(webService.DELETE("/helm-charts/{chart}/versions/{version}").
		Doc("delete helm chart version").
		Param(webService.PathParameter(param.Chart, "deleted chart name").Required(true)).
		Param(webService.PathParameter(param.Version, "deleted chart version").Required(true)).
		To(handler.deleteChartVersions))
}

func bindHelmRepoGetRoutes(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.GET("/helm-repos").
		Doc("list helm repos").
		Param(webService.QueryParameter(param.Page, "page").Required(false).
			DataFormat("page=%d").DefaultValue("page=1")).
		Param(webService.QueryParameter(param.Limit, "limit").Required(false)).
		To(handler.listHelmRepo))

	webService.Route(webService.GET("/helm-repos/{repo}").
		Doc("get specific helm repos").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		Param(webService.QueryParameter(param.Page, "page").Required(false).
			DataFormat("page=%d").DefaultValue("page=1")).
		Param(webService.QueryParameter(param.Limit, "limit").Required(false)).
		To(handler.getHelmRepo))

	webService.Route(webService.GET("/helm-repos/{repo}/sync").
		Doc("get repo sync status").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		To(handler.getRepoSyncStatus))

	webService.Route(webService.GET("/helm-repos/{repo}/charts/{chart}").
		Doc("get chart all versions").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		Param(webService.PathParameter(param.Chart, "helm chart name").Required(true)).
		To(handler.getChartVersions))

	webService.Route(webService.GET("/helm-repos/{repo}/charts/{chart}/versions/{version}").
		Doc("get chart all versions").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		Param(webService.PathParameter(param.Chart, "helm chart name").Required(true)).
		Param(webService.PathParameter(param.Version, "helm chart version").Required(true)).
		To(handler.getChartVersion))

	webService.Route(webService.GET("/helm-repos/{repo}/charts/{chart}/versions/{version}/files").
		Doc("get chart files").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		Param(webService.PathParameter(param.Chart, "helm chart name").Required(true)).
		Param(webService.PathParameter(param.Version, "helm chart version").Required(true)).
		Param(webService.QueryParameter(param.FileType, "file type").Required(false)).
		To(handler.getChartFiles))
}

func bindHelmRepoPostRoutes(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.POST("/helm-repos").
		Doc("create helm repo").
		To(handler.createHelmRepo))

	webService.Route(webService.PUT("/helm-repos/{repo}").
		Doc("update helm repo").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		To(handler.updateHelmRepo))

	webService.Route(webService.DELETE("/helm-repos/{repo}").
		Doc("delete helm repo").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		To(handler.deleteHelmRepo))

	webService.Route(webService.POST("/helm-repos/sync").
		Doc("sync all repos").
		To(handler.syncAllHelmRepo))

	webService.Route(webService.POST("/helm-repos/{repo}/sync").
		Doc("sync repo").
		Param(webService.PathParameter(param.Repository, "helm repo name").Required(true)).
		To(handler.syncHelmRepo))
}
