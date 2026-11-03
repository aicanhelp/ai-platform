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

package v1beta1

import (
	"net/http"

	"github.com/emicklei/go-restful/v3"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/helm"
	helmModel "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/utils/util"
	"marketplace-service/pkg/zlog"
)

// Handler helm handler that contains all the helm operations
type Handler struct {
	HelmHandler helm.Operation
}

func newHandler(handler helm.Operation) *Handler {
	return &Handler{HelmHandler: handler}
}

func (h *Handler) listHelmRepo(request *restful.Request, response *restful.Response) {
	query := param.ParseQueryParameter(request)
	repo := request.QueryParameter(param.Repository)
	if !util.IsValidSearchParam(repo) {
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.GetDefaultClientFailureResponseJson())
		return
	}
	result, status := h.HelmHandler.ListRepo(query, repo)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getHelmRepo(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	result, status := h.HelmHandler.GetRepo(repoName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) createHelmRepo(request *restful.Request, response *restful.Response) {
	repoEntry := &helmModel.SafeRepoEntry{}
	err := request.ReadEntity(repoEntry)
	if err != nil {
		zlog.Warnf("invalid input, %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide proper input",
		})
		return
	}
	sanitizeRepoEntry(repoEntry)
	result, status := h.HelmHandler.CreateRepo(repoEntry)
	_ = response.WriteHeaderAndEntity(status, result)
}

func sanitizeRepoEntry(entry *helmModel.SafeRepoEntry) {
	util.EscapeSpecialChars(entry.Name)
	util.EscapeSpecialChars(entry.URL)
}

func (h *Handler) updateHelmRepo(request *restful.Request, response *restful.Response) {
	repoEntry := &helmModel.SafeRepoEntry{}
	err := request.ReadEntity(repoEntry)
	repoEntry.Name = request.PathParameter(param.Repository)
	if err != nil {
		zlog.Warnf("invalid input, %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "proper input is needed",
		})
		return
	}
	sanitizeRepoEntry(repoEntry)
	result, status := h.HelmHandler.UpdateRepo(repoEntry)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) deleteHelmRepo(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	result, status := h.HelmHandler.DeleteRepo(repoName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) syncAllHelmRepo(request *restful.Request, response *restful.Response) {
	result, status := h.HelmHandler.SyncAllRepos()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) syncHelmRepo(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	result, status := h.HelmHandler.SyncRepo(repoName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getRepoSyncStatus(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	result, status := h.HelmHandler.GetRepoSyncStatus(repoName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getLatestCharts(request *restful.Request, response *restful.Response) {
	searchParam := &helmModel.ChartSearchParam{
		Repositories: util.SanitizeArray(request.QueryParameters(param.Repository)),
		Chart:        util.EscapeSpecialChars(request.QueryParameter(param.Chart)),
		Types:        util.SanitizeArray(request.QueryParameters(param.AppType)),
		Query:        param.ParseQueryParameter(request),
		Scene:        util.SanitizeArray(request.QueryParameters(param.Scene)),
	}
	if !util.IsValidSearchParam(searchParam.Chart) {
		_ = response.WriteHeaderAndEntity(http.StatusOK, httputil.GetDefaultClientFailureResponseJson())
		return
	}

	result, status := h.HelmHandler.GetLatestCharts(searchParam)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getChartsWithOfficialTags(request *restful.Request, response *restful.Response) {
	tags := util.SanitizeArray(request.QueryParameters(param.Tag))
	query := param.ParseQueryParameter(request)

	result, status := h.HelmHandler.GetChartsWithOfficialTags(tags, query)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) uploadChart(request *restful.Request, response *restful.Response) {
	formFile, fileHeader, err := request.Request.FormFile(param.Chart)
	if err != nil {
		response.WriteErrorString(http.StatusBadRequest, "Failed to get chart")
		return
	}
	defer formFile.Close()
	result, status := h.HelmHandler.UploadChart(formFile, fileHeader)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) countChart(request *restful.Request, response *restful.Response) {
	result, status := h.HelmHandler.CountCharts()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) deleteChart(request *restful.Request, response *restful.Response) {
	chartName := util.EscapeSpecialChars(request.PathParameter(param.Chart))
	result, status := h.HelmHandler.DeleteChart(chartName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) deleteChartVersions(request *restful.Request, response *restful.Response) {
	chartName := util.EscapeSpecialChars(request.PathParameter(param.Chart))
	version := util.EscapeSpecialChars(request.PathParameter(param.Version))
	result, status := h.HelmHandler.DeleteChartVersion(chartName, version)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getChartVersions(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	chart := util.EscapeSpecialChars(request.PathParameter(param.Chart))
	result, status := h.HelmHandler.GetChartVersions(repoName, chart)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getChartVersion(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	chart := util.EscapeSpecialChars(request.PathParameter(param.Chart))
	version := util.EscapeSpecialChars(request.PathParameter(param.Version))
	result, status := h.HelmHandler.GetChartVersion(repoName, chart, version)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getChartFiles(request *restful.Request, response *restful.Response) {
	repoName := util.EscapeSpecialChars(request.PathParameter(param.Repository))
	chart := util.EscapeSpecialChars(request.PathParameter(param.Chart))
	version := util.EscapeSpecialChars(request.PathParameter(param.Version))
	fileType := util.EscapeSpecialChars(request.QueryParameter(param.FileType))
	result, status := h.HelmHandler.GetChartFiles(repoName, chart, version, fileType)
	_ = response.WriteHeaderAndEntity(status, result)
}
