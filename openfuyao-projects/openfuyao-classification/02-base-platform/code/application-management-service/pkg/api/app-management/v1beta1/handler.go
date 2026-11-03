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
	"fmt"
	"net/http"
	"strings"

	"github.com/emicklei/go-restful/v3"

	"application-management-service/pkg/audit"
	"application-management-service/pkg/constant"
	"application-management-service/pkg/helm"
	helmModel "application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/client"
	"application-management-service/pkg/server/filters"
	"application-management-service/pkg/server/param"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/zlog"
)

var (
	multiClusterServicePluginAdmittedRole = []string{"platform-admin"}
)

// Handler Component handler
type Handler struct {
	helmOperation helm.Operation
}

func newHandler(operation helm.Operation) *Handler {
	return &Handler{helmOperation: operation}
}

func (h *Handler) getRelease(request *restful.Request, response *restful.Response) {
	// Parse parameters from the request to be used as arguments for the GetReleases function
	result, status := h.helmOperation.GetReleases(param.ReleaseParameter{
		Release:   request.QueryParameter(param.Release),
		Namespace: request.QueryParameter(param.Namespace),
		Extension: request.QueryParameter(param.Extension),
		SortBy:    request.QueryParameter(param.SortBy),
		Ascending: request.QueryParameter(param.Ascending),
		Status:    request.QueryParameters(param.Status),
	})
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getReleaseInfo(request *restful.Request, response *restful.Response) {
	h.processGetOrDeleteAction(request, response, h.helmOperation.GetReleaseInfo)
}

func (h *Handler) deleteRelease(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter(param.Namespace)
	name := request.PathParameter(param.Release)

	// if chart is mcs, check user permission
	releaseInfo, _ := h.helmOperation.GetReleaseInfo(namespace, name)
	if helm.IsMultiClusterServiceChart(releaseInfo.Data) {
		if !filters.AuthorizeByAdmittedRoles(request, multiClusterServicePluginAdmittedRole, client.GetKubeClient()) {
			recordReleaseDeniedAudit(request, "delete", namespace, name)
			httputil.HandleNotAuthorized(response, "current user is unauthorized to delete this release", nil)
			return
		}
	}

	result, status := h.helmOperation.Uninstall(namespace, name)
	writeReleaseOperationResponse(releaseOperationInfo{
		request:      request,
		response:     response,
		action:       "delete",
		detailAction: "delete",
		namespace:    namespace,
		release:      name,
		responseJson: result,
		status:       status,
	})
}

func (h *Handler) createRelease(request *restful.Request, response *restful.Response) {
	h.processCreateOrUpgradeAction(request, response, h.helmOperation.Install)
}

func (h *Handler) upgradeRelease(request *restful.Request, response *restful.Response) {
	h.processCreateOrUpgradeAction(request, response, h.helmOperation.Upgrade)
}

func (h *Handler) getReleaseHistory(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter(param.Namespace)
	name := request.PathParameter(param.Release)

	history, status := h.helmOperation.GetReleaseHistory(namespace, name,
		helm.ParseQueryInt(request.QueryParameter("maxHistory")))
	_ = response.WriteHeaderAndEntity(status, history)
	return
}

func (h *Handler) rollbackRelease(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter(param.Namespace)
	name := request.PathParameter(param.Release)

	// if chart is mcs, check user permission
	releaseInfo, _ := h.helmOperation.GetReleaseInfo(namespace, name)
	if helm.IsMultiClusterServiceChart(releaseInfo.Data) {
		if !filters.AuthorizeByAdmittedRoles(request, multiClusterServicePluginAdmittedRole, client.GetKubeClient()) {
			recordReleaseDeniedAudit(request, "rollback", namespace, name)
			httputil.HandleNotAuthorized(response,
				"current user is unauthorized to rollback this release", nil)
			return
		}
	}

	// Parse parameters from the request to be used as arguments for rolling back
	chartName := helm.GetChartName(releaseInfo.Data)
	result, status := h.helmOperation.RollbackRelease(namespace, name, chartName,
		helm.ParseQueryInt(request.QueryParameter("version")),
		helm.ParseQueryInt(request.QueryParameter("maxHistory")))
	writeReleaseOperationResponse(releaseOperationInfo{
		request:      request,
		response:     response,
		action:       "rollback",
		detailAction: fmt.Sprintf("rollback to version %s", request.QueryParameter("version")),
		namespace:    namespace,
		release:      name,
		responseJson: result,
		status:       status,
	})
}

func (h *Handler) getHelmManifest(request *restful.Request, response *restful.Response) {
	// helper function for getting manifest/values
	h.processGetOrDeleteAction(request, response, h.helmOperation.GetManifest)
}

func (h *Handler) getHelmValues(request *restful.Request, response *restful.Response) {
	// helper function for getting values/manifest
	h.processGetOrDeleteAction(request, response, h.helmOperation.GetValues)
}

func (h *Handler) getReleasePods(request *restful.Request, response *restful.Response) {
	podsInfo, status := h.helmOperation.GetReleasePods(
		request.PathParameter(param.Namespace),
		request.PathParameter(param.Release))

	_ = response.WriteHeaderAndEntity(status, podsInfo)
}

func (h *Handler) processCreateOrUpgradeAction(request *restful.Request, response *restful.Response,
	action func(deployParams helmModel.DeployParam) (*httputil.ResponseJson, int)) {
	var deployParams helmModel.DeployParam
	if err := request.ReadEntity(&deployParams); err != nil {
		zlog.Errorf("%v", err)
		recordAudit(request, audit.Event{
			Action: "apply",
			Result: audit.ResultFailed,
			Detail: "parse helm release request parameter failed",
		})
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, "parse request parameter error")
		return
	}
	deployParams.Release = request.PathParameter(param.Release)
	deployParams.Namespace = request.PathParameter(param.Namespace)

	// if chart is mcs, check user permission
	if deployParams.ChartName == constant.McsChartName {
		if !filters.AuthorizeByAdmittedRoles(request, multiClusterServicePluginAdmittedRole, client.GetKubeClient()) {
			recordReleaseDeniedAudit(request, releaseAction(request), deployParams.Namespace, deployParams.Release)
			httputil.HandleNotAuthorized(response, "current user is unauthorized to create or upgrade this release",
				nil)
			return
		}
	}

	responseJson, status := action(deployParams)
	actionName := releaseAction(request)
	writeReleaseOperationResponse(releaseOperationInfo{
		request:      request,
		response:     response,
		action:       actionName,
		detailAction: actionName,
		namespace:    deployParams.Namespace,
		release:      deployParams.Release,
		responseJson: responseJson,
		status:       status,
	})
}

// releaseOperationInfo carries the context and result of a helm release operation,
// used for audit recording and response writing.
type releaseOperationInfo struct {
	request      *restful.Request
	response     *restful.Response
	action       string
	detailAction string
	namespace    string
	release      string
	responseJson *httputil.ResponseJson
	status       int
}

// recordAudit records the audit event of the release operation.
func (info releaseOperationInfo) recordAudit() {
	recordReleaseAudit(releaseAuditInfo{
		request:   info.request,
		action:    info.action,
		result:    audit.ResultFromStatus(info.status),
		namespace: info.namespace,
		release:   info.release,
		detail: releaseOperationDetail(info.detailAction, info.namespace, info.release,
			info.responseJson, info.status),
	})
}

// writeResponse writes the operation result to the HTTP response.
func (info releaseOperationInfo) writeResponse() {
	_ = info.response.WriteHeaderAndEntity(info.status, info.responseJson)
}

func writeReleaseOperationResponse(info releaseOperationInfo) {
	info.recordAudit()
	info.writeResponse()
}

func recordReleaseDeniedAudit(request *restful.Request, action, namespace, release string) {
	recordReleaseAudit(releaseAuditInfo{
		request:   request,
		action:    action,
		result:    audit.ResultDenied,
		namespace: namespace,
		release:   release,
		detail:    audit.ReleaseDetail(action+" denied for", namespace, release),
	})
}

// releaseAuditInfo carries the information for recording a helm release audit event.
type releaseAuditInfo struct {
	request   *restful.Request
	action    string
	result    string
	namespace string
	release   string
	detail    string
}

func recordReleaseAudit(info releaseAuditInfo) {
	recordAudit(info.request, audit.Event{
		Action:       info.action,
		Result:       info.result,
		ResourceType: "helm-release",
		ResourceName: info.release,
		Namespace:    info.namespace,
		Detail:       info.detail,
	})
}

func recordAudit(request *restful.Request, event audit.Event) {
	audit.Record(request.Request, event)
}

func releaseOperationDetail(action, namespace, release string, responseJson *httputil.ResponseJson, status int) string {
	detail := audit.ReleaseDetail(action, namespace, release)
	if responseJson == nil {
		return detail
	}
	msg := strings.TrimSpace(responseJson.Msg)
	if msg == "" || strings.EqualFold(msg, "success") {
		return detail
	}
	return fmt.Sprintf("%s: %s", detail, msg)
}

func releaseAction(request *restful.Request) string {
	if request == nil || request.Request == nil {
		return "apply"
	}
	switch request.Request.Method {
	case http.MethodPost:
		return "create"
	case http.MethodPut:
		return "update"
	default:
		return "apply"
	}
}

// processGetOrDeleteAction processes common tasks for Helm-related action, e.g. get_manifest, get_values --all .
func (h *Handler) processGetOrDeleteAction(request *restful.Request, response *restful.Response,
	action func(namespace, name string) (*httputil.ResponseJson, int)) {
	namespace := request.PathParameter(param.Namespace)
	name := request.PathParameter(param.Release)

	result, status := action(namespace, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) checkReleaseStatus(request *restful.Request, response *restful.Response) {
	result, status := h.helmOperation.CheckReleaseStatus(request.PathParameter(param.Chart))
	_ = response.WriteHeaderAndEntity(status, result)
}
