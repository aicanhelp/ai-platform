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
	"net/http"
	"strings"

	"github.com/emicklei/go-restful/v3"

	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/utils/httputil"
	"openfuyao-rayservice/pkg/zlog"
)

type Handler struct {
	kuberayHandler Operation
}

func newHandler(handler Operation) *Handler {
	return &Handler{kuberayHandler: handler}
}

type CrdListRequest struct {
	Name       string `json:"name"`
	RayVersion string `json:"rayVersion"`
	Status     string `json:"status"` // 0: all, 1: running, 2: stopped, 3: failed
	CreateBy   string `json:"createBy"`
	// PageNum    int    `json:"page_num"`
	// PageSize   int    `json:"page_size"`
}

func (h *Handler) overviewAllGet(request *restful.Request, response *restful.Response) {
	result, status := h.kuberayHandler.MethodAllOverview()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdCreatePost(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	fileString, err := httputil.ReceiveFile(request.Request, "file")
	if err != nil {
		fileString = request.Request.Form.Get("config")
	}
	createBy := request.Request.Form.Get("createBy")

	isStart := strings.ToLower(request.Request.Form.Get("isStart")) == "true"
	if fileString == "" {
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "Failed",
			Data: nil,
		})
		return
	}
	result, status := h.kuberayHandler.MethodCreateCrd(request.PathParameter("crd"), fileString, createBy, isStart)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdUpdatePost(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	name := request.PathParameter("name")
	namespace := request.PathParameter("namespace")
	fileString, err := httputil.ReceiveFile(request.Request, "file")
	if err != nil {
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "Failed",
			Data: nil,
		})
		return
	}
	result, status := h.kuberayHandler.MethodUpdateCrd(request.PathParameter("crd"), namespace, fileString, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdListGet(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	crdListRequest := &CrdListRequest{}
	crd := request.PathParameter("crd")
	result, status := h.kuberayHandler.MethodListCrd(crd, crdListRequest)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdGetGet(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	name := request.PathParameter("name")
	namespace := request.PathParameter("namespace")
	result, status := h.kuberayHandler.MethodGetCrd(request.PathParameter("crd"), namespace, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdDeleteDelete(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	name := request.PathParameter("name")
	namespace := request.PathParameter("namespace")
	result, status := h.kuberayHandler.MethodDeleteCrd(request.PathParameter("crd"), namespace, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdRestartPost(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	name := request.PathParameter("name")
	namespace := request.PathParameter("namespace")
	result, status := h.kuberayHandler.MethodRestartCrd(request.PathParameter("crd"), namespace, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdStopPost(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	name := request.PathParameter("name")
	namespace := request.PathParameter("namespace")
	result, status := h.kuberayHandler.MethodStopCrd(request.PathParameter("crd"), namespace, name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crdListMetaGet(request *restful.Request, response *restful.Response) {
	if !h.checkCrd(request, response) {
		return
	}
	defer h.handleError(request, response)
	result, status := h.kuberayHandler.MethodListMetaCrd(request.PathParameter("crd"))
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) checkCrd(request *restful.Request, response *restful.Response) bool {
	if _, ok := constant.GetValidCrdResources()[request.PathParameter("crd")]; !ok {
		response.WriteHeaderAndEntity(http.StatusBadRequest, &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "Unknown Resource",
			Data: nil,
		})
		return false
	}
	return true
}

func (h *Handler) handleError(request *restful.Request, response *restful.Response) {
	if r := recover(); r != nil {
		zlog.Errorf("For Request[%s] Error Handler Got error: %v", request.Request.URL.Path, r)
		response.WriteHeaderAndEntity(http.StatusBadRequest, &httputil.ResponseJson{
			Code: constant.ServerError,
			Msg:  "500 Internal Server Error",
			Data: nil,
		})
	}
}
