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
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"go.uber.org/zap"

	"volcano-config-service/pkg/zlog"
)

// Handler volcano handler that contains all the volcano operations
type Handler struct {
	volcanoHandler Operation
}

type policyRequest struct {
	Policy string `json:"policy"`
}

func newHandler(handler Operation) *Handler {
	return &Handler{volcanoHandler: handler}
}

func (h *Handler) numaAwareGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodNumaAwareGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) numaAwarePut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodNumaAwarePut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) numaDistanceGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodNumaDistanceGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) numaDistancePut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodNumaDistancePut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) numaFastGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodNumaFastGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) numaFastPut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodNumaFastPut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) binpackGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodBinpackGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) binpackPut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodBinpackPut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) balanceGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodBalanceGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) balancePut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodBalancePut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) drfGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodDrfGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) drfPut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodDrfPut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) gangGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodGangGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) gangPut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodGangPut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) priorityGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodPriorityGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) priorityPut(request *restful.Request, response *restful.Response) {
	policy, err := getPolicy(request, response)
	if err != nil {
		return
	}

	result, status := h.volcanoHandler.MethodPriorityPut(policy)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) defaultPut(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodDefault()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) nodeListGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodNodeList()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) nodeListGetByName(request *restful.Request, response *restful.Response) {
	name := request.PathParameter("name")
	result, status := h.volcanoHandler.MethodNodeListByName(name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) crGet(request *restful.Request, response *restful.Response) {
	name := request.PathParameter("name")
	result, status := h.volcanoHandler.MethodCRInfo(name)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) podNameGet(request *restful.Request, response *restful.Response) {
	uuid := request.PathParameter("uuid")
	result, status := h.volcanoHandler.MethodUuidToName(uuid)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) schedulerGet(request *restful.Request, response *restful.Response) {
	result, status := h.volcanoHandler.MethodSchedulerGet()
	_ = response.WriteHeaderAndEntity(status, result)
}

func getPolicy(request *restful.Request, response *restful.Response) (string, error) {
	var policyReq policyRequest

	err := request.ReadEntity(&policyReq)
	if err != nil {
		zlog.Error("Failed to read request body", zap.Error(err))
		err = response.WriteError(http.StatusInternalServerError, err)
		return "", err
	}

	policy := policyReq.Policy
	if policy == "" {
		err := response.WriteErrorString(http.StatusBadRequest, "policy is missing")
		zlog.Error("Failed to get policy")
		if err != nil {
			return "", err
		}
		return "", err
	}

	return policy, nil
}
