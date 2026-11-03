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

// Package prerequest provides pre-request plugins for GIE.
package prerequest

import (
	"context"
	"encoding/json"
	"fmt"
	"net"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
)

const (
	PDHeaderHandlerType = "pd-header-handler"
)

// compile-time type assertion
var _ requestcontrol.PreRequest = &PDHeaderHandler{}

// PrefillHeaderHandlerFactory  defines the factory function for the PrefillHeaderHandler
func PDHeaderHandlerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewPDHeaderHandler().WithName(name), nil
}

// NewPDHeaderHandler initializes a new PDHeaderHandler and returns its pointer.
func NewPDHeaderHandler() *PDHeaderHandler {
	return &PDHeaderHandler{
		typedName: plugins.TypedName{Type: PDHeaderHandlerType},
	}
}

// PrefillHeaderHandler PreRequest plugin
type PDHeaderHandler struct {
	typedName plugins.TypedName
}

// TypedName returns the typed name of the plugin.
func (p *PDHeaderHandler) TypedName() plugins.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *PDHeaderHandler) WithName(name string) *PDHeaderHandler {
	p.typedName.Name = name
	return p
}

// PDRoute 描述从 primary profile 中解析出的 PD 三元组。
type PDRoute struct {
	Leader  *backend.Pod
	Prefill *backend.Pod
	Decode  *backend.Pod
}

// ExtractPrimaryPDRoute 根据 schedulingResult.PrimaryProfileName 解析出 primary profile 的
// (leader, prefill, decode) 三元组。若数据不完整或存在 nil，则记录日志并返回 nil。
func ExtractPrimaryPDRoute(ctx context.Context, schedulingResult *types.SchedulingResult) *PDRoute {

	logger := log.FromContext(ctx)
	logger.V(logging.DEBUG).Info("prerequest_pd: starting")

	if schedulingResult == nil {
		return nil
	}

	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primaryProfileResult == nil || primaryProfileResult.TargetPods == nil || len(primaryProfileResult.TargetPods) == 0 {
		logger.Error(fmt.Errorf("nil target pod in primaryProfileResult"), "primaryProfileResult", primaryProfileResult)
		return nil
	}

	logger.V(logging.DEBUG).Info("prerequest_pd: Read scheduling result successfully")

	// 约定顺序：(leader, prefill, decode)。若数量不足，视为配置错误。
	prefillIdx, decodeIdx := 1, 2
	if len(primaryProfileResult.TargetPods) <= decodeIdx ||
		primaryProfileResult.TargetPods[prefillIdx] == nil ||
		primaryProfileResult.TargetPods[decodeIdx] == nil {
		logger.Error(fmt.Errorf("nil target pod in primaryProfileResult"),
			"prefillIdx", prefillIdx, "decodeIdx", decodeIdx)
		return nil
	}

	prefill := primaryProfileResult.TargetPods[prefillIdx].GetPod()
	decode := primaryProfileResult.TargetPods[decodeIdx].GetPod()
	if prefill == nil || decode == nil {
		logger.Error(fmt.Errorf("GetPod() returned nil"), "prefillIdx", prefillIdx, "decodeIdx", decodeIdx)
		return nil
	}

	var leader *backend.Pod
	if len(primaryProfileResult.TargetPods) > 0 && primaryProfileResult.TargetPods[0] != nil {
		leader = primaryProfileResult.TargetPods[0].GetPod()
	}

	logger.V(logging.DEBUG).Info("prerequest_pd: Extract primary pd route successfully", "leader", leader.Address, "prefill", prefill.Address, "decode", decode.Address)

	return &PDRoute{
		Leader:  leader,
		Prefill: prefill,
		Decode:  decode,
	}
}

// InjectPDHeadersFromRoute 根据 PDRoute 将 prefill / decode 地址写入请求头。
func InjectPDHeadersFromRoute(ctx context.Context, request *types.LLMRequest, route *PDRoute) {
	if route == nil || route.Prefill == nil || route.Decode == nil {
		return
	}

	logger := log.FromContext(ctx).V(logging.DEBUG)

	prefillHostPort := net.JoinHostPort(route.Prefill.Address, route.Prefill.Port)
	decodeHostPort := net.JoinHostPort(route.Decode.Address, route.Decode.Port)

	if request.Headers == nil {
		request.Headers = make(map[string]string)
	}
	request.Headers[common.PrefillPodHeader] = prefillHostPort
	request.Headers[common.DecodePodHeader] = decodeHostPort

	logger.Info("injected PD prefill/decode headers",
		"prefill", prefillHostPort,
		"decode", decodeHostPort,
		"prefillPod", route.Prefill.Address,
		"decodePod", route.Decode.Address)
}

// InjectPDHeaders 从 schedulingResult 中解析出 primary profile 的 PDRoute，
// 然后调用 InjectPDHeadersFromRoute 写入请求头。
func InjectPDHeaders(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult) {
	route := ExtractPrimaryPDRoute(ctx, schedulingResult)
	InjectPDHeadersFromRoute(ctx, request, route)
}

func (p *PDHeaderHandler) PreRequest(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult) {
	InjectPDHeaders(ctx, request, schedulingResult)
}
