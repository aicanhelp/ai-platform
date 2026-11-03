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
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// HeaderHandlerType is the plugin type identifier for the D/PD header handler.
	HeaderHandlerType = "pd-header-handler"

	decodeProfileName = "decode"
)

var _ requestcontrol.PreRequest = &HeaderHandler{}

// HeaderHandlerFactory creates a plugin instance from EPP configuration.
func HeaderHandlerFactory(
	name string,
	_ json.RawMessage,
	_ fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	return NewHeaderHandler().WithName(name), nil
}

// NewHeaderHandler initializes a new HeaderHandler.
func NewHeaderHandler() *HeaderHandler {
	return &HeaderHandler{
		typedName: fwkplugin.TypedName{Type: HeaderHandlerType},
	}
}

// HeaderHandler injects D/PD route headers during the pre-request hook.
type HeaderHandler struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the plugin's typed name.
func (handler *HeaderHandler) TypedName() fwkplugin.TypedName {
	return handler.typedName
}

// WithName sets the plugin's configured name.
func (handler *HeaderHandler) WithName(name string) *HeaderHandler {
	handler.typedName.Name = name
	return handler
}

// Route describes the leader and selected members resolved from the primary profile result.
type Route struct {
	Leader  *fwkdl.EndpointMetadata
	Prefill *fwkdl.EndpointMetadata
	Decode  *fwkdl.EndpointMetadata
}

// ExtractPrimaryRoute resolves the selected leader endpoint into a D/PD route.
func ExtractPrimaryRoute(ctx context.Context, schedulingResult *fwkscheduling.SchedulingResult) *Route {
	logger := log.FromContext(ctx)
	if schedulingResult == nil {
		return nil
	}

	primaryProfileResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primaryProfileResult == nil ||
		primaryProfileResult.TargetEndpoints == nil ||
		len(primaryProfileResult.TargetEndpoints) == 0 {
		logger.Error(
			fmt.Errorf("nil target endpoint in primaryProfileResult"),
			"primaryProfileResult", primaryProfileResult,
		)
		return nil
	}

	logger.V(logutil.DEBUG).Info("prerequest_pd: Read scheduling result successfully")
	leader := primaryLeaderEndpoint(primaryProfileResult)
	if route := selectedLeaderRoute(schedulingResult.PrimaryProfileName, leader); route != nil {
		return route
	}

	logger.V(logutil.DEFAULT).Info("primary scheduling result does not contain a valid D/PD route",
		"primaryProfile", schedulingResult.PrimaryProfileName,
		"targets", len(primaryProfileResult.TargetEndpoints))
	return nil
}

func selectedLeaderRoute(primaryProfileName string, leader fwkscheduling.Endpoint) *Route {
	if leader == nil {
		return nil
	}
	leaderMetadata := endpointmeta.MetadataOf(leader)
	if leaderMetadata == nil {
		return nil
	}
	info, ok := pdgroup.GetDirect(leader)
	if !ok || info == nil {
		return nil
	}
	if primaryProfileName == decodeProfileName {
		return selectedDecodeRoute(leaderMetadata, info)
	}
	prefillEP, decodeEP, ok := selectedPDRouteEndpoints(info)
	if !ok {
		return nil
	}
	prefill := endpointmeta.MetadataOf(prefillEP)
	decode := endpointmeta.MetadataOf(decodeEP)
	if prefill == nil || decode == nil {
		return nil
	}
	return &Route{
		Leader:  leaderMetadata,
		Prefill: prefill,
		Decode:  decode,
	}
}

func selectedDecodeRoute(leader *fwkdl.EndpointMetadata, info *pdgroup.Info) *Route {
	if leader == nil || info == nil || info.SelectedPrefillPod != nil || info.SelectedDecodePod == nil {
		return nil
	}
	decode := endpointmeta.MetadataOf(info.SelectedDecodePod.Endpoint)
	if decode == nil {
		return nil
	}
	return &Route{Leader: leader, Decode: decode}
}

func selectedPDRouteEndpoints(info *pdgroup.Info) (fwkscheduling.Endpoint, fwkscheduling.Endpoint, bool) {
	if info.SelectedPrefillPod == nil || info.SelectedDecodePod == nil {
		return nil, nil, false
	}
	prefillEP := info.SelectedPrefillPod.Endpoint
	decodeEP := info.SelectedDecodePod.Endpoint
	return prefillEP, decodeEP, prefillEP != nil && decodeEP != nil
}

func primaryLeaderEndpoint(primaryProfileResult *fwkscheduling.ProfileRunResult) fwkscheduling.Endpoint {
	if primaryProfileResult == nil || len(primaryProfileResult.TargetEndpoints) == 0 {
		return nil
	}
	return primaryProfileResult.TargetEndpoints[0]
}

// InjectHeadersFromRoute writes the selected target addresses into request headers.
func InjectHeadersFromRoute(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	route *Route,
) {
	if request == nil || route == nil {
		return
	}

	logger := log.FromContext(ctx).V(logutil.DEBUG)
	if isDecodeOnlyRoute(route) {
		ensureRequestHeaders(request)
		decodeHostPort := net.JoinHostPort(route.Decode.Address, route.Decode.Port)
		request.Headers[utils.DecodePodHeader] = decodeHostPort
		delete(request.Headers, utils.PrefillPodHeader)
		logger.Info("injected Decode-only route headers", "decode", decodeHostPort)
		return
	}
	if !isPDRoute(route) {
		return
	}

	ensureRequestHeaders(request)
	prefillHostPort := net.JoinHostPort(route.Prefill.Address, route.Prefill.Port)
	decodeHostPort := net.JoinHostPort(route.Decode.Address, route.Decode.Port)
	request.Headers[utils.PrefillPodHeader] = prefillHostPort
	request.Headers[utils.DecodePodHeader] = decodeHostPort
	logger.Info("injected PD route headers", "prefill", prefillHostPort, "decode", decodeHostPort)
}

func isDecodeOnlyRoute(route *Route) bool {
	return route != nil && route.Leader != nil && route.Prefill == nil && route.Decode != nil
}

func isPDRoute(route *Route) bool {
	return route != nil && route.Leader != nil && route.Prefill != nil && route.Decode != nil
}

func ensureRequestHeaders(request *fwkscheduling.LLMRequest) {
	if request.Headers == nil {
		request.Headers = make(map[string]string)
	}
}

// InjectHeaders extracts the primary route and writes the corresponding request headers.
func InjectHeaders(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	schedulingResult *fwkscheduling.SchedulingResult,
) {
	route := ExtractPrimaryRoute(ctx, schedulingResult)
	InjectHeadersFromRoute(ctx, request, route)
}

// PreRequest injects D/PD routing headers before the request is forwarded.
func (handler *HeaderHandler) PreRequest(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	schedulingResult *fwkscheduling.SchedulingResult,
) {
	InjectHeaders(ctx, request, schedulingResult)
}
