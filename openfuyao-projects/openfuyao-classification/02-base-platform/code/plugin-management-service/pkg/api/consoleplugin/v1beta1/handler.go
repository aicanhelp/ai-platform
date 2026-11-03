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
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"plugin-management-service/pkg/audit"
	"plugin-management-service/pkg/constant"
	"plugin-management-service/pkg/plugin"
	"plugin-management-service/pkg/utils/httputil"
	"plugin-management-service/pkg/zlog"
)

// Handler Component handler
type Handler struct {
	config  *rest.Config
	manager *plugin.ConsolePluginManager
}

func newHandler(config *rest.Config) (*Handler, error) {
	cm, err := plugin.NewConsolePluginManager(config)
	if err != nil {
		return nil, err
	}
	return &Handler{
		config:  config,
		manager: cm,
	}, nil
}
func formatOrder(order *int64) *string {
	if order != nil {
		formattedOrder := strconv.FormatInt(*order, constant.BaseTen)
		return &formattedOrder
	}
	return nil
}

// sanitizeLogString sanitizes input strings to prevent log injection attacks
// by removing newline, carriage return, and other potentially dangerous characters
func sanitizeLogString(input string) string {
	if input == "" {
		return ""
	}

	sanitized := make([]rune, 0, len(input))
	var asciiLowerBound rune = 32
	var asciiDel rune = 127
	for _, r := range input {
		// Allow printable ASCII characters (32-126) and extended Unicode
		// Block control characters (0-31) and DEL (127)
		if r >= asciiLowerBound && r != asciiDel {
			sanitized = append(sanitized, r)
		}
	}

	return string(sanitized)
}

// ConsolePluginTrimmed contains only the essential info of a consoleplugin for front-end
type ConsolePluginTrimmed struct {
	Release     string                     `json:"release"`
	DisplayName string                     `json:"displayName"`
	PluginName  string                     `json:"pluginName"`
	Order       *string                    `json:"order,omitempty"`
	SubPages    []plugin.ConsolePluginName `json:"subPages"`
	Entrypoint  string                     `json:"entrypoint"`
	URL         string                     `json:"url"`
	Enabled     bool                       `json:"enabled"`
}

func (h *Handler) listConsolePlugins(request *restful.Request, response *restful.Response) {
	consolePlugins, err := h.manager.ListConsolePlugins()
	if err != nil {
		zlog.Errorf("Error listing ConsolePlugins: %v", err)
		respJson := &httputil.ResponseJson{
			Code: constant.ResourceNotFound,
			Msg:  fmt.Sprintf("Error listing ConsolePlugins: %v", err),
		}
		_ = response.WriteHeaderAndEntity(http.StatusNotFound, respJson)
		return
	}

	consolePluginsTrimmed := make([]ConsolePluginTrimmed, 0)
	for _, cp := range consolePlugins {
		cpTrimmed := ConsolePluginTrimmed{
			DisplayName: cp.Spec.DisplayName,
			PluginName:  cp.Spec.PluginName,
			Order:       formatOrder(cp.Spec.Order),
			SubPages:    cp.Spec.SubPages,
			Entrypoint:  string(cp.Spec.Entrypoint),
			URL:         cp.Status.Link,
			Enabled:     cp.Spec.Enabled,
		}
		if releaseName, ok := cp.ObjectMeta.Annotations["meta.helm.sh/release-name"]; ok {
			cpTrimmed.Release = releaseName
		}
		consolePluginsTrimmed = append(consolePluginsTrimmed, cpTrimmed)
	}

	respJson := &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: consolePluginsTrimmed,
	}
	_ = response.WriteHeaderAndEntity(http.StatusOK, respJson)
}

func (h *Handler) getConsolePlugin(request *restful.Request, response *restful.Response) {
	pluginName := request.PathParameter(constant.PluginName)
	consolePlugin, err := h.manager.GetConsolePlugin(pluginName)
	if err != nil {
		zlog.Errorf("Error getting ConsolePlugin: %v", err)
		respJson := &httputil.ResponseJson{
			Code: constant.ResourceNotFound,
			Msg:  fmt.Sprintf("Error getting ConsolePlugin: %v", err),
		}
		_ = response.WriteHeaderAndEntity(http.StatusNotFound, respJson)
		return
	}

	consolePluginTrimmed := ConsolePluginTrimmed{
		DisplayName: consolePlugin.Spec.DisplayName,
		PluginName:  consolePlugin.Spec.PluginName,
		Order:       formatOrder(consolePlugin.Spec.Order),
		SubPages:    consolePlugin.Spec.SubPages,
		Entrypoint:  string(consolePlugin.Spec.Entrypoint),
		URL:         consolePlugin.Status.Link,
		Enabled:     consolePlugin.Spec.Enabled,
	}
	if releaseName, ok := consolePlugin.ObjectMeta.Annotations["meta.helm.sh/release-name"]; ok {
		consolePluginTrimmed.Release = releaseName
	}

	respJson := &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: consolePluginTrimmed,
	}
	_ = response.WriteHeaderAndEntity(http.StatusOK, respJson)
}

type setEnablementBody struct {
	PluginName string `json:"pluginName"`
	Enabled    bool   `json:"enabled"`
}

func (h *Handler) checkEnablement(request *restful.Request, response *restful.Response) {
	pluginName := request.PathParameter(constant.PluginName)

	pluginEnabled, err := h.manager.CheckPluginEnablementIfInstalled(pluginName)
	if err != nil {
		zlog.Errorf("Error checking ConsolePlugin enablement: %v", err)
		respJson := &httputil.ResponseJson{
			Code: constant.ResourceNotFound,
			Msg:  fmt.Sprintf("Error checking ConsolePlugin enablement: %v", err),
		}
		_ = response.WriteHeaderAndEntity(http.StatusNotFound, respJson)
		return
	}

	respJson := &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: &setEnablementBody{
			PluginName: pluginName,
			Enabled:    pluginEnabled,
		},
	}
	_ = response.WriteHeaderAndEntity(http.StatusOK, respJson)
}

func (h *Handler) setEnablement(request *restful.Request, response *restful.Response) {
	pluginName := request.PathParameter(constant.PluginName)

	body := &setEnablementBody{}
	err := json.NewDecoder(request.Request.Body).Decode(body)
	if err != nil {
		zlog.Errorf("Error parsing request body: %v", err)
		respJson := &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("Error parsing request body: %v", err),
		}
		recordEnablementAudit(request, "set-enablement", audit.ResultFailed, respJson, http.StatusBadRequest)
		writeEnablementResponse(response, http.StatusBadRequest, respJson)
		return
	}

	if pluginName != body.PluginName {
		sanitizedBodyPluginName := sanitizeLogString(body.PluginName)
		zlog.Errorf("PluginName not match: %s, %s", pluginName, sanitizedBodyPluginName)
		respJson := &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("PluginName not match: %s, %s", pluginName, sanitizedBodyPluginName),
		}
		recordEnablementAudit(request, "set-enablement", audit.ResultFailed, respJson, http.StatusBadRequest)
		writeEnablementResponse(response, http.StatusBadRequest, respJson)
		return
	}

	enabledBool := body.Enabled
	action := enablementAction(enabledBool)
	err = h.manager.SetPluginEnablementIfInstalled(pluginName, enabledBool)
	if err != nil {
		zlog.Errorf("Error setting ConsolePlugin enablement: %v", err)
		respJson := &httputil.ResponseJson{
			Code: constant.ServerError,
			Msg:  fmt.Sprintf("Fail to set the ConsolePlugin enablement: %v", err),
		}
		recordEnablementAudit(request, action, audit.ResultFailed, respJson, http.StatusInternalServerError)
		writeEnablementResponse(response, http.StatusInternalServerError, respJson)
		return
	}

	zlog.Infof("Successfully set ConsolePlugin %s enablement to %t", pluginName, enabledBool)
	respJson := &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  fmt.Sprintf("Set ConsolePlugin %s enablement to %t", pluginName, enabledBool),
	}
	recordEnablementAudit(request, action, audit.ResultSuccess, respJson, http.StatusOK)
	writeEnablementResponse(response, http.StatusOK, respJson)
}

func recordEnablementAudit(request *restful.Request, action, result string, respJson *httputil.ResponseJson, status int) {
	pluginName := request.PathParameter(constant.PluginName)
	audit.Record(request.Request, audit.Event{
		Action: action,
		Result: result,
		Detail: enablementOperationDetail(action, pluginName, respJson, status),
	})
}

func writeEnablementResponse(response *restful.Response, status int, respJson *httputil.ResponseJson) {
	_ = response.WriteHeaderAndEntity(status, respJson)
}

func enablementOperationDetail(action, pluginName string, responseJson *httputil.ResponseJson, status int) string {
	detail := fmt.Sprintf("%s consoleplugin %s", action, pluginName)
	if responseJson == nil {
		return detail
	}
	msg := strings.TrimSpace(responseJson.Msg)
	if msg == "" || strings.EqualFold(msg, "success") {
		return detail
	}
	return fmt.Sprintf("%s: %s", detail, msg)
}

func enablementAction(enabled bool) string {
	if enabled {
		return "enable"
	}
	return "disable"
}
