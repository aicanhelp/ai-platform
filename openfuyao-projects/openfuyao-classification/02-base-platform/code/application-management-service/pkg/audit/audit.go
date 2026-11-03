/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package audit

import (
	"fmt"
	"net"
	"net/http"
	"strings"

	"gopkg.openfuyao.cn/common-modules/ologger/seclog"
	"k8s.io/apiserver/pkg/authentication/user"

	"application-management-service/pkg/constant"
)

const (
	ModuleApplicationManagement = "application-management-service"
	ResultSuccess               = "success"
	ResultFailed                = "failed"
	ResultDenied                = "denied"
	unknownUser                 = "unknown"
)

type Event struct {
	Action       string
	Result       string
	ResourceType string
	ResourceName string
	Namespace    string
	Detail       string
}

func Record(r *http.Request, event Event) {
	seclog.WriteOperationLog(seclog.SecLoggerModel{
		UserID:          userID(r),
		TerminalIP:      clientIP(r),
		BusinessModule:  ModuleApplicationManagement,
		OperationType:   event.Action,
		OperationResult: normalizeResult(event.Result),
		DetailInfo:      sanitizeDetail(event.Detail),
	})
}

func ResultFromStatus(status int) string {
	if status >= http.StatusOK && status < http.StatusBadRequest {
		return ResultSuccess
	}
	return ResultFailed
}

func ReleaseDetail(action, namespace, release string) string {
	return fmt.Sprintf("%s helm release %s in namespace %s", action, release, namespace)
}

func userID(r *http.Request) string {
	if r == nil {
		return unknownUser
	}
	currentUser, ok := r.Context().Value(constant.UserKey).(user.Info)
	if !ok || currentUser == nil || strings.TrimSpace(currentUser.GetName()) == "" {
		return unknownUser
	}
	return currentUser.GetName()
}

func clientIP(r *http.Request) string {
	if r == nil {
		return ""
	}
	if realIP := strings.TrimSpace(r.Header.Get("X-Real-IP")); realIP != "" {
		return realIP
	}
	if forwarded := strings.TrimSpace(r.Header.Get("X-Forwarded-For")); forwarded != "" {
		return strings.TrimSpace(strings.Split(forwarded, ",")[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err == nil {
		return host
	}
	return r.RemoteAddr
}

func normalizeResult(result string) string {
	if strings.TrimSpace(result) == "" {
		return ResultFailed
	}
	return result
}

func sanitizeDetail(detail string) string {
	detail = strings.ReplaceAll(detail, "\r", " ")
	detail = strings.ReplaceAll(detail, "\n", " ")
	return strings.TrimSpace(detail)
}
