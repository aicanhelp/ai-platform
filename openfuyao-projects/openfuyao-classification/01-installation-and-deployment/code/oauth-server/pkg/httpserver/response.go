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

// Package httpserver defines the httpserver options, middlewares and responses
package httpserver

import (
	"encoding/json"
	"net/http"

	"openfuyao/oauth-server/pkg/zlog"
)

// HttpResponse defines the http response struct
type HttpResponse struct {
	Code int32                  `json:"Code"`
	Msg  string                 `json:"Msg"`
	Data map[string]interface{} `json:"Data,omitempty"`
}

// RespondWithStatusMsg response with status code and msg
func RespondWithStatusMsg(w http.ResponseWriter, statusCode int, errCode int32, msg string) {
	errResponse := HttpResponse{
		Code: errCode,
		Msg:  msg,
		Data: nil,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(errResponse); err != nil {
		zlog.LogErrorf("failed to encode json, err: %v", err)
	}
	return
}
