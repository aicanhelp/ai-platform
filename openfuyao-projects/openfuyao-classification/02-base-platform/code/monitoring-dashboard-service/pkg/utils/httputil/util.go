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

// Package httputil is used to define http response json
package httputil

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// ResponseJson Http Response
type ResponseJson struct {
	Code int32  `json:"code,omitempty"`
	Msg  string `json:"msg,omitempty"`
	Data any    `json:"data,omitempty"`
}

// GetDefaultServerFailureResponseJson get default failure response json
func GetDefaultServerFailureResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: global.ServerError,
		Msg:  "remote server busy",
		Data: nil,
	}
}

// GetResponseJSONFromHTTPGet 根据uri获取http响应，并将其转为JSON键值对Map对象
func GetResponseJSONFromHTTPGet(uri string) (map[string]interface{}, error) {
	resp, err := http.Get(uri)
	if err != nil {
		zlog.FormatError("http request failed %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("remote server status not ok")
	}
	if resp.ContentLength > 0 && resp.ContentLength > global.MaxExternalResponseBytes {
		return nil, errors.New("response body too large")
	}

	limitedBody := io.LimitReader(resp.Body, global.MaxExternalResponseBytes+1)
	raw, err := io.ReadAll(limitedBody)
	if err != nil {
		zlog.FormatError("read response failed %v", err)
		return nil, err
	}
	if int64(len(raw)) > global.MaxExternalResponseBytes {
		return nil, errors.New("response body too large")
	}

	var body map[string]interface{}
	err = json.Unmarshal(raw, &body)
	if err != nil {
		zlog.FormatError("decode prometheus data failed %v", err)
		return nil, err
	}
	return body, nil
}

// GetMapFromMapByKey 从JSON body 中根据 key 获取值，当值为JSON键值对 对象 时使用
func GetMapFromMapByKey(body map[string]interface{}, key string) map[string]interface{} {
	var rawData interface{}
	var ok bool
	if rawData, ok = body[key]; !ok {
		zlog.FormatInfo("no data found from prometheus body")
		return nil
	}
	data, ok := rawData.(map[string]interface{})
	if !ok || data == nil {
		zlog.FormatError("data type error")
		return nil
	}
	return data
}

// GetMapListFromMapByKey 从JSON body 中根据 key 获取值，当值为JSON键值对 对象数组 时使用
func GetMapListFromMapByKey(body map[string]interface{}, key string) []map[string]interface{} {
	var rawData interface{}
	var ok bool
	if rawData, ok = body[key]; !ok {
		zlog.FormatInfo("no data found from prometheus body")
		return nil
	}
	interfaceList, ok := rawData.([]interface{})
	if !ok {
		zlog.FormatInfo("data of body is not []interface{}")
		return nil
	}
	var data []map[string]interface{}
	for _, item := range interfaceList {
		mapItem, ok := item.(map[string]interface{})
		if !ok || mapItem == nil {
			zlog.FormatInfo("item of interface{} is not map[string]interface{}")
			return nil
		}
		data = append(data, mapItem)
	}
	if len(data) == 0 {
		zlog.FormatError("no data found from prometheus body")
		return nil
	}
	return data
}

// GetInterfaceListFromMapByKey 从JSON body 中根据 key 获取值，当值为 字符串数组 时使用
func GetInterfaceListFromMapByKey(body map[string]interface{}, key string) []interface{} {
	var rawData interface{}
	var ok bool
	if rawData, ok = body[key]; !ok {
		zlog.FormatInfo("no key %s found from prometheus body", key)
		return nil
	}
	data, ok := rawData.([]interface{})
	if !ok || data == nil {
		zlog.FormatInfo("data of body is not []interface{}")
		return nil
	}
	return data
}
