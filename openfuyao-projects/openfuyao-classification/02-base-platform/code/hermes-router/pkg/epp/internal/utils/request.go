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

package utils

import (
	"fmt"
	"unicode/utf8"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// PrefillPodHeader is the request header name that identifies the prefill pod in ip:port format.
	PrefillPodHeader = "x-openfuyao-prefill-pod-address-port"
	// DecodePodHeader is the request header name that identifies the decode pod in ip:port format.
	DecodePodHeader = "x-openfuyao-decode-pod-address-port"
)

// RequestLengthOf calculates the length of an LLM inference request body.
func RequestLengthOf(req *fwkscheduling.LLMRequest) (int, error) {
	if req == nil || req.Body == nil {
		return 0, fmt.Errorf("nil request/body")
	}
	if completions := req.Body.Completions; completions != nil {
		return utf8.RuneCountInString(completions.Prompt.PlainText()), nil
	}
	if chatCompletions := req.Body.ChatCompletions; chatCompletions != nil {
		total := 0
		for _, message := range chatCompletions.Messages {
			total += utf8.RuneCountInString(message.Content.PlainText())
		}
		return total, nil
	}
	return 0, fmt.Errorf("unsupported request body")
}
