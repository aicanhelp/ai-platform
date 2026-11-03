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
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestRequestLengthOfCompletions(t *testing.T) {
	request := &fwkscheduling.LLMRequest{
		Body: &fwkscheduling.LLMRequestBody{
			Completions: &fwkscheduling.CompletionsRequest{
				Prompt: fwkscheduling.Prompt{Raw: "hello世界"},
			},
		},
	}

	got, err := RequestLengthOf(request)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if got != 7 {
		t.Fatalf("expected request length 7, got %d", got)
	}
	if PrefillPodHeader == "" || DecodePodHeader == "" {
		t.Fatal("expected P/D address headers to stay defined")
	}
}

func TestRequestLengthOfChatCompletions(t *testing.T) {
	request := &fwkscheduling.LLMRequest{
		Body: &fwkscheduling.LLMRequestBody{
			ChatCompletions: &fwkscheduling.ChatCompletionsRequest{
				Messages: []fwkscheduling.Message{
					{Role: "user", Content: fwkscheduling.Content{Raw: "hi"}},
					{Role: "assistant", Content: fwkscheduling.Content{Raw: "你好"}},
				},
			},
		},
	}

	got, err := RequestLengthOf(request)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if got != 4 {
		t.Fatalf("expected request length 4, got %d", got)
	}
}
