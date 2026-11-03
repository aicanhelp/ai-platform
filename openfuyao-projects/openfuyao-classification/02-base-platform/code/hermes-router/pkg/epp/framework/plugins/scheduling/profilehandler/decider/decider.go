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

// Package decider defines Prefill scheduling decision strategies consumed by
// ProfileHandlers (for example the disagg handler) after Decode selection.
package decider

import (
	"context"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// Strategy decides whether Prefill should run after Decode selection.
// A non-nil error is treated by the disagg ProfileHandler as shouldRunPrefill=true
// (conservative pd path).
type Strategy interface {
	fwkplugin.Plugin
	Decide(
		ctx context.Context,
		request *fwkscheduling.LLMRequest,
		selectedDecode fwkscheduling.Endpoint,
	) (shouldRunPrefill bool, err error)
}
