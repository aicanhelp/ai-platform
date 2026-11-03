/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package releasefacts provides the embedded static rule catalog (frozen from
// node-checker / env-check / precheck after a one-time review).
package releasefacts

import (
	clusteradapter "gopkg.openfuyao.cn/preflight/adapter/openfuyao/clustercreate"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
)

// StaticRules returns the embedded static rule catalog. These are the rules
// frozen from reviewed current installation requirements. Historical
// node-checker/env-check/precheck candidates are not retained as fallback
// rules. Overly broad residue, sysctl and runtime probes are intentionally not
// part of this catalog.
//
// The rules are data, not a Go plugin; the SDK evaluator consumes them.
func StaticRules() []api.RuleTemplate {
	rules := append([]api.RuleTemplate{}, initadapter.CatalogRules()...)
	rules = append(rules, clusteradapter.CatalogRules()...)
	return rules
}
