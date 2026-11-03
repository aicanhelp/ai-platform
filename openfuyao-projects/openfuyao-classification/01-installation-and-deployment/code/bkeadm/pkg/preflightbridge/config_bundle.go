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

package preflightbridge

import (
	"fmt"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightbundlecompiler"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/bundle"
	"gopkg.openfuyao.cn/preflight/releasefacts"
)

// BuildBundle compiles the human-authored policy into the SDK's neutral
// runtime input. The generated object exists only for this process; it is
// not embedded, written back, or treated as a second configuration source.
// Rule IDs are derived from the reviewed catalog and never appear in the
// human-authored YAML.
func BuildBundle(config preflightconfig.Config) ([]byte, error) {
	// Defensive re-validation: Load/LoadFile already validate, but BuildBundle
	// is an exported API that also accepts hand-constructed configs.
	if err := preflightconfig.Validate(config); err != nil {
		return nil, fmt.Errorf("validate policy: %w", err)
	}
	rules := releasefacts.StaticRules()
	preflightbundlecompiler.ApplyRuntimePolicy(config, rules)
	if preflightbundlecompiler.HasCustomChecks(config) {
		rules = append(rules, preflightbundlecompiler.CustomRules(config)...)
	}
	if preflightbundlecompiler.HasExtensionChecks(config) {
		rules = append(rules, preflightbundlecompiler.ExtensionRules(config)...)
	}
	b := &api.PreflightBundle{
		Operations: []api.Operation{api.OperationInit, api.OperationClusterCreate},
		Rules:      rules,
	}
	if err := bundle.Validate(b); err != nil {
		return nil, fmt.Errorf("compiled policy is invalid: %w", err)
	}
	raw, err := yaml.Marshal(b)
	if err != nil {
		return nil, fmt.Errorf("marshal compiled policy: %w", err)
	}
	return raw, nil
}
