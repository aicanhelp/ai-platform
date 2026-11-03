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

package compatibility

import (
	"context"
	"strings"

	"github.com/Masterminds/semver/v3"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

type Engine struct{}

func NewEngine() *Engine {
	return &Engine{}
}

func (e *Engine) Check(_ context.Context, bundle *manifest.Bundle) Report {
	flat, err := Flatten(bundle)
	if err != nil {
		return Report{Allowed: false, Conflicts: []Conflict{{
			Component: "release", Message: err.Error(),
		}}}
	}
	versions := map[string]string{}
	for _, c := range flat {
		versions[c.Name] = c.Version
	}
	report := Report{Allowed: true, Components: flat}
	for _, c := range flat {
		cv, ok := bundle.Components[manifest.ComponentKey(c.Name, c.Version)]
		if !ok {
			report.Allowed = false
			report.Conflicts = append(report.Conflicts, Conflict{
				Component: c.Name, Version: c.Version,
				Message: "resolved component is missing in bundle",
			})
			continue
		}
		for _, rule := range cv.Spec.Compatibility.Constraints {
			actual, ok := versions[rule.Component]
			if !ok {
				report.Allowed = false
				report.Conflicts = append(report.Conflicts, Conflict{
					Component: rule.Component, RequiredBy: c.Name,
					Rule:    rule.Rule,
					Message: "required component is missing",
				})
				continue
			}
			constraint, err := semver.NewConstraint(normalizeConstraint(rule.Rule))
			if err != nil {
				report.Allowed = false
				report.Conflicts = append(report.Conflicts, Conflict{
					Component: rule.Component,
					Version:   actual, RequiredBy: c.Name,
					Rule:    rule.Rule,
					Message: "invalid compatibility rule: " + err.Error(),
				})
				continue
			}
			version, err := semver.NewVersion(normalizeVersion(actual))
			if err != nil || !constraint.Check(version) {
				report.Allowed = false
				report.Conflicts = append(report.Conflicts, Conflict{
					Component:  rule.Component,
					Version:    actual,
					RequiredBy: c.Name,
					Rule:       rule.Rule,
					Message:    "version does not satisfy constraint",
				})
			}
		}
	}
	return report
}

func normalizeConstraint(rule string) string {
	fields := strings.Fields(rule)
	for i := range fields {
		fields[i] = normalizeVersion(fields[i])
	}
	return strings.Join(fields, " ")
}

func normalizeVersion(v string) string {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	for _, op := range []string{">=", "<=", "!=", ">", "<", "="} {
		if strings.HasPrefix(v, op) {
			return op + normalizeVersion(strings.TrimPrefix(v, op))
		}
	}
	if idx := strings.Index(v, "-of."); idx >= 0 {
		return v[:idx]
	}
	return v
}
