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
	"fmt"
	"strings"

	componentv1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

type Report struct {
	Allowed    bool
	Components []ResolvedComponent
	Conflicts  []Conflict
}

type ResolvedComponent struct {
	Name        string
	Version     string
	Parent      string
	InstallType componentv1.ComponentType
}

type Conflict struct {
	Component  string
	Version    string
	RequiredBy string
	Rule       string
	Message    string
}

func (r Report) Detail() string {
	if r.Allowed {
		return "compatibility check passed"
	}
	var b strings.Builder
	for _, c := range r.Conflicts {
		_, _ = fmt.Fprintf(&b, "component=%s version=%s requiredBy=%s rule=%s message=%s\n",
			c.Component, c.Version, c.RequiredBy, c.Rule, c.Message)
	}
	return strings.TrimSpace(b.String())
}
