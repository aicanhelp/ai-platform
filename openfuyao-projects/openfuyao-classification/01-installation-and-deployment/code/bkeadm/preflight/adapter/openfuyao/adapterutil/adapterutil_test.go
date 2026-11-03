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

package adapterutil

import (
	"reflect"
	"testing"

	"gopkg.openfuyao.cn/preflight/api"
)

func TestSourceLabels(t *testing.T) {
	if got := HumanizeSourceID("image_repo/main"); got != "Image Repo Main" {
		t.Fatalf("HumanizeSourceID() = %q", got)
	}
	if got := HumanizeSourceID("---"); got != "Installation source" {
		t.Fatalf("HumanizeSourceID(empty) = %q", got)
	}
	if got := SourceCheckName("Image repo", " registry.example.com:443 "); got != "Image repo is accessible (registry.example.com:443)" {
		t.Fatalf("SourceCheckName() = %q", got)
	}
	if got := SourceCheckName("Image repo", " "); got != "Image repo is accessible" {
		t.Fatalf("SourceCheckName(empty) = %q", got)
	}
	want := "unable to access Image repo (registry.example.com:443); check network connectivity, " +
		"the endpoint address, proxy/NO_PROXY, and firewall settings, then run preflight again"
	if got := SourceUnreachableFix("Image repo", " registry.example.com:443 "); got != want {
		t.Fatalf("SourceUnreachableFix() = %q", got)
	}
	if got := SourceUnreachableFix("Image repo", ""); got == want {
		t.Fatalf("SourceUnreachableFix(empty) unexpectedly included endpoint: %q", got)
	}
}

func TestUniqueRules(t *testing.T) {
	items := []string{"RULE-A:first", "RULE-B:second", "RULE-A:last"}
	rules := UniqueRules(items, func(item string) api.RuleTemplate {
		return api.RuleTemplate{ID: item[:6], Name: item[7:]}
	})
	want := []api.RuleTemplate{
		{ID: "RULE-A", Name: "first"},
		{ID: "RULE-B", Name: "second"},
	}
	if !reflect.DeepEqual(rules, want) {
		t.Fatalf("UniqueRules() = %#v, want %#v", rules, want)
	}
}

func TestPlanInt(t *testing.T) {
	plan := api.OperationPlan{Values: map[string]api.Value{
		"int":     {Value: int(1)},
		"int64":   {Value: int64(2)},
		"float64": {Value: float64(3)},
		"string":  {Value: "4"},
	}}
	for key, want := range map[string]int{"int": 1, "int64": 2, "float64": 3, "string": 0, "missing": 0} {
		if got := PlanInt(plan, key); got != want {
			t.Errorf("PlanInt(%q) = %d, want %d", key, got, want)
		}
	}
}
