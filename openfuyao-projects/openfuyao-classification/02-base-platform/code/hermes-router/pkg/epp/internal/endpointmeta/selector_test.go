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

package endpointmeta

import (
	"reflect"
	"testing"
)

const (
	testReadyLabelKey = "ready"
	testZoneLabelKey  = "topology.kubernetes.io/zone"
	testZoneLabelA    = "zone-a"
	testZoneLabelB    = "zone-b"
)

func TestCompileLabelSelectorRequiresNonEmptyLabelNames(t *testing.T) {
	t.Run("exists", func(t *testing.T) {
		_, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
			Exists: []string{" ", testRoleLabelKey},
		})
		if err == nil {
			t.Fatal("expected error for empty exists label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})

	t.Run("in", func(t *testing.T) {
		_, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
			In: map[string][]string{
				" ": {testRoleLabelValue},
			},
		})
		if err == nil {
			t.Fatal("expected error for empty in-clause label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})
}

func TestCompileLabelSelectorRejectsInvalidLabelNames(t *testing.T) {
	t.Run("exists", func(t *testing.T) {
		_, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
			Exists: []string{"invalid/key/extra"},
		})
		if err == nil {
			t.Fatal("expected error for invalid exists label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})

	t.Run("in", func(t *testing.T) {
		_, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
			In: map[string][]string{
				"invalid/key/extra": {testRoleLabelValue},
			},
		})
		if err == nil {
			t.Fatal("expected error for invalid in-clause label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})
}

func TestCompileLabelSelectorNormalizesExistsAndInClauses(t *testing.T) {
	selector, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
		Exists: []string{" " + testRoleLabelKey + " ", testZoneLabelKey, testRoleLabelKey},
		In: map[string][]string{
			" " + testRoleLabelKey + " ": {" " + testRoleLabelValue + " ", "", "decode", testRoleLabelValue},
			" " + testZoneLabelKey + " ": {" " + testZoneLabelA + " ", testZoneLabelA, "  "},
		},
	})
	if err != nil {
		t.Fatalf("expected selector compilation to succeed, got error: %v", err)
	}
	if !hasSelector {
		t.Fatal("expected compiled selector to be present")
	}
	if !reflect.DeepEqual(selector.exists, []string{testRoleLabelKey, testZoneLabelKey}) {
		t.Fatalf("unexpected normalized exists clauses: %#v", selector.exists)
	}
	assertSelectorAllowedValues(t, selector.in[testRoleLabelKey], []string{testRoleLabelValue, "decode"})
	assertSelectorAllowedValues(t, selector.in[testZoneLabelKey], []string{testZoneLabelA})
	if _, ok := selector.in[" "+testRoleLabelKey+" "]; ok {
		t.Fatal("expected in-clause keys to be trimmed")
	}
}

func TestLabelSelectorMatchesExistsAndInRules(t *testing.T) {
	var emptySelector LabelSelector
	if !emptySelector.Matches(newTestScoredEndpoint().Endpoint) {
		t.Fatal("expected zero-value selector to match all endpoints")
	}

	selector, hasSelector, err := CompileLabelSelector(&LabelSelectorConfig{
		Exists: []string{testReadyLabelKey},
		In: map[string][]string{
			testRoleLabelKey: {testRoleLabelValue},
			testZoneLabelKey: {testZoneLabelA},
		},
	})
	if err != nil {
		t.Fatalf("expected selector compilation to succeed, got error: %v", err)
	}
	if !hasSelector {
		t.Fatal("expected compiled selector to be present")
	}

	matchingEndpoint := newTestScoredEndpointWithLabels(map[string]string{
		testReadyLabelKey: "true",
		testRoleLabelKey:  testRoleLabelValue,
		testZoneLabelKey:  testZoneLabelA,
	}).Endpoint
	if !selector.Matches(matchingEndpoint) {
		t.Fatal("expected selector to match endpoint that satisfies exists and in clauses")
	}

	missingExistsEndpoint := newTestScoredEndpointWithLabels(map[string]string{
		testRoleLabelKey: testRoleLabelValue,
		testZoneLabelKey: testZoneLabelA,
	}).Endpoint
	if selector.Matches(missingExistsEndpoint) {
		t.Fatal("expected selector to reject endpoint missing required exists label")
	}

	emptyExistsEndpoint := newTestScoredEndpointWithLabels(map[string]string{
		testReadyLabelKey: "",
		testRoleLabelKey:  testRoleLabelValue,
		testZoneLabelKey:  testZoneLabelA,
	}).Endpoint
	if selector.Matches(emptyExistsEndpoint) {
		t.Fatal("expected selector to reject endpoint with empty required exists label")
	}

	invalidValueEndpoint := newTestScoredEndpointWithLabels(map[string]string{
		testReadyLabelKey: "true",
		testRoleLabelKey:  testRoleLabelValue,
		testZoneLabelKey:  testZoneLabelB,
	}).Endpoint
	if selector.Matches(invalidValueEndpoint) {
		t.Fatal("expected selector to reject endpoint with disallowed in-clause value")
	}
}

func TestCompileLegacyRoleSelectorBuildsInClause(t *testing.T) {
	t.Run("requires label when roles configured", func(t *testing.T) {
		_, hasSelector, err := CompileLegacyRoleSelector(" ", []string{testRoleLabelValue})
		if err == nil {
			t.Fatal("expected error when roles are configured without a label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})

	t.Run("rejects invalid label when roles configured", func(t *testing.T) {
		_, hasSelector, err := CompileLegacyRoleSelector("invalid/key/extra", []string{testRoleLabelValue})
		if err == nil {
			t.Fatal("expected error when roles are configured with an invalid label")
		}
		if hasSelector {
			t.Fatal("expected no selector when compilation fails")
		}
	})

	t.Run("builds normalized in clause", func(t *testing.T) {
		selector, hasSelector, err := CompileLegacyRoleSelector(" "+testRoleLabelKey+" ", []string{" " + testRoleLabelValue + " ", "decode", testRoleLabelValue})
		if err != nil {
			t.Fatalf("expected selector compilation to succeed, got error: %v", err)
		}
		if !hasSelector {
			t.Fatal("expected compiled selector to be present")
		}
		if len(selector.exists) != 0 {
			t.Fatalf("expected no exists clauses, got %#v", selector.exists)
		}
		assertSelectorAllowedValues(t, selector.in[testRoleLabelKey], []string{testRoleLabelValue, "decode"})
		if !selector.Matches(newTestScoredEndpoint().Endpoint) {
			t.Fatal("expected legacy selector to match endpoint with allowed role label")
		}
	})
}

func TestCompileLegacyRoleSelectorReturnsNoSelectorWhenLegacyFieldsEmpty(t *testing.T) {
	selector, hasSelector, err := CompileLegacyRoleSelector("", nil)
	if err != nil {
		t.Fatalf("expected no error for empty legacy selector fields, got: %v", err)
	}
	if hasSelector {
		t.Fatal("expected no selector when legacy fields are empty")
	}
	if len(selector.exists) != 0 || len(selector.in) != 0 {
		t.Fatalf("expected zero-value selector, got %#v", selector)
	}
}

func TestCompileLegacyRoleSelectorReturnsNoSelectorWhenRolesUnset(t *testing.T) {
	selector, hasSelector, err := CompileLegacyRoleSelector(testRoleLabelKey, nil)
	if err != nil {
		t.Fatalf("expected no error when legacy roles are unset, got: %v", err)
	}
	if hasSelector {
		t.Fatal("expected no selector when legacy roles are unset")
	}
	if len(selector.exists) != 0 || len(selector.in) != 0 {
		t.Fatalf("expected zero-value selector, got %#v", selector)
	}
}

func TestCompileProducerTargetSelectorRejectsMixedSelectorAndLegacyFields(t *testing.T) {
	_, hasSelector, err := CompileProducerTargetSelector(&LabelSelectorConfig{
		In: map[string][]string{
			testRoleLabelKey: {testRoleLabelValue},
		},
	}, testRoleLabelKey, []string{testRoleLabelValue})
	if err == nil {
		t.Fatal("expected mixed selector error")
	}
	if hasSelector {
		t.Fatal("expected no selector when compilation fails")
	}
}

func TestCompileProducerTargetSelectorRejectsLegacyLabelWithoutRoles(t *testing.T) {
	_, hasSelector, err := CompileProducerTargetSelector(nil, testRoleLabelKey, []string{" ", ""})
	if err == nil {
		t.Fatal("expected incomplete legacy selector error")
	}
	if hasSelector {
		t.Fatal("expected no selector when compilation fails")
	}
}

func TestCompileProducerTargetSelectorBuildsLegacySelector(t *testing.T) {
	selector, hasSelector, err := CompileProducerTargetSelector(nil, " "+testRoleLabelKey+" ", []string{" " + testRoleLabelValue + " ", "decode", testRoleLabelValue})
	if err != nil {
		t.Fatalf("expected selector compilation to succeed, got error: %v", err)
	}
	if !hasSelector {
		t.Fatal("expected compiled selector to be present")
	}
	assertSelectorAllowedValues(t, selector.in[testRoleLabelKey], []string{testRoleLabelValue, "decode"})
}

func assertSelectorAllowedValues(t *testing.T, allowed map[string]struct{}, expected []string) {
	t.Helper()
	if len(allowed) != len(expected) {
		t.Fatalf("expected %d allowed values, got %d", len(expected), len(allowed))
	}
	for _, value := range expected {
		if _, ok := allowed[value]; !ok {
			t.Fatalf("expected allowed values to contain %q, got %#v", value, allowed)
		}
	}
}
