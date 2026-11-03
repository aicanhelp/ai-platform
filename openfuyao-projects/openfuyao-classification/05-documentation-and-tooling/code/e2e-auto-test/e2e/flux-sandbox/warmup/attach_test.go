//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"slices"
	"strings"
	"testing"
)

func TestGateEnabled(t *testing.T) {
	tests := []struct {
		name string
		val  string
		want bool
	}{
		{"unset enables all specs", "", true},
		{"explicit 1 wins", "1", true},
		{"explicit 0 disables", "0", false},
		{"explicit false disables", "false", false},
		{"explicit off disables", "off", false},
		{"explicit no disables", "no", false},
		{"explicit garbage enables", "yes", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("WARM_E2E_TEST_GATE", tt.val)
			if got := gateEnabled("WARM_E2E_TEST_GATE"); got != tt.want {
				t.Fatalf("gateEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDefaultControllerNamespace(t *testing.T) {
	if got := defaultControllerNamespace(); got != "flux-system" {
		t.Fatalf("defaultControllerNamespace() = %q, want flux-system", got)
	}
}

func TestStripWarmFlags(t *testing.T) {
	got := stripWarmFlags([]string{
		"--http-addr=:8080", "--warm-enabled=true", "--warm-postgres-dsn=old",
		"--scheduler-namespace=ns",
	})
	want := []string{"--http-addr=:8080", "--scheduler-namespace=ns"}
	if !slices.Equal(got, want) {
		t.Fatalf("stripWarmFlags() = %v, want %v", got, want)
	}
}

func TestMergeWarmFlags(t *testing.T) {
	merged := mergeWarmFlags([]string{"--http-addr=:8080", "--warm-postgres-dsn=old"})
	warmCount := 0
	for _, a := range merged {
		if len(a) > 7 && a[:7] == "--warm-" {
			warmCount++
		}
	}
	if warmCount != len(warmFlags()) {
		t.Fatalf("merged warm flag count = %d, want %d (duplicates leaked)", warmCount, len(warmFlags()))
	}
	foundDSN := false
	for _, a := range merged {
		if strings.HasPrefix(a, "--warm-postgres-dsn=postgres://") {
			foundDSN = true
		}
	}
	if !foundDSN {
		t.Fatalf("merged args lack canonical DSN: %v", merged)
	}
}
