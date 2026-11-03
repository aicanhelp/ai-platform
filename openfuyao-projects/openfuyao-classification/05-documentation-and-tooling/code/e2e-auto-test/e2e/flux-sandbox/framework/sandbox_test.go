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

package framework

import (
	"regexp"
	"testing"
)

func TestSlugify(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"Agent HTTP API healthz returns ok", "agent-http-api-healthz-returns-ok"},
		{"UPPER CASE", "upper-case"},
		{"  leading and trailing  ", "leading-and-trailing"},
		{"multiple___underscores", "multiple-underscores"},
		{"a", "a"},
		{"", "e2e"},
		{"---", "e2e"},
	}
	for _, tc := range cases {
		if got := Slugify(tc.in); got != tc.want {
			t.Errorf("Slugify(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestSlugifyLength(t *testing.T) {
	long := ""
	for i := 0; i < 30; i++ {
		long += "abcdefghij " // 300 chars
	}
	got := Slugify(long)
	if len(got) > 63 {
		t.Errorf("slug length = %d, want <= 63", len(got))
	}
	if len(got) == 0 || got[0] == '-' || got[len(got)-1] == '-' {
		t.Errorf("slug has bad boundary: %q", got)
	}
}

func TestNewID(t *testing.T) {
	long := ""
	for i := 0; i < 30; i++ {
		long += "abcdefghij " // 300 chars
	}
	for _, name := range []string{"short name", long} {
		id := NewID(name)
		if len(id) > 63 {
			t.Errorf("id length = %d, want <= 63", len(id))
		}
		if !regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`).MatchString(id) {
			t.Errorf("id %q is not RFC1123 compliant", id)
		}
	}
	// Random suffix makes repeated derivations unique.
	seen := map[string]bool{}
	for i := 0; i < 20; i++ {
		seen[NewID("same name")] = true
	}
	if len(seen) < 2 {
		t.Errorf("expected distinct suffixes across calls, got %d distinct", len(seen))
	}
}

func TestBasicSpec(t *testing.T) {
	spec := BasicSpec("sb-1")
	if spec.Image != DefaultImage {
		t.Errorf("Image = %q, want %q", spec.Image, DefaultImage)
	}
	if spec.SandboxID != "sb-1" || spec.SandboxName != "sb-1" {
		t.Errorf("id/name not propagated: %+v", spec)
	}
	if len(spec.Command) == 0 || spec.CPU == "" || spec.Memory == "" {
		t.Errorf("spec incomplete: %+v", spec)
	}
}
