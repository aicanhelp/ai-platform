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
	"math/rand"
	"strings"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
)

// DefaultImage is the image used by e2e sandboxes.
const DefaultImage = "docker.io/library/busybox:latest"

// BasicSpec builds a minimal, working sandbox spec: busybox sleeping for an
// hour with small resource limits.
func BasicSpec(sandboxID string) agentapi.SandboxSpec {
	return agentapi.SandboxSpec{
		SandboxID:   sandboxID,
		SandboxName: sandboxID,
		Image:       DefaultImage,
		Command:     []string{"/bin/sleep", "3600"},
		CPU:         "100m",
		Memory:      "64Mi",
	}
}

// Slugify converts an arbitrary test name into an RFC 1123-compliant label
// fragment: lowercase alphanumerics and hyphens, no leading/trailing hyphen,
// at most 63 characters.
func Slugify(name string) string {
	var b strings.Builder
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'):
			b.WriteByte(c)
		case c >= 'A' && c <= 'Z':
			b.WriteByte(c + ('a' - 'A'))
		default:
			if b.Len() > 0 && !strings.HasSuffix(b.String(), "-") {
				b.WriteByte('-')
			}
		}
	}
	s := strings.Trim(b.String(), "-")
	if len(s) > 63 {
		s = strings.TrimLeft(s[len(s)-63:], "-")
	}
	if s == "" {
		s = "e2e"
	}
	return s
}

const idSuffixChars = "abcdefghijklmnopqrstuvwxyz0123456789"

// NewID derives a unique sandbox ID from a test name: a slug of the name plus
// a short random suffix, so repeated runs never collide with leftovers.
func NewID(testName string) string {
	suffix := make([]byte, 4)
	for i := range suffix {
		suffix[i] = idSuffixChars[rand.Intn(len(idSuffixChars))]
	}
	id := Slugify(testName)
	if len(id) > 58 {
		id = strings.TrimLeft(id[len(id)-58:], "-")
	}
	if id == "" {
		id = "e2e"
	}
	return id + "-" + string(suffix)
}
