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

package bundle

import (
	"bytes"
	"fmt"
	"io"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/preflight/api"
)

// decode unmarshals bundle bytes (YAML or JSON) into a PreflightBundle. YAML
// is the canonical on-disk format; JSON is a strict subset Go's yaml.v3 also
// decodes. Unknown fields are rejected so a typo or a stale field cannot
// silently survive into a release bundle.
func decode(raw []byte) (*api.PreflightBundle, error) {
	var b api.PreflightBundle
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&b); err != nil {
		return nil, err
	}
	// Reject multi-document input: a bundle is a single object. Re-decoding
	// must yield io.EOF (no second document).
	var second api.PreflightBundle
	if err := dec.Decode(&second); err == nil {
		return nil, fmt.Errorf("bundle must be a single YAML document")
	} else if err != io.EOF {
		return nil, fmt.Errorf("bundle trailing content: %w", err)
	}
	return &b, nil
}

// rejectSensitive rejects rule content that looks like a credential. The
// Expected value of a rule must not itself be a credential; a rule comparing
// a password is a design smell.
func rejectSensitive(b *api.PreflightBundle) error {
	for _, r := range b.Rules {
		if r.Fix != "" && looksSensitive(r.Fix) {
			return &LoadError{Field: "rules", Reason: fmt.Sprintf("rule %s fix contains sensitive-looking value", r.ID)}
		}
	}
	return nil
}

// looksSensitive is a lightweight check that a remediation string does not
// embed a literal password. It is intentionally conservative: a false positive
// just means the bundle author must reword the fix.
func looksSensitive(s string) bool {
	// Reuse normalize's fragment list by checking the same substrings. We
	// don't import a private symbol; instead we check common patterns.
	for _, frag := range []string{"password=", "passwd=", "token=", "api_key=", "secret="} {
		if containsFold(s, frag) {
			return true
		}
	}
	return false
}

func containsFold(s, sub string) bool {
	ls := len(s)
	lsub := len(sub)
	if lsub == 0 || lsub > ls {
		return false
	}
	for i := 0; i <= ls-lsub; i++ {
		match := true
		for j := 0; j < lsub; j++ {
			a, b := s[i+j], sub[j]
			if a >= 'A' && a <= 'Z' {
				a += 32
			}
			if b >= 'A' && b <= 'Z' {
				b += 32
			}
			if a != b {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
