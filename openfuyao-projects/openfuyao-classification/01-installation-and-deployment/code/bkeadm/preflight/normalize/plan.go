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

package normalize

import (
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

const schemeSeparatorLength = len("://")

// PlanProjection is the hashable projection of an OperationPlan. It carries
// only resolved business semantics: operation, subjects, values and sources.
// It deliberately excludes Origin (entryKind/absolute input paths) and any
// SourceTrace, so equivalent CLI flags and request YAML produce the same hash.
//
// Subjects are sorted by ID; sources by ID; values is a sorted-key map. The
// projection is what gets canonicalized and SHA-256'd into PlanHash.
type PlanProjection struct {
	Operation string                     `json:"operation"`
	Subjects  []subjectProjection        `json:"subjects,omitempty"`
	Values    map[string]valueProjection `json:"values,omitempty"`
	Sources   []sourceProjection         `json:"sources,omitempty"`
}

type valueProjection struct {
	Key   string        `json:"key"`
	Type  api.ValueType `json:"type"`
	Value any           `json:"value"`
}

type subjectProjection struct {
	ID       string            `json:"id"`
	Role     []string          `json:"role,omitempty"`
	Address  string            `json:"address,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type sourceProjection struct {
	ID         string `json:"id"`
	Endpoint   string `json:"endpoint,omitempty"`
	Kind       string `json:"kind,omitempty"`
	Scheme     string `json:"scheme,omitempty"`
	ObjectPath string `json:"objectPath,omitempty"`
	AuthRef    string `json:"authRef,omitempty"`
}

// ProjectPlan builds the canonical hashable projection from an OperationPlan.
// It sorts subjects and sources by stable ID and normalizes addresses. It
// performs no I/O and creates nothing.
func ProjectPlan(plan *api.OperationPlan) PlanProjection {
	if plan == nil {
		return PlanProjection{}
	}
	p := PlanProjection{
		Operation: string(plan.Operation),
		Values:    make(map[string]valueProjection, len(plan.Values)),
	}
	for k, v := range plan.Values {
		// Values carry typed semantic fields; pass the typed value through.
		// Sensitive keys are rejected later by Canonicalize.
		p.Values[k] = valueProjection{Key: v.Key, Type: v.Type, Value: v.Value}
	}
	// Subjects sorted by ID.
	p.Subjects = make([]subjectProjection, 0, len(plan.Subjects))
	for _, s := range plan.Subjects {
		role := append([]string(nil), s.Role...)
		sortStableStrings(role)
		p.Subjects = append(p.Subjects, subjectProjection{
			ID:       s.ID,
			Role:     role,
			Address:  Address(s.Address),
			Metadata: s.Metadata,
		})
	}
	sortSubjects(p.Subjects)
	// Sources sorted by ID.
	p.Sources = make([]sourceProjection, 0, len(plan.Sources))
	for _, s := range plan.Sources {
		p.Sources = append(p.Sources, sourceProjection{
			ID:         s.ID,
			Endpoint:   s.Endpoint,
			Kind:       s.Kind,
			Scheme:     s.Scheme,
			ObjectPath: s.ObjectPath,
			AuthRef:    s.AuthRef,
		})
	}
	sortSources(p.Sources)
	return p
}

// ComputePlanHash projects, canonicalizes and hashes an OperationPlan. It is
// the primary PlanHash entry point used by the adapter. The returned Digest
// is "sha256:<hex>".
func ComputePlanHash(plan *api.OperationPlan) (api.Digest, error) {
	proj := ProjectPlan(plan)
	return PlanHash(proj)
}

// RedactEndpoint returns an endpoint with any embedded credentials stripped.
// "https://user:pass@host/path" becomes "https://host/path". It is used before
// an endpoint enters Evidence or a Report so secrets never leak.
func RedactEndpoint(endpoint string) string {
	s := strings.TrimSpace(endpoint)
	if s == "" {
		return ""
	}
	// Split scheme.
	scheme, rest := splitScheme(s)
	if rest == "" {
		return s
	}
	// Strip userinfo before '@'.
	if at := strings.Index(rest, "@"); at >= 0 {
		rest = rest[at+1:]
	}
	if scheme != "" {
		return scheme + "://" + rest
	}
	return rest
}

// splitScheme splits "scheme://rest" into ("scheme", "rest"). If there is no
// "://" the scheme is empty and rest is the whole string.
func splitScheme(s string) (scheme, rest string) {
	if i := strings.Index(s, "://"); i >= 0 {
		return s[:i], s[i+schemeSeparatorLength:]
	}
	return "", s
}

// sortStableStrings sorts a slice in place lexicographically.
func sortStableStrings(ss []string) {
	// insertion sort is fine for small role slices; keeps deps minimal.
	for i := 1; i < len(ss); i++ {
		for j := i; j > 0 && ss[j-1] > ss[j]; j-- {
			ss[j-1], ss[j] = ss[j], ss[j-1]
		}
	}
}

func sortSubjects(s []subjectProjection) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1].ID > s[j].ID; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

func sortSources(s []sourceProjection) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1].ID > s[j].ID; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
