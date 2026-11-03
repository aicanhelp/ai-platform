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
	"encoding/json"
	"testing"
)

// FuzzCanonicalizeDeterministic checks that canonicalizing the same JSON input
// twice always yields identical bytes (idempotence), and that canonicalization
// never panics on arbitrary input. Sensitive keys are expected to error
// deterministically, not panic.
//
// The fuzz input is raw JSON bytes, decoded into a generic value before
// canonicalization. This is the standard fuzz shape: []byte is a permitted
// fuzz argument type and exercises the full recursive walk.
func FuzzCanonicalizeDeterministic(f *testing.F) {
	// Seed with representative JSON shapes.
	f.Add([]byte(`"scalar"`))
	f.Add([]byte(`{"a":1,"b":2}`))
	f.Add([]byte(`{"nested":{"x":"y"}}`))
	f.Add([]byte(`[1,2,3]`))
	f.Add([]byte(`{"password":"s3cr3t"}`))
	f.Add([]byte(`{"token":{"access":"abc"}}`))
	f.Add([]byte(`{"port":6443,"hosts":["a","b"]}`))
	f.Add([]byte(``))
	f.Add([]byte(`not json`))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Decode into a generic value. If it isn't valid JSON, canonicalize
		// must still not panic; skip non-JSON rather than asserting success.
		var v any
		if err := json.Unmarshal(data, &v); err != nil {
			// Non-JSON input: canonicalizing a string of the raw bytes is still
			// valid and must be deterministic.
			v = string(data)
		}
		b1, err1 := Canonicalize(v)
		b2, err2 := Canonicalize(v)
		// Both calls must agree: same error or same bytes.
		if (err1 == nil) != (err2 == nil) {
			t.Fatalf("non-deterministic error: %v vs %v", err1, err2)
		}
		if err1 != nil {
			if err1.Error() != err2.Error() {
				t.Fatalf("non-deterministic error message: %v vs %v", err1, err2)
			}
			return
		}
		if string(b1) != string(b2) {
			t.Fatalf("non-deterministic canonical output:\n%s\n%s", b1, b2)
		}
	})
}

// FuzzPlanHashStable checks that computing PlanHash over the same plan twice
// yields the same digest (never panics, never diverges).
func FuzzPlanHashStable(f *testing.F) {
	f.Add(36443, "init", "local")
	f.Add(6443, "cluster-create", "master-01")
	f.Add(0, "init", "")
	f.Add(65535, "cluster-create", "node-7")

	f.Fuzz(func(t *testing.T, port int, op, subject string) {
		plan := mkFuzzPlan(port, op, subject)
		h1, err1 := PlanHash(plan)
		h2, err2 := PlanHash(plan)
		if (err1 == nil) != (err2 == nil) {
			t.Fatalf("non-deterministic error: %v vs %v", err1, err2)
		}
		if err1 != nil {
			if err1.Error() != err2.Error() {
				t.Fatalf("non-deterministic error message: %v vs %v", err1, err2)
			}
			return
		}
		if h1 != h2 {
			t.Fatalf("non-deterministic plan hash: %s vs %s", h1, h2)
		}
	})
}

// mkFuzzPlan builds a small OperationPlan projection for the fuzz test. It
// avoids api.OperationPlan to keep the fuzz input pure (no external deps in
// f.Add). Values with sensitive keys are intentionally possible so the
// sensitive-rejection path is exercised when op/subject happen to contain
// sensitive fragments.
func mkFuzzPlan(port int, op, subject string) any {
	return map[string]any{
		"operation": op,
		"subjects": []any{
			map[string]any{"id": subject, "address": "10.0.0.1"},
		},
		"values": map[string]any{
			"port.k8s": port,
		},
	}
}
