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

// Package normalize implements the deterministic canonical serialization and
// PlanHash computation for the preflight data contract.
//
// The contract (design spec §0.5):
//   - map keys sorted by UTF-8 byte order;
//   - numbers in canonical decimal form; durations to nanosecond integers;
//     capacities to byte integers;
//   - file references use content digests, not absolute paths;
//   - CredentialRef carries only provider/namespace/name/version/purpose/targetRef;
//   - passwords, private keys, tokens, kubeconfig content and credentialed
//     URLs are excluded;
//   - absolute input paths, entryKind and SourceTrace do NOT enter PlanHash
//     (so equivalent CLI flags and request YAML yield the same hash);
//   - canonical JSON uses UTF-8, LF and a fixed field order, then SHA-256.
//
// The hash is formatted as "sha256:" + lowercase hex.
package normalize

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

// HashAlgo is the digest algorithm prefix carried in PlanHash.
const HashAlgo = "sha256"

// ErrSensitiveRejected is returned when a value that must never enter a hash
// (a credential, secret or credentialed URL) is encountered during
// canonicalization. Such a value is a contract violation, not a hash input.
var ErrSensitiveRejected = errors.New("normalize: sensitive value rejected from hash input")

// sensitiveKeyFragments are substrings that mark a map key as sensitive. A
// key containing any of these (case-insensitive) causes canonicalization to
// reject the value rather than risk leaking it into the hash.
var sensitiveKeyFragments = []string{
	"password", "passwd", "secret", "token", "apikey", "api_key",
	"privatekey", "private_key", "kubeconfig", "credential",
}

// Canonicalize produces the canonical JSON bytes for an arbitrary value under
// the PlanHash rules: sorted map keys, no sensitive fields, deterministic
// encoding. It is the single serialization path for PlanHash, RequestHash
// and bundle digests.
//
// The value is first JSON-round-tripped into generic maps/slices so that
// structs are traversed field-by-field; then sensitive map entries are
// rejected; finally the result is re-marshaled with sorted map keys and no
// HTML escaping. Scalars pass straight through.
func Canonicalize(v any) ([]byte, error) {
	// Round-trip through JSON so structs become map[string]any that
	// stripSensitive can walk recursively. This also normalizes types
	// (e.g. int vs float64) consistently.
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("normalize: initial marshal: %w", err)
	}
	var generic any
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	if err := dec.Decode(&generic); err != nil {
		return nil, fmt.Errorf("normalize: decode: %w", err)
	}
	cleaned, err := stripSensitive(generic, "")
	if err != nil {
		return nil, err
	}
	enc, err := marshalCanonical(cleaned)
	if err != nil {
		return nil, fmt.Errorf("normalize: canonical marshal: %w", err)
	}
	return enc, nil
}

// PlanHashBytes computes the PlanHash over already-canonicalized bytes. It is
// the lowest-level entry point, exposed so callers that build canonical input
// incrementally can hash without re-walking.
func PlanHashBytes(canonical []byte) api.Digest {
	sum := sha256.Sum256(canonical)
	return api.Digest(HashAlgo + ":" + hex.EncodeToString(sum[:]))
}

// PlanHash computes the PlanHash for a value by canonicalizing it first. This
// is the primary entry point for hashing an OperationPlan projection.
func PlanHash(v any) (api.Digest, error) {
	b, err := Canonicalize(v)
	if err != nil {
		return "", err
	}
	return PlanHashBytes(b), nil
}

// stripSensitive walks a value recursively and returns a copy with sensitive
// map entries removed. If a sensitive key is encountered, the whole value is
// rejected (the caller should never have placed it in hash input). path is the
// dotted key path for normalization errors.
func stripSensitive(v any, path string) (any, error) {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			if isSensitiveKey(k) {
				return nil, fmt.Errorf("%w: key %q at %s", ErrSensitiveRejected, k, path)
			}
			clean, err := stripSensitive(val, joinPath(path, k))
			if err != nil {
				return nil, err
			}
			out[k] = clean
		}
		return out, nil
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			clean, err := stripSensitive(val, fmt.Sprintf("%s[%d]", path, i))
			if err != nil {
				return nil, err
			}
			out[i] = clean
		}
		return out, nil
	default:
		if text, ok := t.(string); ok && strings.Contains(text, "://") {
			if parsed, err := url.Parse(text); err == nil && parsed.User != nil {
				return nil, fmt.Errorf("%w: credentialed URL at %s", ErrSensitiveRejected, path)
			}
		}
		// Scalars pass through; json.Marshal will encode them canonically.
		return v, nil
	}
}

// isSensitiveKey reports whether a key name marks its value as sensitive.
func isSensitiveKey(k string) bool {
	lower := strings.ToLower(k)
	for _, frag := range sensitiveKeyFragments {
		if strings.Contains(lower, frag) {
			return true
		}
	}
	return false
}

// joinPath builds a dotted normalization-error path.
func joinPath(parent, key string) string {
	if parent == "" {
		return key
	}
	return parent + "." + key
}

// marshalCanonical encodes a value as canonical JSON with sorted map keys.
// json.Marshal from encoding/json already sorts struct fields by their
// declaration order and map keys lexicographically, which satisfies the
// contract when the input maps have string keys. We wrap it to enforce the
// no-HTML-escaping and no-trailing-newline invariants.
func marshalCanonical(v any) ([]byte, error) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	// json.Encoder.Encode appends a newline; strip it for byte-stable output.
	out := strings.TrimSuffix(buf.String(), "\n")
	return []byte(out), nil
}

// SortedKeys returns the keys of a map sorted by UTF-8 byte order. It is a
// helper for callers that build canonical slices from maps before hashing.
func SortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
