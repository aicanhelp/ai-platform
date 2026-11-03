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

// Package bundle loads and validates a PreflightBundle at runtime.
//
// The runtime contract:
//   - rule IDs are unique;
//   - every operation is in the allowed set;
//   - every rule has a remediation fix, and its operator and expected value
//     types are recognized;
//   - dependency references resolve and contain no cycles;
//   - no passwords, tokens, private keys, full kubeconfig or credentialed URLs;
//   - on a missing or unparseable bundle load returns BUNDLE_ERROR and the
//     SDK MUST NOT fall back to an older bundle.
//
// Bundle bytes are supplied by the host, compiled in-process from the embedded
// policy. The SDK never reads a bundle from disk or network on its own.
package bundle

import (
	"errors"
	"fmt"
	"regexp"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/celcheck"
)

var sha256DigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
var kernelPrefixPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+\.$`)

// LoadError is returned when a bundle fails to load. It carries the Overall
// conclusion the caller should surface: BUNDLE_ERROR. The SDK MUST NOT fall
// back to an older bundle when a LoadError occurs.
type LoadError struct {
	Field  string
	Reason string
}

func (e *LoadError) Error() string {
	if e.Field == "" {
		return "bundle load error: " + e.Reason
	}
	return "bundle load error: " + e.Field + ": " + e.Reason
}

// IsLoadError reports whether err is a *LoadError.
func IsLoadError(err error) bool {
	var le *LoadError
	return errors.As(err, &le)
}

// Load decodes and validates a PreflightBundle from raw YAML/JSON bytes. It
// returns the typed bundle on success, or a *LoadError on any contract
// violation. Callers MUST treat any *LoadError as BUNDLE_ERROR and not fall
// back to an older bundle.
func Load(raw []byte) (*api.PreflightBundle, error) {
	if len(raw) == 0 {
		return nil, &LoadError{Reason: "bundle is empty"}
	}
	b, err := decode(raw)
	if err != nil {
		return nil, &LoadError{Field: "yaml", Reason: err.Error()}
	}
	if err := Validate(b); err != nil {
		return nil, err
	}
	return b, nil
}

// Validate checks the loaded bundle against the static contract.
func Validate(b *api.PreflightBundle) error {
	validators := []func(*api.PreflightBundle) error{
		validateBundleHeader,
		validateBundleRules,
		validateVerifiedPlatforms,
	}
	for _, validate := range validators {
		if err := validate(b); err != nil {
			return err
		}
	}
	return rejectSensitive(b)
}

func validateBundleHeader(b *api.PreflightBundle) error {
	if b == nil {
		return &LoadError{Reason: "bundle is nil"}
	}
	if len(b.Operations) == 0 {
		return &LoadError{Field: "operations", Reason: "empty"}
	}
	for _, op := range b.Operations {
		if !api.IsSupportedOperation(op) {
			return &LoadError{Field: "operations", Reason: "unsupported operation " + string(op)}
		}
	}
	return nil
}

func validateBundleRules(b *api.PreflightBundle) error {
	if dup, ok := b.HasDuplicateRuleIDs(); ok {
		return &LoadError{Field: "rules", Reason: "duplicate rule ID " + dup}
	}
	for _, r := range b.Rules {
		if rerr := r.Validate(); rerr != nil {
			return &LoadError{Field: "rules", Reason: rerr.Error()}
		}
		if r.Op == api.OpCEL {
			if _, err := celcheck.Compile(r.Expression); err != nil {
				return &LoadError{Field: "rules", Reason: fmt.Sprintf("rule %s: %v", r.ID, err)}
			}
		}
		// A rule's operation must be one declared by the bundle.
		if !declaresOperation(b, r.Operation) {
			return &LoadError{
				Field:  "rules",
				Reason: fmt.Sprintf("rule %s operation %s not in bundle operations", r.ID, r.Operation),
			}
		}
	}
	if err := validateDependencies(b.Rules); err != nil {
		return err
	}
	return nil
}

func validateVerifiedPlatforms(b *api.PreflightBundle) error {
	seenPlatforms := make(map[string]struct{}, len(b.VerifiedPlatforms))
	for _, platform := range b.VerifiedPlatforms {
		identity := platform.Distribution + "\x00" + platform.Version
		if platform.Distribution == "" || platform.Version == "" || len(platform.Architectures) == 0 ||
			len(platform.KernelPrefixes) == 0 {
			return &LoadError{
				Field:  "verifiedPlatforms",
				Reason: "distribution, version, architectures and kernelPrefixes are required",
			}
		}
		if _, exists := seenPlatforms[identity]; exists {
			return &LoadError{
				Field:  "verifiedPlatforms",
				Reason: "duplicate platform " + platform.Distribution + " " + platform.Version,
			}
		}
		seenPlatforms[identity] = struct{}{}
		for _, architecture := range platform.Architectures {
			if architecture != "amd64" && architecture != "arm64" {
				return &LoadError{Field: "verifiedPlatforms", Reason: "unsupported architecture " + architecture}
			}
		}
		for _, prefix := range platform.KernelPrefixes {
			if !kernelPrefixPattern.MatchString(prefix) {
				return &LoadError{Field: "verifiedPlatforms", Reason: "invalid kernel prefix " + prefix}
			}
		}
	}
	return nil
}

func validateDependencies(rules []api.RuleTemplate) error {
	byID := make(map[string]api.RuleTemplate, len(rules))
	for _, rule := range rules {
		byID[rule.ID] = rule
	}
	if err := validateDependencyReferences(rules, byID); err != nil {
		return err
	}
	checker := dependencyCycleChecker{
		byID:  byID,
		state: make(map[string]uint8, len(rules)),
		stack: make([]string, 0, len(rules)),
	}
	for _, rule := range rules {
		if err := checker.visit(rule.ID); err != nil {
			return err
		}
	}
	return nil
}

func validateDependencyReferences(rules []api.RuleTemplate, byID map[string]api.RuleTemplate) error {
	for _, rule := range rules {
		seen := make(map[string]struct{}, len(rule.DependsOn))
		for _, dependency := range rule.DependsOn {
			if _, duplicate := seen[dependency]; duplicate {
				return &LoadError{
					Field:  "rules",
					Reason: fmt.Sprintf("rule %s has duplicate dependency %s", rule.ID, dependency),
				}
			}
			seen[dependency] = struct{}{}
			dep, ok := byID[dependency]
			if !ok {
				return &LoadError{
					Field:  "rules",
					Reason: fmt.Sprintf("rule %s references missing dependency %s", rule.ID, dependency),
				}
			}
			if dep.Operation != rule.Operation {
				return &LoadError{
					Field: "rules",
					Reason: fmt.Sprintf(
						"rule %s dependency %s belongs to operation %s",
						rule.ID,
						dependency,
						dep.Operation,
					),
				}
			}
		}
	}
	return nil
}

type dependencyCycleChecker struct {
	byID  map[string]api.RuleTemplate
	state map[string]uint8
	stack []string
}

func (c *dependencyCycleChecker) visit(id string) error {
	const (
		visiting = 1
		visited  = 2
	)
	if c.state[id] == visited {
		return nil
	}
	if c.state[id] == visiting {
		return &LoadError{
			Field:  "rules",
			Reason: fmt.Sprintf("dependency cycle includes %s (path %v)", id, append(c.stack, id)),
		}
	}
	c.state[id] = visiting
	c.stack = append(c.stack, id)
	for _, dependency := range c.byID[id].DependsOn {
		if err := c.visit(dependency); err != nil {
			return err
		}
	}
	c.stack = c.stack[:len(c.stack)-1]
	c.state[id] = visited
	return nil
}

// declaresOperation reports whether op is in the bundle's operations.
func declaresOperation(b *api.PreflightBundle, op api.Operation) bool {
	for _, o := range b.Operations {
		if o == op {
			return true
		}
	}
	return false
}
