/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgrade

import (
	"sort"
	"sync"
)

// Decision is the VersionContext-driven action for a component.
type Decision string

const (
	// DecisionSkip means current already matches target; Executor should no-op.
	DecisionSkip Decision = "Skip"
	// DecisionUpgrade means the component should run the upgrade path
	// (including first-time align when current is empty but target is set).
	// The declarative DAG is upgrade-only; there is no separate Install decision.
	DecisionUpgrade Decision = "Upgrade"
)

// VersionContext holds per-component current and target versions for declarative upgrade.
type VersionContext struct {
	mu      sync.RWMutex
	Current map[string]string
	Target  map[string]string
}

// NewVersionContext creates an empty VersionContext.
func NewVersionContext() *VersionContext {
	return &VersionContext{
		Current: make(map[string]string),
		Target:  make(map[string]string),
	}
}

// SetCurrent records the running version of a component.
func (vc *VersionContext) SetCurrent(name, version string) {
	if vc == nil {
		return
	}
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.Current[name] = version
}

// SetTarget records the desired version of a component.
func (vc *VersionContext) SetTarget(name, version string) {
	if vc == nil {
		return
	}
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.Target[name] = version
}

// GetCurrent returns the running version of a component.
func (vc *VersionContext) GetCurrent(name string) string {
	v, _ := vc.CurrentVersion(name)
	return v
}

// GetTarget returns the desired version of a component.
func (vc *VersionContext) GetTarget(name string) string {
	v, _ := vc.TargetVersion(name)
	return v
}

// CurrentVersion returns the running version and whether a non-empty value is set.
func (vc *VersionContext) CurrentVersion(name string) (string, bool) {
	if vc == nil {
		return "", false
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	v := vc.Current[name]
	if v == "" {
		return "", false
	}
	return v, true
}

// TargetVersion returns the desired version and whether a non-empty value is set.
func (vc *VersionContext) TargetVersion(name string) (string, bool) {
	if vc == nil {
		return "", false
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	v := vc.Target[name]
	if v == "" {
		return "", false
	}
	return v, true
}

// HasCurrent reports whether a non-empty current version is set for the component.
func (vc *VersionContext) HasCurrent(name string) bool {
	_, ok := vc.CurrentVersion(name)
	return ok
}

// HasTarget reports whether a non-empty target version is set for the component.
func (vc *VersionContext) HasTarget(name string) bool {
	_, ok := vc.TargetVersion(name)
	return ok
}

// NeedsUpgrade reports whether current and target differ for the component.
// Missing / empty target means no upgrade needed.
func (vc *VersionContext) NeedsUpgrade(name string) bool {
	if vc == nil {
		return false
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	target := vc.Target[name]
	if target == "" {
		return false
	}
	return vc.Current[name] != target
}

// Decide returns Skip or Upgrade for Executor decisions on the declarative upgrade DAG.
// - nil VC: Upgrade (context not wired; do not block execution)
// - empty target: Skip (nothing to upgrade to; aligns with NeedsUpgrade == false)
// - current == target: Skip
// - otherwise: Upgrade (including empty current aligning to target)
func Decide(vc *VersionContext, name string) Decision {
	if vc == nil {
		return DecisionUpgrade
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	if vc.Current[name] == vc.Target[name] || vc.Target[name] == "" {
		return DecisionSkip
	}
	return DecisionUpgrade
}

// NeedsExecution reports whether Executor / Apply should run for the component.
func NeedsExecution(vc *VersionContext, name string) bool {
	return Decide(vc, name) != DecisionSkip
}

// AnyTargetNeedsUpgrade reports whether any component in Target has a different Current version.
func (vc *VersionContext) AnyTargetNeedsUpgrade() bool {
	if vc == nil {
		return false
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	for name, target := range vc.Target {
		if target == "" {
			continue
		}
		if vc.Current[name] != target {
			return true
		}
	}
	return false
}

// TargetNames returns sorted component names that have a non-empty target version.
func (vc *VersionContext) TargetNames() []string {
	if vc == nil {
		return nil
	}
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	names := make([]string, 0, len(vc.Target))
	for name, ver := range vc.Target {
		if ver != "" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
