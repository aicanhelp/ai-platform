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

package api

// Cause codes are stable identifiers carried in Result.CauseCode and
// referenced by downstream SKIP results. They name the root-cause category
// for a failure or skip; one root cause yields one FAIL/WARN, dependents
// yield SKIP referencing the root.
const (
	// CauseInputError is the root cause for an input-shape problem.
	CauseInputError = "INPUT_ERROR"
	// CauseBundleError is the root cause for a bundle load/validation problem.
	CauseBundleError = "BUNDLE_ERROR"
	// CauseCanceledBeforeStart marks an instance skipped because the run was
	// canceled before it started.
	CauseCanceledBeforeStart = "CANCELED_BEFORE_START"
	// CauseCanceledDuringCollection marks an instance skipped because
	// cancellation arrived mid-collection with no trusted fact.
	CauseCanceledDuringCollection = "CANCELED_DURING_COLLECTION"
	// CauseDependencyFailed marks an instance skipped because a dependency
	// instance failed; DependsOn references the root.
	CauseDependencyFailed = "DEPENDENCY_FAILED"
	// CauseNotApplicable is retained for report compatibility; new runs omit
	// non-applicable instances instead of emitting a result with this cause.
	CauseNotApplicable = "NOT_APPLICABLE"
	// CauseUnknownFact marks an instance whose fact could not be obtained and
	// was resolved by the unknown policy.
	CauseUnknownFact = "UNKNOWN_FACT"
)

// FrameworkError is returned by the engine only for SDK/collector internal
// faults (overall=INTERNAL_ERROR). Input, bundle and check failures MUST be
// carried in the Report, not as a FrameworkError.
type FrameworkError struct {
	Code    string
	Message string
	Cause   error
}

func (e *FrameworkError) Error() string {
	if e.Cause != nil {
		return e.Code + ": " + e.Message + ": " + e.Cause.Error()
	}
	return e.Code + ": " + e.Message
}

// Unwrap returns the underlying framework error cause.
func (e *FrameworkError) Unwrap() error { return e.Cause }

// Framework error codes.
const (
	FrameworkErrInternal       = "INTERNAL"
	FrameworkErrBundleLoad     = "BUNDLE_LOAD"
	FrameworkErrPlanBuild      = "PLAN_BUILD"
	FrameworkErrCollectorPanic = "COLLECTOR_PANIC"
)
