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

// Status is the single public per-check result state.
//
// The four values are the only states a Report Finding may carry. Internal
// evaluation results (PASS/FAIL/UNKNOWN/SKIPPED) are normalized into these
// four public states by the evaluator; callers MUST NOT recompute status.
type Status string

const (
	// StatusPass means a fact was obtained and satisfies the expectation.
	StatusPass Status = "PASS"
	// StatusFail means a fact was obtained and does not satisfy a blocking
	// expectation, or an unknown-policy of BLOCK could not obtain a trusted
	// fact for a blocking rule.
	StatusFail Status = "FAIL"
	// StatusWarn means a fact was obtained and does not satisfy a
	// non-blocking expectation, or an unknown-policy of WARN could not obtain
	// a trusted fact for a warning rule.
	StatusWarn Status = "WARN"
	// StatusSkip means an applicable check was blocked by a dependency failure,
	// unavailable fact under unknown=SKIP, or cancellation/timeout before it
	// completed. Non-applicable rules are not results. SKIP is not a pass.
	StatusSkip Status = "SKIP"
)

// Valid reports whether s is one of the four public statuses.
func (s Status) Valid() bool {
	switch s {
	case StatusPass, StatusFail, StatusWarn, StatusSkip:
		return true
	default:
		return false
	}
}

// String returns the status as its canonical upper-case token.
func (s Status) String() string { return string(s) }

// statusesInSeverityOrder lists statuses from most to least severe, which is
// the order used by AggregateOverall. FAIL is more severe than WARN, which is
// more severe than PASS; SKIP never raises the overall conclusion on its own.
var statusesInSeverityOrder = []Status{StatusFail, StatusWarn, StatusPass, StatusSkip}

// SeverityRank returns a rank for s used by aggregation. Higher means more
// severe. Unknown statuses return -1 so they never aggregate up.
func SeverityRank(s Status) int {
	for i, v := range statusesInSeverityOrder {
		if v == s {
			return len(statusesInSeverityOrder) - i
		}
	}
	return -1
}

// FailureSeverity describes what a FAIL evaluation means for the user.
//
// ERROR maps a failed evaluation to a blocking FAIL; WARNING maps it to a
// non-blocking WARN; INFO is observation-only and normally only produces PASS
// or SKIP. This is the rule-side severity, distinct from the public Status.
type FailureSeverity string

// Supported rule failure severities.
const (
	SeverityError   FailureSeverity = "ERROR"   // SeverityError maps a mismatch to FAIL.
	SeverityWarning FailureSeverity = "WARNING" // SeverityWarning maps a mismatch to WARN.
	SeverityInfo    FailureSeverity = "INFO"    // SeverityInfo records an observation.
)

// Valid reports whether fs is a recognized failure severity.
func (fs FailureSeverity) Valid() bool {
	switch fs {
	case SeverityError, SeverityWarning, SeverityInfo:
		return true
	default:
		return false
	}
}

// UnknownPolicy describes what happens when a required fact cannot be obtained
// with sufficient evidence. It MUST NOT be used to turn a genuine rule
// mismatch into "unknown": a fact that is present but does not satisfy the
// expectation is a FAIL/WARN per FailureSeverity, never unknown.
type UnknownPolicy string

const (
	// UnknownBlock turns an unavailable fact into FAIL (blocking).
	UnknownBlock UnknownPolicy = "BLOCK"
	// UnknownWarn turns an unavailable fact into WARN (non-blocking).
	UnknownWarn UnknownPolicy = "WARN"
	// UnknownIgnore turns an unavailable fact into SKIP.
	UnknownIgnore UnknownPolicy = "IGNORE"
)

// Valid reports whether p is a recognized unknown policy.
func (p UnknownPolicy) Valid() bool {
	switch p {
	case UnknownBlock, UnknownWarn, UnknownIgnore:
		return true
	default:
		return false
	}
}

// NormalizeUnknown maps an unknown-policy to the public Status produced when
// a fact is unavailable. It is the only sanctioned way to convert an
// unavailable fact into a status; callers MUST NOT treat unknown as PASS.
func NormalizeUnknown(p UnknownPolicy) Status {
	switch p {
	case UnknownBlock:
		return StatusFail
	case UnknownWarn:
		return StatusWarn
	case UnknownIgnore:
		return StatusSkip
	default:
		// An invalid unknown policy is treated conservatively as blocking.
		return StatusFail
	}
}

// NormalizeFailure maps a failed evaluation and its severity to the public
// Status. ERROR -> FAIL, WARNING -> WARN. INFO failures are a contract error
// (INFO is observation-only) and are conservatively reported as FAIL so a
// misconfigured rule can never be silently swallowed.
func NormalizeFailure(sev FailureSeverity) Status {
	switch sev {
	case SeverityError:
		return StatusFail
	case SeverityWarning:
		return StatusWarn
	case SeverityInfo:
		// INFO rules must not produce a FAIL; if they do, surface it.
		return StatusFail
	default:
		return StatusFail
	}
}
