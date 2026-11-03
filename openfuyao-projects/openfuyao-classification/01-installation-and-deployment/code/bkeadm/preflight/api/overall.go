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

import (
	"errors"
	"sort"
)

// Overall is the final conclusion of a preflight run.
//
// The four check-level outcomes (PASS/FAIL/WARN/SKIP) are separate from the
// run conclusion so that input errors and framework faults are never
// disguised as a check failure. READY / READY_WITH_WARNINGS / BLOCKED are the
// conclusions for a fully completed run; INPUT_ERROR / INTERNAL_ERROR /
// (BUNDLE_ERROR) cover failure modes that occur before or outside check
// evaluation; TIMED_OUT / CANCELED cover incomplete runs.
type Overall string

// Supported overall outcomes.
const (
	// OverallPass (READY): at least one applicable check and all are PASS.
	OverallPass Overall = "PASS"
	// OverallWarn (READY_WITH_WARNINGS): no FAIL, at least one WARN.
	OverallWarn Overall = "WARN"
	// OverallFail (BLOCKED): at least one FAIL.
	OverallFail Overall = "FAIL"
	// OverallSkip: no applicable checks, or all were skipped.
	OverallSkip Overall = "SKIP"
	// OverallInputError: the request or business input could not form an
	// OperationPlan. Run issues carry the cause; no checks are run.
	OverallInputError Overall = "INPUT_ERROR"
	// OverallBundleError: the bundle is missing, mismatched, has a bad
	// digest, or contains unparseable rules.
	OverallBundleError Overall = "BUNDLE_ERROR"
	// OverallInternalError: an SDK or collector internal fault.
	OverallInternalError Overall = "INTERNAL_ERROR"
	// OverallTimedOut: the overall timeout elapsed before completion.
	OverallTimedOut Overall = "TIMED_OUT"
	// OverallCanceled: the caller canceled the run.
	OverallCanceled Overall = "CANCELED"
)

// Exit codes are stable and the only values the outermost CLI layer may map a
// Report to. They MUST align with Overall as defined by ExitCode.
const (
	ExitSuccess       = 0   // PASS / READY
	ExitWarn          = 2   // WARN / READY_WITH_WARNINGS, or SKIP
	ExitFail          = 3   // FAIL / BLOCKED
	ExitInputError    = 4   // INPUT_ERROR or BUNDLE_ERROR
	ExitInternalError = 5   // INTERNAL_ERROR
	ExitTimedOut      = 124 // overall timeout
	ExitCanceled      = 130 // SIGINT / caller cancel
)

// ExitCode maps an Overall conclusion to its stable exit code.
//
// It is the single source of truth for the CLI exit-code mapping; text and
// JSON renderers MUST consume the same Report and MUST NOT recompute it.
func ExitCode(o Overall) int {
	switch o {
	case OverallPass:
		return ExitSuccess
	case OverallWarn, OverallSkip:
		return ExitWarn
	case OverallFail:
		return ExitFail
	case OverallInputError, OverallBundleError:
		return ExitInputError
	case OverallInternalError:
		return ExitInternalError
	case OverallTimedOut:
		return ExitTimedOut
	case OverallCanceled:
		return ExitCanceled
	default:
		// An unknown overall is a contract violation; surface it loudly.
		return ExitInternalError
	}
}

// Summary tallies per-status counts over a set of statuses. It is used by the
// Report and satisfies total = pass + fail + warn + skip. Unknown statuses are
// ignored and MUST NOT appear in production reports.
type Summary struct {
	Total int `json:"total"`
	Pass  int `json:"pass"`
	Fail  int `json:"fail"`
	Warn  int `json:"warn"`
	Skip  int `json:"skip"`
}

// Summarize counts the given statuses into a Summary. The order of statuses
// does not affect the result; the summary is order-independent.
func Summarize(statuses []Status) Summary {
	s := Summary{Total: len(statuses)}
	for _, st := range statuses {
		switch st {
		case StatusPass:
			s.Pass++
		case StatusFail:
			s.Fail++
		case StatusWarn:
			s.Warn++
		case StatusSkip:
			s.Skip++
		default:
			continue
		}
	}
	return s
}

// AggregateOverall computes the Overall conclusion for a completed run from
// the per-check statuses. The aggregation order is FAIL > WARN > PASS; when
// there are no applicable checks (empty input) the result is SKIP. When all
// checks were skipped, the result is also SKIP (a run that produced no
// actionable result is not a PASS).
//
// AggregateOverall is only for a COMPLETED run. Timeout, cancellation and
// framework faults set Overall directly and MUST NOT be routed through here.
func AggregateOverall(statuses []Status) Overall {
	s := Summarize(statuses)
	if s.Total == 0 {
		return OverallSkip
	}
	switch {
	case s.Fail > 0:
		return OverallFail
	case s.Warn > 0:
		return OverallWarn
	case s.Pass > 0:
		return OverallPass
	default:
		// Total > 0 but no Pass/Fail/Warn: all skipped.
		return OverallSkip
	}
}

// FailedStatuses returns statuses that are FAIL or WARN, deduplicated and
// ordered by severity (FAIL before WARN). It is a helper for surfacing the
// root-cause findings; downstream SKIP results reference the root cause
// rather than re-aggregating risk.
func FailedStatuses(statuses []Status) []Status {
	seen := make(map[Status]struct{})
	var out []Status
	for _, st := range statuses {
		if st == StatusFail || st == StatusWarn {
			if _, ok := seen[st]; ok {
				continue
			}
			seen[st] = struct{}{}
			out = append(out, st)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		return SeverityRank(out[i]) > SeverityRank(out[j])
	})
	return out
}

// AssertExitCode is a test helper that returns an error when the given overall
// does not map to the expected exit code. It keeps the contract table in one
// place and is used by the status/exit-code table tests.
func AssertExitCode(o Overall, want int) error {
	if got := ExitCode(o); got != want {
		return errors.New("exit code mismatch")
	}
	return nil
}
