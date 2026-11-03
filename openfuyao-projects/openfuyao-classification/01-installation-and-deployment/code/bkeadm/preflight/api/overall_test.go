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
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestExitCodeTable freezes the overall -> exit-code mapping. This is the
// single source of truth the CLI consumes; changing any line is a contract
// change requiring a version bump.
func TestExitCodeTable(t *testing.T) {
	tests := []struct {
		name    string
		overall Overall
		want    int
	}{
		{"pass", OverallPass, 0},
		{"warn", OverallWarn, 2},
		{"skip", OverallSkip, 2},
		{"fail", OverallFail, 3},
		{"input error", OverallInputError, 4},
		{"bundle error", OverallBundleError, 4},
		{"internal error", OverallInternalError, 5},
		{"timed out", OverallTimedOut, 124},
		{"canceled", OverallCanceled, 130},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ExitCode(tt.overall))
			assert.NoError(t, AssertExitCode(tt.overall, tt.want))
		})
	}
}

// TestExitCodeUnknownIsInternal freezes the conservative default for an
// unrecognized overall: it surfaces as INTERNAL_ERROR (5), never 0.
func TestExitCodeUnknownIsInternal(t *testing.T) {
	assert.Equal(t, ExitInternalError, ExitCode(Overall("NOPE")))
}

// TestAggregateOverallOrder freezes the aggregation algorithm:
// FAIL > WARN > PASS; empty input -> SKIP; all-skipped -> SKIP (not PASS).
func TestAggregateOverallOrder(t *testing.T) {
	tests := []struct {
		name     string
		statuses []Status
		want     Overall
	}{
		{"empty is skip", nil, OverallSkip},
		{"empty is skip", []Status{}, OverallSkip},
		{"all pass", []Status{StatusPass, StatusPass}, OverallPass},
		{"single pass", []Status{StatusPass}, OverallPass},
		{"warn with pass", []Status{StatusPass, StatusWarn}, OverallWarn},
		{"fail dominates warn", []Status{StatusWarn, StatusFail}, OverallFail},
		{"fail dominates pass", []Status{StatusPass, StatusFail}, OverallFail},
		{"only skip", []Status{StatusSkip}, OverallSkip},
		{"skip with pass", []Status{StatusSkip, StatusPass}, OverallPass},
		{"skip with warn", []Status{StatusSkip, StatusWarn}, OverallWarn},
		{"all skip is skip", []Status{StatusSkip, StatusSkip}, OverallSkip},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, AggregateOverall(tt.statuses))
		})
	}
}

// TestSummarizeCounts freezes the summary invariant total = pass+fail+warn+skip
// and that the summary is order-independent.
func TestSummarizeCounts(t *testing.T) {
	statuses := []Status{StatusPass, StatusFail, StatusWarn, StatusSkip, StatusPass}
	s := Summarize(statuses)
	assert.Equal(t, 5, s.Total)
	assert.Equal(t, 2, s.Pass)
	assert.Equal(t, 1, s.Fail)
	assert.Equal(t, 1, s.Warn)
	assert.Equal(t, 1, s.Skip)
	assert.Equal(t, s.Total, s.Pass+s.Fail+s.Warn+s.Skip)

	// Order-independent.
	reordered := []Status{StatusSkip, StatusFail, StatusPass, StatusWarn, StatusPass}
	assert.Equal(t, s, Summarize(reordered))

	// Unknown statuses are counted in Total but not in any bucket.
	mixed := []Status{StatusPass, Status("BOGUS")}
	sm := Summarize(mixed)
	assert.Equal(t, 2, sm.Total)
	assert.Equal(t, 1, sm.Pass)
}

// TestFailedStatusesDedupAndOrder freezes the root-cause surfacing helper.
func TestFailedStatusesDedupAndOrder(t *testing.T) {
	out := FailedStatuses([]Status{StatusPass, StatusFail, StatusWarn, StatusFail, StatusWarn})
	// FAIL before WARN, deduplicated.
	assert.Equal(t, []Status{StatusFail, StatusWarn}, out)

	// No failures -> empty.
	assert.Empty(t, FailedStatuses([]Status{StatusPass, StatusSkip}))
}

func TestIssueGates(t *testing.T) {
	assert.False(t, HasStartBlocker(nil))
	assert.False(t, HasStartBlocker([]RunIssue{{Severity: IssueError, Effect: IssueContinue}}))
	assert.True(t, HasStartBlocker([]RunIssue{{Severity: IssueWarn, Effect: IssueBlockStart}}))
	assert.False(t, HasErrorIssue([]RunIssue{{Severity: IssueWarn, Effect: IssueContinue}}))
	assert.True(t, HasErrorIssue([]RunIssue{{Severity: IssueError, Effect: IssueContinue}}))
}

// TestAssertExitCode freezes the test helper contract for both branches.
func TestAssertExitCode(t *testing.T) {
	assert.NoError(t, AssertExitCode(OverallPass, 0))
	assert.NoError(t, AssertExitCode(OverallFail, 3))
	err := AssertExitCode(OverallPass, 3)
	assert.ErrorContains(t, err, "mismatch")
}
