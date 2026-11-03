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

// TestSchemaVersionFrozen freezes the contract: the schema version and
// apiVersion string are part of the frozen contract and must not change
// without a version bump.
func TestSchemaVersionFrozen(t *testing.T) {
	assert.Equal(t, "v1", SchemaVersion)
	assert.Equal(t, "preflight.openfuyao.io/v1", Version())
	assert.Equal(t, "PreflightBundle", PreflightBundleKind)
	assert.Equal(t, "PreflightReport", PreflightReportKind)
}

// TestStatusValid enumerates the four and only four public statuses.
func TestStatusValid(t *testing.T) {
	valid := []Status{StatusPass, StatusFail, StatusWarn, StatusSkip}
	for _, s := range valid {
		assert.True(t, s.Valid(), "%s should be valid", s)
		assert.Equal(t, string(s), s.String())
	}
	assert.False(t, Status("").Valid())
	assert.False(t, Status("pass").Valid()) // case-sensitive
}

// TestSeverityRankOrder freezes the aggregation order: FAIL > WARN > PASS > SKIP.
func TestSeverityRankOrder(t *testing.T) {
	assert.Greater(t, SeverityRank(StatusFail), SeverityRank(StatusWarn))
	assert.Greater(t, SeverityRank(StatusWarn), SeverityRank(StatusPass))
	assert.Greater(t, SeverityRank(StatusPass), SeverityRank(StatusSkip))
	assert.Equal(t, -1, SeverityRank(Status("BOGUS")))
}

// TestNormalizeUnknown freezes the unknown -> status mapping. Unknown MUST
// NEVER normalize to PASS.
func TestNormalizeUnknown(t *testing.T) {
	assert.Equal(t, StatusFail, NormalizeUnknown(UnknownBlock))
	assert.Equal(t, StatusWarn, NormalizeUnknown(UnknownWarn))
	assert.Equal(t, StatusSkip, NormalizeUnknown(UnknownIgnore))
	// Invalid policy is conservatively blocking.
	assert.Equal(t, StatusFail, NormalizeUnknown(UnknownPolicy("BOGUS")))
}

// TestNormalizeFailure freezes the failure-severity -> status mapping.
func TestNormalizeFailure(t *testing.T) {
	assert.Equal(t, StatusFail, NormalizeFailure(SeverityError))
	assert.Equal(t, StatusWarn, NormalizeFailure(SeverityWarning))
	// INFO failures are a contract error but surfaced, never swallowed.
	assert.Equal(t, StatusFail, NormalizeFailure(SeverityInfo))
	assert.Equal(t, StatusFail, NormalizeFailure(FailureSeverity("BOGUS")))
}

// TestUnknownNeverPass is the invariant the design insists on: no path through
// NormalizeUnknown or NormalizeFailure yields PASS.
func TestUnknownNeverPass(t *testing.T) {
	for _, p := range []UnknownPolicy{UnknownBlock, UnknownWarn, UnknownIgnore, "BOGUS"} {
		assert.NotEqual(t, StatusPass, NormalizeUnknown(p))
	}
	for _, s := range []FailureSeverity{SeverityError, SeverityWarning, SeverityInfo, "BOGUS"} {
		assert.NotEqual(t, StatusPass, NormalizeFailure(s))
	}
}
