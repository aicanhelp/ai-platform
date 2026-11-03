/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package version

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPrintVersion(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	GitCommitID = "abc123"
	Version = "v1.0.0"
	Architecture = "amd64"
	BuildTime = "2024-01-01"

	PrintVersion()
}

func TestLogPrintVersion(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	GitCommitID = "def456"
	Version = "v2.0.0"
	Architecture = "arm64"
	BuildTime = "2024-06-15"

	LogPrintVersion()
}

func TestString(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	GitCommitID = "ghi789"
	Version = "v3.0.0"
	Architecture = "386"
	BuildTime = "2024-12-31"

	result := String()
	assert.Equal(t, "v3.0.0 ghi789 386 2024-12-31", result)
}

func TestStringWithDefaultValues(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	GitCommitID = "dev"
	Version = "v1.0.0"
	Architecture = "unknown"
	BuildTime = "unknown"

	result := String()
	assert.Equal(t, "v1.0.0 dev unknown unknown", result)
}

func TestVariables(t *testing.T) {
	assert.Equal(t, "dev", GitCommitID)
	assert.Equal(t, "v1.0.0", Version)
	assert.Equal(t, "unknown", Architecture)
	assert.Equal(t, "unknown", BuildTime)
}

func TestVariablesModification(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	GitCommitID = "test-commit"
	Version = "v0.0.1-test"
	Architecture = "arm"
	BuildTime = "test-time"

	assert.Equal(t, "test-commit", GitCommitID)
	assert.Equal(t, "v0.0.1-test", Version)
	assert.Equal(t, "arm", Architecture)
	assert.Equal(t, "test-time", BuildTime)
}

func TestStringFormat(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	Version = ""
	GitCommitID = ""
	Architecture = ""
	BuildTime = ""

	result := String()
	assert.Equal(t, "   ", result)
}

func TestStringWithDifferentVersions(t *testing.T) {
	originalGitCommitID := GitCommitID
	originalVersion := Version
	originalArchitecture := Architecture
	originalBuildTime := BuildTime
	defer func() {
		GitCommitID = originalGitCommitID
		Version = originalVersion
		Architecture = originalArchitecture
		BuildTime = originalBuildTime
	}()

	testCases := []struct {
		version     string
		gitCommitID string
		arch        string
		buildTime   string
		expected    string
	}{
		{"v1.0.0", "abc123", "amd64", "2024-01-01", "v1.0.0 abc123 amd64 2024-01-01"},
		{"v2.0.0-beta", "def456", "arm64", "2024-02-15", "v2.0.0-beta def456 arm64 2024-02-15"},
		{"v0.0.1-alpha", "ghi789", "386", "", "v0.0.1-alpha ghi789 386 "},
	}

	for _, tc := range testCases {
		Version = tc.version
		GitCommitID = tc.gitCommitID
		Architecture = tc.arch
		BuildTime = tc.buildTime

		result := String()
		assert.Equal(t, tc.expected, result)
	}
}
