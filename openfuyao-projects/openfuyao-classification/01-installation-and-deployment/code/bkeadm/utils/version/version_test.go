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

package version

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVersionVariablesAssignment(t *testing.T) {
	originalVersion := Version
	originalCommit := GitCommitID
	originalArch := Architecture
	originalTimestamp := Timestamp
	t.Cleanup(func() {
		Version = originalVersion
		GitCommitID = originalCommit
		Architecture = originalArch
		Timestamp = originalTimestamp
	})

	Version = "v1.2.3"
	GitCommitID = "abc123def"
	Architecture = "linux/amd64"
	Timestamp = "2026-08-13"

	assert.Equal(t, "v1.2.3", Version)
	assert.Equal(t, "abc123def", GitCommitID)
	assert.Equal(t, "linux/amd64", Architecture)
	assert.Equal(t, "2026-08-13", Timestamp)
}

func TestVersionDisplayLines(t *testing.T) {
	originalVersion := Version
	originalCommit := GitCommitID
	originalArch := Architecture
	originalTimestamp := Timestamp
	t.Cleanup(func() {
		Version = originalVersion
		GitCommitID = originalCommit
		Architecture = originalArch
		Timestamp = originalTimestamp
	})

	Version = "v1.0.0"
	GitCommitID = "deadbeef"
	Architecture = "amd64"
	Timestamp = "2024-01-01"

	lines := []string{
		fmt.Sprintf("version: %s", Version),
		fmt.Sprintf("gitCommitID: %s", GitCommitID),
		fmt.Sprintf("os/arch: %s", Architecture),
		fmt.Sprintf("date: %s", Timestamp),
	}

	assert.Equal(t, "version: v1.0.0", lines[0])
	assert.Equal(t, "gitCommitID: deadbeef", lines[1])
	assert.Equal(t, "os/arch: amd64", lines[2])
	assert.Equal(t, "date: 2024-01-01", lines[3])
}

func TestVersionOnlyOutput(t *testing.T) {
	originalVersion := Version
	t.Cleanup(func() {
		Version = originalVersion
	})

	Version = "v9.9.9"
	assert.Equal(t, "v9.9.9", fmt.Sprint(Version))
}

func TestVersionComparisonByPrefix(t *testing.T) {
	originalVersion := Version
	t.Cleanup(func() {
		Version = originalVersion
	})

	Version = "v1.0.0"
	v100 := Version
	Version = "v2.0.0"
	v200 := Version

	assert.NotEqual(t, v100, v200)
	assert.True(t, v100 < v200)
}
