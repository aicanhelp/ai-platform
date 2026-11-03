/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestGetOutputFilename(t *testing.T) {
	assert := assert.New(t)
	name := getOutputFilename("base-name", "json")
	assert.Equal("base-name.json", name)

	name = getOutputFilename("base-name", "text")
	assert.Equal("", name)
}

func TestSaveResultToFile(t *testing.T) {
	assert := assert.New(t)
	dir := t.TempDir()
	outPath := filepath.Join(dir, "sub", "res.json")

	err := saveResultToFile(outPath, "hello")
	assert.NoError(err)

	data, err := os.ReadFile(outPath)
	assert.NoError(err)
	assert.Equal("hello", string(data))
}

func TestRunFileQueryStatusForResidualPaths(t *testing.T) {
	log := logger.NewLogger("")
	dir := t.TempDir()
	existingPath := filepath.Join(dir, "residual.txt")
	assert.NoError(t, os.WriteFile(existingPath, []byte("data"), config.FileMode))

	result := runFileQuery(&config.AppConfig{Paths: []string{existingPath}}, log)
	assert.Equal(t, "fail", result.Status)
	assert.Contains(t, result.Detail, "residual paths exist")

	missingPath := filepath.Join(dir, "missing.txt")
	result = runFileQuery(&config.AppConfig{Paths: []string{missingPath}}, log)
	assert.Equal(t, "pass", result.Status)
	assert.Contains(t, result.Detail, "residual paths are absent")
}

func TestRunProgramCheckStatusUsesExpectedState(t *testing.T) {
	log := logger.NewLogger("")

	result := runProgramCheck(&config.AppConfig{
		ProgramList: []config.ProgramConfig{{
			Name:        "no-such-program-env-check-test",
			ShouldExist: false,
		}},
	}, log)
	assert.Equal(t, "pass", result.Status)
	assert.Contains(t, result.Detail, "match expected state")

	result = runProgramCheck(&config.AppConfig{
		ProgramList: []config.ProgramConfig{{
			Name:        "no-such-program-env-check-test",
			ShouldExist: true,
		}},
	}, log)
	assert.Equal(t, "fail", result.Status)
	assert.Contains(t, result.Detail, "do not match expected state")
}
