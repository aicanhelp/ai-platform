/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpandEnvSimple(t *testing.T) {
	os.Setenv("ENV_CHECK_SIMPLE_VAR", "myvalue")
	defer os.Unsetenv("ENV_CHECK_SIMPLE_VAR")

	expander := NewPathExpander()
	result := expander.ExpandEnv("$ENV_CHECK_SIMPLE_VAR")

	assert.Equal(t, "myvalue", result)
}

func TestExpandEnvNoVar(t *testing.T) {
	expander := NewPathExpander()
	result := expander.ExpandEnv("/usr/local/bin")

	assert.Equal(t, "/usr/local/bin", result)
}

func TestExpandEnvMultipleVars(t *testing.T) {
	os.Setenv("ENV_CHECK_VAR_A", "aaa")
	os.Setenv("ENV_CHECK_VAR_B", "bbb")
	defer os.Unsetenv("ENV_CHECK_VAR_A")
	defer os.Unsetenv("ENV_CHECK_VAR_B")

	expander := NewPathExpander()
	result := expander.ExpandEnv("$ENV_CHECK_VAR_A/$ENV_CHECK_VAR_B")

	assert.Equal(t, "aaa/bbb", result)
}

func TestExpandEnvEmptyString(t *testing.T) {
	expander := NewPathExpander()
	result := expander.ExpandEnv("")

	assert.Equal(t, "", result)
}

func TestExpandEnvTilde(t *testing.T) {
	expander := NewPathExpander()
	result := expander.ExpandEnv("~/test")

	assert.NotEqual(t, "~/test", result)
	assert.True(t, filepath.IsAbs(result))
	assert.True(t, strings.HasSuffix(result, "test"))
}

func TestExpandEnvUnsetVar(t *testing.T) {
	os.Unsetenv("UNSET_VAR_ENV_CHECK_TEST")

	expander := NewPathExpander()
	result := expander.ExpandEnv("$UNSET_VAR_ENV_CHECK_TEST/path")

	assert.Equal(t, "/path", result)
}
