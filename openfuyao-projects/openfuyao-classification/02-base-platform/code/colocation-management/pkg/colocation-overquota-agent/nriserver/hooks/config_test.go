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
package hooks

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReadHookActionConfigFileEmptyPath(t *testing.T) {
	cfg, err := readHookActionConfigFile("")
	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "empty")
}

func TestReadHookActionConfigFileNotExist(t *testing.T) {
	cfg, err := readHookActionConfigFile(filepath.Join(t.TempDir(), "missing.json"))
	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "unable to read")
}

func TestReadHookActionConfigFileInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	assert.NoError(t, os.WriteFile(path, []byte("{not json}"), 0644))

	cfg, err := readHookActionConfigFile(path)
	assert.Error(t, err)
	assert.Nil(t, cfg)
	assert.Contains(t, err.Error(), "error decoding")
}

func TestReadHookActionConfigFileValidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	// the config struct currently has no exported fields; an empty JSON
	// object is the minimal valid document.
	assert.NoError(t, os.WriteFile(path, []byte("{}"), 0644))

	cfg, err := readHookActionConfigFile(path)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
}

func TestValidateHookActionConfigFile(t *testing.T) {
	// validateHookActionConfigFile is currently a no-op that always returns
	// an empty slice of errors.
	errs := validateHookActionConfigFile(&config{})
	assert.Empty(t, errs)
}
