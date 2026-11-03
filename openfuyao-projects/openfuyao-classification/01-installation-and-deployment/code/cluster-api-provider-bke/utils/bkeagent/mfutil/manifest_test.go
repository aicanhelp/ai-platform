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

package mfutil

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPathForManifestWithDefaultPath(t *testing.T) {
	component := &BKEComponent{
		Name:   "test-component",
		MfPath: "",
	}
	pathForManifest(component)

}

func TestPathForManifestCreatesDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	component := &BKEComponent{
		Name:   "test-component",
		MfPath: tmpDir,
	}
	result := pathForManifest(component)
	expected := filepath.Join(tmpDir, "test-component.yaml")
	assert.Equal(t, expected, result)

	info, err := os.Stat(tmpDir)
	assert.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestPathForHAManifestConf(t *testing.T) {
	tmpDir := t.TempDir()
	component := &BKEHAComponent{
		Name:     "haproxy",
		ConfPath: tmpDir,
		ConfName: "haproxy.cfg",
	}
	result := pathForHAManifestConf(component)
	expected := filepath.Join(tmpDir, "haproxy.cfg")
	assert.Equal(t, filepath.ToSlash(expected), filepath.ToSlash(result))

	info, err := os.Stat(tmpDir)
	assert.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestPathForHAManifestScript(t *testing.T) {
	tmpDir := t.TempDir()
	component := &BKEHAComponent{
		Name:     "keepalived",
		ConfPath: tmpDir,
	}
	result := pathForHAManifestScript(component, "check-master.sh")
	expected := filepath.Join(tmpDir, "check-master.sh")
	assert.Equal(t, filepath.ToSlash(expected), filepath.ToSlash(result))

	info, err := os.Stat(tmpDir)
	assert.NoError(t, err)
	assert.True(t, info.IsDir())
}

func TestPathForHAManifestCreatesDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	component := &BKEHAComponent{
		Name:   "test-component",
		MfPath: tmpDir,
	}
	result := pathForHAManifest(component)
	expected := filepath.Join(tmpDir, "test-component.yaml")
	assert.Equal(t, expected, result)

	info, err := os.Stat(tmpDir)
	assert.NoError(t, err)
	assert.True(t, info.IsDir())
}
