/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package disk

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestNewChecker(t *testing.T) {
	cfg := &config.DiskCheckConfig{}
	log := logger.NewLogger("")
	roles := []string{"master"}
	checker := NewChecker(cfg, log, roles)
	assert.NotNil(t, checker)
	assert.Equal(t, cfg, checker.cfg)
	assert.Equal(t, log, checker.logger)
	assert.Equal(t, roles, checker.roles)
}

func TestShouldCheckForRole(t *testing.T) {
	tests := []struct {
		name      string
		itemRoles []string
		nodeRoles []string
		want      bool
	}{
		{"emptyItemRoles", []string{}, []string{"master"}, true},
		{"matchSingle", []string{"master"}, []string{"master"}, true},
		{"matchMultiple", []string{"master", "worker"}, []string{"worker"}, true},
		{"noMatch", []string{"master"}, []string{"worker"}, false},
		{"emptyNodeRolesNoItemRoles", []string{}, []string{}, true},
		{"emptyNodeRolesWithItemRoles", []string{"master"}, []string{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checker := NewChecker(&config.DiskCheckConfig{}, logger.NewLogger(""), tt.nodeRoles)
			got := checker.shouldCheckForRole(tt.itemRoles)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCheckDiskSpacePathExists(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.DiskCheckConfig{}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})

	item := config.DiskCheckItem{Path: dir, MinFreeGB: 0}
	space := checker.checkDiskSpace(item)

	assert.Equal(t, dir, space.Path)
	assert.Empty(t, space.Error)
	assert.True(t, space.Total > 0)
	assert.True(t, space.Free > 0)
	assert.True(t, space.IsSufficient)
}

func TestCheckDiskSpacePathNotExist(t *testing.T) {
	dir := t.TempDir()
	nonExistPath := filepath.Join(dir, "notexist")
	cfg := &config.DiskCheckConfig{}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})

	item := config.DiskCheckItem{Path: nonExistPath, MinFreeGB: 0}
	space := checker.checkDiskSpace(item)

	assert.Equal(t, dir, space.Path)
	assert.Empty(t, space.Error)
	assert.True(t, space.Total > 0)
}

func TestCheckDiskSpacePathIsFile(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "testfile")
	err := os.WriteFile(filePath, []byte("hello"), 0644)
	assert.NoError(t, err)

	cfg := &config.DiskCheckConfig{}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})

	item := config.DiskCheckItem{Path: filePath, MinFreeGB: 0}
	space := checker.checkDiskSpace(item)

	assert.Equal(t, dir, space.Path)
	assert.Empty(t, space.Error)
	assert.True(t, space.Total > 0)
}

func TestCheckDiskSpaceInsufficient(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.DiskCheckConfig{}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})

	item := config.DiskCheckItem{Path: dir, MinFreeGB: 1024 * 1024}
	space := checker.checkDiskSpace(item)

	assert.False(t, space.IsSufficient)
	assert.Equal(t, item.MinFreeGB*gbBytes, space.MinFree)
}

func TestCheckDiskSpaceRootNotExist(t *testing.T) {
	cfg := &config.DiskCheckConfig{}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})

	root := filepath.VolumeName(os.TempDir()) + string(os.PathSeparator)
	item := config.DiskCheckItem{Path: root, MinFreeGB: 0}
	space := checker.checkDiskSpace(item)

	assert.NotEmpty(t, space.Path)
}

func TestCheckerExecute(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.DiskCheckConfig{
		CheckItems: []config.DiskCheckItem{
			{Path: dir, MinFreeGB: 0, Roles: []string{}},
		},
	}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})
	result, err := checker.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.Spaces, 1)
	assert.Equal(t, 1, result.Summary.TotalChecked)
	assert.Equal(t, 1, result.Summary.SufficientPath)
	assert.Equal(t, 0, result.Summary.InsufficientPath)
}

func TestCheckerExecuteSkipRole(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.DiskCheckConfig{
		CheckItems: []config.DiskCheckItem{
			{Path: dir, MinFreeGB: 0, Roles: []string{"master"}},
		},
	}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{"worker"})
	result, err := checker.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.Spaces, 0)
	assert.Equal(t, 0, result.Summary.TotalChecked)
}

func TestCheckerExecuteMultipleItems(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.DiskCheckConfig{
		CheckItems: []config.DiskCheckItem{
			{Path: dir, MinFreeGB: 0, Roles: []string{}},
			{Path: dir, MinFreeGB: 1024 * 1024, Roles: []string{}},
		},
	}
	checker := NewChecker(cfg, logger.NewLogger(""), []string{})
	result, err := checker.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.Spaces, 2)
	assert.Equal(t, 2, result.Summary.TotalChecked)
	assert.Equal(t, 1, result.Summary.SufficientPath)
	assert.Equal(t, 1, result.Summary.InsufficientPath)
}
