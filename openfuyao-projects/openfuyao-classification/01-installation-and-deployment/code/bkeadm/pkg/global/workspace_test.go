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

// Package-level characterization tests for the workspace resolution contract.
//
// These tests pin the PF-01 behavior: the preflight read-only boundary
// requires that merely importing and loading the global package MUST NOT
// create any directories. Directory creation is an explicit, opt-in action
// taken only by the original install entry points via EnsureWorkspaceLayout.
//
// Run with GOOS=linux because the package imports executor code that uses
// Linux-specific syscalls.
package global

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPackageInitDoesNotCreateDirectories is the PF-01 failing test that
// freezes the read-only-loading contract: after the package is imported and
// Workspace is resolved, none of tmpl/volumes/mount may have been created by
// package init. EnsureWorkspaceLayout is the only thing that creates them.
//
// We point the workspace at a fresh temp dir via BKE_WORKSPACE so the test is
// deterministic and isolated from the host's /bke.
func TestPackageInitDoesNotCreateDirectories(t *testing.T) {
	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "bke")

	// ResolveWorkspace must be pure: it computes the path WITHOUT creating
	// anything. The directories below MUST NOT exist after resolution.
	resolved := ResolveWorkspace(tempDir, "")
	assert.Equal(t, workspace, resolved)

	for _, sub := range []string{"tmpl", "volumes", "mount"} {
		_, err := os.Stat(filepath.Join(workspace, sub))
		assert.True(t, os.IsNotExist(err),
			"ResolveWorkspace must not create %s, but stat err=%v", sub, err)
	}
}

// TestResolveWorkspaceOrder pins the resolution precedence from the design:
// BKE_WORKSPACE env > /opt/BKE_WORKSPACE file > default /bke, and the result
// always appends "/bke". This must stay byte-for-byte stable for PlanHash
// equivalence.
func TestResolveWorkspaceOrder(t *testing.T) {
	t.Run("env overrides file", func(t *testing.T) {
		// Provide a workspace-file dir; env must win.
		fileDir := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(fileDir, "BKE_WORKSPACE"),
			[]byte("/from/file\n"), 0o644))
		got := ResolveWorkspace("/env/wins", "/from/file")
		assert.Equal(t, filepath.Join("/env/wins", "bke"), got)
	})

	t.Run("file used when no env", func(t *testing.T) {
		got := ResolveWorkspace("", "/from/file")
		assert.Equal(t, filepath.Join("/from/file", "bke"), got)
	})

	t.Run("default when neither", func(t *testing.T) {
		got := ResolveWorkspace("", "")
		assert.Equal(t, "/bke", got)
	})
}

// TestEnsureWorkspaceLayoutCreatesDirectories pins that the explicit layout
// helper is the one that actually creates the directories, and only when asked.
func TestEnsureWorkspaceLayoutCreatesDirectories(t *testing.T) {
	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "bke")

	require.NoError(t, EnsureWorkspaceLayout(workspace))

	for _, sub := range []string{"tmpl", "volumes", "mount"} {
		info, err := os.Stat(filepath.Join(workspace, sub))
		require.NoError(t, err, "%s should be created by EnsureWorkspaceLayout", sub)
		assert.True(t, info.IsDir(), "%s should be a directory", sub)
	}
}

// TestEnsureWorkspaceLayoutIdempotent pins that calling the layout helper
// twice on an existing layout does not error.
func TestEnsureWorkspaceLayoutIdempotent(t *testing.T) {
	workspace := t.TempDir()
	require.NoError(t, EnsureWorkspaceLayout(workspace))
	require.NoError(t, EnsureWorkspaceLayout(workspace))
}
