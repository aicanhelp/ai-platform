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

package kubelet

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

func TestResolveDeliveredFilePath(t *testing.T) {
	t.Run("absolute path", func(t *testing.T) {
		path, err := resolveDeliveredFilePath("/etc/kubernetes/extra.conf", "/etc/kubernetes")
		require.NoError(t, err)
		assert.Equal(t, "/etc/kubernetes/extra.conf", path)
	})

	t.Run("relative path joins base path", func(t *testing.T) {
		path, err := resolveDeliveredFilePath("extra/kubelet.env", "/etc/kubernetes")
		require.NoError(t, err)
		assert.Equal(t, filepath.Join("/etc/kubernetes", "extra/kubelet.env"), path)
	})

	t.Run("reject path traversal", func(t *testing.T) {
		_, err := resolveDeliveredFilePath("../secret", "/etc/kubernetes")
		assert.Error(t, err)
	})
}

func TestParseFilePermissions(t *testing.T) {
	t.Run("default permissions", func(t *testing.T) {
		mode, err := parseFilePermissions("")
		require.NoError(t, err)
		assert.Equal(t, os.FileMode(RwRR), mode)
	})

	t.Run("custom permissions", func(t *testing.T) {
		mode, err := parseFilePermissions("0755")
		require.NoError(t, err)
		assert.Equal(t, os.FileMode(0755), mode)
	})

	t.Run("invalid permissions", func(t *testing.T) {
		_, err := parseFilePermissions("invalid")
		assert.Error(t, err)
	})
}

func TestProcessKubeletFiles(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "subdir", "kubelet.env")

	kp := &kubeletPlugin{exec: &testExecutor{}}
	err := kp.processKubeletFiles([]confv1beta1.FileSpec{
		{
			Path:        targetFile,
			Content:     "KUBELET_EXTRA_ARGS=--foo=bar",
			Permissions: "0644",
		},
	}, map[string]string{})
	require.NoError(t, err)

	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "KUBELET_EXTRA_ARGS=--foo=bar", string(content))

	info, err := os.Stat(targetFile)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0644), info.Mode().Perm())
}

func TestProcessKubeletFilesWithRelativePathAndOwner(t *testing.T) {
	tempDir := t.TempDir()
	executor := &testExecutor{}

	kp := &kubeletPlugin{exec: executor}
	err := kp.processKubeletFiles([]confv1beta1.FileSpec{
		{
			Path:        "relative.conf",
			Content:     "hello",
			Permissions: "0600",
			Owner:       "root:root",
		},
	}, map[string]string{
		"fileBasePath": tempDir,
	})
	require.NoError(t, err)

	targetFile := filepath.Join(tempDir, "relative.conf")
	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "hello", string(content))
	assert.Contains(t, executor.execCalls, "chown root:root "+targetFile)
}

func TestProcessKubeletFilesWithVariableSubstitution(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "substituted.conf")

	kp := &kubeletPlugin{exec: &testExecutor{
		outputMap: map[string]string{
			"echo host": "node-1",
		},
	}}
	err := kp.processKubeletFiles([]confv1beta1.FileSpec{
		{
			Path:    targetFile,
			Content: "hostname=${HOSTNAME}",
		},
	}, map[string]string{
		"enableVariableSubstitution": "true",
		"HOSTNAME":                   "node-1",
	})
	require.NoError(t, err)

	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "hostname=node-1", string(content))
}

func TestProcessKubeletCommands(t *testing.T) {
	executor := &testExecutor{}
	kp := &kubeletPlugin{exec: executor}

	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{Command: "echo delivered"},
	}, map[string]string{})
	require.NoError(t, err)
	assert.Contains(t, executor.execCalls, "sh -c echo delivered")
}

func TestProcessKubeletCommandsWithArgs(t *testing.T) {
	executor := &testExecutor{}
	kp := &kubeletPlugin{exec: executor}

	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{Command: "echo", Args: []string{"hello", "world"}},
	}, map[string]string{})
	require.NoError(t, err)
	assert.Contains(t, executor.execCalls, "echo hello world")
}

func TestProcessKubeletCommandsWithWorkingDir(t *testing.T) {
	tempDir := t.TempDir()
	kp := &kubeletPlugin{exec: &testExecutor{}}

	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{
			Command:    "pwd",
			WorkingDir: tempDir,
		},
	}, map[string]string{})
	require.NoError(t, err)
}

func TestProcessKubeletCommandsMultiline(t *testing.T) {
	executor := &testExecutor{}
	kp := &kubeletPlugin{exec: executor}
	script := "echo first\necho second"

	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{
			Command: "/bin/sh",
			Args: []string{
				"-c",
				script,
			},
		},
	}, map[string]string{})
	require.NoError(t, err)
	require.Len(t, executor.execCalls, 1)
	assert.Equal(t, "/bin/sh -c "+script, executor.execCalls[0])
}

func TestProcessKubeletCommandsMultilineWithWorkingDirAndArgs(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "multiline.log")

	kp := &kubeletPlugin{exec: &testExecutor{}}
	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{
			Command: "/bin/sh",
			Args: []string{
				"-c",
				fmt.Sprintf("echo first >> %q\nif [ -f %q ]; then\n  echo second >> %q\nfi", targetFile, targetFile, targetFile),
			},
			WorkingDir: tempDir,
		},
	}, map[string]string{})
	require.NoError(t, err)

	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "first\nsecond\n", string(content))
}

func TestDeliveredCommandInvocation(t *testing.T) {
	command, args := deliveredCommandInvocation("echo delivered", nil)
	assert.Equal(t, "sh", command)
	assert.Equal(t, []string{"-c", "echo delivered"}, args)

	command, args = deliveredCommandInvocation("/bin/bash", []string{"-c", "echo hello"})
	assert.Equal(t, "/bin/bash", command)
	assert.Equal(t, []string{"-c", "echo hello"}, args)
}

func TestProcessKubeletCommandsFailure(t *testing.T) {
	kp := &kubeletPlugin{exec: &testExecutor{
		shouldFail:  true,
		failCommand: "sh",
	}}

	err := kp.processKubeletCommands([]confv1beta1.CommandSpec{
		{Command: "exit 1"},
	}, map[string]string{})
	assert.Error(t, err)
}
