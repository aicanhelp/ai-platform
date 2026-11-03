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

package resetutil

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

func TestCleanDirWithNonExistentPath(t *testing.T) {
	err := CleanDir("/nonexistent/path")
	assert.NoError(t, err)
}

func TestCleanDirWithFileInsteadOfDirectory(t *testing.T) {
	tmpFile := t.TempDir() + "/testfile"
	err := os.WriteFile(tmpFile, []byte("test"), 0644)
	assert.NoError(t, err)

	err = CleanDir(tmpFile)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not a directory")
}

func TestCleanDirWithEmptyDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	err := CleanDir(tmpDir)
	assert.NoError(t, err)
}

func TestCleanDirWithFiles(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.WriteFile(filepath.Join(tmpDir, "file1.txt"), []byte("content1"), 0644)
	assert.NoError(t, err)
	err = os.WriteFile(filepath.Join(tmpDir, "file2.txt"), []byte("content2"), 0644)
	assert.NoError(t, err)
	err = os.MkdirAll(filepath.Join(tmpDir, "subdir"), 0755)
	assert.NoError(t, err)

	err = CleanDir(tmpDir)
	assert.NoError(t, err)

	entries, err := os.ReadDir(tmpDir)
	assert.NoError(t, err)
	assert.Empty(t, entries)
}

func TestCleanDirWithNestedDirectories(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.MkdirAll(filepath.Join(tmpDir, "level1", "level2", "level3"), 0755)
	assert.NoError(t, err)
	err = os.WriteFile(filepath.Join(tmpDir, "level1", "file.txt"), []byte("content"), 0644)
	assert.NoError(t, err)

	err = CleanDir(tmpDir)
	assert.NoError(t, err)

	entries, err := os.ReadDir(tmpDir)
	assert.NoError(t, err)
	assert.Empty(t, entries)
}

func TestCleanFileWithNonExistentFile(t *testing.T) {
	err := CleanFile("/nonexistent/file.txt")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "does not exist")
}

func TestCleanFileWithDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	err := CleanFile(tmpDir)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "is a directory")
}

func TestCleanFileWithExistingFile(t *testing.T) {
	tmpFile := t.TempDir() + "/testfile"
	err := os.WriteFile(tmpFile, []byte("test content"), 0644)
	assert.NoError(t, err)

	err = CleanFile(tmpFile)
	assert.NoError(t, err)

	_, err = os.Stat(tmpFile)
	assert.True(t, os.IsNotExist(err))
}

func TestShouldUnmount(t *testing.T) {
	tests := []struct {
		name       string
		mountPoint string
		target     string
		expected   bool
	}{
		{
			name:       "exact match returns false",
			mountPoint: "/var/lib/kubelet",
			target:     "/var/lib/kubelet",
			expected:   false,
		},
		{
			name:       "subdirectory returns true",
			mountPoint: "/var/lib/kubelet/pods",
			target:     "/var/lib/kubelet",
			expected:   true,
		},
		{
			name:       "different path returns false",
			mountPoint: "/var/lib/other",
			target:     "/var/lib/kubelet",
			expected:   false,
		},
		{
			name:       "parent path returns false",
			mountPoint: "/var/lib",
			target:     "/var/lib/kubelet",
			expected:   false,
		},
		{
			name:       "with trailing slash subdirectory",
			mountPoint: "/var/lib/kubelet/pods/volumes",
			target:     "/var/lib/kubelet/",
			expected:   true,
		},
		{
			name:       "exact match with trailing slash target",
			mountPoint: "/var/lib/kubelet/",
			target:     "/var/lib/kubelet/",
			expected:   false,
		},
		{
			name:       "target without slash matches subdirectory",
			mountPoint: "/var/lib/kubelet",
			target:     "/var/lib/kubelet/",
			expected:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shouldUnmount(tt.mountPoint, tt.target)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSecureUnmount(t *testing.T) {
	err := secureUnmount("/some/path")
	assert.Error(t, err)
}

func TestUnmountKubeletDirectory(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readMountPoints, func() ([]string, error) {
		return []string{
			"/var/lib/kubelet",
			"/var/lib/kubelet/pods",
			"/var/lib/kubelet/plugins",
			"/var/log/pods",
		}, nil
	})

	err := UnmountKubeletDirectory("/var/lib/kubelet")
	assert.NoError(t, err)
}

func TestUnmountKubeletDirectoryWithReadMountPointsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readMountPoints, func() ([]string, error) {
		return nil, os.ErrPermission
	})

	err := UnmountKubeletDirectory("/var/lib/kubelet")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to access mount information")
}

func TestUnmountKubeletDirectoryWithNoMatchingMounts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readMountPoints, func() ([]string, error) {
		return []string{
			"/var/lib/other",
			"/var/log/messages",
		}, nil
	})

	err := UnmountKubeletDirectory("/var/lib/kubelet")
	assert.NoError(t, err)
}

func TestReadMountPoints(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		content := `proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
/dev/sda1 / ext4 rw,relatime 0 0
/dev/sda2 /var/lib/kubelet ext4 rw,relatime 0 0
/dev/sda3 /var/lib/kubelet/pods ext4 rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev,noexec,relatime,mode=755 0 0
`
		return []byte(content), nil
	})

	mounts, err := readMountPoints()
	assert.NoError(t, err)
	assert.NotEmpty(t, mounts)
	assert.Contains(t, mounts, "/proc")
	assert.Contains(t, mounts, "/sys")
	assert.Contains(t, mounts, "/var/lib/kubelet")
	assert.Contains(t, mounts, "/var/lib/kubelet/pods")
}

func TestReadMountPointsWithEmptyContent(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte(""), nil
	})

	mounts, err := readMountPoints()
	assert.NoError(t, err)
	assert.Empty(t, mounts)
}

func TestReadMountPointsWithError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return nil, os.ErrPermission
	})

	mounts, err := readMountPoints()
	assert.Error(t, err)
	assert.Nil(t, mounts)
}

func TestLeastFieldNums(t *testing.T) {
	assert.Equal(t, 2, LeastFieldNums)
}
