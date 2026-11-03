/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package timezone

import (
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"
)

const (
	testIPv4SegmentA = 127
	testIPv4SegmentB = 0
	testIPv4SegmentC = 0
	testIPv4SegmentD = 1
)

const (
	testFileModeReadOnly = 0644
	testFileModeExec     = 0755
)

var testLoopbackIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
)

func TestFindTimezoneFileNotFound(t *testing.T) {
	result, err := findTimezoneFile("Invalid/Timezone")
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestCopyFileSuccess(t *testing.T) {
	src := t.TempDir() + "/src"
	dst := t.TempDir() + "/dst"
	content := []byte("test content")

	err := os.WriteFile(src, content, testFileModeReadOnly)
	assert.NoError(t, err)

	err = copyFile(src, dst)
	assert.NoError(t, err)

	result, err := os.ReadFile(dst)
	assert.NoError(t, err)
	assert.Equal(t, content, result)
}

func TestCopyFileReadError(t *testing.T) {
	dst := t.TempDir() + "/dst"
	err := copyFile("/nonexistent/file", dst)
	assert.Error(t, err)
}

func TestCopyFileWriteError(t *testing.T) {
	src := t.TempDir() + "/src"
	dst := t.TempDir() + "/dst"
	content := []byte("test content")

	err := os.WriteFile(src, content, testFileModeReadOnly)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return assert.AnError
	})

	err = copyFile(src, dst)
	assert.Error(t, err)
}

func TestNTPServerLocalName(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(startLocalNTPServer, func(string) (string, error) {
		return testLoopbackIP.String() + ":123", nil
	})

	result, err := NTPServer("local", testLoopbackIP.String(), false)
	assert.NoError(t, err)
	assert.NotEmpty(t, result)
}

func TestNTPServerExternalSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		return nil
	})

	result, err := NTPServer("pool.ntp.org", testLoopbackIP.String(), false)
	assert.NoError(t, err)
	assert.Equal(t, "pool.ntp.org", result)
}

func TestNTPServerExternalFailOffline(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		return assert.AnError
	})

	result, err := NTPServer("pool.ntp.org", testLoopbackIP.String(), false)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestNTPServerDefaultServer(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		return assert.AnError
	})

	patches.ApplyFunc(startLocalNTPServer, func(string) (string, error) {
		return testLoopbackIP.String() + ":123", nil
	})

	result, err := NTPServer("cn.pool.ntp.org:123", testLoopbackIP.String(), false)
	assert.NoError(t, err)
	assert.NotEmpty(t, result)
}

func TestNTPServerUnreachableWithOnline(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(ntp.Date, func(string) error {
		return assert.AnError
	})

	result, err := NTPServer("unreachable.server", testLoopbackIP.String(), true)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestTryConnectExternalNTPSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		return nil
	})

	result, err := tryConnectExternalNTP("pool.ntp.org", false)
	assert.NoError(t, err)
	assert.Equal(t, "pool.ntp.org", result)
}

func TestTryConnectExternalNTPFailOffline(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		return assert.AnError
	})

	result, err := tryConnectExternalNTP("pool.ntp.org", false)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestTryConnectExternalNTPRetrySuccess(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	callCount := 0
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(ntp.Date, func(string) error {
		callCount++
		if callCount < 2 {
			return assert.AnError
		}
		return nil
	})

	result, err := tryConnectExternalNTP("pool.ntp.org", true)
	assert.NoError(t, err)
	assert.Equal(t, "pool.ntp.org", result)
}

func TestRetryNTPConnectionAllFail(t *testing.T) {
	sleepPatches := gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
	defer sleepPatches.Reset()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(ntp.Date, func(string) error {
		return assert.AnError
	})

	result, err := retryNTPConnection("pool.ntp.org")
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestStartLocalNTPServerSuccess(t *testing.T) {
	originalK8s := global.K8s
	global.K8s = nil
	defer func() { global.K8s = originalK8s }()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.ExecPath, func() (string, error) {
		return "/fake/path", nil
	})

	patches.ApplyFunc(os.Getenv, func(string) string {
		return ""
	})

	patches.ApplyFunc(os.Getpid, func() int {
		return 12345
	})

	patches.ApplyFunc((*exec.Cmd).Start, func(*exec.Cmd) error {
		return nil
	})

	patches.ApplyFunc(server.TryConnectNTPServer, func(string) bool {
		return true
	})

	result, err := startLocalNTPServer(testLoopbackIP.String())
	assert.NoError(t, err)
	assert.Contains(t, result, testLoopbackIP.String())
}

func TestStartLocalNTPServerExecPathError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.ExecPath, func() (string, error) {
		return "", assert.AnError
	})

	result, err := startLocalNTPServer(testLoopbackIP.String())
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestSetTimeZoneAlreadySet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte("Asia/Shanghai"), nil
	})

	err := SetTimeZone()
	assert.NoError(t, err)
}

func TestSetTimeZoneReadTimezoneError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return nil, assert.AnError
	})

	err := SetTimeZone()
	assert.Error(t, err)
}

func TestSetTimeZoneFindTimezoneFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool {
		return false
	})

	patches.ApplyFunc(findTimezoneFile, func(string) (string, error) {
		return "", assert.AnError
	})

	err := SetTimeZone()
	assert.Error(t, err)
}

func TestSetTimeZoneSymlinkSuccess(t *testing.T) {
	tempDir := t.TempDir()
	zoneInfoPath := filepath.Join(tempDir, "Asia/Shanghai")

	err := os.MkdirAll(filepath.Dir(zoneInfoPath), testFileModeExec)
	assert.NoError(t, err)

	err = os.WriteFile(zoneInfoPath, []byte("tzdata"), testFileModeReadOnly)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == timezonePath
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte("America/New_York"), nil
	})

	patches.ApplyFunc(findTimezoneFile, func(zone string) (string, error) {
		return zoneInfoPath, nil
	})

	patches.ApplyFunc(backupTimezoneState, func() error {
		return nil
	})

	patches.ApplyFunc(removeExistingTimezoneFile, func(string) error {
		return nil
	})

	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, assert.AnError
	})

	patches.ApplyFunc(os.Rename, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.Symlink, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	err = SetTimeZone()
	assert.NoError(t, err)
}

func TestSetTimeZoneSymlinkErrorCopySuccess(t *testing.T) {
	copyFileCalled := false

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == timezonePath
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte("America/New_York"), nil
	})

	patches.ApplyFunc(findTimezoneFile, func(zone string) (string, error) {
		return "/usr/share/zoneinfo/Asia/Shanghai", nil
	})

	patches.ApplyFunc(backupTimezoneState, func() error {
		return nil
	})

	patches.ApplyFunc(removeExistingTimezoneFile, func(string) error {
		return nil
	})

	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, nil
	})

	patches.ApplyFunc(os.Rename, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.Symlink, func(target, link string) error {
		return assert.AnError
	})

	patches.ApplyFunc(copyFile, func(src, dst string) error {
		copyFileCalled = true
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(path string, data []byte, perm os.FileMode) error {
		return nil
	})

	err := SetTimeZone()
	assert.NoError(t, err)
	assert.True(t, copyFileCalled, "copyFile should have been called")
}

func TestSetTimeZoneCopyFileError(t *testing.T) {
	tempDir := t.TempDir()
	zoneInfoPath := filepath.Join(tempDir, "Asia/Shanghai")

	err := os.MkdirAll(filepath.Dir(zoneInfoPath), testFileModeExec)
	assert.NoError(t, err)

	err = os.WriteFile(zoneInfoPath, []byte("tzdata"), testFileModeReadOnly)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == timezonePath || path == localtimePath
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte("America/New_York"), nil
	})

	patches.ApplyFunc(findTimezoneFile, func(zone string) (string, error) {
		return zoneInfoPath, nil
	})

	patches.ApplyFunc(backupTimezoneState, func() error {
		return nil
	})

	patches.ApplyFunc(removeExistingTimezoneFile, func(string) error {
		return nil
	})

	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, nil
	})

	patches.ApplyFunc(os.Symlink, func(string, string) error {
		return assert.AnError
	})

	patches.ApplyFunc(os.Rename, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(copyFile, func(string, string) error {
		return assert.AnError
	})

	err = SetTimeZone()
	assert.Error(t, err)
}

func TestSetTimeZoneWriteFileError(t *testing.T) {
	tempDir := t.TempDir()
	zoneInfoPath := filepath.Join(tempDir, "Asia/Shanghai")

	err := os.MkdirAll(filepath.Dir(zoneInfoPath), testFileModeExec)
	assert.NoError(t, err)

	err = os.WriteFile(zoneInfoPath, []byte("tzdata"), testFileModeReadOnly)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(findTimezoneFile, func(zone string) (string, error) {
		return zoneInfoPath, nil
	})

	patches.ApplyFunc(backupTimezoneState, func() error {
		return nil
	})

	patches.ApplyFunc(removeExistingTimezoneFile, func(string) error {
		return nil
	})

	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, assert.AnError
	})

	patches.ApplyFunc(os.Rename, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.Symlink, func(string, string) error {
		return assert.AnError
	})

	patches.ApplyFunc(copyFile, func(string, string) error {
		return nil
	})

	writeFileCalled := false
	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		writeFileCalled = true
		return assert.AnError
	})

	err = SetTimeZone()
	assert.Error(t, err)
	assert.True(t, writeFileCalled)
}

func TestSetTimeZoneRenameError(t *testing.T) {
	tempDir := t.TempDir()
	zoneInfoPath := filepath.Join(tempDir, "Asia/Shanghai")

	err := os.MkdirAll(filepath.Dir(zoneInfoPath), testFileModeExec)
	assert.NoError(t, err)

	err = os.WriteFile(zoneInfoPath, []byte("tzdata"), testFileModeReadOnly)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path == timezonePath
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		return []byte("America/New_York"), nil
	})

	patches.ApplyFunc(findTimezoneFile, func(zone string) (string, error) {
		return zoneInfoPath, nil
	})

	patches.ApplyFunc(backupTimezoneState, func() error {
		return nil
	})

	patches.ApplyFunc(removeExistingTimezoneFile, func(string) error {
		return nil
	})

	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, nil
	})

	patches.ApplyFunc(os.Rename, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.Symlink, func(string, string) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	err = SetTimeZone()
	assert.NoError(t, err)
}

func TestBackupTimezoneStateBackupsLocaltimeFileAndTimezone(t *testing.T) {
	tempDir := t.TempDir()
	originalWorkspace := global.Workspace
	originalLocaltimePath := localtimePath
	originalTimezonePath := timezonePath
	global.Workspace = tempDir
	localtimePath = filepath.Join(tempDir, "localtime")
	timezonePath = filepath.Join(tempDir, "timezone")
	defer func() {
		global.Workspace = originalWorkspace
		localtimePath = originalLocaltimePath
		timezonePath = originalTimezonePath
	}()

	err := os.WriteFile(localtimePath, []byte("old localtime"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(timezonePath, []byte("old timezone"), testFileModeReadOnly)
	assert.NoError(t, err)

	err = backupTimezoneState()
	assert.NoError(t, err)

	localtimeBackup, err := os.ReadFile(localtimeBackupPath())
	assert.NoError(t, err)
	assert.Equal(t, "old localtime", string(localtimeBackup))

	timezoneBackup, err := os.ReadFile(timezoneBackupPath())
	assert.NoError(t, err)
	assert.Equal(t, "old timezone", string(timezoneBackup))

	marker, err := os.ReadFile(timezoneMarkerPath())
	assert.NoError(t, err)
	assert.Contains(t, string(marker), "timezone=Asia/Shanghai")
	assert.Contains(t, string(marker), "localtime_backup=")
	assert.Contains(t, string(marker), "localtime_symlink=")
	assert.Contains(t, string(marker), "timezone_backup=")
}

func TestBackupTimezoneStateBackupsLocaltimeSymlink(t *testing.T) {
	tempDir := t.TempDir()
	originalWorkspace := global.Workspace
	originalLocaltimePath := localtimePath
	originalTimezonePath := timezonePath
	global.Workspace = tempDir
	localtimePath = filepath.Join(tempDir, "localtime")
	timezonePath = filepath.Join(tempDir, "timezone")
	defer func() {
		global.Workspace = originalWorkspace
		localtimePath = originalLocaltimePath
		timezonePath = originalTimezonePath
	}()

	targetPath := filepath.Join(tempDir, "zoneinfo")
	err := os.WriteFile(targetPath, []byte("tzdata"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.Symlink(targetPath, localtimePath)
	if err != nil {
		t.Skipf("symlink is not supported in this environment: %v", err)
	}

	err = backupTimezoneState()
	assert.NoError(t, err)

	symlinkBackup, err := os.ReadFile(localtimeSymlinkPath())
	assert.NoError(t, err)
	assert.Equal(t, targetPath, string(symlinkBackup))
	assert.NoFileExists(t, localtimeBackupPath())
}

func TestRestoreTimeZoneRestoresBackups(t *testing.T) {
	tempDir := t.TempDir()
	originalWorkspace := global.Workspace
	originalLocaltimePath := localtimePath
	originalTimezonePath := timezonePath
	global.Workspace = tempDir
	localtimePath = filepath.Join(tempDir, "localtime")
	timezonePath = filepath.Join(tempDir, "timezone")
	defer func() {
		global.Workspace = originalWorkspace
		localtimePath = originalLocaltimePath
		timezonePath = originalTimezonePath
	}()

	err := os.MkdirAll(bkeadmTimezoneStateDir(), testFileModeExec)
	assert.NoError(t, err)
	err = os.WriteFile(timezoneMarkerPath(), []byte("timezone=Asia/Shanghai\n"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(localtimeBackupPath(), []byte("old localtime"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(timezoneBackupPath(), []byte("old timezone"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(localtimePath, []byte("new localtime"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(timezonePath, []byte("new timezone"), testFileModeReadOnly)
	assert.NoError(t, err)

	RestoreTimeZone()

	localtimeData, err := os.ReadFile(localtimePath)
	assert.NoError(t, err)
	assert.Equal(t, "old localtime", string(localtimeData))

	timezoneData, err := os.ReadFile(timezonePath)
	assert.NoError(t, err)
	assert.Equal(t, "old timezone", string(timezoneData))
}

func TestRestoreTimeZoneRemovesTimezoneWhenNoBackup(t *testing.T) {
	tempDir := t.TempDir()
	originalWorkspace := global.Workspace
	originalLocaltimePath := localtimePath
	originalTimezonePath := timezonePath
	global.Workspace = tempDir
	localtimePath = filepath.Join(tempDir, "localtime")
	timezonePath = filepath.Join(tempDir, "timezone")
	defer func() {
		global.Workspace = originalWorkspace
		localtimePath = originalLocaltimePath
		timezonePath = originalTimezonePath
	}()

	err := os.MkdirAll(bkeadmTimezoneStateDir(), testFileModeExec)
	assert.NoError(t, err)
	err = os.WriteFile(timezoneMarkerPath(), []byte("timezone=Asia/Shanghai\n"), testFileModeReadOnly)
	assert.NoError(t, err)
	err = os.WriteFile(timezonePath, []byte("new timezone"), testFileModeReadOnly)
	assert.NoError(t, err)

	RestoreTimeZone()

	assert.NoFileExists(t, timezonePath)
}
