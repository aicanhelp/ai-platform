/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package backup

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	numZero  = 0
	numOne   = 1
	numTwo   = 2
	numThree = 3

	testFileName = "test.txt"
	testDirName  = "test"
	relativePath = "relative"
)

var (
	testAbsPath   string
	testAbsPath2  string
	backupSaveTo  string
	backupSaveTo2 string
	nonexistPath  string
)

func init() {
	tmpDir := os.TempDir()
	testAbsPath = filepath.Join(tmpDir, "test")
	testAbsPath2 = filepath.Join(tmpDir, "test2")
	backupSaveTo = filepath.Join(tmpDir, "backup")
	backupSaveTo2 = filepath.Join(tmpDir, "backup2")
	nonexistPath = filepath.Join(tmpDir, "nonexist")
}

type mockBackupExecutor struct {
	exec.Executor
	executeCommandCalled           bool
	executeCommandError            error
	executeCommandWithOutputCalled bool
	executeCommandWithOutput       string
	executeCommandWithOutputErr    error
}

func (m *mockBackupExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockBackupExecutor) ExecuteCommand(_ string, _ ...string) error {
	m.executeCommandCalled = true
	return m.executeCommandError
}

func (m *mockBackupExecutor) ExecuteCommandWithOutput(_ string, _ ...string) (string, error) {
	m.executeCommandWithOutputCalled = true
	return m.executeCommandWithOutput, m.executeCommandWithOutputErr
}

func TestBackupPluginName(t *testing.T) {
	plugin := &BackupPlugin{}
	assert.Equal(t, Name, plugin.Name())
}

func TestBackupPluginParam(t *testing.T) {
	plugin := &BackupPlugin{}
	params := plugin.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "backupDirs")
	assert.Contains(t, params, "backupFiles")
	assert.Contains(t, params, "saveTo")
}

func TestBackupPluginParamDefaults(t *testing.T) {
	plugin := &BackupPlugin{}
	params := plugin.Param()
	assert.Equal(t, defaultBackupSaveTo, params["saveTo"].Default)
}

func TestNewBackupPlugin(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, Name, plugin.Name())
}

func TestBackupPluginExecuteEmptyBackup(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=", "backupFiles="}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupPluginExecuteBothEmpty(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=", "backupFiles=", "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupPluginExecuteOnlyFilesEmpty(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=", "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupPluginExecuteOnlyDirsEmpty(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=", "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupPluginExecuteInvalidSaveTo(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + relativePath}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
	assert.Empty(t, result)
}

func TestBackupPluginExecuteInvalidBackupDir(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=" + relativePath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
	assert.Empty(t, result)
}

func TestBackupPluginExecuteInvalidBackupFile(t *testing.T) {
	plugin := &BackupPlugin{}
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + relativePath, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
	assert.Empty(t, result)
}

func TestBackupDirEmpty(t *testing.T) {
	plugin := &BackupPlugin{}
	err := plugin.backupDir("", backupSaveTo)
	assert.NoError(t, err)
}

func TestBackupDirRelativePath(t *testing.T) {
	plugin := &BackupPlugin{}
	err := plugin.backupDir(relativePath, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
}

func TestBackupDirNotExists(t *testing.T) {
	patches := gomonkey.ApplyFunc(utils.IsDir, func(path string) bool {
		return false
	})
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	plugin := &BackupPlugin{}
	err := plugin.backupDir(nonexistPath, backupSaveTo)
	assert.NoError(t, err)
}

func TestBackupDirNotDir(t *testing.T) {
	patches := gomonkey.ApplyFunc(utils.IsDir, func(path string) bool {
		return false
	})
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	plugin := &BackupPlugin{}
	err := plugin.backupDir(nonexistPath, backupSaveTo)
	assert.NoError(t, err)
}

func TestBackupDirMkdirFail(t *testing.T) {
	patches := gomonkey.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	plugin := &BackupPlugin{}
	err := plugin.backupDir(testAbsPath, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
}

func TestBackupDirSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	err := plugin.backupDir(testAbsPath, backupSaveTo)
	assert.NoError(t, err)
	assert.True(t, mockExec.executeCommandWithOutputCalled)
}

func TestBackupDirExecuteCommandFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{
		executeCommandWithOutputErr: errors.New("execute failed"),
	}
	plugin := New(mockExec)
	err := plugin.backupDir(testAbsPath, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execute failed")
}

func TestBackupFileEmpty(t *testing.T) {
	plugin := &BackupPlugin{}
	err := plugin.backupFile("", backupSaveTo)
	assert.NoError(t, err)
}

func TestBackupFileRelativePath(t *testing.T) {
	plugin := &BackupPlugin{}
	err := plugin.backupFile(relativePath, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
}

func TestBackupFileNotExists(t *testing.T) {
	patches := gomonkey.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})
	defer patches.Reset()

	plugin := &BackupPlugin{}
	err := plugin.backupFile(nonexistPath, backupSaveTo)
	assert.NoError(t, err)
}

func TestBackupFileMkdirFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	plugin := &BackupPlugin{}
	err := plugin.backupFile(testAbsPath+"/"+testFileName, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
}

func TestBackupFileSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	err := plugin.backupFile(testAbsPath+"/"+testFileName, backupSaveTo)
	assert.NoError(t, err)
	assert.True(t, mockExec.executeCommandCalled)
}

func TestBackupFileExecuteCommandFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{
		executeCommandError: errors.New("execute failed"),
	}
	plugin := New(mockExec)
	err := plugin.backupFile(testAbsPath+"/"+testFileName, backupSaveTo)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execute failed")
}

func TestBackupDirBackupFileSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupDirBackupFileBackupDirFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir failed")
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestBackupDirBackupFileBackupFileFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{
		executeCommandError: errors.New("execute failed"),
	}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestBackupExecuteSaveToNotExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo2}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupExecuteSaveToExistsNotDir(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=" + testAbsPath + "/" + testFileName, "saveTo=" + backupSaveTo2}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupExecuteMultipleDirs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath + "," + testAbsPath2, "backupFiles=", "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupExecuteMultipleFiles(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=", "backupFiles=" + testAbsPath + "/" + testFileName + "," + testAbsPath2 + "/" + testFileName, "saveTo=" + backupSaveTo}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestBackupExecuteSaveToMkdirFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(utils.IsDir, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return errors.New("mkdir saveTo failed")
	})

	mockExec := &mockBackupExecutor{}
	plugin := New(mockExec)
	commands := []string{Name, "backupDirs=" + testAbsPath, "backupFiles=", "saveTo=" + backupSaveTo2}
	result, err := plugin.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir saveTo failed")
	assert.Empty(t, result)
}
