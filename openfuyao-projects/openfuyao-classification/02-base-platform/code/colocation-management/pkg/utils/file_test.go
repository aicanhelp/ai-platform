/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
* rubik licensed under the Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*     http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
* PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: Xiang Li
* Create: 2021-04-17
* Description: filepath related common functions
*
* Original file: https://gitee.com/openeuler/rubik/blob/master/pkg/common/util/file_test.go
 */

// Package utils is common utilitization
package utils

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/common"
)

// TestIsDir is IsDir function test
func TestIsDir(t *testing.T) {
	err := os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)
	defer os.RemoveAll(common.TmpTestDir)
	directory, err := ioutil.TempDir(common.TmpTestDir, t.Name())
	assert.NoError(t, err)

	filePath, err := ioutil.TempFile(directory, t.Name())
	assert.NoError(t, err)

	type args struct {
		path string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "TC1-directory is exist",
			args: args{path: directory},
			want: true,
		},
		{
			name: "TC2-directory is not exist",
			args: args{path: "/directory/is/not/exist"},
			want: false,
		},
		{
			name: "TC3-test file path",
			args: args{path: filePath.Name()},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDir(tt.args.path); got != tt.want {
				t.Errorf("IsDirectory() = %v, want %v", got, tt.want)
			}
		})
	}
	err = filePath.Close()
	assert.NoError(t, err)
}

// TestPathIsExist is PathExist function test
func TestPathIsExist(t *testing.T) {
	err := os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)
	defer os.RemoveAll(common.TmpTestDir)
	filePath, err := ioutil.TempDir(common.TmpTestDir, "file_exist")
	assert.NoError(t, err)

	type args struct {
		path string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "TC1-path is exist",
			args: args{path: filePath},
			want: true,
		},
		{
			name: "TC2-path is not exist",
			args: args{path: "/path/is/not/exist"},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PathExist(tt.args.path); got != tt.want {
				t.Errorf("PathExist() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestReadSmallFile is test for read file
func TestReadSmallFile(t *testing.T) {
	err := os.RemoveAll(common.TmpTestDir)
	assert.NoError(t, err)
	err = os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)
	defer os.RemoveAll(common.TmpTestDir)
	filePath, err := ioutil.TempDir(common.TmpTestDir, "read_file")
	assert.NoError(t, err)

	// case1: ok
	err = ioutil.WriteFile(filepath.Join(filePath, "ok"), []byte{}, common.DefaultFileMode)
	assert.NoError(t, err)
	_, err = ReadSmallFile(filepath.Join(filePath, "ok"))
	assert.NoError(t, err)

	// case2: too big
	size := 20000000
	big := make([]byte, size)
	err = ioutil.WriteFile(filepath.Join(filePath, "big"), big, common.DefaultFileMode)
	assert.NoError(t, err)
	_, err = ReadSmallFile(filepath.Join(filePath, "big"))
	assert.Error(t, err)

	// case3: file not exist
	_, err = ReadSmallFile(filepath.Join(filePath, "missing"))
	assert.Error(t, err)
}

// TestCreateLockFile is CreateLockFile function test
func TestCreateLockFile(t *testing.T) {
	lockFile := filepath.Join(common.TmpTestDir, "rubik.lock")
	err := os.RemoveAll(lockFile)
	assert.NoError(t, err)

	lock, err := LockFile(lockFile)
	assert.NoError(t, err)
	err = UnlockFile(lock)
	assert.NoError(t, err)
	assert.NoError(t, lock.Close())
	assert.NoError(t, os.Remove(lockFile))
}

// TestLockFail is CreateLockFile fail test
func TestLockFail(t *testing.T) {
	lockFile := filepath.Join(common.TmpTestDir, "rubik.lock")
	err := os.RemoveAll(common.TmpTestDir)
	assert.NoError(t, err)
	err = os.MkdirAll(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)

	lockFile = filepath.Join(common.TmpTestDir, "rubik-lock")
	_, err = os.OpenFile(lockFile, os.O_CREATE|os.O_WRONLY, 0700)
	assert.NoError(t, err)
	_, err = LockFile(filepath.Join(common.TmpTestDir, "rubik-lock", "rubik.lock"))
	assert.Equal(t, true, err != nil)
	err = os.RemoveAll(filepath.Join(common.TmpTestDir, "rubik-lock"))
	assert.NoError(t, err)

	err = os.MkdirAll(lockFile, common.DefaultDirMode)
	assert.NoError(t, err)
	_, err = LockFile(lockFile)
	assert.Equal(t, true, err != nil)
	err = os.RemoveAll(lockFile)
	assert.NoError(t, err)

	_, err = LockFile(lockFile)
	assert.NoError(t, err)
	_, err = LockFile(lockFile)
	assert.Equal(t, true, err != nil)
	err = os.RemoveAll(lockFile)
	assert.NoError(t, err)
}

// TestReadFile tests ReadFile
func TestReadFile(t *testing.T) {
	err := os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	if err != nil && !os.IsExist(err) {
		assert.NoError(t, err)
	}
	defer os.RemoveAll(common.TmpTestDir)
	type args struct {
		path string
	}
	tests := []struct {
		name    string
		args    args
		pre     func(t *testing.T)
		post    func(t *testing.T)
		want    []byte
		wantErr bool
	}{
		{
			name: "TC1-path is dir",
			args: args{
				path: common.TmpTestDir,
			},
			pre: func(t *testing.T) {
				_, err := ioutil.TempDir(common.TmpTestDir, "TC1")
				assert.NoError(t, err)
			},
			post: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(common.TmpTestDir))
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pre != nil {
				tt.pre(t)
			}
			got, err := ReadFile(tt.args.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReadFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ReadFile() = %v, want %v", got, tt.want)
			}
			if tt.post != nil {
				tt.post(t)
			}

		})
	}
}

// TestWriteFile tests WriteFile
func TestWriteFile(t *testing.T) {
	err := os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)
	defer os.RemoveAll(common.TmpTestDir)
	var filePath = filepath.Join(common.TmpTestDir, "cpu", "kubepods", "PodXXX")
	type args struct {
		path    string
		content string
	}
	tests := []struct {
		name    string
		args    args
		pre     func(t *testing.T)
		post    func(t *testing.T)
		wantErr bool
	}{
		{
			name: "TC1-path is dir",
			args: args{
				path: common.TmpTestDir,
			},
			pre: func(t *testing.T) {
				_, err := ioutil.TempDir(common.TmpTestDir, "TC1")
				assert.NoError(t, err)
			},
			post: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(common.TmpTestDir))
			},
			wantErr: true,
		},
		{
			name: "TC2-create dir & write file",
			args: args{
				path:    filePath,
				content: "1",
			},
			pre: func(t *testing.T) {
				assert.NoError(t, os.Mkdir(common.TmpTestDir, common.DefaultDirMode))
			},
			post: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(common.TmpTestDir))
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pre != nil {
				tt.pre(t)
			}
			if err := WriteFile(tt.args.path, tt.args.content); (err != nil) != tt.wantErr {
				t.Errorf("WriteFile() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.post != nil {
				tt.post(t)
			}
		})
	}
}

// TestWriteFileMkdirError covers the MkdirAll error branch of WriteFile that
// occurs when the parent directory path cannot be created (because a path
// component is a regular file rather than a directory).
func TestWriteFileMkdirError(t *testing.T) {
	baseDir := t.TempDir()
	// Create a regular file that will be used as a path prefix, so MkdirAll on
	// a sub-path beneath it fails with "not a directory".
	blocker := filepath.Join(baseDir, "blocker")
	assert.NoError(t, ioutil.WriteFile(blocker, []byte("x"), common.DefaultFileMode))

	// Target a path whose parent dir traversal hits the regular file above.
	target := filepath.Join(blocker, "sub", "file")
	err := WriteFile(target, "data")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error creating dir")
}

// TestAppendFileOpenError covers the os.OpenFile error branch of AppendFile.
// The file exists (PathExist true, IsDir false) but cannot be opened for
// writing because it is read-only.
func TestAppendFileOpenError(t *testing.T) {
	// Skip when running as root, since root bypasses file permission checks.
	if os.Geteuid() == 0 {
		t.Skip("AppendFile open-error test is not meaningful when running as root")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "readonly")
	assert.NoError(t, ioutil.WriteFile(target, []byte("x"), common.DefaultFileMode))
	// Strip all write permissions so OpenFile(O_WRONLY) fails with permission denied.
	assert.NoError(t, os.Chmod(target, 0400))

	err := AppendFile(target, "data")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error opening file")
}

// TestFileSizeLstatError covers the os.Lstat error branch of FileSize. It is
// only reachable via a TOCTOU race (PathExist true, then the path is removed
// before Lstat), which we approximate here by removing the file between the
// PathExist check and the Lstat call is not directly possible. Instead we
// confirm FileSize behaves on a normal file and on a missing path.
func TestFileSizeBranches(t *testing.T) {
	dir := t.TempDir()
	// Existing file: returns size, no error.
	f := filepath.Join(dir, "sized")
	assert.NoError(t, ioutil.WriteFile(f, []byte("hello"), common.DefaultFileMode))
	size, err := FileSize(f)
	assert.NoError(t, err)
	assert.Equal(t, int64(5), size)

	// Missing path: PathExist false -> error.
	_, err = FileSize(filepath.Join(dir, "missing"))
	assert.Error(t, err)
}

// TestAppendFile tests AppendFile
func TestAppendFile(t *testing.T) {
	err := os.Mkdir(common.TmpTestDir, common.DefaultDirMode)
	assert.NoError(t, err)
	defer os.RemoveAll(common.TmpTestDir)
	var (
		dirPath  = filepath.Join(common.TmpTestDir, "cpu", "kubepods", "PodXXX")
		filePath = filepath.Join(dirPath, "cpu.cfs_quota_us")
	)
	type args struct {
		path    string
		content string
	}
	tests := []struct {
		name    string
		args    args
		pre     func(t *testing.T)
		post    func(t *testing.T)
		wantErr bool
	}{
		{
			name: "TC1-path is dir",
			args: args{
				path: common.TmpTestDir,
			},
			pre: func(t *testing.T) {
				_, err := ioutil.TempDir(common.TmpTestDir, "TC1")
				assert.NoError(t, err)
			},
			post: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(filepath.Join(common.TmpTestDir, "TC1")))
			},
			wantErr: true,
		},
		{
			name: "TC2-empty path",
			args: args{
				path: dirPath,
			},
			pre: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(common.TmpTestDir))
			},
			wantErr: true,
		},
		{
			name: "TC3-write file success",
			args: args{
				path:    filePath,
				content: "1",
			},
			pre: func(t *testing.T) {
				assert.NoError(t, os.MkdirAll(dirPath, common.DefaultDirMode))
				assert.NoError(t, ioutil.WriteFile(filePath, []byte(""), common.DefaultFileMode))
			},
			post: func(t *testing.T) {
				assert.NoError(t, os.RemoveAll(common.TmpTestDir))
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pre != nil {
				tt.pre(t)
			}
			err := AppendFile(tt.args.path, tt.args.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("AppendFile() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				fmt.Printf("error: %v\n", err)
			}
			if tt.post != nil {
				tt.post(t)
			}
		})
	}
}
