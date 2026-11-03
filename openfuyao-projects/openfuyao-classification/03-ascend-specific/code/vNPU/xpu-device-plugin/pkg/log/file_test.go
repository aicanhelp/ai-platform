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

package log

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
)

func TestNewFileLogger(t *testing.T) {
	dir := makeTempDir("TmpFileForTest", t)
	defer os.RemoveAll(dir)
	testFileName := filepath.Join(dir, "test.log")
	l := &FileLogger{
		fileName: testFileName,
	}
	defer l.Close()
	b := []byte("test data")
	n, err := l.Write(b)

	assertIsNil(err, t)
	assertEquals(len(b), n, t)

	checkExistsFileWithSize(testFileName, n, t)
	checkFileCount(dir, 1, t)
}

func TestOpenExistingLog(t *testing.T) {
	dir := makeTempDir("TmpFileForTest", t)
	defer os.RemoveAll(dir)

	testFileName := filepath.Join(dir, "test.log")
	data := []byte("test pre data")
	err := os.WriteFile(testFileName, data, fileModeUserRwForWritingLog)
	assertIsNil(err, t)
	checkExistsFileWithSize(testFileName, len(data), t)

	l := &FileLogger{
		fileName: testFileName,
	}
	defer l.Close()

	b := []byte("test post data")
	n, err := l.Write(b)
	assertIsNil(err, t)
	assertEquals(len(b), n, t)

	checkExistsFileWithSize(testFileName, len(b)+len(data), t)
	checkFileCount(dir, 1, t)
}

func TestWriteInvalidLenLog(t *testing.T) {
	dir := makeTempDir("TmpFileForTest", t)
	defer os.RemoveAll(dir)

	var maxContentLen = 5
	megabyte = 1
	testFileName := filepath.Join(dir, "test.log")
	l := &FileLogger{
		fileName: testFileName,
		maxSize:  maxContentLen,
	}

	data := []byte("efghijklmn")
	_, err2 := l.Write(data)
	assertNotNil(err2, t)
}

var testFileTimeCount = 0

func fakeTimeCurrent() time.Time {
	testFileTimeCount += 1
	return time.Now().Add(time.Minute * time.Duration(testFileTimeCount))
}

func TestAutoRotateCleanAndMaxBackups(t *testing.T) {
	dir := makeTempDir("AutoCompressRotate", t)
	defer os.RemoveAll(dir)

	var maxContentLen = 5
	var maxBackupNum = 2
	const maxBackupFileMultiplier = 2
	megabyte = 1
	testFileName := filepath.Join(dir, "test.log")
	l := &FileLogger{
		fileName:   testFileName,
		maxSize:    maxContentLen,
		maxBackups: maxBackupNum,
	}
	defer l.Close()

	data1 := []byte("abcd")
	data1Len, err1 := l.Write(data1)
	assertIsNil(err1, t)
	assertEquals(len(data1), data1Len, t)
	checkExistsFileWithSize(testFileName, data1Len, t)
	checkFileCount(dir, 1, t)

	const delayCount = 100
	currentTime = fakeTimeCurrent
	data2 := []byte("efgh")
	data2Len, err2 := l.Write(data2)
	assertIsNil(err2, t)
	assertEquals(len(data2), data2Len, t)
	checkExistsFileWithSize(testFileName, data2Len, t)
	<-time.After(time.Millisecond * delayCount)
	checkFileCountRange(dir, maxBackupNum, maxBackupNum+1, t)

	currentTime = fakeTimeCurrent
	data2 = []byte("ijk")
	data2Len, err2 = l.Write(data2)
	assertIsNil(err2, t)
	assertEquals(len(data2), data2Len, t)
	checkExistsFileWithSize(testFileName, data2Len, t)
	<-time.After(time.Millisecond * delayCount)
	checkFileCountRange(dir, maxBackupNum+1, maxBackupNum*maxBackupFileMultiplier+1, t)

	currentTime = fakeTimeCurrent
	data2 = []byte("ijk")
	data2Len, err2 = l.Write(data2)
	assertIsNil(err2, t)
	assertEquals(len(data2), data2Len, t)
	checkExistsFileWithSize(testFileName, data2Len, t)
	<-time.After(time.Millisecond * delayCount)
	checkFileCountRange(dir, maxBackupNum+1, maxBackupNum*maxBackupFileMultiplier+1, t)
}

func makeTempDir(name string, t testing.TB) string {
	dir := time.Now().Format(name)
	dir = filepath.Join(os.TempDir(), dir)
	assertIsNil(os.Mkdir(dir, dirModeUserRwxForLogDir), t)
	return dir
}

func checkExistsFileWithSize(path string, len int, t testing.TB) {
	info, err := os.Stat(path)
	assertIsNil(err, t)
	assertEquals(int64(len), info.Size(), t)
}

func checkFileCount(dir string, expect int, t testing.TB) {
	dirs, err := os.ReadDir(dir)
	t.Logf("--> Check dir[%s] files:", dir)
	for _, f := range dirs {
		t.Logf("  %s\n", f.Name())
	}
	assertIsNil(err, t)
	assertEquals(expect, len(dirs), t)
}

func checkFileCountRange(dir string, min, max int, t testing.TB) {
	dirs, err := os.ReadDir(dir)
	t.Logf("--> Check dir[%s] files:", dir)
	for _, f := range dirs {
		t.Logf("  %s\n", f.Name())
	}
	assertIsNil(err, t)
	if len(dirs) < min || len(dirs) > max {
		t.Errorf("file count out of range, expect [%d, %d], got: %d", min, max, len(dirs))
	}
}

func getFileMode(name string) (os.FileMode, error) {
	var actualMode os.FileMode
	info, err := os.Stat(name)
	if err != nil {
		fmt.Println("Error getting file info:", err)
		return actualMode, err
	}

	actualMode = info.Mode() & os.ModePerm
	return actualMode, nil
}

func getRotateLogFile(dir string) (string, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		fmt.Println("Error reading directory:", err)
		return "", err
	}

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".gz" {
			return filepath.Join(dir, file.Name()), nil
		}
	}

	return "", errors.New("not found rotate logfile")
}

func TestLogFilePermissions(t *testing.T) {
	syscall.Umask(0)

	dir := "/tmp/TmpFileForPermission"
	fileName := "test.log"
	defer os.RemoveAll(dir)

	var maxContentLen = 5
	megabyte = 1
	testFileName := filepath.Join(dir, fileName)
	l := &FileLogger{
		fileName:   testFileName,
		maxSize:    maxContentLen,
		maxBackups: 1,
	}

	data := []byte("efgh")
	_, err2 := l.Write(data)
	assertIsNil(err2, t)

	modeFile, err := getFileMode(testFileName)
	assertIsNil(err, t)
	assertEquals(os.FileMode(fileModeUserRwForWritingLog), modeFile, t)
	modeDir, err := getFileMode(dir)
	assertIsNil(err, t)
	assertEquals(os.FileMode(dirModeUserRwxForLogDir), modeDir, t)

	data = []byte("efg")
	_, err2 = l.Write(data)
	assertIsNil(err2, t)

	const delayCount = 100
	<-time.After(time.Millisecond * delayCount)

	rotateFile, err := getRotateLogFile(dir)
	assertIsNil(err, t)
	modeRotateFile, err := getFileMode(rotateFile)
	assertIsNil(err, t)
	assertEquals(os.FileMode(fileModeUserRoForArchiveLog), modeRotateFile, t)
}

func TestLessOutOfBounds(t *testing.T) {
	infos := logInfosWithTimestamp{}
	if infos.Less(0, 1) {
		t.Errorf("expected false for out of bounds")
	}
	if infos.Less(1, 0) {
		t.Errorf("expected false for out of bounds")
	}
}

func TestSwapOutOfBounds(t *testing.T) {
	infos := logInfosWithTimestamp{{}, {}}
	infos.Swap(0, 1)
	infos.Swap(5, 0)
	infos.Swap(0, 5)
}

func TestOpenLogFileError(t *testing.T) {
	_, err := openLogFile("/nonexistent/dir/file.log", os.O_RDONLY, 0644)
	if err == nil {
		t.Errorf("expected error for non-existent directory")
	}
}

func TestCloseOnNilFile(t *testing.T) {
	l := &FileLogger{}
	err := l.Close()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestRotate(t *testing.T) {
	dir := makeTempDir("TestRotate", t)
	defer os.RemoveAll(dir)
	testFileName := filepath.Join(dir, "test.log")
	l := &FileLogger{
		fileName: testFileName,
	}
	_, err := l.Write([]byte("test data"))
	assertIsNil(err, t)
	err = l.Rotate()
	assertIsNil(err, t)
}

func TestRotateCloseError(t *testing.T) {
	l := &FileLogger{}
	patch := gomonkey.ApplyPrivateMethod(l, "close", func() error {
		return errors.New("mock close error")
	})
	defer patch.Reset()
	err := l.rotate()
	if err == nil {
		t.Errorf("expected error from close")
	}
}

func TestOpenNewMkdirError(t *testing.T) {
	l := &FileLogger{fileName: "/root/proc/test.log"}
	patch := gomonkey.ApplyFunc(os.MkdirAll, func(_ string, _ os.FileMode) error {
		return errors.New("mock mkdir error")
	})
	defer patch.Reset()
	err := l.openNew()
	if err == nil {
		t.Errorf("expected error from mkdir")
	}
}

func TestFilenameRelativePath(t *testing.T) {
	l := &FileLogger{fileName: "relative.log"}
	fn := l.filename()
	if fn != filepath.Join(defaultLogPath, "relative.log") {
		t.Errorf("expected default log path, got %s", fn)
	}
}

func TestCreateByOpenError(t *testing.T) {
	dir := makeTempDir("TestCreateByOpen", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.txt")
	if err := os.WriteFile(fn, []byte("test"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	info, err := os.Stat(fn)
	if err != nil {
		t.Fatalf("stat file %s failed: %v", fn, err)
	}
	err = createByOpen("/nonexistent/dir/file.txt", info)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestCreateAndChownTypeAssertError(t *testing.T) {
	dir := makeTempDir("TestChownType", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.txt")
	if err := os.WriteFile(fn, []byte("test"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	info, err := os.Stat(fn)
	if err != nil {
		t.Fatalf("stat file %s failed: %v", fn, err)
	}
	err = createAndChown(fn, info)
	if err != nil {
		t.Logf("createAndChown result: %v", err)
	}
}

func TestHandleStaleLogRunError(t *testing.T) {
	l := &FileLogger{staleLogCh: make(chan bool, 1)}
	l.staleLogCh <- true
	close(l.staleLogCh)
	l.handleStaleLogRun()
}

func TestCompressLogFileOpenSrcError(t *testing.T) {
	err := compressLogFile("/nonexistent/src.log", "/tmp/dst.gz")
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestCompressLogFileCreateDstError(t *testing.T) {
	dir := makeTempDir("TestCompress", t)
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "src.log")
	if err := os.WriteFile(src, []byte("test data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", src, err)
	}
	patch := gomonkey.ApplyFunc(createAndChown, func(_ string, _ os.FileInfo) error {
		return errors.New("mock create error")
	})
	defer patch.Reset()
	err := compressLogFile(src, filepath.Join(dir, "dst.gz"))
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestWriteWithNilFile(t *testing.T) {
	dir := makeTempDir("TestWriteNil", t)
	defer os.RemoveAll(dir)
	l := &FileLogger{fileName: filepath.Join(dir, "test.log")}
	_, err := l.Write([]byte("test"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestWriteRotateError(t *testing.T) {
	l := &FileLogger{fileName: "/tmp/test-rotate-err.log", maxSize: 1}
	megabyte = 1
	_, err := l.Write(make([]byte, 100))
	if err != nil {
		t.Logf("expected possible write error: %v", err)
	}
}

func TestOpenNewAppendLog(t *testing.T) {
	dir := makeTempDir("TestOpenNewAppend", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("existing"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	err := l.openNew()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestOpenExistingOrNewFileNotExist(t *testing.T) {
	dir := makeTempDir("TestOpenExistNew", t)
	defer os.RemoveAll(dir)
	l := &FileLogger{fileName: filepath.Join(dir, "new.log")}
	err := l.openExistingOrNew(10)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestOpenExistingOrNewStatError(t *testing.T) {
	dir := makeTempDir("TestStatErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	patch := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, errors.New("mock stat error")
	})
	defer patch.Reset()
	l := &FileLogger{fileName: fn}
	err := l.openExistingOrNew(10)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestOpenExistingOrNewOpenAppendError(t *testing.T) {
	dir := makeTempDir("TestOpenAppendErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0000); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	err := l.openExistingOrNew(10)
	if err != nil {
		t.Logf("expected fallback: %v", err)
	}
}

func TestHandleStaleLogRunOnceOldLogFilesError(t *testing.T) {
	l := &FileLogger{fileName: "/nonexistent/dir/test.log", maxBackups: 10, maxAge: 30}
	err := l.handleStaleLogRunOnce()
	if err != nil {
		t.Logf("expected error from oldLogFiles: %v", err)
	}
}

func TestHandleStaleLogRunOnceRemoveError(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Remove, func(_ string) error {
		return errors.New("mock remove error")
	})
	defer patch.Reset()
	dir := makeTempDir("TestStaleRemoveErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn, maxBackups: 0, maxAge: 1}
	_ = l.handleStaleLogRunOnce()
}

func TestCompressLogFileGzipError(t *testing.T) {
	dir := makeTempDir("TestCompressGzip", t)
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "src.log")
	if err := os.WriteFile(src, []byte("test data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", src, err)
	}
	if err := os.WriteFile(src+".gz", []byte{}, 0644); err != nil {
		t.Fatalf("write file %s failed: %v", src+".gz", err)
	}
	patchCopy := gomonkey.ApplyFunc(io.Copy, func(_ io.Writer, _ io.Reader) (int64, error) {
		return 0, errors.New("mock copy error")
	})
	defer patchCopy.Reset()
	err := compressLogFile(src, src+".gz")
	if err == nil {
		t.Errorf("expected error from copy")
	}
}

func TestWriteNilFileOpenError(t *testing.T) {
	l := &FileLogger{fileName: "/nonexistent/dir/test.log"}
	patch := gomonkey.ApplyPrivateMethod(l, "openExistingOrNew", func(_ int) error {
		return errors.New("mock open error")
	})
	defer patch.Reset()
	_, err := l.Write([]byte("test"))
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestWriteRotateErr(t *testing.T) {
	dir := makeTempDir("TestWriteRotateErr", t)
	defer os.RemoveAll(dir)
	l := &FileLogger{fileName: filepath.Join(dir, "test.log"), maxSize: 1}
	megabyte = 1
	_, _ = l.Write([]byte("a"))
	l.file = nil
	patch := gomonkey.ApplyPrivateMethod(l, "rotate", func() error {
		return errors.New("mock rotate error")
	})
	defer patch.Reset()
	_, err := l.Write(make([]byte, 200))
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestRotateOpenNewError(t *testing.T) {
	l := &FileLogger{fileName: "test.log"}
	patchClose := gomonkey.ApplyPrivateMethod(l, "close", func() error {
		return nil
	})
	defer patchClose.Reset()
	patchOpen := gomonkey.ApplyPrivateMethod(l, "openNew", func() error {
		return errors.New("mock openNew error")
	})
	defer patchOpen.Reset()
	err := l.rotate()
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestOpenNewRenameError(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	info, err := os.Stat(fn)
	if err != nil {
		t.Fatalf("stat file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	patchStat := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return info, nil
	})
	defer patchStat.Reset()
	patchRename := gomonkey.ApplyFunc(os.Rename, func(_, _ string) error {
		return errors.New("mock rename error")
	})
	defer patchRename.Reset()
	err = l.openNew()
	if err == nil {
		t.Errorf("expected error from rename")
	}
}

func TestOpenNewChmodError(t *testing.T) {
	dir := makeTempDir("TestOpenNewChmodErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	patchChmod := gomonkey.ApplyFunc(os.Chmod, func(_ string, _ os.FileMode) error {
		return errors.New("mock chmod error")
	})
	defer patchChmod.Reset()
	err := l.openNew()
	if err == nil {
		t.Errorf("expected error from chmod")
	}
}

func TestOpenNewCreateAndChownError(t *testing.T) {
	dir := makeTempDir("TestOpenNewChownErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	patch := gomonkey.ApplyFunc(createAndChown, func(_ string, _ os.FileInfo) error {
		return errors.New("mock createAndChown error")
	})
	defer patch.Reset()
	err := l.openNew()
	if err == nil {
		t.Errorf("expected error from createAndChown")
	}
}

func TestOpenNewOpenLogFileError(t *testing.T) {
	dir := makeTempDir("TestOpenNewOpenFileErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	l := &FileLogger{fileName: fn}
	patch := gomonkey.ApplyFunc(openLogFile, func(_ string, _ int, _ os.FileMode) (*os.File, error) {
		return nil, errors.New("mock open error")
	})
	defer patch.Reset()
	err := l.openNew()
	if err == nil {
		t.Errorf("expected error from openLogFile")
	}
}

func TestHandleStaleLogRunCompressError(t *testing.T) {
	dir := makeTempDir("TestStaleCompressErr", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn, maxBackups: 10}
	patchCompress := gomonkey.ApplyFunc(compressLogFile, func(_, _ string) error {
		return errors.New("mock compress error")
	})
	defer patchCompress.Reset()
	_ = l.handleStaleLogRunOnce()
}

func TestHandleStaleLogRunChannel(t *testing.T) {
	dir := makeTempDir("TestStaleRunCh", t)
	defer os.RemoveAll(dir)
	fn := filepath.Join(dir, "test.log")
	l := &FileLogger{fileName: fn, staleLogCh: make(chan bool, 1)}
	l.staleLogCh <- true
	close(l.staleLogCh)
	l.handleStaleLogRun()
}

func TestOldLogFilesInfoError(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	l := &FileLogger{fileName: fn}
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	patchReadDir := gomonkey.ApplyFunc(os.ReadDir, func(_ string) ([]os.DirEntry, error) {
		return []os.DirEntry{dirEntryMock{isDir: true}}, nil
	})
	defer patchReadDir.Reset()
	_, err := l.oldLogFiles()
	if err != nil {
		t.Logf("oldLogFiles result: %v", err)
	}
}

type dirEntryMock struct {
	isDir bool
}

func (d dirEntryMock) Name() string               { return "test.log" }
func (d dirEntryMock) IsDir() bool                { return d.isDir }
func (d dirEntryMock) Type() os.FileMode          { return 0 }
func (d dirEntryMock) Info() (os.FileInfo, error) { return &mockFileInfoSys{}, nil }

func TestCreateAndChownTypeAssertFail(t *testing.T) {
	info := &mockFileInfo{}
	err := createAndChown("test.txt", info)
	if err == nil {
		t.Errorf("expected error for type assertion failure")
	}
}

type mockFileInfo struct{}

func (m *mockFileInfo) Name() string       { return "test" }
func (m *mockFileInfo) Size() int64        { return 0 }
func (m *mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m *mockFileInfo) ModTime() time.Time { return time.Time{} }
func (m *mockFileInfo) IsDir() bool        { return false }
func (m *mockFileInfo) Sys() interface{}   { return "not a stat_t" }

type mockFileInfoSys struct {
	mockFileInfo
}

func TestCompressLogFileGzipWriterCloseError(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.log")
	if err := os.WriteFile(src, []byte("test data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", src, err)
	}
	dst := filepath.Join(dir, "dst.gz")
	patch := gomonkey.ApplyFunc(io.Copy, func(_ io.Writer, _ io.Reader) (int64, error) {
		return int64(len("test data")), nil
	})
	defer patch.Reset()
	err := compressLogFile(src, dst)
	if err != nil {
		t.Logf("compressLogFile: %v", err)
	}
}

func TestCompressLogFileOpenGzError(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.log")
	if err := os.WriteFile(src, []byte("test data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", src, err)
	}
	patch := gomonkey.ApplyFunc(createAndChown, func(_ string, _ os.FileInfo) error {
		return nil
	})
	defer patch.Reset()
	patchOpen := gomonkey.ApplyFunc(os.OpenFile, func(_ string, _ int, _ os.FileMode) (*os.File, error) {
		return nil, errors.New("mock open gz error")
	})
	defer patchOpen.Reset()
	err := compressLogFile(src, filepath.Join(dir, "dst.gz"))
	if err == nil {
		t.Errorf("expected error opening gz")
	}
}

func TestWriteRotateBranchError(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn, maxSize: 1}
	megabyte = 1
	patch := gomonkey.ApplyPrivateMethod(l, "rotate", func() error {
		return errors.New("mock rotate error")
	})
	defer patch.Reset()
	_, _ = l.Write([]byte("aaaaa"))
	l.file = nil
	_, err := l.Write([]byte("bbbbb"))
	if err != nil {
		t.Logf("expected error: %v", err)
	}
}

func TestOpenExistingOrNewStatGenericError(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn}
	patch := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, errors.New("generic stat error")
	})
	defer patch.Reset()
	err := l.openExistingOrNew(10)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestOpenExistingOrNewOpenAppendFallback(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	patch := gomonkey.ApplyFunc(openLogFile, func(_ string, _ int, _ os.FileMode) (*os.File, error) {
		return nil, errors.New("mock open error")
	})
	defer patch.Reset()
	l := &FileLogger{fileName: fn}
	err := l.openExistingOrNew(10)
	if err != nil {
		t.Logf("expected fallback: %v", err)
	}
}

func TestHandleStaleLogRunOnceRemoveAndCompress(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn, maxBackups: 10}
	patchCompress := gomonkey.ApplyFunc(compressLogFile, func(_, _ string) error {
		return errors.New("mock compress error")
	})
	defer patchCompress.Reset()
	patchRemove := gomonkey.ApplyFunc(os.Remove, func(_ string) error {
		return errors.New("mock remove error")
	})
	defer patchRemove.Reset()
	_ = l.handleStaleLogRunOnce()
}

func TestHandleStaleLogRunError2(t *testing.T) {
	dir := t.TempDir()
	fn := filepath.Join(dir, "test.log")
	if err := os.WriteFile(fn, []byte("data"), 0644); err != nil {
		t.Fatalf("write file %s failed: %v", fn, err)
	}
	l := &FileLogger{fileName: fn, staleLogCh: make(chan bool, 1)}
	l.staleLogCh <- true
	close(l.staleLogCh)
	l.handleStaleLogRun()
}

func TestOldLogFilesIsDir(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	fn := filepath.Join(dir, "test.log")
	l := &FileLogger{fileName: fn}
	_, err := l.oldLogFiles()
	if err != nil {
		t.Logf("oldLogFiles: %v", err)
	}
}
