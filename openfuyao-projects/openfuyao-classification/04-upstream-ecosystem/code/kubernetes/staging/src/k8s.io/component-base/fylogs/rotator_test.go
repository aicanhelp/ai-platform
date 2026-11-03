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

package fylogs

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

const (
	defaultFilePerm = 0644
	totalCountLimit = 2
	totalSizeLimit  = 1
	sleepTime       = 10 * time.Second
)

func TestRotator(t *testing.T) {
	file, tempDir, err := setUp()
	if err != nil {
		t.Fatalf("unable to create temp file: %v", err)
	}

	stat, err := file.Stat()
	if err != nil {
		t.Fatalf("unable to stat temp file: %v", err)
	}

	fw := newFileWrap(file, filepath.Join(tempDir, "test"), defaultFilePerm)
	wfg := NewWrapFileConfig()
	wfg.rotateFileConfig.EnableCompress = true
	wfg.rotateFileConfig.TotalCountLimit = totalCountLimit
	wfg.rotateFileConfig.TotalSizeLimit = totalSizeLimit
	rt := newRotator(fw, wfg, stat)

	fileNames := []string{"test_01.gz", "test_02.tmp", "test_03"}
	var files []*os.File
	for i := 0; i < len(fileNames); i++ {
		files = append(files, newTestFile(t, tempDir, fileNames[i]))
	}
	defer func() {
		for _, f := range files {
			if err := os.Remove(filepath.Join(tempDir, f.Name())); err != nil {
				t.Logf("unable to remove file: %v", err)
			}
		}
	}()

	rt.rotate()

	time.Sleep(sleepTime)
}

func newTestFile(t *testing.T, tempDir, name string) *os.File {
	file, err := os.OpenFile(filepath.Join(tempDir, name), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("unable to create file: %v", err)
	}

	defer deferCloseAndRemoveFile(file, t)

	return file
}

func TestRemoveExcessLogsSize(t *testing.T) {
	type testCase struct {
		name      string
		r         *rotator
		pathExist bool
		hasErr    bool
	}

	// 初始化测试对象
	var path string

	testCases := []testCase{
		{
			"totalSizeLimit is 0",
			&rotator{
				totalSizeLimit: int64(0),
			},
			false,
			false,
		},
		{
			"error when stat",
			&rotator{
				totalSizeLimit: int64(1),
			},
			false,
			true,
		},
		{
			"sizeCounter < reserveTotalSize and no error return",
			&rotator{
				totalSizeLimit: int64(1000),
				eachFileSize:   int64(1),
			},
			true,
			false,
		},
		{
			"no error and return",
			&rotator{
				totalSizeLimit: int64(0),
			},
			true,
			false,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			path = t.TempDir()
			var logs []string
			var err error
			if tc.pathExist {
				logs, err = tc.r.removeExcessLogsSize([]string{path})
			} else {
				logs, err = tc.r.removeExcessLogsSize([]string{"not/exist/path"})
			}
			if tc.hasErr {
				assert.Nil(t, logs)
				assert.Error(t, err)
			} else {
				assert.NotNil(t, logs)
				assert.NoError(t, err)
			}
		})
	}
}

func TestRotate(t *testing.T) {
	// 初始化测试对象
	var r *rotator
	tempFile := t.TempDir()

	type testCase struct {
		name         string
		writtenBytes int
		eachFileSize int64
		filePath     string
		checkFunc    func()
	}
	testCases := []testCase{
		{
			"counter < eachFileSize and return",
			1,
			10,
			"",
			func() {
				assert.Equal(t, int64(1), r.counter)
			},
		},
		{
			"filepath not exist error",
			10,
			10,
			"/not/exist/path",
			func() {
				assert.Equal(t, int64(10), r.counter)
			},
		},
		{
			"no error and return",
			0,
			0,
			tempFile,
			func() {
				assert.NotNil(t, r.target)
			},
		},
		{
			"rename error for file already exist and return",
			0,
			0,
			tempFile,
			func() {
				assert.Nil(t, r.target)
			},
		},
	}

	// 删除可能遗留的临时文件
	timestamp := time.Now().Format(timestampFormat)
	rotated := fmt.Sprintf("%s_%s", timestamp, timestamp)
	err := os.RemoveAll(rotated)
	if err != nil {
		t.Fatalf("failed to remove temp file: %v", err)
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r = &rotator{
				fileWrap: &fileWrap{},
			}
			r.writtenBytes = tc.writtenBytes
			r.eachFileSize = tc.eachFileSize
			r.filePath = tc.filePath

			r.rotate()

			tc.checkFunc()
		})
	}
}

func TestRotateLogs(t *testing.T) {
	// 初始化测试对象
	r := &rotator{
		fileWrap:       &fileWrap{},
		enableCompress: true,
	}

	gz, err := os.OpenFile("TestRotateLogs_test.gz", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Errorf("fail to create gz file: %v", err)
	}
	defer deferCloseAndRemoveFile(gz, t)

	log, err := os.OpenFile("TestRotateLogs_test.log", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Errorf("fail to create gz file: %v", err)
	}
	defer deferCloseAndRemoveFile(log, t)

	type testCase struct {
		name     string
		filePath string
		hasErr   bool
	}
	testCases := []testCase{
		{
			"filePath error",
			"[",
			true,
		},
		{
			"no error and return",
			"TestRotateLogs",
			false,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r.filePath = tc.filePath

			err := r.rotateLogs()
			if tc.hasErr {
				if err == nil {
					t.Errorf("expect an error when %s, but got nil", tc.name)
				}
			} else {
				if err != nil {
					t.Errorf("unexpect error when %s: %v", tc.name, err)
				}
			}
		})
	}
}

func TestCompressLog(t *testing.T) {
	// 初始化测试对象
	r := &rotator{
		fileWrap: &fileWrap{},
	}
	tempFile, err := os.CreateTemp("", "TestCompressLog")
	if err != nil {
		t.Errorf("fail to create temp file: %v", err)
	}
	defer deferCloseAndRemoveFile(tempFile, t)

	type testCase struct {
		name   string
		log    string
		hasErr bool
	}
	testCases := []testCase{
		{
			"file not exist error when open",
			"/not/exist/file/path",
			true,
		},
		{
			"no error and return",
			tempFile.Name(),
			false,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := r.compressLog(tc.log)
			if tc.hasErr {
				if err == nil {
					t.Errorf("expect an error when %s, but got nil", tc.name)
				}
			} else {
				if err != nil {
					t.Errorf("unexpect error when %s: %v", tc.name, err)
				}
			}
		})
	}
}

func TestStartRotateLog(t *testing.T) {
	// 初始化测试对象
	r := &rotator{
		fileWrap: &fileWrap{},
	}
	gz, err := os.OpenFile("TestStartRotateLog_test.gz", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Errorf("fail to create gz file: %v", err)
	}
	defer deferCloseAndRemoveFile(gz, t)

	type testCase struct {
		name           string
		rotateCounting uint
		filePath       string
		expectCount    uint
	}
	testCases := []testCase{
		{
			"rotateCounting != 1 and skip",
			1,
			"",
			2,
		},
		{
			"rotateCounting == 1 but has error",
			0,
			"/not/exist/file/path",
			0,
		},
		{
			"rotateCounting == 1 and no error",
			0,
			"TestStartRotateLog",
			0,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r.rotateCounting = tc.rotateCounting
			r.filePath = tc.filePath

			r.startRotateLog()

			if tc.expectCount != r.rotateCounting {
				t.Errorf("expect rotateCounting become %d after startRotateLog, but got %d", tc.expectCount, r.rotateCounting)
			}
		})
	}
}

func TestAfterRotateLog(t *testing.T) {
	// 初始化测试对象
	gz, err := os.OpenFile("TestAfterRotateLog_test.gz", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Errorf("fail to create gz file: %v", err)
	}
	defer deferCloseAndRemoveFile(gz, t)
	r := &rotator{
		fileWrap: &fileWrap{
			filePath: "TestAfterRotateLog",
		},
	}

	type testCase struct {
		name           string
		err            error
		rotateCounting uint
	}
	testCases := []testCase{
		{
			"no more retate and no error",
			nil,
			1,
		},
		{
			"more rotate and has error",
			fmt.Errorf("error"),
			2,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r.rotateCounting = tc.rotateCounting

			r.afterRotateLog(tc.err)

			if r.rotateCounting != 0 {
				t.Errorf("expect retateCounting become 0 after AfterRotateLog, but got %d", r.rotateCounting)
			}
		})
	}
}

type mockCloser struct {
	isClosed bool
	count    int
}

func (mc *mockCloser) Close() error {
	mc.count++
	if mc.isClosed {
		return os.ErrClosed
	}
	mc.isClosed = true
	return nil
}

func TestCloseIgnoreErr(t *testing.T) {
	// 初始化测试对象
	r := &rotator{
		fileWrap: &fileWrap{},
	}
	mc := &mockCloser{}

	type testCase struct {
		name   string
		ignore bool
	}
	testCases := []testCase{
		{
			"no error",
			false,
		},
		{
			"has error and ignore",
			true,
		},
		{
			"has error and not ignore",
			false,
		},
	}

	// 执行测试
	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r.closeIgnoreErr(mc, "", tc.ignore)

			if !mc.isClosed {
				t.Errorf("error when closeIgnoreErr: expect file to be closed, but not")
			}
			if mc.count != i+1 {
				t.Errorf("expect file to be closed %d times, but got %d", i+1, mc.count)
			}
		})
	}
}

func TestRemoveExcessLogsCount(t *testing.T) {
	// 初始化测试对象
	r := &rotator{}
	gz, err := os.OpenFile("TestRemoveExcessLogs_test.gz", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Errorf("fail to create gz file: %v", err)
	}
	defer deferCloseAndRemoveFile(gz, t)

	type testCase struct {
		name            string
		totalCountLimit int
		logs            []string
		isNil           bool
		length          int
		hasErr          bool
	}
	testCases := []testCase{
		{
			"totalCountLimit == 0 and skip",
			0,
			[]string{},
			false,
			0,
			false,
		},
		{
			"error when remove log file",
			-1,
			[]string{"/not/exist/file/path"},
			true,
			1,
			true,
		},
		{
			"no error and return",
			-1,
			[]string{gz.Name()},
			false,
			0,
			false,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r.totalCountLimit = tc.totalCountLimit

			logs, err := r.removeExcessLogs(tc.logs)
			if tc.isNil {
				if logs != nil {
					t.Errorf("expect result logs is nil, but got %v", logs)
				}
			} else {
				if len(logs) != tc.length {
					t.Errorf("expect result length %d, but got %d", tc.length, len(logs))
				}
			}
			if tc.hasErr {
				if err == nil {
					t.Errorf("expect an error when %s, but got nil", tc.name)
				}
			} else {
				if err != nil {
					t.Errorf("expect no error when %s, but got %v", tc.name, err)
				}
			}
		})
	}
}

func TestIsInUse(t *testing.T) {
	// 初始化测试对象
	r := &rotator{}

	type testCase struct {
		name     string
		fileName string
		logs     []string
		result   bool
	}
	testCases := []testCase{
		{
			"has suffix HasSuffix",
			"a.tmp",
			[]string{},
			false,
		},
		{
			"has suffix compressSuffix",
			"b.gz",
			[]string{},
			true,
		},
		{
			"has suffix compressSuffix in logs",
			"c",
			[]string{"c.gz"},
			false,
		},
		{
			"default",
			"d",
			[]string{},
			true,
		},
	}

	// 执行测试
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := r.isInUse(tc.fileName, tc.logs)
			if result != tc.result {
				t.Errorf("expect result %v, but got %v", tc.result, result)
			}
		})
	}
}

func TestFilterUnusedLogs(t *testing.T) {
	r := &rotator{}
	logs := []string{"a.tmp", "b.gz", "c", "c.gz", "d"}

	inuse, unused := r.filterUnusedLogs(logs)

	if len(inuse) != 3 {
		t.Errorf("expect len 3, but got %d", len(inuse))
		if inuse[0] != "b.gz" {
			t.Errorf("expect the first log of inuse is b.gz, but got %s", inuse[0])
		}
		if inuse[1] != "c.gz" {
			t.Errorf("expect the second log of inuse is c.gz, but got %s", inuse[1])
		}
		if inuse[2] != "d" {
			t.Errorf("expect the third log of inuse is d, but got %s", inuse[2])
		}
	}
	if len(unused) != 2 {
		t.Errorf("expect len 2, but got %d", len(unused))
		if unused[0] != "a.tmp" {
			t.Errorf("expect the first log of unused is a.tmp, but got %s", inuse[0])
		}
		if unused[1] != "c" {
			t.Errorf("expect the second log of unused is c, but got %s", inuse[1])
		}
	}
}

func deferCloseAndRemoveFile(file *os.File, t *testing.T) {
	err := file.Close()
	if err != nil {
		t.Errorf("fail to close gz file: %v", err)
	}
	err = os.RemoveAll(file.Name())
	if err != nil {
		t.Errorf("fail to remove gz file: %v", err)
	}
}
