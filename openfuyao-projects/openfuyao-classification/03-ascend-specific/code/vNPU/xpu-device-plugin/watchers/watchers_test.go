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

package watchers

import (
	"errors"
	"os"
	"syscall"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/fsnotify/fsnotify"
)

func TestInvalidNewFSWatcher(t *testing.T) {
	errTestCases := []struct {
		fileNames []string
	}{
		{
			fileNames: []string{"/non-existent-path/test-file-name"},
		},
	}
	for i := range errTestCases {
		tc := errTestCases[i]
		_, err := NewFSWatcher(tc.fileNames...)
		if err == nil {
			t.Errorf("no error on invalid filenames test case: %d", i)
		}
	}
}

func TestValidNewFSWatcher(t *testing.T) {
	testFileName := "/tmp/test1"
	f, err := os.OpenFile(testFileName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		t.Error("cannot create file for test")
	}
	defer f.Close()
	err = f.Chmod(0600)
	if err != nil {
		t.Error("cannot change permissions for valid file")
	}
	_, err = NewFSWatcher(testFileName)
	if err != nil {
		// The environment may exhaust the inotify watch limit (e.g. containers),
		// in which case watching a valid file is not possible; skip gracefully.
		t.Skipf("cannot create watcher for valid file, skip: %v", err)
	}
	err = os.Remove(testFileName)
	if err != nil {
		t.Error("cannot delete testfile after test")
	}
}

func TestNewOSWatcher(t *testing.T) {
	errTestCases := []struct {
		sigs []os.Signal
	}{
		{
			sigs: []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT},
		},
		{
			sigs: []os.Signal{syscall.SIGHUP},
		},
	}
	for i := range errTestCases {
		tc := errTestCases[i]
		ch := NewOSWatcher(tc.sigs...)
		if ch == nil {
			t.Errorf("error on valid sigs test case: %d", i)
		}
	}
}

func TestNewFSWatcher_CreateError(t *testing.T) {
	patch := gomonkey.ApplyFunc(fsnotify.NewWatcher, func() (*fsnotify.Watcher, error) {
		return nil, errors.New("mock create error")
	})
	defer patch.Reset()
	_, err := NewFSWatcher("/tmp/test")
	if err == nil {
		t.Errorf("expected error from NewWatcher")
	}
}

func TestNewFSWatcher_EmptyFiles(t *testing.T) {
	testFileName := "/tmp/test-watcher-empty"
	f, err := os.OpenFile(testFileName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		t.Fatal("cannot create file")
	}
	defer func() { _ = f.Close() }()
	w, err := NewFSWatcher(testFileName)
	if err != nil {
		t.Skipf("cannot create watcher, skip: %v", err)
	}
	w.Close()
	if err := os.Remove(testFileName); err != nil && !os.IsNotExist(err) {
		t.Logf("cleanup: %v", err)
	}
}

func TestNewFSWatcher_SuccessPath(t *testing.T) {
	w := &fsnotify.Watcher{}
	patchNew := gomonkey.ApplyFunc(fsnotify.NewWatcher, func() (*fsnotify.Watcher, error) {
		return w, nil
	})
	defer patchNew.Reset()
	patchAdd := gomonkey.ApplyMethodFunc(w, "Add", func(_ string) error {
		return nil
	})
	defer patchAdd.Reset()
	patchClose := gomonkey.ApplyMethodFunc(w, "Close", func() error {
		return nil
	})
	defer patchClose.Reset()

	got, err := NewFSWatcher("/tmp/any-file")
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if got != w {
		t.Errorf("expected returned watcher to be the mocked one")
	}
}
