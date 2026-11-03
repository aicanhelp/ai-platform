// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package contentprovider

import (
	"bytes"
	"os"
	"testing"
	"time"
)

const testCallBackDuration = 1 * time.Second
const testCallBackDurationLong = 3 * time.Second

// watch file A, we should get notification when A changed
func TestWatchFile(t *testing.T) {
	// start watcher
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	defer stopWatcher()

	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// watch file A
	var newContentRead []byte
	watcher.AddListener(file.Name(), func() {
		newContentRead, _ = os.ReadFile(file.Name())
	})

	// change file A
	newContent := []byte("new")
	os.WriteFile(file.Name(), newContent, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead, newContent) {
		t.Errorf("expected '%s' but got '%s'", newContent, newContent)
	}
}

// watch multiple files
func TestWatchFileWhenMultiFiles(t *testing.T) {
	// start watcher
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	defer stopWatcher()

	// create tmp file A,B
	file1, _ := os.CreateTemp("", "testfile-*")
	file2, _ := os.CreateTemp("", "testfile-*")
	defer file1.Close()
	defer file2.Close()

	// watch file A,B
	var newContentRead1 []byte
	watcher.AddListener(file1.Name(), func() {
		newContentRead1, _ = os.ReadFile(file1.Name())
	})
	var newContentRead2 []byte
	watcher.AddListener(file2.Name(), func() {
		newContentRead2, _ = os.ReadFile(file2.Name())
	})

	// change file A
	newContent1 := []byte("new1")
	os.WriteFile(file1.Name(), newContent1, 0600)
	newContent2 := []byte("new2")
	os.WriteFile(file2.Name(), newContent2, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead1, newContent1) {
		t.Errorf("expected '%s' but found '%s'", newContent1, newContentRead1)
	}
	if !bytes.Equal(newContent2, newContent2) {
		t.Errorf("expected '%s' but found '%s'", newContent2, newContentRead2)
	}
}

// register multiple listener for the same file
func TestWatchFileWhenRegisterMultiListener(t *testing.T) {
	// start watcher
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	defer stopWatcher()

	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// watch file A
	var newContentRead []byte
	watcher.AddListener(file.Name(), func() {
		newContentRead, _ = os.ReadFile(file.Name())
	})
	var newContentRead2 []byte
	watcher.AddListener(file.Name(), func() {
		newContentRead2, _ = os.ReadFile(file.Name())
	})

	// change file A
	newContent := []byte("new1")
	os.WriteFile(file.Name(), newContent, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead, newContent) || !bytes.Equal(newContentRead2, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead)
	}
}

// file changes multiple times, we should get same times notification sequentially
func TestWatchFileWhenFileChangeMultiTimes(t *testing.T) {
	// start watcher
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	defer stopWatcher()

	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// watch file A
	var newContentRead1 []byte
	var extras = ""
	var newContentRead2 []byte
	var found = false
	count := 0
	watcher.AddListener(file.Name(), func() {
		if count == 0 {
			newContentRead1, _ = os.ReadFile(file.Name())
			time.Sleep(3 * time.Second)
			extras = "OK"
			count++
		} else {
			newContentRead2, _ = os.ReadFile(file.Name())
			if extras == "OK" {
				found = true
			}
		}
	})

	// change file A
	newContent := []byte("new")
	os.WriteFile(file.Name(), newContent, 0600)

	time.Sleep(testCallBackDuration + 2*time.Second)
	newContent2 := []byte("new2")
	os.WriteFile(file.Name(), newContent2, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead1, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead1)
	}
	if !bytes.Equal(newContent2, newContent2) {
		t.Errorf("expected '%s' but found '%s'", newContent2, newContentRead2)
	}
	if !found {
		t.Errorf("expected getting notification sequentially")
	}
}

// file changes in close time, we should get one time notification
func TestWatchFileWhenFileChangesInCloseTime(t *testing.T) {
	// start watcher
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	defer stopWatcher()

	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// watch file A
	var newContentRead []byte
	count := 0
	watcher.AddListener(file.Name(), func() {
		newContentRead, _ = os.ReadFile(file.Name())
		count++
	})

	// change file A
	newContent1 := []byte("new1")
	os.WriteFile(file.Name(), newContent1, 0600)

	newContent2 := []byte("new2")
	os.WriteFile(file.Name(), newContent2, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead, newContent2) || count != 1 {
		t.Errorf("expected '%s' but found '%s'", newContent2, newContentRead)
	}
	if count != 1 {
		t.Errorf("expected call back %d times but found %d times", 1, count)
	}
}

// initWatcher init watcher, eliminate the impact of initialization
func initWatcher(callBack time.Duration) (FileWatcher, func()) {
	watcher, _ := CreateAndRunFileWatcher(callBack)
	time.Sleep(100 * time.Millisecond)
	return watcher, func() {
		watcher.Stop()
	}
}
