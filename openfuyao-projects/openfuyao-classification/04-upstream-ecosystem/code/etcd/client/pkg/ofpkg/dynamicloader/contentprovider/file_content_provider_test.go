package contentprovider

import (
	"bytes"
	"context"
	"os"
	"testing"
	"time"
)

// TestFileContentProvider watch file A, we should get notification when A changed
func TestFileContentProvider(t *testing.T) {
	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// write origin content
	originContent := []byte("origin")
	os.WriteFile(file.Name(), originContent, 0600)

	// create dynamic provider
	provider := NewFileContentProvider(file.Name())
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	provider.(*fileContentProvider).fileWatcher = watcher
	defer stopWatcher()

	// load file content
	provider.RunOnce()

	// expect get correct content
	cueerntContent := provider.CurrentContent()
	if !bytes.Equal(provider.CurrentContent(), originContent) {
		t.Errorf("expected '%s' but found '%s'", originContent, cueerntContent)
	}

	// register listener
	var newContentRead []byte
	provider.AddListener(func(content []byte) {
		newContentRead = content
	})

	// run provider
	provider.Run(context.Background())

	// change file A
	newContent := []byte("new")
	os.WriteFile(file.Name(), newContent, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead)
	}
}

// TestFileContentProviderWhenRegisterMultiListener register multiple listener for file A
func TestFileContentProviderWhenRegisterMultiListener(t *testing.T) {
	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// create dynamic provider
	provider := NewFileContentProvider(file.Name())
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	provider.(*fileContentProvider).fileWatcher = watcher
	defer stopWatcher()

	// load file content
	provider.RunOnce()

	// register listener
	var newContentRead1 []byte
	provider.AddListener(func(content []byte) {
		newContentRead1 = content
	})
	var newContentRead2 []byte
	provider.AddListener(func(content []byte) {
		newContentRead2 = content
	})

	// run provider
	provider.Run(context.Background())

	// change file A
	newContent := []byte("new")
	os.WriteFile(file.Name(), newContent, 0600)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead1, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead1)
	}
	if !bytes.Equal(newContentRead2, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead2)
	}
}

// TestFileContentProviderWhenFileChangesMultiTimes file changes multiple times, we should get same times notification sequentially
func TestFileContentProviderWhenFileChangesMultiTimes(t *testing.T) {
	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// create dynamic provider
	provider := NewFileContentProvider(file.Name())
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	provider.(*fileContentProvider).fileWatcher = watcher
	defer stopWatcher()

	// load file content
	provider.RunOnce()

	// register listener
	var newContentRead1 []byte
	var extras = ""
	var newContentRead2 []byte
	var found = false
	count := 0
	provider.AddListener(func(content []byte) {
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

	// run provider
	provider.Run(context.Background())

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
	if !bytes.Equal(newContentRead2, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent2, newContentRead2)
	}
	if !found {
		t.Errorf("expected getting notification sequentially")
	}
}

// TestFileContentProviderWhenFileChangesMultiTimesButContentNotChange file changes multiple times, but content not change, we should get one times notification
func TestFileContentProviderWhenFileChangesMultiTimesButContentNotChange(t *testing.T) {
	// create tmp file A
	file, _ := os.CreateTemp("", "testfile-*")
	defer file.Close()

	// create dynamic provider
	provider := NewFileContentProvider(file.Name())
	watcher, stopWatcher := initWatcher(testCallBackDuration)
	provider.(*fileContentProvider).fileWatcher = watcher
	defer stopWatcher()

	// load file content
	provider.RunOnce()

	// register listener
	var newContentRead []byte
	count := 0
	provider.AddListener(func(content []byte) {
		newContentRead, _ = os.ReadFile(file.Name())
		count++
	})

	// run provider
	provider.Run(context.Background())

	// change file A
	newContent := []byte("new")
	os.WriteFile(file.Name(), newContent, 0600)

	time.Sleep(testCallBackDuration + 2*time.Second)
	os.WriteFile(file.Name(), newContent, 0777)

	// we get new content of file A
	time.Sleep(testCallBackDuration + 2*time.Second)
	if !bytes.Equal(newContentRead, newContent) {
		t.Errorf("expected '%s' but found '%s'", newContent, newContentRead)
	}
	if count != 1 {
		t.Errorf("expected call back %d times, but found %d times", 1, count)
	}
}
