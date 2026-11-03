// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package contentprovider

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"github.com/coredns/coredns/plugin/pkg/log"
)

type fileContentProvider struct {
	filePath    string
	fileContent atomic.Value
	contentLock sync.Mutex

	fileWatcher FileWatcher
	listeners   []Listener
}

func NewFileContentProvider(path string) DynamicContentProvider {
	return &fileContentProvider{
		filePath: path,
	}
}

func (f *fileContentProvider) CurrentContent() []byte {
	return f.loadContent()
}

func (f *fileContentProvider) AddListener(listener Listener) {
	f.listeners = append(f.listeners, listener)
}

func (f *fileContentProvider) RunOnce() error {
	content, err := os.ReadFile(f.filePath)
	if err != nil {
		return err
	}
	f.fileContent.Store(content)
	return nil
}

func (f *fileContentProvider) Run(ctx context.Context) error {
	if f.fileWatcher == nil {
		if watcher, err := DefaultFileWatcher(); err != nil {
			return fmt.Errorf("error creating FileWatcher: %v", err)
		} else {
			f.fileWatcher = watcher
		}
	}
	f.fileWatcher.AddListener(f.filePath, func() {
		f.contentLock.Lock()
		defer f.contentLock.Unlock()

		content, err := os.ReadFile(f.filePath)
		if err != nil {
			log.Errorf("read file failed: %v", err)
			return
		}
		fileContent := f.loadContent()
		if bytes.Equal(content, fileContent) {
			return
		}
		f.fileContent.Store(content)
		for _, listener := range f.listeners {
			listener(content)
		}
	})
	return nil
}

func (f *fileContentProvider) loadContent() []byte {
	v := f.fileContent.Load()
	if v == nil {
		return nil
	}
	if s, ok := v.([]byte); !ok {
		return nil
	} else {
		return s
	}
}
