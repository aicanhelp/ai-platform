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

package contentprovider

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"k8s.io/klog/v2"
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
			klog.Errorf("read file failed: %v", err)
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
