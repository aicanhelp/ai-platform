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
	"context"
	"os"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/fy-ext/signal"
)

var (
	defaultFileWatcher FileWatcher
	lock               sync.Mutex
)

const DefaultCallBackDuration = 30 * time.Second
const ChangedFilesChanSize = 100

func DefaultFileWatcher() (FileWatcher, error) {
	if defaultFileWatcher == nil {
		lock.Lock()
		defer lock.Unlock()
		if defaultFileWatcher == nil {
			ctx := signal.FyGlobalContext()
			if watcher, err := CreateAndRunFileWatcher(DefaultCallBackDuration, ctx); err != nil {
				return nil, err
			} else {
				defaultFileWatcher = watcher
			}
		}
	}
	return defaultFileWatcher, nil
}

func CreateAndRunFileWatcher(callBack time.Duration, ctx context.Context) (FileWatcher, error) {
	watcher := NewFileWatcher(callBack)
	if err := watcher.Run(ctx); err != nil {
		return nil, err
	}
	return watcher, nil
}

type FileChangedListener func()

// FileWatcher should call back multiple changes for same file sequentially
type FileWatcher interface {
	AddListener(filePath string, listener FileChangedListener)
	Run(ctx context.Context) error
}

type fileWatcher struct {
	callbackDuration time.Duration

	filePaths    map[string][]FileChangedListener
	lastModTime  map[string]int64
	changedFiles chan string

	filePathLock sync.Mutex
}

func NewFileWatcher(callBack time.Duration) FileWatcher {
	return &fileWatcher{
		filePaths:        map[string][]FileChangedListener{},
		lastModTime:      map[string]int64{},
		changedFiles:     make(chan string, ChangedFilesChanSize),
		callbackDuration: callBack,
	}
}

func (f *fileWatcher) AddListener(filePath string, listener FileChangedListener) {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	listeners, ok := f.filePaths[filePath]
	if !ok {
		f.filePaths[filePath] = []FileChangedListener{listener}
	} else {
		f.filePaths[filePath] = append(listeners, listener)
	}
}

func (f *fileWatcher) Run(ctx context.Context) error {
	go wait.Until(f.watchFile, f.callbackDuration, ctx.Done())
	go wait.Until(func() { f.handleFileChanged(ctx.Done()) }, time.Second, ctx.Done())
	return nil
}

func (f *fileWatcher) watchFile() {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	for filePath := range f.filePaths {
		if f.checkFileChange(filePath) {
			f.changedFiles <- filePath
		}
	}
}

func (f *fileWatcher) checkFileChange(filePath string) bool {
	getFileModTime := func(path string) int64 {
		fi, err := os.Stat(path)
		if err != nil {
			return time.Now().Unix()
		}
		return fi.ModTime().Unix()
	}

	modTimeNew := getFileModTime(filePath)
	if modTime, ok := f.lastModTime[filePath]; !ok || modTime != modTimeNew {
		f.lastModTime[filePath] = modTimeNew
		return true
	}
	return false
}

func (f *fileWatcher) handleFileChanged(stopCh <-chan struct{}) {
	for {
		select {
		case filePath := <-f.changedFiles:
			f.fireFileChangeEventFor(filePath)
		case <-stopCh:
			return
		}
	}
}

func (f *fileWatcher) fireFileChangeEventFor(filePath string) {
	f.filePathLock.Lock()
	defer f.filePathLock.Unlock()

	listeners, ok := f.filePaths[filePath]
	if !ok {
		return
	}
	for _, listener := range listeners {
		listener()
	}
}
