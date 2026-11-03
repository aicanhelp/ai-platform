package contentprovider

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"go.uber.org/zap"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/log"
)

type fileContentProvider struct {
	filePath    string
	fileContent atomic.Value
	contentLock sync.Mutex

	fileWatcher FileWatcher
	listeners   []Listener
}

// NewFileContentProvider 创建文件内容提供器，监控指定文件的变化
func NewFileContentProvider(path string) DynamicContentProvider {
	return &fileContentProvider{
		filePath: path,
	}
}

// CurrentContent 获取当前文件内容的副本
func (f *fileContentProvider) CurrentContent() []byte {
	return f.loadContent()
}

// AddListener 添加文件内容变化监听器
func (f *fileContentProvider) AddListener(listener Listener) {
	f.listeners = append(f.listeners, listener)
}

// RunOnce 一次性读取文件内容并存储
func (f *fileContentProvider) RunOnce() error {
	content, err := os.ReadFile(f.filePath)
	if err != nil {
		return err
	}
	f.fileContent.Store(content)
	return nil
}

// Run 启动文件内容监控，检测到文件变化时通知所有监听器
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
			log.GetLogger().Error("read file failed", zap.Error(err))
			return
		}
		fileContent := f.loadContent()
		if bytes.Equal(content, fileContent) {
			return
		}
		f.fileContent.Store(fileContent)
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
