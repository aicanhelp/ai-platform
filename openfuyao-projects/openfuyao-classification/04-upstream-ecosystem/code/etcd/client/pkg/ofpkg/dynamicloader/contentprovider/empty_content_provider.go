package contentprovider

import (
	"context"
)

type emptyContentProvider struct{}

// NewEmptyContentProvider 创建一个空内容提供器（不监控任何文件）
func NewEmptyContentProvider() DynamicContentProvider {
	return &emptyContentProvider{}
}

// CurrentContent 返回空内容（nil字节数组）
func (e emptyContentProvider) CurrentContent() []byte { return nil }

// AddListener 空实现，不添加任何监听器
func (e emptyContentProvider) AddListener(listener Listener) {}

// RunOnce 空实现，总是返回nil
func (e emptyContentProvider) RunOnce() error { return nil }

// Run 空实现，总是返回nil
func (e emptyContentProvider) Run(ctx context.Context) error { return nil }
