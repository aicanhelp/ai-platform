package contentprovider

import "context"

// ContentProvider 内容提供器接口，提供获取当前内容的能力
type ContentProvider interface {
	CurrentContent() []byte
}

// Listener 内容变化监听器函数类型
type Listener func(content []byte)

// Notifier 通知器接口，用于添加内容变化监听器
type Notifier interface {
	// 添加内容变化监听器
	AddListener(listener Listener)
}

// ControllerRunner 控制器运行接口，支持一次性运行和持续运行
type ControllerRunner interface {
	// 一次性运行初始化
	RunOnce() error

	// 持续运行，接受上下文控制
	Run(ctx context.Context) error
}

// DynamicContentProvider should call back multiple changes sequentially
type DynamicContentProvider interface {
	// 继承内容提供功能
	ContentProvider

	// 继承通知功能
	Notifier

	// 继承控制器运行功能
	ControllerRunner
}
