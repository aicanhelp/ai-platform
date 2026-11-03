// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package contentprovider

import "context"

type ContentProvider interface {
	CurrentContent() []byte
}

type Listener func(content []byte)

type Notifier interface {
	AddListener(listener Listener)
}

type ControllerRunner interface {
	RunOnce() error
	Run(ctx context.Context) error
}

// DynamicContentProvider should call back multiple changes sequentially
type DynamicContentProvider interface {
	ContentProvider
	Notifier
	ControllerRunner
}
