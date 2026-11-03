// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package contentprovider

import "context"

type emptyContentProvider struct{}

func NewEmptyContentProvider() DynamicContentProvider {
	return &emptyContentProvider{}
}

func (e emptyContentProvider) CurrentContent() []byte { return nil }

func (e emptyContentProvider) AddListener(listener Listener) {}

func (e emptyContentProvider) RunOnce() error { return nil }

func (e emptyContentProvider) Run(ctx context.Context) error { return nil }
