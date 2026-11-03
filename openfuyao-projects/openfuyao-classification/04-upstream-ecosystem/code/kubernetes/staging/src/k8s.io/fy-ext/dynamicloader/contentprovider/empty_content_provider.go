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

import "context"

type emptyContentProvider struct{}

func NewEmptyContentProvider() DynamicContentProvider {
	return &emptyContentProvider{}
}

func (e emptyContentProvider) CurrentContent() []byte { return nil }

func (e emptyContentProvider) AddListener(listener Listener) {}

func (e emptyContentProvider) RunOnce() error { return nil }

func (e emptyContentProvider) Run(ctx context.Context) error { return nil }
