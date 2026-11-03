/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package tokenizer

import (
	"context"
	"fmt"
)

// FakeTokenizer implements Backend for unit tests.
type FakeTokenizer struct {
	InitializeFn   func(ctx context.Context, req InitializeRequest) error
	TokenizeFn     func(ctx context.Context, req Request) (*TokenizationResult, error)
	CompleteFn     func(ctx context.Context, req CompleteRequest) (*TokenizationResult, error)
	ChatCompleteFn func(ctx context.Context, req ChatCompleteRequest) (*TokenizationResult, error)
	CloseFn        func() error
}

func (f *FakeTokenizer) Close() error {
	if f == nil {
		return nil
	}
	if f.CloseFn != nil {
		return f.CloseFn()
	}
	return nil
}

var _ Backend = (*FakeTokenizer)(nil)

func (f *FakeTokenizer) Initialize(ctx context.Context, req InitializeRequest) error {
	if f.InitializeFn != nil {
		return f.InitializeFn(ctx, req)
	}
	return fmt.Errorf("Initialize not configured")
}

func (f *FakeTokenizer) Tokenize(ctx context.Context, req Request) (*TokenizationResult, error) {
	if f.TokenizeFn != nil {
		return f.TokenizeFn(ctx, req)
	}
	return nil, fmt.Errorf("Tokenize not configured")
}

func (f *FakeTokenizer) Complete(ctx context.Context, req CompleteRequest) (*TokenizationResult, error) {
	if f.CompleteFn != nil {
		return f.CompleteFn(ctx, req)
	}
	return nil, fmt.Errorf("Complete not configured")
}

func (f *FakeTokenizer) ChatComplete(ctx context.Context, req ChatCompleteRequest) (*TokenizationResult, error) {
	if f.ChatCompleteFn != nil {
		return f.ChatCompleteFn(ctx, req)
	}
	return nil, fmt.Errorf("ChatComplete not configured")
}
