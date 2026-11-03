/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	toolscache "k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// WriteThroughClient wraps client.Client so that every successful write also
// synchronously updates the local Informer cache, closing the cache_miss window
// between a same-controller Create/Update and the next Get.
//
// Pass cache = nil to disable write-through (no-op wrapper).
type WriteThroughClient struct {
	client.Client
	cache cache.Cache
}

func NewWriteThroughClient(c client.Client, cache cache.Cache) *WriteThroughClient {
	return &WriteThroughClient{Client: c, cache: cache}
}

// storeFor resolves the toolscache.Store for obj's type. Returns nil on any
// failure (cache disabled, informer not registered, type assertion failed) —
// the caller silently drops the write-through attempt and lets the WATCH
// event reconcile.
//
// controller-runtime's public cache.Informer interface does not expose
// GetStore(); we type-assert to toolscache.SharedIndexInformer which does.
func (w *WriteThroughClient) storeFor(ctx context.Context, obj client.Object) toolscache.Store {
	if w.cache == nil {
		return nil
	}
	informer, err := w.cache.GetInformer(ctx, obj)
	if err != nil {
		return nil
	}
	sii, ok := informer.(toolscache.SharedIndexInformer)
	if !ok {
		return nil
	}
	return sii.GetStore()
}

func (w *WriteThroughClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	err := w.Client.Create(ctx, obj, opts...)
	if err == nil {
		if s := w.storeFor(ctx, obj); s != nil {
			_ = s.Add(obj)
		}
		return nil
	}
	// AlreadyExists: object is on the API server but may not be in the Informer
	// cache yet. Seed the store with the create payload so a same-reconcile Get
	// (e.g. HA kubeconfig follow-up) does not race with the WATCH event.
	if apierrors.IsAlreadyExists(err) {
		if s := w.storeFor(ctx, obj); s != nil {
			if _, exists, _ := s.Get(obj); !exists {
				_ = s.Add(obj)
			}
		}
	}
	return err
}

func (w *WriteThroughClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	if err := w.Client.Update(ctx, obj, opts...); err != nil {
		return err
	}
	if s := w.storeFor(ctx, obj); s != nil {
		_ = s.Update(obj)
	}
	return nil
}

func (w *WriteThroughClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	if err := w.Client.Delete(ctx, obj, opts...); err != nil {
		return err
	}
	if s := w.storeFor(ctx, obj); s != nil {
		_ = s.Delete(obj)
	}
	return nil
}
