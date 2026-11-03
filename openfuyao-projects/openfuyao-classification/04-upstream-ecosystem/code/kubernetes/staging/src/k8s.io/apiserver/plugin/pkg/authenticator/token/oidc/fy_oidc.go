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

package oidc

import (
	"context"
	"crypto/tls"

	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/fy-ext/dynamicloader"
	"k8s.io/fy-ext/dynamicloader/contentprovider"
)

type listenerAdapter struct {
	listenFunc contentprovider.Listener
	caContent  CAContentProvider
}

func (l *listenerAdapter) Enqueue() {
	l.listenFunc(l.caContent.CurrentCABundleContent())
}

type fyDynamicContentLoaderAdapter struct {
	caContentProvider CAContentProvider
}

func (f *fyDynamicContentLoaderAdapter) CurrentContent() []byte {
	return f.caContentProvider.CurrentCABundleContent()
}

func (f *fyDynamicContentLoaderAdapter) AddListener(listener contentprovider.Listener) {
	if notifier, ok := f.caContentProvider.(dynamiccertificates.Notifier); ok {
		notifier.AddListener(&listenerAdapter{listenFunc: listener, caContent: f.caContentProvider})
	}
}

func (f *fyDynamicContentLoaderAdapter) RunOnce() error {
	if runner, ok := f.caContentProvider.(dynamiccertificates.ControllerRunner); ok {
		ctx, _ := context.WithCancel(context.Background())
		return runner.RunOnce(ctx)
	}
	return nil
}

func (f *fyDynamicContentLoaderAdapter) Run(ctx context.Context) error {
	if runner, ok := f.caContentProvider.(dynamiccertificates.ControllerRunner); ok {
		go runner.Run(ctx, 1)
		return nil
	}
	return nil
}

func fyCreateDynamicTLSConfig(caContentProvider CAContentProvider) (*tls.Config, error) {
	dynamicContentProvider := &fyDynamicContentLoaderAdapter{caContentProvider: caContentProvider}
	tlsConfig := &tls.Config{}
	if err := dynamicloader.UpdateCADynamicallyForClientByProvider(tlsConfig, dynamicContentProvider); err != nil {
		return nil, err
	}
	return tlsConfig, nil
}
