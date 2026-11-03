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

package certificate

import (
	"context"
	"crypto/tls"
	"sync"
	"sync/atomic"
)

type DynamicServerCertificateProvider struct {
	baseTLSConfig      *tls.Config
	currentTLSConfig   atomic.Value
	tlsConfigLock      sync.RWMutex
	certificateContent *DynamicCertificateContent
}

func NewDynamicCertificateProvider(tlsConfig *tls.Config, certFile, keyFile, trustCAFile string) *DynamicServerCertificateProvider {
	return &DynamicServerCertificateProvider{
		baseTLSConfig:      tlsConfig,
		certificateContent: NewDynamicCertificateContentFromFile(certFile, keyFile, trustCAFile),
	}
}

func (p *DynamicServerCertificateProvider) GetConfigForClient(clientHello *tls.ClientHelloInfo) (*tls.Config, error) {
	return p.loadCurrentTLSConfig().Clone(), nil
}

func (p *DynamicServerCertificateProvider) RunOnce() error {
	err := p.certificateContent.RunOnce()
	if err != nil {
		return err
	}
	p.refreshTLSConfig()
	return nil
}

func (p *DynamicServerCertificateProvider) Run(ctx context.Context) error {
	p.certificateContent.AddKeyPairListener(p.refreshTLSConfig)
	p.certificateContent.AddCAListener(p.refreshTLSConfig)
	return p.certificateContent.Run(ctx)
}

func (p *DynamicServerCertificateProvider) refreshTLSConfig() {
	p.tlsConfigLock.Lock()
	defer p.tlsConfigLock.Unlock()

	tlsConfigNew := p.baseTLSConfig.Clone()
	if p.certificateContent.KeyPair() != nil {
		tlsConfigNew.Certificates = []tls.Certificate{*p.certificateContent.KeyPair()}
	}
	if p.certificateContent.CAPool() != nil {
		tlsConfigNew.ClientCAs = p.certificateContent.CAPool()
	}
	p.currentTLSConfig.Store(tlsConfigNew)
}

func (p *DynamicServerCertificateProvider) loadCurrentTLSConfig() *tls.Config {
	v := p.currentTLSConfig.Load()
	if v == nil {
		return nil
	}
	if p, ok := v.(*tls.Config); !ok {
		return nil
	} else {
		return p
	}
}
