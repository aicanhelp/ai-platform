// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

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

func (p *DynamicServerCertificateProvider) GetCertificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	return p.certificateContent.KeyPair(), nil
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
