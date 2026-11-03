package certificate

import (
	"context"
	"crypto/tls"
	"sync"
	"sync/atomic"
)

// DynamicServerCertificateProvider 动态服务器证书提供器，管理服务端TLS配置的动态更新
type DynamicServerCertificateProvider struct {
	baseTLSConfig      *tls.Config
	currentTLSConfig   atomic.Value
	tlsConfigLock      sync.RWMutex
	certificateContent *DynamicCertificateContent
}

// NewDynamicCertificateProvider 创建动态证书提供器（服务端使用）
func NewDynamicCertificateProvider(tlsConfig *tls.Config, certFile, keyFile,
	trustCAFile string) *DynamicServerCertificateProvider {
	return &DynamicServerCertificateProvider{
		baseTLSConfig:      tlsConfig,
		certificateContent: NewDynamicCertificateContentFromFile(certFile, keyFile, trustCAFile),
	}
}

// GetConfigForClient 获取针对客户端的TLS配置（实现tls.Config.GetConfigForClient接口）
func (p *DynamicServerCertificateProvider) GetConfigForClient(clientHello *tls.ClientHelloInfo) (*tls.Config, error) {
	return p.loadCurrentTLSConfig().Clone(), nil
}

// RunOnce 一次性运行初始化，加载证书并刷新TLS配置
func (p *DynamicServerCertificateProvider) RunOnce() error {
	err := p.certificateContent.RunOnce()
	if err != nil {
		return err
	}
	p.refreshTLSConfig()
	return nil
}

// Run 启动动态证书提供器的后台运行
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
