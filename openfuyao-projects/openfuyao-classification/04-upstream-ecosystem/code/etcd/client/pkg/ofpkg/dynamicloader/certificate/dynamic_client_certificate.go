package certificate

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"sync"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader/certificate/cache"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader/contentprovider"
)

var (
	clientCertificatesCache = map[cache.CertificateCacheKey]*DynamicClientCertificateProvider{}
	clientCertificatesLock  sync.Mutex
)

// DynamicClientCertificateProvider be called concurrently, it should handle concurrency properly
type DynamicClientCertificateProvider struct {
	certificateContent *DynamicCertificateContent

	runOnceInit bool
	runInit     bool
	runLock     sync.Mutex
}

// GetDynamicClientCertificateProviderFromCache 从缓存获取或创建动态客户端证书提供器
func GetDynamicClientCertificateProviderFromCache(certFile, keyFile,
	trustCAFile string) *DynamicClientCertificateProvider {
	clientCertificatesLock.Lock()
	defer clientCertificatesLock.Unlock()

	cacheKey := cache.NewCacheKey(certFile, keyFile, trustCAFile)
	if p, ok := clientCertificatesCache[cacheKey]; ok {
		return p
	} else {
		p = NewDynamicClientCertificateProviderFromFile(certFile, keyFile, trustCAFile)
		clientCertificatesCache[cacheKey] = p
		return p
	}
}

// NewDynamicClientCertificateProviderFromFile 从文件创建动态客户端证书提供器
func NewDynamicClientCertificateProviderFromFile(certFile, keyFile,
	trustCAFile string) *DynamicClientCertificateProvider {
	return &DynamicClientCertificateProvider{
		certificateContent: NewDynamicCertificateContentFromFile(certFile, keyFile, trustCAFile),
	}
}

// NewDynamicClientCertificateProvider 从内容提供器创建动态客户端证书提供器
func NewDynamicClientCertificateProvider(certProvider, keyProvider,
	caProvider contentprovider.DynamicContentProvider) *DynamicClientCertificateProvider {
	return &DynamicClientCertificateProvider{
		certificateContent: NewDynamicCertificateContent(certProvider, keyProvider, caProvider),
	}
}

// GetClientCertificate 获取客户端证书（实现tls.Config.GetClientCertificate接口）
func (p *DynamicClientCertificateProvider) GetClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return p.certificateContent.KeyPair(), nil
}

// GetCACertificate 获取CA证书池
func (p *DynamicClientCertificateProvider) GetCACertificate() *x509.CertPool {
	return p.certificateContent.CAPool()
}

// UpdateCADynamically 动态更新TLS配置中的CA证书池
func (p *DynamicClientCertificateProvider) UpdateCADynamically(tlsConfig *tls.Config) {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	p.certificateContent.AddCAListener(func() {
		*tlsConfig.RootCAs = *p.certificateContent.CAPool().Clone()
	})
}

// RunOnce 一次性运行初始化，加载证书内容
func (p *DynamicClientCertificateProvider) RunOnce() error {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	if p.runOnceInit {
		return nil
	}
	p.runOnceInit = true
	return p.certificateContent.RunOnce()
}

// Run 启动动态证书提供器的后台运行
func (p *DynamicClientCertificateProvider) Run(ctx context.Context) error {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	if p.runInit {
		return nil
	}
	p.runInit = true
	return p.certificateContent.Run(ctx)
}
