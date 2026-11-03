package certificate

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"sync/atomic"

	"go.uber.org/zap"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader/contentprovider"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/log"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/timer"
)

// DynamicCertificateContent 动态证书内容管理器，用于管理证书、私钥和CA证书的动态加载和更新
type DynamicCertificateContent struct {
	ctx context.Context

	certProvider contentprovider.DynamicContentProvider
	keyProvider  contentprovider.DynamicContentProvider
	caProvider   contentprovider.DynamicContentProvider

	keyPair atomic.Value
	caPool  atomic.Value

	keyPairRefreshRunner timer.RetryableRunner

	keyPairChangedListeners []func()
	caChangedListeners      []func()
}

// KeyPair 获取当前TLS密钥对
func (c *DynamicCertificateContent) KeyPair() *tls.Certificate {
	return c.loadKeyPair()
}

// CAPool 获取当前CA证书池
func (c *DynamicCertificateContent) CAPool() *x509.CertPool {
	return c.loadCAPool()
}

// NewDynamicCertificateContentFromFile 从文件创建动态证书内容管理器
func NewDynamicCertificateContentFromFile(certFile, keyFile, trustCAFile string) *DynamicCertificateContent {
	var certProvider contentprovider.DynamicContentProvider
	if len(certFile) > 0 {
		certProvider = contentprovider.NewFileContentProvider(certFile)
	}
	var keyProvider contentprovider.DynamicContentProvider
	if len(keyFile) > 0 {
		keyProvider = contentprovider.NewFileContentProvider(keyFile)
	}
	var caProvider contentprovider.DynamicContentProvider
	if len(trustCAFile) > 0 {
		caProvider = contentprovider.NewFileContentProvider(trustCAFile)
	}
	return NewDynamicCertificateContent(certProvider, keyProvider, caProvider)
}

// NewDynamicCertificateContent 从内容提供器创建动态证书内容管理器
func NewDynamicCertificateContent(certProvider, keyProvider,
	caProvider contentprovider.DynamicContentProvider) *DynamicCertificateContent {
	if certProvider == nil {
		certProvider = contentprovider.NewEmptyContentProvider()
	}
	if keyProvider == nil {
		keyProvider = contentprovider.NewEmptyContentProvider()
	}
	if caProvider == nil {
		caProvider = contentprovider.NewEmptyContentProvider()
	}
	return &DynamicCertificateContent{
		certProvider:            certProvider,
		keyProvider:             keyProvider,
		caProvider:              caProvider,
		keyPairChangedListeners: []func(){},
		caChangedListeners:      []func(){},
	}
}

// AddCAListener 添加CA证书变化监听器
func (c *DynamicCertificateContent) AddCAListener(listener func()) {
	c.caChangedListeners = append(c.caChangedListeners, listener)
}

// AddKeyPairListener 添加密钥对变化监听器
func (c *DynamicCertificateContent) AddKeyPairListener(listener func()) {
	c.keyPairChangedListeners = append(c.keyPairChangedListeners, listener)
}

// RunOnce 一次性运行初始化，加载所有证书内容
func (c *DynamicCertificateContent) RunOnce() error {
	if err := c.caProvider.RunOnce(); err != nil {
		return err
	}
	if err := c.certProvider.RunOnce(); err != nil {
		return err
	}
	if err := c.keyProvider.RunOnce(); err != nil {
		return err
	}

	if err := c.refreshCAPool(); err != nil {
		return fmt.Errorf("init cert pool failed: %v", err)
	}
	if err := c.refreshKeyPair(); err != nil {
		return fmt.Errorf("init key pair failed: %v", err)
	}

	return nil
}

// Run 启动动态证书内容管理器的后台运行
func (c *DynamicCertificateContent) Run(ctx context.Context) error {
	c.ctx = ctx

	c.caProvider.AddListener(c.onTrustCAChanged)
	err := c.caProvider.Run(ctx)
	if err != nil {
		return fmt.Errorf("run ca provider failed: %v", err)
	}

	c.certProvider.AddListener(c.onCertChanged)
	c.keyProvider.AddListener(c.onKeyChanged)

	err = c.certProvider.Run(ctx)
	if err != nil {
		return fmt.Errorf("run cert provider failed: %v", err)
	}
	err = c.keyProvider.Run(ctx)
	if err != nil {
		return fmt.Errorf("run key provider failed: %v", err)
	}

	return nil
}

func (c *DynamicCertificateContent) onTrustCAChanged([]byte) {
	// listener of content provider would be call back sequentially, so we don't need to use lock
	if err := c.refreshCAPool(); err != nil {
		log.GetLogger().Error("failed to refresh ca in dynamic certificates", zap.Error(err))
	} else {
		for _, listener := range c.caChangedListeners {
			listener()
		}
	}
}

func (c *DynamicCertificateContent) onCertChanged([]byte) {
	c.onCertOrKeyChanged()
}

func (c *DynamicCertificateContent) onKeyChanged([]byte) {
	c.onCertOrKeyChanged()
}

func (c *DynamicCertificateContent) onCertOrKeyChanged() {
	c.keyPairRefreshRunner.Run(c.ctx, func(context.Context) error {
		if err := c.refreshKeyPair(); err != nil {
			return err
		}
		for _, listener := range c.keyPairChangedListeners {
			listener()
		}
		return nil
	})
}

func (c *DynamicCertificateContent) refreshCAPool() error {
	caContent := c.caProvider.CurrentContent()
	if caContent == nil {
		return nil
	}
	caPool := x509.NewCertPool()
	ok := caPool.AppendCertsFromPEM(caContent)
	if !ok {
		return fmt.Errorf("parse ca failed")
	}
	c.caPool.Store(caPool)
	return nil
}

func (c *DynamicCertificateContent) refreshKeyPair() error {
	certContent := c.certProvider.CurrentContent()
	keyContent := c.keyProvider.CurrentContent()
	if certContent == nil || keyContent == nil {
		return nil
	}
	keyContent, err := decryption.OfTryDecryptCert(keyContent)
	if err != nil {
		return err
	}
	keyPair, err := tls.X509KeyPair(certContent, keyContent)
	if err != nil {
		return err
	}
	c.keyPair.Store(&keyPair)
	return nil
}

func (c *DynamicCertificateContent) loadKeyPair() *tls.Certificate {
	v := c.keyPair.Load()
	if v == nil {
		return nil
	}
	if p, ok := v.(*tls.Certificate); !ok {
		return nil
	} else {
		return p
	}
}

func (c *DynamicCertificateContent) loadCAPool() *x509.CertPool {
	v := c.caPool.Load()
	if v == nil {
		return nil
	}
	if p, ok := v.(*x509.CertPool); !ok {
		return nil
	} else {
		return p
	}
}
