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
	"crypto/x509"
	"fmt"
	"sync/atomic"

	"k8s.io/fy-ext/dynamicloader/contentprovider"
	"k8s.io/fy-ext/security/crypto/decryption"
	"k8s.io/fy-ext/util/timer"
	"k8s.io/klog/v2"
)

type DynamicCertificateContent struct {
	ctx context.Context

	certProvider contentprovider.DynamicContentProvider
	keyProvider  contentprovider.DynamicContentProvider
	caProvider   contentprovider.DynamicContentProvider

	keyPair atomic.Value
	caPool  atomic.Value

	keyPairRefreshRunner timer.RetryableRunner

	keyPairChangedListener []func()
	caChangedListener      []func()
}

func (c *DynamicCertificateContent) KeyPair() *tls.Certificate {
	return c.loadKeyPair()
}

func (c *DynamicCertificateContent) CAPool() *x509.CertPool {
	return c.loadCAPool()
}

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

func NewDynamicCertificateContent(certProvider, keyProvider, caProvider contentprovider.DynamicContentProvider) *DynamicCertificateContent {
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
		certProvider:           certProvider,
		keyProvider:            keyProvider,
		caProvider:             caProvider,
		keyPairChangedListener: []func(){},
		caChangedListener:      []func(){},
	}
}

func (c *DynamicCertificateContent) AddCAListener(listener func()) {
	c.caChangedListener = append(c.caChangedListener, listener)
}

func (c *DynamicCertificateContent) AddKeyPairListener(listener func()) {
	c.keyPairChangedListener = append(c.keyPairChangedListener, listener)
}

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
		klog.Errorf("failed to refresh ca in dynamic certificates: %v", err)
	} else {
		for _, listener := range c.caChangedListener {
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
		for _, listener := range c.keyPairChangedListener {
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
	keyContent, err := decryption.FYTryDecryptCert(keyContent)
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
