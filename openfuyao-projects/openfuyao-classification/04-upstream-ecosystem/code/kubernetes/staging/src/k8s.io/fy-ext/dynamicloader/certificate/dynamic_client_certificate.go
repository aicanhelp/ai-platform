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
	"sync"

	"k8s.io/fy-ext/dynamicloader/certificate/cache"
	"k8s.io/fy-ext/dynamicloader/contentprovider"
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

func GetDynamicClientCertificateProviderFromCache(certFile, keyFile, trustCAFile string) *DynamicClientCertificateProvider {
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

func NewDynamicClientCertificateProviderFromFile(certFile, keyFile, trustCAFile string) *DynamicClientCertificateProvider {
	return &DynamicClientCertificateProvider{
		certificateContent: NewDynamicCertificateContentFromFile(certFile, keyFile, trustCAFile),
	}
}

func NewDynamicClientCertificateProvider(certProvider, keyProvider, caProvider contentprovider.DynamicContentProvider) *DynamicClientCertificateProvider {
	return &DynamicClientCertificateProvider{
		certificateContent: NewDynamicCertificateContent(certProvider, keyProvider, caProvider),
	}
}

func (p *DynamicClientCertificateProvider) GetClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return p.certificateContent.KeyPair(), nil
}

func (p *DynamicClientCertificateProvider) GetCACertificate() *x509.CertPool {
	return p.certificateContent.CAPool()
}

func (p *DynamicClientCertificateProvider) UpdateCADynamically(tlsConfig *tls.Config) {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	p.certificateContent.AddCAListener(func() {
		*tlsConfig.RootCAs = *p.certificateContent.CAPool().Clone()
	})
}

func (p *DynamicClientCertificateProvider) RunOnce() error {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	if p.runOnceInit {
		return nil
	}
	p.runOnceInit = true
	return p.certificateContent.RunOnce()
}

func (p *DynamicClientCertificateProvider) Run(ctx context.Context) error {
	p.runLock.Lock()
	defer p.runLock.Unlock()

	if p.runInit {
		return nil
	}
	p.runInit = true
	return p.certificateContent.Run(ctx)
}
