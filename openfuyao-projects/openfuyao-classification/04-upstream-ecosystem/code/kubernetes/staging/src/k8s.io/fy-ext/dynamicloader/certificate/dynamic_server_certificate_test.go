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
	"bytes"
	"context"
	"crypto/tls"
	"testing"
	"time"
)

func TestDynamicServerCertificateProvider(t *testing.T) {
	certProvider := &mockDynamicContentProvider{
		content: certData1,
	}
	keyProvider := &mockDynamicContentProvider{
		content: keyData1,
	}
	caProvider := &mockDynamicContentProvider{
		content: caData1,
	}
	originTlsConfig := &tls.Config{}
	provider := &DynamicServerCertificateProvider{
		baseTLSConfig:      originTlsConfig,
		certificateContent: NewDynamicCertificateContent(certProvider, keyProvider, caProvider),
	}
	provider.RunOnce()

	// check after runonce
	tmpTLSConfig, _ := provider.GetConfigForClient(nil)
	initCertificate := tmpTLSConfig.Certificates
	if len(initCertificate) != 1 {
		t.Fatal("expected key pair not nil")
	}
	initCACertificate := tmpTLSConfig.ClientCAs
	if tmpTLSConfig.ClientCAs == nil {
		t.Fatal("expected ca pool not nil")
	}

	ctx, _ := context.WithCancel(context.Background())
	provider.Run(ctx)

	// check after update certificate
	certProvider.SetContent(certData2)
	time.Sleep(100 * time.Millisecond)
	tmpTLSConfig, _ = provider.GetConfigForClient(nil)
	tmpCertificate := tmpTLSConfig.Certificates
	if len(tmpCertificate) != 1 || !bytes.Equal(tmpCertificate[0].Certificate[0], initCertificate[0].Certificate[0]) {
		t.Fatal("expected key pair equal with old one")
	}

	// check after update key
	keyProvider.SetContent(keyData2)
	time.Sleep(100 * time.Millisecond)
	tmpTLSConfig, _ = provider.GetConfigForClient(nil)
	tmpCertificate = tmpTLSConfig.Certificates
	if len(tmpCertificate) != 1 || bytes.Equal(tmpCertificate[0].Certificate[0], initCertificate[0].Certificate[0]) {
		t.Fatal("expected key pair not nil and not equal with old one")
	}

	// check after update ca certificate
	caProvider.SetContent(caData2)
	time.Sleep(100 * time.Millisecond)
	tmpTLSConfig, _ = provider.GetConfigForClient(nil)
	if tmpTLSConfig.ClientCAs == nil || tmpTLSConfig.ClientCAs.Equal(initCACertificate) {
		t.Fatal("expected ca pool not nil and not equal with old one")
	}
}
