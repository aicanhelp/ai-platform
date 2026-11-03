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
	"os"
	"testing"
	"time"

	"k8s.io/fy-ext/dynamicloader/contentprovider"
)

func TestDynamicClientCertificateProvider(t *testing.T) {
	certProvider := &mockDynamicContentProvider{
		content: certData1,
	}
	keyProvider := &mockDynamicContentProvider{
		content: keyData1,
	}
	caProvider := &mockDynamicContentProvider{
		content: caData1,
	}
	provider := NewDynamicClientCertificateProvider(certProvider, keyProvider, caProvider)
	provider.RunOnce()

	// check after runonce
	keyPair, _ := provider.GetClientCertificate(nil)
	if keyPair == nil {
		t.Fatal("expected key pair not nil")
	}
	caPool := provider.GetCACertificate()
	if caPool == nil {
		t.Fatal("expected ca pool not nil")
	}
	tlsConfig := &tls.Config{RootCAs: caPool}
	initCAPool := caPool.Clone()

	provider.UpdateCADynamically(tlsConfig)
	ctx, _ := context.WithCancel(context.Background())
	provider.Run(ctx)

	// check after update certificate
	certProvider.SetContent(certData2)
	time.Sleep(100 * time.Millisecond)
	keyPairNew, _ := provider.GetClientCertificate(nil)
	if keyPairNew != keyPair {
		t.Fatal("expected key pair equal with old one")
	}

	// check after update key
	keyProvider.SetContent(keyData2)
	time.Sleep(100 * time.Millisecond)
	keyPairNew, _ = provider.GetClientCertificate(nil)
	if keyPairNew == nil || keyPairNew == keyPair {
		t.Fatal("expected key pair not nil and not equal with old one")
	}

	// check after update ca certificate
	caProvider.SetContent(caData2)
	time.Sleep(100 * time.Millisecond)
	caPoolNew := provider.GetCACertificate()
	if caPoolNew == nil || caPoolNew.Equal(initCAPool) {
		t.Fatal("expected ca pool not nil and not equal with old one")
	}
	if tlsConfig.RootCAs == nil || tlsConfig.RootCAs.Equal(initCAPool) {
		t.Fatal("expected tls.Config.RootCAs not nil and not equal with old one")
	}
}

func TestDynamicClientCertificateProviderWhenUsingCache(t *testing.T) {
	caFile, _ := os.CreateTemp("", "testfile-*")
	defer os.Remove(caFile.Name())
	os.WriteFile(caFile.Name(), caData1, 0600)

	certFile, _ := os.CreateTemp("", "testfile-*")
	defer os.Remove(certFile.Name())
	os.WriteFile(certFile.Name(), certData1, 0600)

	keyFile, _ := os.CreateTemp("", "testfile-*")
	defer os.Remove(keyFile.Name())
	os.WriteFile(keyFile.Name(), keyData1, 0600)

	// create provider
	provider := GetDynamicClientCertificateProviderFromCache(certFile.Name(), keyFile.Name(), caFile.Name())
	provider2 := GetDynamicClientCertificateProviderFromCache(certFile.Name(), keyFile.Name(), caFile.Name())
	if provider != provider2 {
		t.Fatal("expected get same provider instance")
	}

	// runOnce
	provider.RunOnce()
	keyPair1, _ := provider.GetClientCertificate(nil)
	provider.RunOnce()
	keyPair2, _ := provider.GetClientCertificate(nil)

	if keyPair1 == nil || keyPair1 != keyPair2 {
		t.Fatal("expected keypair1 equal with keypair2")
	}

	tlsConfig1 := &tls.Config{RootCAs: provider.GetCACertificate()}
	initCA1 := *tlsConfig1.RootCAs
	provider.UpdateCADynamically(tlsConfig1)
	ctx1, _ := context.WithCancel(context.Background())
	provider.Run(ctx1)

	tlsConfig2 := &tls.Config{RootCAs: provider.GetCACertificate()}
	initCA2 := *tlsConfig2.RootCAs
	provider.UpdateCADynamically(tlsConfig2)
	ctx2, _ := context.WithCancel(context.Background())
	provider.Run(ctx2)

	time.Sleep(time.Second)
	os.WriteFile(caFile.Name(), caData2, 0600)
	os.WriteFile(certFile.Name(), certData2, 0600)
	os.WriteFile(keyFile.Name(), keyData2, 0600)

	// check after update certificate
	time.Sleep(contentprovider.DefaultCallBackDuration + 2*time.Second)
	keyPairNew1, _ := provider.GetClientCertificate(nil)
	keyPairNew2, _ := provider.GetClientCertificate(nil)
	if keyPairNew1 == nil || keyPairNew1 != keyPairNew2 {
		t.Fatal("expected keypair1 equal with keypair2 after key/cert updated")
	}

	newCA1 := *tlsConfig1.RootCAs
	newCA2 := *tlsConfig2.RootCAs

	// check after update ca certificate
	if initCA1.Equal(&newCA1) || initCA2.Equal(&newCA2) || !newCA1.Equal(&newCA2) {
		t.Fatal("expected new CA is equal with another new ca")
	}
}
