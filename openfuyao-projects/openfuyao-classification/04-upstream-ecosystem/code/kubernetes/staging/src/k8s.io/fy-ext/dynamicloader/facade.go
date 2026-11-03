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

// Package dynamicloader provide facade of dynamic loader to others
package dynamicloader

import (
	"crypto/tls"
	"fmt"

	"k8s.io/fy-ext/dynamicloader/certificate"
	"k8s.io/fy-ext/dynamicloader/contentprovider"
	"k8s.io/fy-ext/signal"
	"k8s.io/klog/v2"
)

type TLS struct {
	CertFile      string
	KeyFile       string
	TrustedCAFile string
}

type Listener func(content []byte)

func ConfigureDynamicCertificateForServer(tlsConfig *tls.Config, tlsInfo *TLS) error {
	klog.V(4).Infof("configure dynamic certificate for server with : key file %s", tlsInfo.KeyFile)
	certificateProvider := certificate.NewDynamicCertificateProvider(tlsConfig.Clone(), tlsInfo.CertFile, tlsInfo.KeyFile, tlsInfo.TrustedCAFile)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic server certificate provider failed: %v", err)
	}

	err = certificateProvider.Run(signal.FyGlobalContext())
	if err != nil {
		return fmt.Errorf("run dynamic server certificate provider failed: %v", err)
	}

	tlsConfig.GetConfigForClient = certificateProvider.GetConfigForClient
	return nil
}

func ConfigureDynamicCertificateForClient(tlsConfig *tls.Config, certFile string, keyFile string) error {
	klog.V(4).Infof("configure dynamic certificate for client with : key file %s", keyFile)
	certificateProvider := certificate.GetDynamicClientCertificateProviderFromCache(certFile, keyFile, "")
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic client certificate provider failed: %v", err)
	}

	err = certificateProvider.Run(signal.FyGlobalContext())
	if err != nil {
		return fmt.Errorf("run dynamic client certificate provider failed: %v", err)
	}

	tlsConfig.GetClientCertificate = certificateProvider.GetClientCertificate
	return nil
}

func UpdateCADynamicallyForClient(tlsConfig *tls.Config, trustedCAFile string) error {
	klog.V(4).Infof("support updating ca certificates dynamically for client with ca file %s", trustedCAFile)
	certificateProvider := certificate.GetDynamicClientCertificateProviderFromCache("", "", trustedCAFile)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic client certificate provider failed: %v", err)
	}

	tlsConfig.RootCAs = certificateProvider.GetCACertificate().Clone()
	certificateProvider.UpdateCADynamically(tlsConfig)

	err = certificateProvider.Run(signal.FyGlobalContext())
	if err != nil {
		return fmt.Errorf("run dynamic client certificate provider failed: %v", err)
	}
	return nil
}

func UpdateCADynamicallyForClientByProvider(tlsConfig *tls.Config, provider contentprovider.DynamicContentProvider) error {
	klog.V(4).Info("support updating ca certificates dynamically for client with ca provider")
	certificateProvider := certificate.NewDynamicClientCertificateProvider(nil, nil, provider)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic client certificate provider failed: %v", err)
	}

	tlsConfig.RootCAs = certificateProvider.GetCACertificate().Clone()
	certificateProvider.UpdateCADynamically(tlsConfig)

	err = certificateProvider.Run(signal.FyGlobalContext())
	if err != nil {
		return fmt.Errorf("run dynamic client certificate provider failed: %v", err)
	}
	return nil
}

// ListenFileChangedEvent should call back multiple changes to the same file sequentially
func ListenFileChangedEvent(filePath string, listener Listener) error {
	klog.V(4).Infof("listen content change on file %s", filePath)
	p := contentprovider.NewFileContentProvider(filePath)
	err := p.RunOnce()
	if err != nil {
		return fmt.Errorf("init file content provider failed: %v", err)
	}
	p.AddListener(contentprovider.Listener(listener))
	err = p.Run(signal.FyGlobalContext())
	if err != nil {
		return fmt.Errorf("run file content provider failed for %s: %v", filePath, err)
	}
	return nil
}
