// Package dynamicloader provide facade of dynamic loader to others
package dynamicloader

import (
	"context"
	"crypto/tls"
	"fmt"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader/certificate"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/featuregate"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/log"
)

// TLS 定义 TLS 配置相关的文件路径信息
type TLS struct {
	// CertFile 指定证书文件的路径
	CertFile string
	// KeyFile 指定私钥文件的路径
	KeyFile string
	// TrustedCAFile 指定受信任 CA 证书文件的路径
	TrustedCAFile string
}

// ConfigureDynamicCertificateForServer 为服务端配置动态证书加载功能
func ConfigureDynamicCertificateForServer(tlsConfig *tls.Config, tlsInfo *TLS) error {
	if !featuregate.CertHotLoading {
		return nil
	}
	log.GetLogger().Info("configure dynamic certificate for server with key file" + tlsInfo.KeyFile)
	certificateProvider := certificate.NewDynamicCertificateProvider(tlsConfig.Clone(), tlsInfo.CertFile,
		tlsInfo.KeyFile, tlsInfo.TrustedCAFile)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic server certificate provider failed: %v", err)
	}

	// we don't need to cancel the goroutine, but context must be cancelable
	err = certificateProvider.Run(context.Background())
	if err != nil {
		return fmt.Errorf("run dynamic server certificate provider failed: %v", err)
	}

	tlsConfig.GetConfigForClient = certificateProvider.GetConfigForClient
	return nil
}

// ConfigureDynamicCertificateForClient 为客户端配置动态证书加载功能
func ConfigureDynamicCertificateForClient(tlsConfig *tls.Config, certFile string, keyFile string) error {
	if !featuregate.CertHotLoading {
		return nil
	}
	log.GetLogger().Info("configure dynamic certificate for client with key file" + keyFile)
	certificateProvider := certificate.GetDynamicClientCertificateProviderFromCache(certFile, keyFile, "")
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic client certificate provider failed: %v", err)
	}

	// we don't need to cancel the goroutine, but context must be cancelable
	err = certificateProvider.Run(context.Background())
	if err != nil {
		return fmt.Errorf("run dynamic client certificate provider failed: %v", err)
	}

	tlsConfig.GetClientCertificate = certificateProvider.GetClientCertificate
	return nil
}

// UpdateCADynamicallyForClient 为客户端配置动态 CA 证书更新功能
func UpdateCADynamicallyForClient(tlsConfig *tls.Config, trustedCAFile string) error {
	if !featuregate.CertHotLoading {
		return nil
	}
	log.GetLogger().Info("support updating ca certificates dynamically for client with ca file" + trustedCAFile)
	certificateProvider := certificate.GetDynamicClientCertificateProviderFromCache("", "", trustedCAFile)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic client certificate provider failed: %v", err)
	}

	tlsConfig.RootCAs = certificateProvider.GetCACertificate().Clone()
	certificateProvider.UpdateCADynamically(tlsConfig)

	// we don't need to cancel the goroutine, but context must be cancelable
	err = certificateProvider.Run(context.Background())
	if err != nil {
		return fmt.Errorf("run dynamic client certificate provider failed: %v", err)
	}
	return nil
}
