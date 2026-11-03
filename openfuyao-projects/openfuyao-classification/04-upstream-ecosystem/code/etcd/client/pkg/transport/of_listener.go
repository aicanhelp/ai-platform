// Package transport is used to protect the security of data transmission
package transport

import (
	"crypto/tls"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader"
)

func ofConfigureDynamicCertificateForServer(tlsConfig *tls.Config, info TLSInfo) error {
	var tlsInfo = dynamicloader.TLS{
		CertFile:      info.CertFile,
		KeyFile:       info.KeyFile,
		TrustedCAFile: info.TrustedCAFile,
	}
	return dynamicloader.ConfigureDynamicCertificateForServer(tlsConfig, &tlsInfo)
}
