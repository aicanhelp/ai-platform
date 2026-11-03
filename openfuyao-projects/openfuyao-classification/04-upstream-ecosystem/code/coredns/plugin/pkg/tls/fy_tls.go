// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package tls

import (
	"crypto/tls"
	"net/http"

	"github.com/coredns/coredns/fypkg/dynamicloader"
)

// FyTLSArgs hold tls params
type FyTLSArgs struct {
	ServerCertPath   string
	ServerKeyPath    string
	ClientCACertPath string
	CRLPath          string
	// only support NoClientCert and RequireAndVerifyClientCert
	ClientAuth tls.ClientAuthType
}

func NewFyTLSArgs() *FyTLSArgs {
	return &FyTLSArgs{ClientAuth: tls.NoClientCert}
}

// FYSetTLSConfigForServer create tls config for server if tls params exist
func (c *FyTLSArgs) FYSetTLSConfigForServer(server *http.Server) error {
	if c == nil || len(c.ServerCertPath) == 0 {
		return nil
	}
	tlsConfig, err := c.fyConvertToTLSConfig()
	if err != nil {
		return err
	}
	server.TLSConfig = tlsConfig
	return nil
}

// fyConvertToTLSConfig parse cert file, create and return tls config
func (c *FyTLSArgs) fyConvertToTLSConfig() (*tls.Config, error) {
	tlsConfig := &tls.Config{
		ClientAuth: c.ClientAuth,
		MinVersion: tls.VersionTLS12,
		MaxVersion: tls.VersionTLS13,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		},
		NextProtos: []string{"h2", "http/1.1"},
	}

	// set crl verify call back func if needed
	SetVerifyCRLCallback(c.CRLPath, tlsConfig)

	// configure dynamic certificates
	tlsInfo := &dynamicloader.TLS{
		CertFile:      c.ServerCertPath,
		KeyFile:       c.ServerKeyPath,
		TrustedCAFile: c.ClientCACertPath,
	}
	if err := dynamicloader.ConfigureDynamicCertificateForServer(tlsConfig, tlsInfo); err != nil {
		return nil, err
	}

	return tlsConfig, nil
}
