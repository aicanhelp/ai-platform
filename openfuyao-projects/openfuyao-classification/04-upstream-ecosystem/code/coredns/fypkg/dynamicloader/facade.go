// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package dynamicloader provider facade of dynamic loader others
package dynamicloader

import (
	"context"
	"crypto/tls"
	"fmt"

	"github.com/coredns/coredns/fypkg/dynamicloader/certificate"
	"github.com/coredns/coredns/plugin/pkg/log"
)

type TLS struct {
	CertFile      string
	KeyFile       string
	TrustedCAFile string
}

func ConfigureDynamicCertificateForServer(tlsConfig *tls.Config, tlsInfo *TLS) error {
	log.Infof("configure dynamic certificate for server with : key file %s", tlsInfo.KeyFile)
	certificateProvider := certificate.NewDynamicCertificateProvider(tlsConfig.Clone(), tlsInfo.CertFile, tlsInfo.KeyFile, tlsInfo.TrustedCAFile)
	err := certificateProvider.RunOnce()
	if err != nil {
		return fmt.Errorf("init dynamic server certificate provider failed: %v", err)
	}

	// we don't need to cancel the goroutine, but context must be cancelable
	ctx, _ := context.WithCancel(context.Background())
	err = certificateProvider.Run(ctx)
	if err != nil {
		return fmt.Errorf("run dynamic server certificate provider failed: %v", err)
	}

	tlsConfig.GetConfigForClient = certificateProvider.GetConfigForClient

	// this is not used when tls shaking hands, it's used to let http.Server.ServerTLS to use dynamic certificates
	tlsConfig.GetCertificate = certificateProvider.GetCertificate
	return nil
}
