// Package etcdmain is used to start etcd and configuration file parsing
package etcdmain

import (
	"crypto/tls"
	"fmt"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/dynamicloader"
)

func ofConfigureDynamicCertificateForClient(tlsConfig *tls.Config, cert, key, ca string) error {
	if err := dynamicloader.ConfigureDynamicCertificateForClient(tlsConfig, cert, key); err != nil {
		return fmt.Errorf("failed to support updating keys dynamically: %v", err)
	}
	if err := dynamicloader.UpdateCADynamicallyForClient(tlsConfig, ca); err != nil {
		return fmt.Errorf("failed to support updating CA dynamically: %v", err)
	}
	return nil
}
