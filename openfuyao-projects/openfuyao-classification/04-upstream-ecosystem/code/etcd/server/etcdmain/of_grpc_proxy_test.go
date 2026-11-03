package etcdmain

import (
	"crypto/tls"
	"os"
	"testing"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/featuregate"
)

func TestOfConfigureDynamicCertificateForClient(t *testing.T) {
	certHotLoading := featuregate.CertHotLoading
	featuregate.CertHotLoading = true
	defer func() {
		featuregate.CertHotLoading = certHotLoading
	}()
	// Prepare parameters
	tlsConfig := &tls.Config{}

	certFile, _ := os.CreateTemp("", "testFile-*")
	keyFile, _ := os.CreateTemp("", "testFile-*")
	caFile, _ := os.CreateTemp("", "testFile-*")

	if err := ofConfigureDynamicCertificateForClient(
		tlsConfig, certFile.Name(), keyFile.Name(), caFile.Name(),
	); err != nil {
		t.Fatalf("failed to configure dynamic certificate: %v", err)
	}
}
