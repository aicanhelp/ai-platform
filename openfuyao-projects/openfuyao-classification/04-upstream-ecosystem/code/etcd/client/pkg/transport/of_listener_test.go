package transport

import (
	"crypto/tls"
	"testing"
)

func TestOfConfigureDynamicCertificateForServer(t *testing.T) {
	// prepare parameters
	tlsconfig := &tls.Config{}
	tlsInfo := TLSInfo{}

	// call target method
	if err := ofConfigureDynamicCertificateForServer(tlsconfig, tlsInfo); err != nil {
		t.Fatal("expected no error, but got", err)
	}
}
