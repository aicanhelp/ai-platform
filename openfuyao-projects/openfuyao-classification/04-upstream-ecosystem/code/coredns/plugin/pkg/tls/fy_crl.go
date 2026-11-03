// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package tls provider tls support
package tls

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

const pemType = "X509 CRL"

var pemPrefix = []byte("-----BEGIN X509 CRL")

type safetyLoader struct {
	mutex sync.Mutex

	filename string

	crl    *x509.RevocationList
	modify time.Time
}

// SetVerifyCRLCallback if crl filename and tls.Config was not empty
func SetVerifyCRLCallback(filename string, config *tls.Config) {
	if len(filename) == 0 {
		klog.V(4).InfoS("Skip to verify crl because crl filename was not configured")
		return
	}
	if config == nil {
		klog.ErrorS(errors.New("tls.Config is nil"), "Set crl verify callback failed")
		return
	}
	callBack, err := getVerifyCallBackFromFile(filename)
	if err != nil {
		// There is no need to return when setting tls options VerifyPeerCertificate failed
		klog.ErrorS(err, "Failed to set ctl verify callback", "filename", filename)
		return
	}
	config.VerifyPeerCertificate = callBack
}

// verifyCertRevokeCallBack verify the cert has revoked or not
type verifyCertRevokeCallBack func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error

// parseFromFile parse a revoked cert list from a given crl file
func parseFromFile(file string) (*x509.RevocationList, error) {
	crlBytes, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	if bytes.HasPrefix(crlBytes, pemPrefix) {
		block, _ := pem.Decode(crlBytes)
		if block != nil && block.Type == pemType {
			crlBytes = block.Bytes
		}
	}
	return x509.ParseRevocationList(crlBytes)
}

// hasExpired return whether the revoked cert list has expired or not
func hasExpired(crl *x509.RevocationList) bool {
	return crl.NextUpdate.Before(time.Now().UTC())
}

// revoked return whether the cert SerialNumber has revoked or not
func revoked(sn *big.Int, crl *x509.RevocationList) bool {
	if crl == nil {
		return false
	}
	for _, cert := range crl.RevokedCertificates {
		if cert.SerialNumber.Cmp(sn) == 0 && cert.RevocationTime.Before(time.Now().UTC()) {
			return true
		}
	}
	return false
}

// verifyPeerCertRevoke verify the given cert according to the given cl
func verifyPeerCertRevoke(rawCerts [][]byte, verifiedChains [][]*x509.Certificate, crl *x509.RevocationList) error {
	if len(verifiedChains) > 0 && !checkCertMatchWithCRL(verifiedChains, crl) {
		klog.V(4).InfoS("Skip to verify crl because crl file not match with cert")
		return nil
	}
	for _, rawCert := range rawCerts {
		cert, err := x509.ParseCertificate(rawCert)
		if err != nil {
			continue
		}
		if revoked(cert.SerialNumber, crl) {
			return fmt.Errorf("peer cert[SN:%v] is revoked", cert.SerialNumber)
		}
	}
	for _, certs := range verifiedChains {
		for _, cert := range certs {
			if revoked(cert.SerialNumber, crl) {
				return fmt.Errorf("peer cert[SN:%v] in verified chains is revoked", cert.SerialNumber)
			}
		}
	}
	return nil
}

// checkCertMatchWithCRL check the crl is matching with cert
func checkCertMatchWithCRL(verifiedChains [][]*x509.Certificate, crl *x509.RevocationList) bool {
	for _, verifyCerts := range verifiedChains {
		for _, verifyCert := range verifyCerts {
			if err := crl.CheckSignatureFrom(verifyCert); err == nil {
				return true
			}
		}
	}
	return false
}

// getVerifyCallBackFromFile parse a given crl and return a verifyCertRevokeCallBack func based on the parsed cl
func getVerifyCallBackFromFile(filename string) (verifyCertRevokeCallBack, error) {
	loader := newLoader(filename)
	if _, err := loader.latest(); err != nil {
		return nil, err
	}
	return func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
		crl, err := loader.latest()
		if err != nil {
			klog.ErrorS(err, "Reload crl error")
			crl, _ = loader.cache()
		}
		return verifyPeerCertRevoke(rawCerts, verifiedChains, crl)
	}, nil
}

func newLoader(filename string) *safetyLoader {
	return &safetyLoader{filename: filename}
}

func (l *safetyLoader) latest() (*x509.RevocationList, error) {
	info, err := os.Stat(l.filename)
	if err != nil {
		return nil, fmt.Errorf("get crl file modify time error: %v", err)
	}
	modify := info.ModTime()

	lastCRL, lastModify := l.cache()
	if lastModify.Equal(modify) {
		return lastCRL, nil
	}

	l.mutex.Lock()
	defer l.mutex.Unlock()

	if l.modify.Equal(modify) {
		return l.crl, nil
	}
	l.modify = modify

	crl, err := parseFromFile(l.filename)
	if err != nil {
		return nil, fmt.Errorf("parse crl file error: %v", err)
	}
	if hasExpired(crl) {
		// Continue to use crl which has expired, and not return error.
		klog.Warningf("CRL file has expired, nextUpdate: %s", crl.NextUpdate)
	}
	l.crl = crl
	return crl, nil
}

func (l *safetyLoader) cache() (*x509.RevocationList, time.Time) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	return l.crl, l.modify
}
