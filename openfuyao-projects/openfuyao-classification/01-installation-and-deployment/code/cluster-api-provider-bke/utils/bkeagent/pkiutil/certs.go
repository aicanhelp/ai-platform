/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pkiutil

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
	certutil "k8s.io/client-go/util/cert"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	netutil "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// GenerateCACert  generate CA certificates and save to file
// crt and key will be saved to pkiPath/ca.crt and pkiPath/ca.key
func GenerateCACert(certSpec *BKECert) error {
	if certSpec.CAName != "" {
		return errors.New("this func only support generate CA cert")
	}
	// if the cert and key exists, skip generate
	if isCertAndKeyExists(certSpec) {
		log.Infof("use exists CA cert and key %q", certSpec.Name)
		return nil
	}
	caCert, caKey, err := NewCertificateAuthority(certSpec)
	if err != nil {
		return err
	}
	log.Infof("generate CA cert and key for %q", certSpec.Name)
	if err := WriteCertAndKey(certSpec, caCert, caKey); err != nil {
		return err
	}
	return nil
}

// GenerateRSACert generate RSA certificates and save to file
// pub and key will be saved to pkiPath/pub and pkiPath/key
func GenerateRSACert(certSpec *BKECert) error {
	// if the pub and key exists, skip generate
	if isPubAndKeyExists(certSpec) {
		log.Infof("RSA key and pub exists %q", certSpec.Name)
		return nil
	}
	pub, key, err := NewRSACertAndKey(certSpec)
	if err != nil {
		return err
	}
	log.Infof("generate RSA key and pub for %q", certSpec.Name)
	if err := writePubAndKey(certSpec, nil, pub, key, true); err != nil {
		return err
	}
	return nil
}

// GenerateCertWithCA generates a new certificate signed by the given CA
// crt and key will be saved to pkiPath/crt and pkiPath/key
func GenerateCertWithCA(certSpec *BKECert, caCertSpec *BKECert) error {
	caCrt, caKey, err := loadCACertificateAuthority(caCertSpec)
	if err != nil {
		return err
	}
	cert, key, err := NewCertAndKey(certSpec, caCrt, caKey)
	if err != nil {
		return err
	}
	log.Infof("generate cert and key for %q", certSpec.Name)
	if err := WriteCertAndKey(certSpec, cert, key); err != nil {
		return err
	}

	return nil
}

// NewRSACertAndKey creates a new RSA certificate and key
func NewRSACertAndKey(certSpec *BKECert) (*rsa.PublicKey, *rsa.PrivateKey, error) {
	key, err := newPrivateKey()
	if err != nil {
		return nil, nil, err
	}
	return &key.PublicKey, key, nil
}

// NewCertificateAuthority generates a new certificate authority.
func NewCertificateAuthority(cert *BKECert) (*x509.Certificate, *rsa.PrivateKey, error) {
	key, err := newPrivateKey()
	if err != nil {
		return nil, nil, err
	}

	crt, err := newSelfSignedCACert(cert, key)
	if err != nil {
		return nil, nil, err
	}

	return crt, key, nil
}

// newPrivateKey creates a new private key
func newPrivateKey() (*rsa.PrivateKey, error) {
	return rsa.GenerateKey(rand.Reader, DefaultRSAKeySize)
}

// newSignedCert creates a new signed certificate using the given CA certificate and key
func newSignedCert(certSpec *BKECert, key *rsa.PrivateKey, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, error) {
	serial, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64))
	if err != nil {
		return nil, err
	}
	keyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature

	if certSpec.IsCA {
		keyUsage |= x509.KeyUsageCertSign
	}

	if certSpec.Name == "ca" {
		keyUsage |= x509.KeyUsageCRLSign
	}

	// 合并certSpec.Config.BaseUsages中的值到keyUsage
	for _, usage := range certSpec.Config.BaseUsages {
		keyUsage |= usage
	}

	var notBefore, notAfter time.Time
	if certSpec.Config.Validity != 0 {
		// 如果 Validity 不为空，使用 Validity 来设置证书有效期
		now := time.Now().UTC()
		notBefore = now.Add(-time.Hour * utils.OneMonthHour)
		notAfter = now.Add(certSpec.Config.Validity)
	} else {
		// 如果 Validity 为空，使用原来的逻辑（CA证书的有效期）
		notBefore = caCert.NotBefore
		notAfter = caCert.NotAfter
	}

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName:         certSpec.Config.CommonName,
			Organization:       certSpec.Config.Organization,
			Country:            certSpec.Config.Country,
			Province:           certSpec.Config.Province,
			Locality:           certSpec.Config.Locality,
			OrganizationalUnit: certSpec.Config.OrganizationalUnit,
		},
		DNSNames:              certSpec.Config.AltNames.DNSNames,
		IPAddresses:           certSpec.Config.AltNames.IPs,
		SerialNumber:          serial,
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              keyUsage,
		ExtKeyUsage:           certSpec.Config.Usages,
		BasicConstraintsValid: true,
		IsCA:                  certSpec.IsCA,
	}
	certDERBytes, err := x509.CreateCertificate(rand.Reader, &certTmpl, caCert, key.Public(), caKey)
	if err != nil {
		return nil, err
	}
	return x509.ParseCertificate(certDERBytes)
}

// newSelfSignedCACert creates a CA certificate.
func newSelfSignedCACert(certSpec *BKECert, key *rsa.PrivateKey) (*x509.Certificate, error) {
	now := time.Now().UTC()

	tmpl := x509.Certificate{
		SerialNumber: new(big.Int).SetInt64(0),
		Subject: pkix.Name{
			CommonName:   certSpec.Config.CommonName,
			Organization: certSpec.Config.Organization,
		},
		// NotBefore 在当前时间基础上减去一年
		NotBefore:             now.Add(-utils.OneYearHour * time.Hour),
		NotAfter:              now.Add(certSpec.Config.Validity), // 100 years
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	certDERBytes, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, key.Public(), key)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create self signed CA certificate: %+v", tmpl)
	}

	return x509.ParseCertificate(certDERBytes)
}

// WriteCertAndKey writes a certificate and key to disk
func WriteCertAndKey(certSpec *BKECert, cert *x509.Certificate, key *rsa.PrivateKey) error {
	if err := writeKey(certSpec, key); err != nil {
		return errors.Wrapf(err, "couldn't write key %s", certSpec.BaseName)
	}
	if err := writeCert(certSpec, cert); err != nil {
		return errors.Wrapf(err, "couldn't write cert %s", certSpec.BaseName)
	}
	if HasServerAuth(cert) {
		log.Infof("%q serving cert is signed for DNS names %v and IPs %v", certSpec.BaseName, cert.DNSNames, cert.IPAddresses)
	}
	return nil
}

func writePubAndKey(certSpec *BKECert, certificateKey []*x509.Certificate, pub *rsa.PublicKey, key *rsa.PrivateKey, isRSACertificate bool) error {
	if err := writeKey(certSpec, key); err != nil {
		return errors.Wrapf(err, "couldn't write key %s", certSpec.BaseName)
	}
	if isRSACertificate {
		if err := writePubKey(certSpec, pub); err != nil {
			return errors.Wrapf(err, "couldn't write pub %s", certSpec.BaseName)
		}
	} else {
		if err := writeCertificatePubKey(certSpec, certificateKey[0]); err != nil {
			return errors.Wrapf(err, "couldn't write certificate pub %s", certSpec.BaseName)
		}
	}
	return nil
}

// writeKey writes a private key to disk
func writeKey(certSpec *BKECert, key *rsa.PrivateKey) error {
	privateKeyPath := pathForKey(certSpec)
	encoded := EncodeKeyToPEM(key)
	if err := os.MkdirAll(filepath.Dir(privateKeyPath), os.FileMode(utils.RwxRxRx)); err != nil {
		return err
	}
	return os.WriteFile(privateKeyPath, encoded, utils.RwRR)
}

func EncodeKeyToPEM(key *rsa.PrivateKey) []byte {
	block := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	}
	return pem.EncodeToMemory(block)
}

// writeCert writes a certificate to disk
func writeCert(certSpec *BKECert, cert *x509.Certificate) error {
	certPath := pathForCert(certSpec)
	encoded := EncodeCertToPEM(cert)
	if err := os.MkdirAll(certSpec.PkiPath, utils.RwxRxRx); err != nil {
		return err
	}
	return os.WriteFile(certPath, encoded, utils.RwRR)
}

func EncodeCertToPEM(cert *x509.Certificate) []byte {
	block := &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert.Raw,
	}
	return pem.EncodeToMemory(block)
}

func EncodePublicKeyToPEM(key *rsa.PublicKey) ([]byte, error) {
	b, err := x509.MarshalPKIXPublicKey(key)
	if err != nil {
		return nil, errors.Wrap(err, "failed to marshal public key")
	}
	block := &pem.Block{
		Type:  "PUBLIC KEY",
		Bytes: b,
	}
	return pem.EncodeToMemory(block), nil
}

// writeCertificatePubKey writes a certificate public key to disk
func writeCertificatePubKey(certSpec *BKECert, cert *x509.Certificate) error {
	publicKeyPath := pathForPublicKey(certSpec)
	encoded := EncodeCertToPEM(cert)
	if err := os.MkdirAll(certSpec.PkiPath, utils.RwxRxRx); err != nil {
		return err
	}
	return os.WriteFile(publicKeyPath, encoded, utils.RwRR)
}

// writePubKey writes a public key to disk
func writePubKey(certSpec *BKECert, key *rsa.PublicKey) error {
	publicKeyPath := pathForPublicKey(certSpec)
	encoded, err := EncodePublicKeyToPEM(key)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(certSpec.PkiPath, utils.RwxRxRx); err != nil {
		return err
	}
	return os.WriteFile(publicKeyPath, encoded, utils.RwRR)
}

// loadCACertificateAuthority loads a CA certificate and key from disk
func loadCACertificateAuthority(caCert *BKECert) (*x509.Certificate, *rsa.PrivateKey, error) {
	privateKeyPath := pathForKey(caCert)
	certPath := pathForCert(caCert)

	// load cert
	certPemBlock, err := os.ReadFile(certPath)
	if err != nil {
		return nil, nil, err
	}
	certs, err := ParseCertsPEM(certPemBlock)
	if err != nil {
		return nil, nil, err
	}

	// load key
	keyPemBlock, err := os.ReadFile(privateKeyPath)
	if err != nil {
		return nil, nil, err
	}
	key, err := ParsePrivateKeyPEM(keyPemBlock)
	if err != nil {
		return nil, nil, err
	}
	// Make sure the loaded CA cert actually is a CA
	if !certs[0].IsCA {
		return nil, nil, errors.Errorf("%s certificate is not a certificate authority", caCert.BaseName)
	}

	return certs[0], key, nil
}

// ParseCertsPEM returns the x509.Certificates contained in the given PEM-encoded byte array
// Returns an error if a certificate could not be parsed, or if the data does not contain any certificates
func ParseCertsPEM(pemCerts []byte) ([]*x509.Certificate, error) {
	ok := false
	certs := make([]*x509.Certificate, 0)
	for len(pemCerts) > 0 {
		var block *pem.Block
		block, pemCerts = pem.Decode(pemCerts)
		if block == nil {
			break
		}

		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return certs, err
		}

		certs = append(certs, cert)
		ok = true
	}

	if !ok {
		return certs, errors.New("data does not contain any valid certificates")
	}
	return certs, nil
}

// ParsePrivateKeyPEM returns a private key parsed from a PEM block in the supplied data.
// Recognizes PEM blocks for "RSA PRIVATE KEY"
func ParsePrivateKeyPEM(keyData []byte) (*rsa.PrivateKey, error) {
	var privateKeyPemBlock *pem.Block
	for {
		privateKeyPemBlock, keyData = pem.Decode(keyData)
		if privateKeyPemBlock == nil {
			break
		}
		if privateKeyPemBlock.Type != "RSA PRIVATE KEY" {
			continue
		}
		// RSA Private Key in PKCS#1 format
		if key, err := x509.ParsePKCS1PrivateKey(privateKeyPemBlock.Bytes); err == nil {
			return key, nil
		}
	}

	return nil, fmt.Errorf("data does not contain a valid RSA private key")
}

// ParsePublicKeyPEM returns a public key parsed from a PEM block in the supplied data.
func ParsePublicKeyPEM(keyData []byte) (*rsa.PublicKey, error) {
	var publicKeyPemBlock *pem.Block
	for {
		publicKeyPemBlock, keyData = pem.Decode(keyData)
		if publicKeyPemBlock == nil {
			break
		}
		if publicKeyPemBlock.Type != "PUBLIC KEY" {
			continue
		}

		var pubKey *rsa.PublicKey
		var ok bool
		if key, err := x509.ParsePKIXPublicKey(publicKeyPemBlock.Bytes); err == nil {
			if pubKey, ok = key.(*rsa.PublicKey); !ok {
				return nil, fmt.Errorf("data doesn't contain valid RSA Public Key")
			}
			return pubKey, nil
		}
	}
	return nil, fmt.Errorf("data does not contain a valid RSA Public Key")
}

// NewCertAndKey creates new certificate and key by passing the certificate authority certificate and key
func NewCertAndKey(certSpec *BKECert, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {

	key, err := newPrivateKey()
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to create private key")
	}

	cert, err := newSignedCert(certSpec, key, caCert, caKey)
	if err != nil {
		return nil, nil, errors.Wrap(err, "unable to sign certificate")
	}

	return cert, key, nil
}

// HasServerAuth returns true if the given certificate is a ServerAuth
func HasServerAuth(cert *x509.Certificate) bool {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageServerAuth {
			return true
		}
	}
	return false
}

// isCertAndKeyExists returns true if the cert and key exists
func isCertAndKeyExists(cert *BKECert) bool {
	return utils.Exists(pathForCert(cert)) && utils.Exists(pathForKey(cert))
}

// isPubAndKeyExists returns true if the public key and key exists
func isPubAndKeyExists(cert *BKECert) bool {
	return utils.Exists(pathForPublicKey(cert)) && utils.Exists(pathForKey(cert))
}

func pathForCert(cert *BKECert) string {
	if cert.PkiPath == "" {
		cert.PkiPath = GetDefaultPkiPath()
	}
	return filepath.Join(cert.PkiPath, fmt.Sprintf("%s.crt", cert.BaseName))
}

func pathForKey(cert *BKECert) string {
	if cert.PkiPath == "" {
		cert.PkiPath = GetDefaultPkiPath()
	}
	return filepath.Join(cert.PkiPath, fmt.Sprintf("%s.key", cert.BaseName))
}

func pathForPublicKey(cert *BKECert) string {
	if cert.PkiPath == "" {
		cert.PkiPath = GetDefaultPkiPath()
	}
	return filepath.Join(cert.PkiPath, fmt.Sprintf("%s.pub", cert.BaseName))
}

// CertExists returns true if the cert key and its associated certificate exists
func CertExists(cert *BKECert) error {

	crtPath := pathForCert(cert)
	if !utils.Exists(crtPath) {
		return errors.Errorf("cert %q certificate file does not exist at path %q", cert.Name, crtPath)
	}

	keyPath := pathForKey(cert)
	if !utils.Exists(keyPath) {
		return errors.Errorf("cert %q key file does not exist at path %q", cert.Name, keyPath)
	}

	return nil
}

// GetMasterNodeAltNames returns the AltNames object for controller-manager and scheduler certificates
// It includes all master node IPs, similar to GetAPIServerCertAltNamesFromBkeConfig but without service IPs
func GetMasterNodeAltNames(bkeConfig *bkev1beta1.BKEConfig) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}

	// append all the master node IPs to altnames.IPs
	nodesData, err := cluster.GetNodesData(bkeConfig.Cluster.ContainerdConfigRef.Namespace, bkeConfig.Cluster.ContainerdConfigRef.Name)
	if err != nil {
		return nil, err
	}
	return GetMasterNodeAltNamesWithNodes(bkenode.Nodes(nodesData))
}

// GetMasterNodeAltNamesWithNodes returns the AltNames for controller-manager and scheduler certificates using provided nodes
func GetMasterNodeAltNamesWithNodes(nodes bkenode.Nodes) (*certutil.AltNames, error) {
	altNames := &certutil.AltNames{}
	for _, node := range nodes.Master() {
		altNames.IPs = append(altNames.IPs, net.ParseIP(node.IP))
		altNames.DNSNames = append(altNames.DNSNames, node.Hostname)
	}

	// remove repeat ips in altnames.IPs
	altNames.IPs = netutil.RemoveRepIP(altNames.IPs)
	altNames.DNSNames = netutil.RemoveRepDomain(altNames.DNSNames)
	return altNames, nil
}
