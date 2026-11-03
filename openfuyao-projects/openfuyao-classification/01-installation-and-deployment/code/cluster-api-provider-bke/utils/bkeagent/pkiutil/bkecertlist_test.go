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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

const (
	numZero = 0
	numFour = 4
	numTen  = 10
)

func TestBKECertStruct(t *testing.T) {
	cert := &BKECert{
		Name:     "test-cert",
		LongName: "Test Certificate",
		BaseName: "test",
		CAName:   "ca",
		PkiPath:  "/pki",
		IsCA:     false,
		Config:   CertConfig{},
	}

	assert.Equal(t, "test-cert", cert.Name)
	assert.Equal(t, "Test Certificate", cert.LongName)
	assert.Equal(t, "test", cert.BaseName)
	assert.Equal(t, "ca", cert.CAName)
	assert.Equal(t, "/pki", cert.PkiPath)
	assert.False(t, cert.IsCA)
}

func TestCertConfigStruct(t *testing.T) {
	config := CertConfig{
		Validity:           365 * time.Hour,
		PublicKeyAlgorithm: 0,
		KeySize:            2048,
		Country:            []string{"CN"},
		Province:           []string{"Beijing"},
		Locality:           []string{"Beijing"},
		OrganizationalUnit: []string{"IT"},
	}

	assert.Equal(t, 365*time.Hour, config.Validity)
	assert.Equal(t, 2048, config.KeySize)
	assert.Equal(t, []string{"CN"}, config.Country)
	assert.Equal(t, []string{"Beijing"}, config.Province)
}

func TestCertificatesSetPkiPath(t *testing.T) {
	certs := Certificates{
		{Name: "cert1", PkiPath: ""},
		{Name: "cert2", PkiPath: ""},
		{Name: "cert3", PkiPath: ""},
	}

	certs.SetPkiPath("/etc/kubernetes/pki")

	for _, cert := range certs {
		assert.Equal(t, "/etc/kubernetes/pki", cert.PkiPath)
	}
}

func TestCertificatesSetPkiPathEmpty(t *testing.T) {
	certs := Certificates{}

	certs.SetPkiPath("/test/path")

	assert.Empty(t, certs)
}

func TestCertificatesExport(t *testing.T) {
	certs := Certificates{
		{BaseName: "test-cert", PkiPath: ""},
		{BaseName: ServiceAccountKeyBaseName, PkiPath: ""},
	}

	files := certs.Export("/pki")

	assert.NotEmpty(t, files)
	assert.Equal(t, numFour, len(files))
}

func TestCertificatesExportEmpty(t *testing.T) {
	certs := Certificates{}

	files := certs.Export("/pki")

	assert.Empty(t, files)
}

func TestGetDefaultCertList(t *testing.T) {
	certs := GetDefaultCertList()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 10)
}

func TestGetUserCustomCerts(t *testing.T) {
	certs := GetUserCustomCerts()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, numTen)
}

func TestGetTargetClusterCertList(t *testing.T) {
	certs := GetTargetClusterCertList()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 11)
}

func TestGetCertsWithoutCA(t *testing.T) {
	certs := GetCertsWithoutCA()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 7)
}

func TestGetCACerts(t *testing.T) {
	certs := GetCACerts()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 3)
}

func TestGetCertsWithoutEtcd(t *testing.T) {
	certs := GetCertsWithoutEtcd()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 6)
}

func TestGetEtcdCerts(t *testing.T) {
	certs := GetEtcdCerts()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 5)
}

func TestBKECertRootCA(t *testing.T) {
	cert := BKECertRootCA()

	assert.Equal(t, "ca", cert.Name)
	assert.Equal(t, "Root Certificate Authority", cert.LongName)
	assert.Equal(t, CACertAndKeyBaseName, cert.BaseName)
}

func TestBKECertAPIServer(t *testing.T) {
	cert := BKECertAPIServer()

	assert.Equal(t, "apiserver", cert.Name)
	assert.Equal(t, APIServerCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "ca", cert.CAName)
	assert.NotEmpty(t, cert.Config.AltNames.DNSNames)
	assert.NotEmpty(t, cert.Config.AltNames.IPs)
}

func TestBKECertKubeletClient(t *testing.T) {
	cert := BKECertKubeletClient()

	assert.Equal(t, "apiserver-kubelet-client", cert.Name)
	assert.Equal(t, APIServerKubeletClientCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "ca", cert.CAName)
}

func TestBKECertFrontProxyCA(t *testing.T) {
	cert := BKECertFrontProxyCA()

	assert.Equal(t, "proxy", cert.Name)
	assert.Equal(t, FrontProxyCACertAndKeyBaseName, cert.BaseName)
}

func TestBKECertFrontProxyClient(t *testing.T) {
	cert := BKECertFrontProxyClient()

	assert.Equal(t, "front-proxy-client", cert.Name)
	assert.Equal(t, FrontProxyClientCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "front-proxy-ca", cert.CAName)
}

func TestBKECertEtcdCA(t *testing.T) {
	cert := BKECertEtcdCA()

	assert.Equal(t, "etcd", cert.Name)
	assert.Equal(t, EtcdCACertAndKeyBaseName, cert.BaseName)
}

func TestBKECertEtcdServer(t *testing.T) {
	cert := BKECertEtcdServer()

	assert.Equal(t, "etcd-server", cert.Name)
	assert.Equal(t, EtcdServerCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "etcd-ca", cert.CAName)
}

func TestBKECertEtcdPeer(t *testing.T) {
	cert := BKECertEtcdPeer()

	assert.Equal(t, "etcd-peer", cert.Name)
	assert.Equal(t, EtcdPeerCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "etcd-ca", cert.CAName)
}

func TestBKECertEtcdHealthcheck(t *testing.T) {
	cert := BKECertEtcdHealthcheck()

	assert.Equal(t, "etcd-healthcheck-client", cert.Name)
	assert.Equal(t, EtcdHealthcheckClientCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "etcd-ca", cert.CAName)
}

func TestBKECertEtcdAPIClient(t *testing.T) {
	cert := BKECertEtcdAPIClient()

	assert.Equal(t, "apiserver-etcd-client", cert.Name)
	assert.Equal(t, APIServerEtcdClientCertAndKeyBaseName, cert.BaseName)
	assert.Equal(t, "etcd-ca", cert.CAName)
}

func TestBKECertServiceAccount(t *testing.T) {
	cert := BKECertServiceAccount()

	assert.Equal(t, "sa", cert.Name)
	assert.Equal(t, ServiceAccountKeyBaseName, cert.BaseName)
}

func TestGetEtcdAltNames(t *testing.T) {
	altNames := getEtcdAltNames()

	assert.Contains(t, altNames.DNSNames, "localhost")
	assert.NotEmpty(t, altNames.IPs)
}

func TestBKEAdminKubeConfig(t *testing.T) {
	cert := BKEAdminKubeConfig()

	assert.Equal(t, "kubeconfig", cert.Name)
	assert.Equal(t, AdminKubeConfigFileName, cert.BaseName)
	assert.Equal(t, KubernetesDir, cert.PkiPath)
	assert.NotEmpty(t, cert.Config.Usages)
}

func TestBKEKubeletKubeConfig(t *testing.T) {
	cert := BKEKubeletKubeConfig()

	assert.Equal(t, "kubelet", cert.Name)
	assert.Equal(t, KubeletKubeConfigFileName, cert.BaseName)
	assert.Equal(t, KubernetesDir, cert.PkiPath)
	assert.Contains(t, cert.Config.CommonName, "system:node:")
}

func TestBKEControllerKubeConfig(t *testing.T) {
	cert := BKEControllerKubeConfig()

	assert.Equal(t, "controller-manager", cert.Name)
	assert.Equal(t, ControllerManagerKubeConfigFileName, cert.BaseName)
	assert.Equal(t, KubernetesDir, cert.PkiPath)
}

func TestBKESchedulerKubeConfig(t *testing.T) {
	cert := BKESchedulerKubeConfig()

	assert.Equal(t, "scheduler", cert.Name)
	assert.Equal(t, SchedulerKubeConfigFileName, cert.BaseName)
	assert.Equal(t, KubernetesDir, cert.PkiPath)
}

func TestCertificatesLength(t *testing.T) {
	certs := GetDefaultCertList()
	assert.True(t, len(certs) > numZero)

	certs = GetTargetClusterCertList()
	assert.True(t, len(certs) > numZero)

	certs = GetClusterAPICertList()
	assert.True(t, len(certs) > numZero)

	certs = GetCertsWithoutCA()
	assert.True(t, len(certs) > numZero)

	certs = GetCACerts()
	assert.True(t, len(certs) > numZero)

	certs = GetCertsWithoutEtcd()
	assert.True(t, len(certs) > numZero)

	certs = GetEtcdCerts()
	assert.True(t, len(certs) > numZero)

	certs = GetKubeConfigs()
	assert.True(t, len(certs) > numZero)
}
