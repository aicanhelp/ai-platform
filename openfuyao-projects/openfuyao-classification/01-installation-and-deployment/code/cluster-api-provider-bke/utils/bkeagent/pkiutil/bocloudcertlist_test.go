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

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func TestBocloudCertStruct(t *testing.T) {
	cert := &BocloudCert{
		Name:     "test-cert",
		CertName: "test.pem",
		KeyName:  "test-key.pem",
		PkiPath:  "/pki",
	}

	assert.Equal(t, "test-cert", cert.Name)
	assert.Equal(t, "test.pem", cert.CertName)
	assert.Equal(t, "test-key.pem", cert.KeyName)
	assert.Equal(t, "/pki", cert.PkiPath)
}

func TestBocloudCertificatesSetPkiPath(t *testing.T) {
	certs := BocloudCertificates{
		{Name: "cert1", PkiPath: ""},
		{Name: "cert2", PkiPath: ""},
		{Name: "cert3", PkiPath: ""},
	}

	certs.SetPkiPath("/etc/kubernetes/pki")

	for _, cert := range certs {
		assert.Equal(t, "/etc/kubernetes/pki", cert.PkiPath)
	}
}

func TestBocloudCertificatesSetPkiPathEmpty(t *testing.T) {
	certs := BocloudCertificates{}

	certs.SetPkiPath("/test/path")

	assert.Empty(t, certs)
}

func TestGetBocloudCertList(t *testing.T) {
	certs := GetBocloudCertList()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 10)
}

func TestGetBocloudCertListForEtcd(t *testing.T) {
	certs := GetBocloudCertListForEtcd()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 4)
}

func TestGetBocloudCertListWithoutEtcd(t *testing.T) {
	certs := GetBocloudCertListWithoutEtcd()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 6)
}

func TestGetBocloudEtcdClientCerts(t *testing.T) {
	certs := GetBocloudEtcdClientCerts()

	assert.NotEmpty(t, certs)
	assert.Len(t, certs, 2)
}

func TestBocloudCertCA(t *testing.T) {
	cert := BocloudCertCA()

	assert.Equal(t, "ca", cert.Name)
	assert.Equal(t, "ca.pem", cert.CertName)
	assert.Equal(t, "ca-key.pem", cert.KeyName)
}

func TestBocloudCertEtcdCA(t *testing.T) {
	cert := BocloudCertEtcdCA()

	assert.Equal(t, "etcd", cert.Name)
	assert.Equal(t, "ca.pem", cert.CertName)
	assert.Equal(t, "ca-key.pem", cert.KeyName)
}

func TestBocloudCertEtcdServer(t *testing.T) {
	cert := BocloudCertEtcdServer()

	assert.Equal(t, "etcd-server", cert.Name)
	assert.Equal(t, "server.pem", cert.CertName)
	assert.Equal(t, "server-key.pem", cert.KeyName)
}

func TestBocloudCertEtcdPeer(t *testing.T) {
	cert := BocloudCertEtcdPeer()

	assert.Equal(t, "etcd-peer", cert.Name)
	assert.Equal(t, "peer.pem", cert.CertName)
	assert.Equal(t, "peer-key.pem", cert.KeyName)
}

func TestBocloudCertEtcdAPIClient(t *testing.T) {
	cert := BocloudCertEtcdAPIClient()

	assert.Equal(t, "apiserver-etcd-client", cert.Name)
	assert.Equal(t, "client.pem", cert.CertName)
	assert.Equal(t, "client-key.pem", cert.KeyName)
}

func TestBocloudCertKubeletClient(t *testing.T) {
	cert := BocloudCertKubeletClient()

	assert.Equal(t, "apiserver-kubelet-client", cert.Name)
	assert.Equal(t, "kubelet-client.pem", cert.CertName)
	assert.Equal(t, "kubelet-client-key.pem", cert.KeyName)
}

func TestBocloudCertAPIServer(t *testing.T) {
	cert := BocloudCertAPIServer()

	assert.Equal(t, "apiserver", cert.Name)
	assert.Equal(t, "apiserver.pem", cert.CertName)
	assert.Equal(t, "apiserver-key.pem", cert.KeyName)
}

func TestBocloudCertFrontProxyCA(t *testing.T) {
	cert := BocloudCertFrontProxyCA()

	assert.Equal(t, "proxy", cert.Name)
	assert.Equal(t, "ca.pem", cert.CertName)
	assert.Equal(t, "ca-key.pem", cert.KeyName)
}

func TestBocloudCertFrontProxyClient(t *testing.T) {
	cert := BocloudCertFrontProxyClient()

	assert.Equal(t, "front-proxy-client", cert.Name)
	assert.Equal(t, "apiserver.pem", cert.CertName)
	assert.Equal(t, "apiserver-key.pem", cert.KeyName)
}

func TestBocloudCertServiceAccount(t *testing.T) {
	cert := BocloudCertServiceAccount()

	assert.Equal(t, "sa", cert.Name)
	assert.Equal(t, "ca.pem", cert.CertName)
	assert.Equal(t, "ca-key.pem", cert.KeyName)
}

func TestPathForBocloudCert(t *testing.T) {
	cert := &BocloudCert{
		Name:     "test",
		CertName: "test.pem",
		PkiPath:  "/pki",
	}

	result := pathForBocloudCert(cert)
	assert.True(t, result == "/pki/test.pem" || result == "\\pki\\test.pem")
}

func TestPathForBocloudKey(t *testing.T) {
	cert := &BocloudCert{
		Name:    "test",
		KeyName: "test-key.pem",
		PkiPath: "/pki",
	}

	result := pathForBocloudKey(cert)
	assert.True(t, result == "/pki/test-key.pem" || result == "\\pki\\test-key.pem")
}

func TestBocloudCertExistsWithExistingCert(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	cert := &BocloudCert{
		Name:     "test",
		CertName: "test.pem",
		KeyName:  "test-key.pem",
		PkiPath:  "/pki",
	}

	err := BocloudCertExists(cert)
	assert.NoError(t, err)
}

func TestBocloudCertExistsWithMissingCert(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	cert := &BocloudCert{
		Name:     "test",
		CertName: "test.pem",
		KeyName:  "test-key.pem",
		PkiPath:  "/pki",
	}

	err := BocloudCertExists(cert)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "certificate")
}

func TestBocloudCertExistsWithMissingKey(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	callCount := 0
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		callCount++
		if callCount == 1 {
			return true
		}
		return false
	})

	cert := &BocloudCert{
		Name:     "test",
		CertName: "test.pem",
		KeyName:  "test-key.pem",
		PkiPath:  "/pki",
	}

	err := BocloudCertExists(cert)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "key")
}

func TestBocloudCertificatesLength(t *testing.T) {
	certs := GetBocloudCertList()
	assert.Equal(t, 10, len(certs))

	certs = GetBocloudCertListForEtcd()
	assert.Equal(t, 4, len(certs))

	certs = GetBocloudCertListWithoutEtcd()
	assert.Equal(t, 6, len(certs))

	certs = GetBocloudEtcdClientCerts()
	assert.Equal(t, 2, len(certs))
}
