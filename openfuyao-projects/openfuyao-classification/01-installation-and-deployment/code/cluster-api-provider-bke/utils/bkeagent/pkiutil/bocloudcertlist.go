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
	"path/filepath"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type BocloudCert struct {
	Name     string
	CertName string
	KeyName  string
	PkiPath  string
}

type BocloudCertificates []*BocloudCert

func (c *BocloudCertificates) SetPkiPath(pkiPath string) {
	for _, cert := range *c {
		cert.PkiPath = pkiPath
	}
}

func GetBocloudCertList() BocloudCertificates {
	return BocloudCertificates{
		BocloudCertCA(),
		BocloudCertAPIServer(),
		BocloudCertKubeletClient(),

		BocloudCertFrontProxyCA(),
		BocloudCertFrontProxyClient(),

		BocloudCertServiceAccount(),

		BocloudCertEtcdCA(),
		BocloudCertEtcdServer(),
		BocloudCertEtcdPeer(),
		BocloudCertEtcdAPIClient(),
		// no etcd health cert
	}
}

func GetBocloudCertListForEtcd() BocloudCertificates {
	return BocloudCertificates{
		BocloudCertEtcdCA(),
		BocloudCertEtcdServer(),
		BocloudCertEtcdPeer(),
		BocloudCertEtcdAPIClient(),
		// no etcd health cert
	}
}

func GetBocloudCertListWithoutEtcd() BocloudCertificates {
	return BocloudCertificates{
		// CA
		BocloudCertCA(),
		// certificates
		BocloudCertAPIServer(),
		BocloudCertKubeletClient(),
		// SA
		BocloudCertServiceAccount(),
		// Front Proxy certs
		BocloudCertFrontProxyCA(),
		BocloudCertFrontProxyClient(),
	}
}

func GetBocloudEtcdClientCerts() BocloudCertificates {
	return BocloudCertificates{
		BocloudCertEtcdCA(),
		BocloudCertEtcdAPIClient(),
	}
}

func BocloudCertCA() *BocloudCert {
	return &BocloudCert{
		Name:     "ca",
		CertName: "ca.pem",
		KeyName:  "ca-key.pem",
	}
}

func BocloudCertEtcdCA() *BocloudCert {
	return &BocloudCert{
		Name:     "etcd",
		CertName: "ca.pem",
		KeyName:  "ca-key.pem",
	}
}

func BocloudCertEtcdServer() *BocloudCert {
	return &BocloudCert{
		Name:     "etcd-server",
		CertName: "server.pem",
		KeyName:  "server-key.pem",
	}
}

func BocloudCertEtcdPeer() *BocloudCert {
	return &BocloudCert{
		Name:     "etcd-peer",
		CertName: "peer.pem",
		KeyName:  "peer-key.pem",
	}
}

func BocloudCertEtcdAPIClient() *BocloudCert {
	return &BocloudCert{
		Name:     "apiserver-etcd-client",
		CertName: "client.pem",
		KeyName:  "client-key.pem",
	}
}

func BocloudCertKubeletClient() *BocloudCert {
	return &BocloudCert{
		Name:     "apiserver-kubelet-client",
		CertName: "kubelet-client.pem",
		KeyName:  "kubelet-client-key.pem",
	}
}

func BocloudCertAPIServer() *BocloudCert {
	return &BocloudCert{
		Name:     "apiserver",
		CertName: "apiserver.pem",
		KeyName:  "apiserver-key.pem",
	}
}

func BocloudCertFrontProxyCA() *BocloudCert {
	return &BocloudCert{
		Name:     "proxy",
		CertName: "ca.pem",
		KeyName:  "ca-key.pem",
	}
}

func BocloudCertFrontProxyClient() *BocloudCert {
	return &BocloudCert{
		Name:     "front-proxy-client",
		CertName: "apiserver.pem",
		KeyName:  "apiserver-key.pem",
	}
}

func BocloudCertServiceAccount() *BocloudCert {
	return &BocloudCert{
		Name:     "sa",
		CertName: "ca.pem",
		KeyName:  "ca-key.pem",
	}
}

func pathForBocloudCert(cert *BocloudCert) string {
	return filepath.Join(cert.PkiPath, cert.CertName)
}

func pathForBocloudKey(cert *BocloudCert) string {
	return filepath.Join(cert.PkiPath, cert.KeyName)
}

func BocloudCertExists(cert *BocloudCert) error {
	if !utils.Exists(pathForBocloudCert(cert)) {
		return errors.Errorf("certificate %s does not exist", pathForBocloudCert(cert))
	}
	if !utils.Exists(pathForBocloudKey(cert)) {
		return errors.Errorf("key %s does not exist", pathForBocloudKey(cert))
	}
	return nil
}
