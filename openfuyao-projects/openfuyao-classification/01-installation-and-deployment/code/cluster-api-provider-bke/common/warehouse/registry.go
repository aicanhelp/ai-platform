/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package warehouse

import (
	"fmt"
	"os"
	"path"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
)

const (
	script = `
mkdir -p certs
openssl req -newkey rsa:4096 -nodes -sha256 -keyout certs/deploy.bocloud.k8s.key --addext "subjectAltName=DNS:*.bocloud.k8s\
,IP:0.0.0.0" -x509 -days 36500 -out certs/deploy.bocloud.k8s.crt
`
	script2 = `
openssl x509 -noout -text -in deploy.bocloud.k8s.crt
`

	DeployBoCloudK8sCrt = `

`

	registryConf = `
version: 0.1
log:
  fields:
    service: registry
storage:
  delete:
    enabled: true
  cache:
    blobdescriptor: inmemory
  filesystem:
    rootdirectory: /var/lib/registry
#http:
#  addr: 0.0.0.0:5000
#  headers:
#    X-Content-Type-Options: [nosniff]
http:
  addr: 0.0.0.0:443
  tls:
    certificate: /etc/docker/registry/deploy.bocloud.k8s.crt
    key: /etc/docker/registry/deploy.bocloud.k8s.key
health:
  storagedriver:
    enabled: true
    interval: 10s
    threshold: 3
`
	serverCrtFile = "deploy.bocloud.k8s.crt"
	serverKeyFile = "deploy.bocloud.k8s.key"
	clientCrtFile = "/etc/docker/certs.d/deploy.bocloud.k8s:40443/ca.crt"
)

const (
	DirPerm  os.FileMode = 0755 // 目录权限
	FilePerm os.FileMode = 0644 // 文件权限
)

// ensureDir creates the directory if it does not exist.
func ensureDir(dir string) error {
	if utils.Exists(dir) {
		return nil
	}
	return os.MkdirAll(dir, DirPerm)
}

// writeContentIfNotExists writes content to a file under dir if it doesn't already exist.
func writeContentIfNotExists(dir, filename, content string) error {
	filePath := path.Join(dir, filename)
	if utils.Exists(filePath) {
		return nil
	}
	f, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE, FilePerm)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}

// SetClientCertificate writes client certificate for deploy.bocloud.k8s on given port.
func SetClientCertificate(port string) error {
	clientCrt := fmt.Sprintf("/etc/docker/certs.d/deploy.bocloud.k8s:%s", port)
	if err := ensureDir(clientCrt); err != nil {
		return err
	}
	return writeContentIfNotExists(clientCrt, "ca.crt", DeployBoCloudK8sCrt)
}

// SetClientLocalCertificate writes client certificate for localhost on given port.
func SetClientLocalCertificate(port string) error {
	clientCrt := fmt.Sprintf("/etc/docker/certs.d/0.0.0.0:%s", port)
	if err := ensureDir(clientCrt); err != nil {
		return err
	}
	return writeContentIfNotExists(clientCrt, "ca.crt", DeployBoCloudK8sCrt)
}

// SetServerCertificate writes server certificate under certPath (defaults to /etc/docker/registry).
func SetServerCertificate(certPath string) error {
	if certPath == "" {
		certPath = "/etc/docker/registry"
	}
	if err := ensureDir(certPath); err != nil {
		return err
	}
	return writeContentIfNotExists(certPath, serverCrtFile, DeployBoCloudK8sCrt)
}

// SetRegistryConfig writes registry configuration under certPath (defaults to /etc/docker/registry).
func SetRegistryConfig(certPath string) error {
	if certPath == "" {
		certPath = "/etc/docker/registry"
	}
	if err := ensureDir(certPath); err != nil {
		return err
	}
	return writeContentIfNotExists(certPath, "config.yml", registryConf)
}
