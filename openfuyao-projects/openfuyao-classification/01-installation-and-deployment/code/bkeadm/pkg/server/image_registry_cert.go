/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package server

import (
	_ "embed"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

var (
	//go:embed config.yml
	configYml string
	//go:embed generate-registry-certs.sh
	certGen string
)

const (
	serverCrtFile = "deploy.bocloud.k8s.crt"
)

func getCertContent(path string) (string, error) {
	filePath := filepath.Join(path, serverCrtFile)
	// 读取文件内容
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read certificate file: %v", err)
	}
	return string(data), nil
}

func setCommonCert(srcPath, dstPath string) error {
	if !utils.FileExists(dstPath) {
		err := os.MkdirAll(dstPath, utils.DefaultDirPermission)
		if err != nil {
			return err
		}
	}
	if !utils.FileExists(dstPath + "/ca.crt") {
		srcString, err := getCertContent(srcPath)
		if err != nil {
			return err
		}
		err = utils.WriteCommon(dstPath+"/ca.crt", srcString)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetClientCertificate write client certificate
func SetClientCertificate(certPath, port string) error {
	clientCrt := fmt.Sprintf("%s/certs.d/deploy.bocloud.k8s:%s", k3s.DefaultK3sDataDir, port)
	return setCommonCert(certPath, clientCrt)
}

// SetClientLocalCertificate write client certificate
func SetClientLocalCertificate(certPath, port string) error {
	clientCrt := fmt.Sprintf("%s/certs.d/127.0.0.1:%s", k3s.DefaultK3sDataDir, port)
	return setCommonCert(certPath, clientCrt)
}

// SetServerCertificate write server certificate
func SetServerCertificate(certPath string) error {
	var (
		output string
		err    error
	)

	if len(certPath) == 0 {
		certPath = fmt.Sprintf("%s/registry", k3s.DefaultK3sDataDir)
	}
	genShFile := filepath.Join(certPath, "generate-registry-certs.sh")
	err = os.WriteFile(genShFile, []byte(certGen), utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("write generate-registry-certs.sh failed: %w", err)
	}

	executor := &exec.CommandExecutor{}
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c",
		fmt.Sprintf("cd %s && chmod +x ./generate-registry-certs.sh && ./generate-registry-certs.sh &&"+
			"chmod -x ./generate-registry-certs.sh", certPath))
	if err != nil {
		return fmt.Errorf("generate registry tls cert failed, output: %s, err: %w", output, err)
	}
	return nil
}

// SetRegistryConfig write registry config
func SetRegistryConfig(certPath string) error {
	if len(certPath) == 0 {
		certPath = fmt.Sprintf("%s/registry", k3s.DefaultK3sDataDir)
	}
	if !utils.FileExists(certPath) {
		err := os.MkdirAll(certPath, utils.DefaultDirPermission)
		if err != nil {
			return err
		}
	}
	conf := path.Join(certPath, "config.yml")
	if !utils.FileExists(conf) {
		err := utils.WriteCommon(conf, configYml)
		if err != nil {
			return err
		}
	}
	return nil
}
