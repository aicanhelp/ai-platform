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

package tls

import (
	"crypto/tls"
	"os"

	"k8s.io/fy-ext/security/crypto/decryption"
	"k8s.io/fy-ext/security/sensitive"
)

// FYX509KeyPair warps tls.X509KeyPair to support load encrypted info
func FYX509KeyPair(certPEMBlock, keyPEMBlock []byte) (tls.Certificate, error) {
	decryptedKeyPEMBlock, err := decryption.FYTryDecryptCert(keyPEMBlock)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.X509KeyPair(certPEMBlock, decryptedKeyPEMBlock)
}

// FYLoadX509KeyPair warps tls.X509KeyPair to support load encrypted info
func FYLoadX509KeyPair(certFile, keyFile string) (tls.Certificate, error) {
	certPEMBlock, err := os.ReadFile(certFile)
	if err != nil {
		return tls.Certificate{}, err
	}
	decryptedKeyPEMBlock, err := decryption.FYTryDecryptKeyFromDisk(keyFile)
	if err != nil {
		return tls.Certificate{}, err
	}
	defer sensitive.CleanSensitiveData(&decryptedKeyPEMBlock)
	return tls.X509KeyPair(certPEMBlock, decryptedKeyPEMBlock)
}
