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

package decryption

import (
	"fmt"
	"time"

	"k8s.io/fy-ext/security/crypto/decryption/api"
	"k8s.io/fy-ext/util/env"
)

const (
	// the env key of cert private key decrypt uds endpoint.
	udsKey = "FY_CERT_KEY_DECRYPT_UDS"

	// the option env key of uds API timeout, unit is second
	timeoutKey = "FY_CERT_KEY_DECRYPT_UDS_TIMEOUT_SECOND"

	// the default timeout is 3 second
	defaultTimeout = 3
)

// The factory to create decryption service. This is to make custom and write test easier.
var decryptServiceFactory func(endpoint string, callTimeout time.Duration) (api.Service, error)

// FYTryDecryptCert decrypt cert by plugin.
// When no plugin is specified return raw data to adapter community version with plain key.
func FYTryDecryptCert(key []byte) ([]byte, error) {
	return decrypt(key, true)
}

// FYTryDecryptSecret decrypt Secret by plugin.
// When no plugin is specified return raw data to adapter community version with plain key.
func FYTryDecryptSecret(key []byte) ([]byte, error) {
	return decrypt(key, false)
}

func decrypt(key []byte, certOrSecret bool) ([]byte, error) {
	enable, service, err := tryCreateDecryptService()
	// decryption is disable
	if !enable {
		return key, nil
	}
	// create failed return origin data
	if err != nil {
		return key, err
	}
	defer func() { service.Release() }()
	if certOrSecret {
		return service.DecryptCert(key)
	}
	return service.DecryptCert(key)
}

func tryCreateDecryptService() (bool, api.Service, error) {
	endpoint := env.GetEnvAsStringOrDefault(udsKey, "")
	if endpoint == "" {
		// adapter community version without encryption
		return false, nil, nil
	}
	service, err := createGRPCService(endpoint)
	return true, service, err
}

func createGRPCService(endpoint string) (api.Service, error) {
	timeoutValue, err := env.GetEnvAsInt64OrDefault(timeoutKey, int64(defaultTimeout))
	if err != nil || timeoutValue < 0 {
		return nil, fmt.Errorf("timeout should be a positive number: %v", timeoutValue)
	}
	timeout := time.Duration(timeoutValue) * time.Second
	return decryptServiceFactory(endpoint, timeout)
}
