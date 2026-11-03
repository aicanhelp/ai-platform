// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package decryption

import (
	"fmt"
	"time"

	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/api"
	"github.com/coredns/coredns/fypkg/crypto/caascore/util/env"
)

const (
	// the env key of cert private key decrypt uds endpoints.
	udsKey = "FY_CERT_KEY_DECRYPT_UDS"

	// the option env key of uds API timeout, unit is second.
	timeoutKey = "FY_CERT_KEY_DECRYPT_UDS_TIMEOUT_SECOND"

	// the default timeout is 3 second
	defaultTimeout = 3
)

// the factory to create decryption service. This is to make custom and write test easier.
var decryptServiceFactory func(endpoint string, callTimeout time.Duration) (api.Service, error)

// FYTryDecryptCert decrypt cert by plugin.
// When no plugin is specified return raw data to adapter community version with plain key.
func FYTryDecryptCert(key []byte) ([]byte, error) {
	return decrypt(key, true)
}

// FYTryDecryptSecret decrypt cert by plugin.
// When no plugin is specified return raw data to adapter community version with plain key.
func FYTryDecryptSecret(key []byte) ([]byte, error) {
	return decrypt(key, false)
}

func decrypt(key []byte, certOrSecret bool) ([]byte, error) {
	enable, service, err := tryCreateDecryptService()
	// decryption is disabel
	if !enable {
		return key, nil
	}
	// create failed return origin data
	if err != nil {
		return key, err
	}
	if certOrSecret {
		return service.DecryptCert(key)
	}
	return service.DecryptSecret(key)
}

func tryCreateDecryptService() (bool, api.Service, error) {
	endpoint := env.GetEnvAsStringOrDefault(udsKey, "")
	if endpoint == "" {
		// adapter community version without encryption
		return false, nil, nil
	}
	service, err := createGRPService(endpoint)
	return true, service, err
}

func createGRPService(endpoint string) (api.Service, error) {
	timeoutValue, err := env.GetEnvAsInt64OrDefault(timeoutKey, int64(defaultTimeout))
	if err != nil || timeoutValue < 0 {
		return nil, fmt.Errorf("timeout should be a positive number: %v", timeoutValue)
	}
	timeout := time.Duration(timeoutValue) * time.Second
	return decryptServiceFactory(endpoint, timeout)
}

func grpcEnable() bool {
	endpoint := env.GetEnvAsStringOrDefault(udsKey, "")
	return endpoint != ""
}
