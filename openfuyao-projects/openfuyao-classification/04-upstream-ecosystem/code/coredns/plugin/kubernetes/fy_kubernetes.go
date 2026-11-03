// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package kubernetes plugins
package kubernetes

import (
	"errors"

	"github.com/coredns/coredns/fypkg/crypto"

	"k8s.io/client-go/tools/clientcmd/api"
)

var ErrLimitExceeded = errors.New("the number of concurrent requests exceeded the maximum")

type FyConcurrentParam struct {
	concurrentLimitEnabled bool // Concurrent limit switch.
	kubernetesHandlerDelay int
	tokenBucket            chan struct{}
	maxConcurrent          int64
	maxWaitQueueSize       int64
	waitQueueSize          int64
}

var fyAdjustAuthInfo = func(k *Kubernetes, authInfo *api.AuthInfo) error {
	if len(k.APIClientKey) > 0 {
		decryptKeyHook := crypto.GetDecryptKeyHookFunc()
		if decryptKeyHook != nil {
			decryData, err := decryptKeyHook(k.APIClientKey)
			if err != nil {
				return err
			}
			authInfo.ClientKeyData = decryData
			authInfo.ClientKey = ""
		}
	}
	return nil
}
