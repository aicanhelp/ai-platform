/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package securityinit initializes the BKENode password encryption key source
// for the bkeadm host process. It replaces the former env-var based bridge
// (read Secret -> os.Setenv) with a direct Secret-backed KeyProvider.
package securityinit

import (
	"context"
	"errors"
	"sync"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	initOnce sync.Once
	initErr  error
)

// Init initializes the BKENode password encryption key source (idempotent
// per process):
//  1. builds a client-go backed SecretKeyProvider that talks to the
//     management cluster apiserver directly;
//  2. eagerly loads the key: an existing Secret is reused as-is; a missing
//     Secret is generated after the guard detects key loss (console WARN
//     with the affected BKENodes) and created.
//
// It must be called before the first security.AesEncrypt/AesDecrypt call in
// the process (replaces the ordering contract of the former
// ensurePasswordEncryptionKeyEnv).
func Init() error {
	initOnce.Do(func() {
		if global.K8s == nil {
			initErr = errors.New("kubernetes client is not initialized")
			return
		}

		clientset := global.K8s.GetClient()
		dynamicClient := global.K8s.GetDynamicClient()

		get, create := security.ClientGoSecretIO(
			clientset, security.DefaultSecretNamespace, security.DefaultSecretName)

		provider, err := security.NewSecretKeyProvider(security.SecretKeyProviderConfig{
			GetSecret:    get,
			CreateSecret: create,
			Guard:        security.DynamicBKENodeGuard(dynamicClient),
		})
		if err != nil {
			initErr = err
			return
		}
		security.SetKeyProvider(provider)

		// Eager load: surface problems (missing/invalid Secret, deferred
		// generation) at the command entry point instead of failing per
		// node later.
		if _, err := provider.Key(context.Background()); err != nil {
			initErr = err
			return
		}
		log.Infof("BKENode password encryption key ready (secret %s/%s)",
			security.DefaultSecretNamespace, security.DefaultSecretName)
	})
	return initErr
}

// resetForTesting resets the once-guard so tests can re-run Init.
func resetForTesting() {
	initOnce = sync.Once{}
	initErr = nil
	security.SetKeyProvider(nil)
}
