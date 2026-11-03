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

package serviceaccount

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"gopkg.in/go-jose/go-jose.v2"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/fy-ext/dynamicloader"
	"k8s.io/klog/v2"
)

const cacheAgeMaxSeconds = 180

type FyPrivateKeysUpdatable interface {
	FyUpdatePrivateKeys(privateKey interface{}) error
}

func (j *jwtTokenGenerator) FyUpdatePrivateKeys(privateKey interface{}) error {
	if tokenGenerator, err := JWTTokenGenerator(j.iss, privateKey); err != nil {
		return fmt.Errorf("failed to create tokenGenerator: %v", err)
	} else {
		j.signer.Store(tokenGenerator.(*jwtTokenGenerator).fyLoadSigner())
		return nil
	}
}

func (j *jwtTokenGenerator) fyLoadSigner() jose.Signer {
	v := j.signer.Load()
	if v == nil {
		return nil
	}
	if p, ok := v.(jose.Signer); !ok {
		return nil
	} else {
		return p
	}
}

func fyCreateAtomicSigner(signer jose.Signer) atomic.Value {
	var atomicSigner atomic.Value
	atomicSigner.Store(signer)
	return atomicSigner
}

// fyDynamicPublicKeysGetter constructs an implementation of PublicKeysGetter and dynamic loading of public keys.
type fyDynamicPublicKeysGetter struct {
	publicKeysGetter atomic.Value
	listeners        []Listener
}

// FyDynamicPublicKeysGetter constructs an implementation of PublicKeysGetter and dynamic loading of public keys.
func FyDynamicPublicKeysGetter(keyFiles []string) (PublicKeysGetter, error) {
	getter := &fyDynamicPublicKeysGetter{}
	// load the public key for the first time
	if err := getter.loadPublicKeysGetter(keyFiles); err != nil {
		return nil, err
	}
	var lock sync.Mutex
	for _, keyFile := range keyFiles {
		err := dynamicloader.ListenFileChangedEvent(keyFile, func(content []byte) {
			lock.Lock()
			defer lock.Unlock()
			if err := getter.loadPublicKeysGetter(keyFiles); err != nil {
				klog.Error(err)
			}

			for _, listener := range getter.listeners {
				listener.Enqueue()
			}

		})
		if err != nil {
			return nil, err
		}
	}
	return getter, nil
}

func (f *fyDynamicPublicKeysGetter) loadPublicKeysGetter(keyFiles []string) error {
	if pubKeys, err := fyReadPublicKeys(keyFiles); err != nil {
		return err
	} else {
		if tmp, err1 := StaticPublicKeysGetter(pubKeys); err1 != nil {
			return err1
		} else {
			f.publicKeysGetter.Store(tmp)
		}
	}

	return nil
}

func (f *fyDynamicPublicKeysGetter) AddListener(listener Listener) {
	f.listeners = append(f.listeners, listener)
}

func (f *fyDynamicPublicKeysGetter) GetCacheAgeMaxSeconds() int {
	// hard-coded to match cache max-age set in OIDC discovery
	return cacheAgeMaxSeconds
}

func (f *fyDynamicPublicKeysGetter) GetPublicKeys(ctx context.Context, keyID string) []PublicKey {
	if getter, ok := f.publicKeysGetter.Load().(*staticPublicKeysGetter); !ok {
		return nil
	} else {
		return getter.GetPublicKeys(ctx, keyID)
	}
}

func fyReadPublicKeys(keyFiles []string) ([]interface{}, error) {
	var allPublicKeys []interface{}
	for _, keyfile := range keyFiles {
		publicKeys, err := keyutil.PublicKeysFromFile(keyfile)
		if err != nil {
			return nil, err
		}
		allPublicKeys = append(allPublicKeys, publicKeys...)
	}
	return allPublicKeys, nil
}
