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

package options

import (
	"fmt"

	"k8s.io/client-go/util/keyutil"
	"k8s.io/fy-ext/dynamicloader"
	"k8s.io/fy-ext/security/crypto/decryption"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/fyprivatekeyspublisher"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

func fyUpdateKeysDynamically(tokenGenerator serviceaccount.TokenGenerator, keyFile string) error {
	sk, ok := tokenGenerator.(serviceaccount.FyPrivateKeysUpdatable)
	if !ok {
		return fmt.Errorf("tokenGenerator not support updating keys")
	}
	return dynamicloader.ListenFileChangedEvent(keyFile, func(content []byte) {
		const errFmt = "[onchange] failed to parse service-account-issuer-key-file: %v"
		data, err := decryption.FYTryDecryptKeyFromDisk(keyFile)
		if err != nil {
			klog.Errorf(errFmt, err)
			return
		}
		privateKey, err := keyutil.ParsePrivateKeyPEM(data)
		if err != nil {
			err = fmt.Errorf("error loading private key file %s :%v", keyFile, err)
			klog.Errorf(errFmt, err)
			return
		}

		if err = sk.FyUpdatePrivateKeys(privateKey); err != nil {
			klog.Errorf("[onchange] failed to update service account issuer key: %v", err)
			return
		}
		fyprivatekeyspublisher.Publish(data)
	})
}
