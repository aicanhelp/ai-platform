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

package app

import (
	"context"
	"fmt"

	"k8s.io/client-go/util/keyutil"
	"k8s.io/fy-ext/dynamicloader"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

func fyUpdateKeysDynamically(tokenGenerator serviceaccount.TokenGenerator, keyFile string) error {
	sk, ok := tokenGenerator.(serviceaccount.FyPrivateKeysUpdatable)
	if !ok {
		return fmt.Errorf("tokenGenerator not support updating keys")
	}
	return dynamicloader.ListenFileChangedEvent(keyFile, func(content []byte) {
		if privateKey, err := keyutil.PrivateKeyFromFile(keyFile); err != nil {
			klog.Errorf("[onchange] failed to lparse service-account-issuer-key-file: %v", err)
		} else {
			if err = sk.FyUpdatePrivateKeys(privateKey); err != nil {
				klog.Errorf("[onchange] failed to update service account issuer key: %v", err)
			}
		}
	})
}

func fyUpdateCADynamically(ctx context.Context, controller *serviceaccountcontroller.TokensController, keyFile string) error {
	if keyFile == "" {
		return fmt.Errorf("[onchange] failed to parse root-ca-file beacuse RootCAFile invalid")
	}
	return dynamicloader.ListenFileChangedEvent(keyFile, func(content []byte) {
		if rootCA, err := readCA(keyFile); err != nil {
			klog.Errorf("[onchange] failed to parse root-ca-file: %v", err)
		} else {
			controller.UpdateSecretAndCA(ctx, rootCA)
		}
	})
}

func fyUpdatePublishCADynamically(publisher *rootcacertpublisher.Publisher, controllerContext ControllerContext) error {
	keyFile := controllerContext.ComponentConfig.SAController.RootCAFile
	if keyFile == "" {
		return fmt.Errorf("[onchange] failed to parse root-ca-file because RootCAFile invalid")
	}
	return dynamicloader.ListenFileChangedEvent(keyFile, func(content []byte) {
		if rootCA, err := readCA(keyFile); err != nil {
			klog.Errorf("[onchange] failed to parse root-ca-file: %v", err)
		} else {
			publisher.SetConfigmapAndCA(rootCA, controllerContext.InformerFactory.Core().V1().Namespaces())
		}
	})
}
