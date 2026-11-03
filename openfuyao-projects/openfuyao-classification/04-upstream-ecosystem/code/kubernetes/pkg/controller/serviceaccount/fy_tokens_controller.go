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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

func (e *TokensController) UpdateSecretAndCA(ctx context.Context, currentCA []byte) {
	e.rootCA = currentCA
	e.updateSecret(ctx)
}

func (e *TokensController) updateSecret(ctx context.Context) {
	nsList, err := e.client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		klog.Errorf("[onchange] failed to list namespace to update secret root-ca-file: %v", err)
		return
	}
	for _, ns := range nsList.Items {
		scList, err := e.client.CoreV1().Secrets(ns.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			klog.Errorf("[onchange] failed to update root-ca-file: %v", err)
			return
		}
		for _, sc := range scList.Items {
			if sc.Type != v1.SecretTypeServiceAccountToken {
				continue
			}
			e.syncSecretQueue.Add(makeSecretQueueKey(&sc))

			// To tigger token update in the secret
			sc.Data[v1.ServiceAccountTokenKey] = nil
			_, err := e.client.CoreV1().Secrets(ns.Name).Update(context.TODO(), &sc, metav1.UpdateOptions{})
			if err != nil {
				klog.Errorf("[onchange] failed to update tigger token update in the secret: %v", err)
				return
			}
		}
	}
}
