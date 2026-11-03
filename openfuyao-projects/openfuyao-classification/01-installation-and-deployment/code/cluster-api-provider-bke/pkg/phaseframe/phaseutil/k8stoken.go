/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	agentutils "gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func NewK8sTokenSecret(ctx context.Context, token string, c client.Client, bkeCluster *bkev1beta1.BKECluster) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-k8s-token", bkeCluster.Name),
			Namespace: bkeCluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: bkeCluster.APIVersion,
					Kind:       bkeCluster.Kind,
					Name:       bkeCluster.Name,
					UID:        bkeCluster.UID,
				},
			},
		},
		StringData: map[string]string{
			"token": token,
		},
		Type: agentutils.BKESecretType,
	}

	if err := c.Create(ctx, secret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			if err = c.Delete(ctx, secret); err != nil {
				return fmt.Errorf("delete %q k8s token secret failed: %w", utils.ClientObjNS(bkeCluster), err)
			}
			if err = c.Create(ctx, secret); err != nil {
				return fmt.Errorf("create %q k8s token secret failed: %w", utils.ClientObjNS(bkeCluster), err)
			}
			return nil
		}
		return fmt.Errorf("create %q k8s token secret failed: %w", utils.ClientObjNS(bkeCluster), err)
	}
	return nil
}

func GetK8sTokenSecret(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*corev1.Secret, error) {
	secret := &corev1.Secret{}
	key := client.ObjectKey{
		Name:      fmt.Sprintf("%s-k8s-token", bkeCluster.Name),
		Namespace: bkeCluster.Namespace,
	}
	if err := c.Get(ctx, key, secret); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, errors.Errorf("failed to retrieve k8s-token secret for BKECluster %s: %v", utils.ClientObjNS(bkeCluster), err)
		}
		return nil, errors.Errorf("get %q k8s token secret failed: %v", utils.ClientObjNS(bkeCluster), err)
	}
	return secret, nil
}

func DeleteK8sTokenSecret(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-k8s-token", bkeCluster.Name),
			Namespace: bkeCluster.Namespace,
		},
	}
	if err := c.Delete(ctx, secret); err != nil {
		return fmt.Errorf("delete %q k8s token secret failed: %w", utils.ClientObjNS(bkeCluster), err)
	}
	return nil
}
