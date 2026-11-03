/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clusterversion

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

// GetClusterVersionForBKECluster returns the ClusterVersion associated with a BKECluster.
// Primary lookup: ClusterVersion.ownerReferences -> BKECluster (1:1 per design).
// Fallback: ClusterVersion with the same name/namespace as the cluster (install examples).
func GetClusterVersionForBKECluster(
	ctx context.Context,
	c client.Client,
	bkeCluster *bkev1beta1.BKECluster,
) (*cvv1alpha1.ClusterVersion, error) {
	if bkeCluster == nil {
		return nil, fmt.Errorf("bke cluster is nil")
	}
	if c == nil {
		return nil, fmt.Errorf("client is nil")
	}

	cvList := &cvv1alpha1.ClusterVersionList{}
	if err := c.List(ctx, cvList, client.InNamespace(bkeCluster.Namespace)); err != nil {
		return nil, err
	}
	for i := range cvList.Items {
		cv := &cvList.Items[i]
		if clusterVersionOwnsBKECluster(cv, bkeCluster) {
			return cv.DeepCopy(), nil
		}
	}

	cv := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(ctx, types.NamespacedName{
		Namespace: bkeCluster.Namespace,
		Name:      bkeCluster.Name,
	}, cv); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, apierrors.NewNotFound(
				cvv1alpha1.GroupVersion.WithResource("clusterversions").GroupResource(),
				bkeCluster.Name,
			)
		}
		return nil, err
	}
	return cv, nil
}

// GetBKEClusterForClusterVersion returns the BKECluster managed by a ClusterVersion.
func GetBKEClusterForClusterVersion(
	ctx context.Context,
	c client.Client,
	cv *cvv1alpha1.ClusterVersion,
) (*bkev1beta1.BKECluster, error) {
	if cv == nil {
		return nil, fmt.Errorf("cluster version is nil")
	}

	for _, ref := range cv.GetOwnerReferences() {
		if ref.Kind != "BKECluster" {
			continue
		}
		bc := &bkev1beta1.BKECluster{}
		if err := c.Get(ctx, types.NamespacedName{
			Namespace: cv.Namespace,
			Name:      ref.Name,
		}, bc); err != nil {
			return nil, err
		}
		return bc, nil
	}

	bc := &bkev1beta1.BKECluster{}
	if err := c.Get(ctx, types.NamespacedName{
		Namespace: cv.Namespace,
		Name:      cv.Name,
	}, bc); err != nil {
		return nil, err
	}
	return bc, nil
}

func clusterVersionOwnsBKECluster(cv *cvv1alpha1.ClusterVersion, bc *bkev1beta1.BKECluster) bool {
	if cv == nil || bc == nil {
		return false
	}
	for _, ref := range cv.GetOwnerReferences() {
		if ref.Kind != "BKECluster" || ref.Name != bc.Name {
			continue
		}
		if ref.UID != "" && ref.UID != bc.UID {
			continue
		}
		return true
	}
	return false
}
