/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package cluster provides a suite og tools adn utilities for managing cluster objects.
package cluster

import (
	"context"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// Unjoin performs the necessagr operations to unjoin a managed cluster from the karmada controlplane.
// 1. Releasing resources: ServiceAccounts, Clusterrole, Clusterrolebinding.
// 2. Deleting the cluster object from karmada controlplane and multicluster-service.
func (c *OperationClient) Unjoin(ctx context.Context, opt UnjoinOptions) error {
	var err error
	exist, err := utils.IsClusterUIDValid(c.KarmadaVersionedClient, opt.ClusterName)
	if !exist || err != nil {
		zlog.LogErrorf("Error Checking cluster object : %v", err)
		return err
	}

	// delete cluster
	err = unregsiterHostCluster(c.KarmadaVersionedClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error deleting cluster object : %v", err)
		return err
	}

	// ensure cluster has been deleted
	err = utils.IsClusterDeleted(c.KarmadaVersionedClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error deleting cluster object from from karmada controlplane, timeout : %v", err)
		return err
	}

	err = utils.DeleteKarmadaResources(c.KarmadaClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error deleting karmada resources: %v", err)
		return err
	}

	// retrieving kubeconfig from cluster object
	clusterKClient, err := utils.RetriveKubeconfigClient(c.MgrClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error Creating cluster client from kubeconfig data: %v", err)
		return err
	}

	err = utils.DeleteClusterResources(clusterKClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error deleting cluster resources from cluster: %v", err)
		return err
	}

	err = deleteAPICluster(c.MgrClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error Deleting v1beta1 Cluster Object : %v", err)
		return err
	}

	zlog.LogInfo("All created resources have been successfully cleaned up.")
	return nil
}

func deleteAPICluster(mgrClient client.Client, tag string) error {
	clusterObj := v1beta1.Cluster{}
	if err := mgrClient.Get(context.TODO(), client.ObjectKey{Name: tag}, &clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object: %v", err)
		return err
	}

	if err := mgrClient.Delete(context.TODO(), &clusterObj); err != nil {
		zlog.LogErrorf("Error Deleted cluster object: %v", err)
		return err
	}

	zlog.LogInfof("Cluster object %s deleted.", clusterObj.Name)
	return nil
}

func unregsiterHostCluster(kclient *versioned.Clientset, tag string) error {
	err := kclient.ClusterV1alpha1().Clusters().Delete(context.TODO(), tag, v1.DeleteOptions{})
	if err != nil {
		zlog.LogErrorf("Error Deleting cluster object : %v", err)
		return err
	}

	zlog.LogInfof("Karmada Cluster Object %s unjoin request sended.", tag)
	return nil
}
