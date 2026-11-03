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
	"fmt"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// Edit updates the tags for a specified cluster.
func (c *OperationClient) Edit(ctx context.Context, opt EditOptions) error {
	var err error
	clusterObj := v1beta1.Cluster{}
	if err = c.MgrClient.Get(context.TODO(), client.ObjectKey{Name: opt.ClusterName}, &clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object: %v", err)
		return err
	}

	exist, err := utils.IsClusterUIDValid(c.KarmadaVersionedClient, opt.ClusterName)
	if err != nil {
		zlog.LogErrorf("Error Checking cluster object : %v", err)
		return err
	}
	if !exist {
		zlog.LogErrorf("Error Checking cluster object invalid : %v", err)
		return fmt.Errorf("cluster ID no exists")
	}

	karmadaClusterObj, err := c.KarmadaVersionedClient.ClusterV1alpha1().Clusters().
		Get(context.TODO(), opt.ClusterName, v1.GetOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return err
	}

	// make nil
	clusterObj.Labels = make(map[string]string)
	karmadaClusterObj.Labels = make(map[string]string)

	for key, value := range opt.ClusterLabels {
		clusterObj.Labels[key] = value
		karmadaClusterObj.Labels[key] = value
	}

	err = c.MgrClient.Update(context.TODO(), &clusterObj)
	if err != nil {
		zlog.LogErrorf("Error Tagging cluster object %s: %v", clusterObj.Name, err)
		return err
	}
	zlog.LogInfof("Tagging the controller cluster %s successfully", opt.ClusterName)

	karmadaClusterObj, err = c.KarmadaVersionedClient.ClusterV1alpha1().Clusters().
		Update(context.TODO(), karmadaClusterObj, v1.UpdateOptions{})
	if err != nil {
		zlog.LogErrorf("Error update karmada cluster object label: %v", err)
		return err
	}
	zlog.LogInfof("Tagging the karmada cluster %s successfully", opt.ClusterName)

	return nil
}
