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

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/resource"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// RetrieveClusterStatus retrieves the current status and resource usage information of a kubernetes cluster.
func RetrieveClusterStatus(clusterTag string, c *versioned.Clientset) (*ResourceUsage, []v1.Condition, error) {
	// first check cluster status
	if isOk := checkClusterstatus(clusterTag, c); !isOk {
		return nil, nil, fmt.Errorf("cluster object is not ready")
	}

	clusterObj, err := c.ClusterV1alpha1().Clusters().Get(context.TODO(), clusterTag, v1.GetOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return nil, nil, err
	}

	if clusterObj.Status.ResourceSummary == nil {
		zlog.LogErrorf("Error Reconciling cluster object not ready")
		return nil, nil, fmt.Errorf("cluster object is not ready")
	}

	resourceUsage := &ResourcesInCluster{
		Allocatable: clusterObj.Status.ResourceSummary.Allocatable,
		Allocated:   clusterObj.Status.ResourceSummary.Allocated,
	}

	cpuAndMemoryUsage := &ResourceUsage{}
	memStr := resourceUsage.Allocatable.Memory().String()
	cpuAndMemoryUsage.AllocatableMemory = utils.ScaleToGiB(memStr)

	usedMemStr := resourceUsage.Allocated.Memory().String()
	cpuAndMemoryUsage.AllocatedMemory = utils.ScaleToGiB(usedMemStr)

	cpuStr := resourceUsage.Allocatable.Cpu().String()
	cpuQuantity := resource.MustParse(cpuStr)
	cpuAndMemoryUsage.AllocatableCPU = fmt.Sprintf("%.1f", float64(cpuQuantity.MilliValue())/utils.MilliCoresPerCore)

	usedCpuStr := resourceUsage.Allocated.Cpu().String()
	usedCpuQuantity := resource.MustParse(usedCpuStr)
	cpuAndMemoryUsage.AllocatedCPU = fmt.Sprintf("%.1f", float64(usedCpuQuantity.MilliValue())/utils.MilliCoresPerCore)

	newCondition := make([]v1.Condition, len(clusterObj.Status.Conditions))
	copy(newCondition, clusterObj.Status.Conditions)
	currentCondition := newCondition

	return cpuAndMemoryUsage, currentCondition, nil
}

func checkClusterstatus(clusterTag string, c *versioned.Clientset) bool {
	interval := utils.WatchClusterResourceInterval
	timeout := utils.WatchTimeout

	var err error
	err = wait.PollUntilContextTimeout(context.TODO(), interval, timeout, false,
		func(context.Context) (done bool, err error) {
			clusterObj, err := c.ClusterV1alpha1().Clusters().Get(context.TODO(), clusterTag, v1.GetOptions{})
			if err != nil {
				return false, err
			}
			if clusterObj.Status.ResourceSummary != nil {
				return true, nil
			}

			return false, nil
		})
	if err != nil {
		return false
	}
	return true
}
