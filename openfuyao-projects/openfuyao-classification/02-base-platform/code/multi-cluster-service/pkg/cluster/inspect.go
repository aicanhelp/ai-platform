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
	"encoding/json"
	"fmt"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// Inspect check karmada cluster objects and return clusterlist information.
func (c *OperationClient) Inspect(ctx context.Context, opt InspectOptions) (*ListOfCluster, error) {
	var err error

	// before get list, check karmada api service if ok
	if err = c.checkHostClusterReady(); err != nil {
		zlog.LogErrorf("Error Waiting autoregistering service ok: %v", err)
		return nil, err
	}

	clusterList := &ListOfCluster{}
	clusterList.Info = make(map[string]*InformationOfCluster)
	if opt.ClusterName != "" {
		clusterInformation, err := inspect(opt.ClusterName, c.KarmadaVersionedClient, c.MgrClient)
		if err != nil {
			zlog.LogErrorf("Error Inspecting cluster object %s : %v", opt.ClusterName, err)
			return nil, err
		}
		clusterList.Info[clusterInformation.ClusterName] = clusterInformation
		return clusterList, nil
	}

	clusterObjs, err := c.KarmadaVersionedClient.ClusterV1alpha1().Clusters().List(context.TODO(), v1.ListOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster object list : %v", err)
		return nil, err
	}

	for _, clusterobj := range clusterObjs.Items {
		clusterInformation, err := inspect(clusterobj.Name, c.KarmadaVersionedClient, c.MgrClient)
		if err != nil {
			return nil, err
		}
		clusterList.Info[clusterInformation.ClusterName] = clusterInformation
	}

	return clusterList, err
}

func inspect(clusterTag string, karmadaClient *versioned.Clientset, mgr client.Client) (*InformationOfCluster, error) {
	var err error
	exist, err := isClusterIDExist(clusterTag, mgr)
	if err != nil {
		zlog.LogErrorf("Error Checking cluster object : %v", err)
		return nil, err
	}
	if !exist {
		zlog.LogErrorf("Error Checking cluster object invalid : %v", err)
		return nil, fmt.Errorf("cluster ID no exists")
	}

	clusterObj, err := karmadaClient.ClusterV1alpha1().Clusters().Get(context.TODO(), clusterTag, v1.GetOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return nil, err
	}

	var labels []string
	for key, values := range clusterObj.Labels {
		labels = append(labels, key+":"+values)
	}

	// sync from karmada cluster object
	cpuAndMemoryUsage, err := retrieveClusterStatus(clusterTag, mgr)
	if err != nil {
		if err.Error() == "cluster object is not ready" {
			return &InformationOfCluster{
				ClusterName:           clusterObj.Name,
				ClusterLabels:         labels,
				ClusterResourcesUsage: &ResourceUsage{},
			}, fmt.Errorf("cluster object is not ready")
		}
		zlog.LogErrorf("Error Retrieving cluster object resources: %v", err)
		return nil, err
	}

	config, err := retrieveClusterConfig(clusterTag, mgr)
	if err != nil {
		return nil, err
	}

	clusterInformation := &InformationOfCluster{
		ClusterName:           clusterObj.Name,
		ClusterLabels:         labels,
		ClusterResourcesUsage: cpuAndMemoryUsage,
		ClusterConfig:         config,
	}

	return clusterInformation, nil
}

func (c *OperationClient) checkHostClusterReady() error {
	// before get host, check karmada api service if ok
	clusters, err := c.KarmadaVersionedClient.ClusterV1alpha1().Clusters().List(context.TODO(), v1.ListOptions{})
	if err != nil || clusters == nil {
		zlog.LogErrorf("Error Waiting karmada service ok: %v", err)
		return fmt.Errorf("karmada service is not ready")
	}

	for _, cluster := range clusters.Items {
		if cluster.Name == "host" && cluster.Status.ResourceSummary != nil {
			return nil
		}
	}

	return fmt.Errorf("host Cluster is being onboarded")
}

// Inspection return a specific clsuter resources usage.
func (c *OperationClient) Inspection(clusterTag string) (*ResourcesInCluster, error) {
	var err error
	exist, err := utils.IsClusterUIDValid(c.KarmadaVersionedClient, clusterTag)
	if err != nil {
		zlog.LogErrorf("Error Checking cluster object : %v", err)
		return nil, err
	}
	if !exist {
		zlog.LogErrorf("Error Checking cluster object invalid : %v", err)
		return nil, fmt.Errorf("cluster ID no exists")
	}

	clusterObj, err := c.KarmadaVersionedClient.ClusterV1alpha1().Clusters().
		Get(context.TODO(), clusterTag, v1.GetOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return nil, err
	}

	resourceUsage := &ResourcesInCluster{
		Allocatable: clusterObj.Status.ResourceSummary.Allocatable,
		Allocated:   clusterObj.Status.ResourceSummary.Allocated,
	}

	return resourceUsage, nil
}

// FilterClusterList filters clusters information based on user member cluster list.
func FilterClusterList(clusterInfo *ListOfCluster, memList []string) *ListOfCluster {
	clusterList := &ListOfCluster{}
	clusterList.Info = make(map[string]*InformationOfCluster)

	if len(memList) == 0 {
		return clusterList
	}

	if len(memList) == 1 && memList[0] == "*" {
		return clusterInfo
	}

	for _, memCluster := range memList {
		if values, exists := clusterInfo.Info[memCluster]; exists {
			clusterList.Info[memCluster] = values
		}
	}
	return clusterList
}

func retrieveClusterConfig(clusterTag string, mgr client.Client) (json.RawMessage, error) {
	clusterObj := &v1beta1.Cluster{}
	if err := mgr.Get(context.TODO(), types.NamespacedName{Name: clusterTag}, clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object from MgrClient : %v", err)
		return nil, err
	}

	return json.RawMessage(clusterObj.Spec.OutClusterConfig.Raw), nil
}

func retrieveClusterStatus(clusterTag string, mgr client.Client) (*ResourceUsage, error) {
	if isOk := checkMgrclusterOK(clusterTag, mgr); !isOk {
		return nil, fmt.Errorf("cluster object is not ready")
	}

	clusterObj := &v1beta1.Cluster{}
	if err := mgr.Get(context.TODO(), types.NamespacedName{Name: clusterTag}, clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object from MgrClient : %v", err)
		return nil, fmt.Errorf("cluster object is not ready")
	}

	cpuAndMemoryUsage := &ResourceUsage{}
	cpuAndMemoryUsage.AllocatableCPU = clusterObj.Status.ResourceList.AllocatableCPU
	cpuAndMemoryUsage.AllocatableMemory = clusterObj.Status.ResourceList.AllocatableMemory
	cpuAndMemoryUsage.AllocatedCPU = clusterObj.Status.ResourceList.AllocatedCPU
	cpuAndMemoryUsage.AllocatedMemory = clusterObj.Status.ResourceList.AllocatedMemory

	return cpuAndMemoryUsage, nil
}

func checkMgrclusterOK(clusterTag string, mgr client.Client) bool {
	interval := utils.WatchClusterResourceInterval
	timeout := utils.WatchTimeout

	var err error
	err = wait.PollUntilContextTimeout(context.TODO(), interval, timeout, false,
		func(context.Context) (done bool, err error) {
			clusterObj := &v1beta1.Cluster{}
			err = mgr.Get(context.TODO(), types.NamespacedName{Name: clusterTag}, clusterObj)
			if err != nil {
				return false, err
			}
			if clusterObj.Status.ResourceList.AllocatableCPU != "" {
				return true, nil
			}

			return false, nil
		})
	if err != nil {
		return false
	}
	return true
}

func isClusterIDExist(clusterTag string, mgr client.Client) (bool, error) {
	clusterObj := &v1beta1.Cluster{}
	err := mgr.Get(context.TODO(), types.NamespacedName{Name: clusterTag}, clusterObj)
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.Warnf("NotFound cluster object.")
			return false, nil
		}
		zlog.LogErrorf("Error Retrieving cluster object : %v", err)
		return false, err
	}

	return true, nil
}
