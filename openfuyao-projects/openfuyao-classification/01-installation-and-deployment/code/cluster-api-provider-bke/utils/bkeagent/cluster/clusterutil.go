/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cluster

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/kubeclient"
)

var kubeconfig = fmt.Sprintf("%s/%s", utils.Workspace, "config")

type ClusterData struct {
	Cluster *bkev1beta1.BKECluster
	Nodes   bkenode.Nodes
}

func GetClusterData(namespace string, name string) (*ClusterData, error) {
	if namespace == "" || name == "" {
		return nil, errors.New("invalid namespace or name")
	}

	k8sClient, err := kubeclient.NewClient(kubeconfig)
	if err != nil {
		return nil, err
	}

	gvr := schema.GroupVersionResource{
		Group:    bkev1beta1.GVK.Group,
		Version:  bkev1beta1.GVK.Version,
		Resource: strings.ToLower(bkev1beta1.GVK.Kind) + "s",
	}

	bkeCluster, err := k8sClient.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get BKECluster %s/%s", namespace, name)
	}

	bc := &bkev1beta1.BKECluster{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(bkeCluster.Object, bc); err != nil {
		return nil, errors.Wrap(err, "failed to convert unstructured BKECluster")
	}

	bkeNodeGVR := schema.GroupVersionResource{
		Group:    bkev1beta1.BKENodeGVK.Group,
		Version:  bkev1beta1.BKENodeGVK.Version,
		Resource: "bkenodes",
	}
	labelSelector := fmt.Sprintf("cluster.x-k8s.io/cluster-name=%s", bkeCluster.GetName())

	unstructuredList, err := k8sClient.DynamicClient.Resource(bkeNodeGVR).Namespace(bkeCluster.GetNamespace()).List(context.Background(), v1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to list BKENodes for cluster %s", bkeCluster.GetName())
	}
	bkeNodeList := &bkev1beta1.BKENodeList{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredList.UnstructuredContent(), bkeNodeList); err != nil {
		return nil, errors.Wrap(err, "failed to convert unstructured BKENodeList")
	}

	nodes := bkenode.ConvertBKENodeListToNodes(bkeNodeList)
	nodes = bkenode.SetDefaultsForNodes(nodes)

	cd := &ClusterData{
		Cluster: bc,
		Nodes:   nodes,
	}
	return cd, nil

}

//go:noinline
func GetNodesData(namespace string, name string) (bkenode.Nodes, error) {
	clusterData, err := GetClusterData(namespace, name)
	if err != nil {
		return nil, err
	}
	return clusterData.Nodes, nil
}
