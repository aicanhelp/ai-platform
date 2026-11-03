/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package join

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
)

func (n *NodeJoinClusterOperate) addLabels(node *cluster.Node, addLabels []cluster.Labels) error {
	clusterNode, err := n.k8sClient.Kubernetes().CoreV1().Nodes().Get(context.TODO(), node.Hostname, v1.GetOptions{})
	if err != nil {
		return fmt.Errorf("error getting node %s when add labels: %v", node.Hostname, err)
	}

	labels := clusterNode.GetLabels()
	for _, label := range addLabels {
		labels[label.Name] = label.Value
	}
	clusterNode.SetLabels(labels)

	_, err = n.k8sClient.Kubernetes().CoreV1().Nodes().Update(context.TODO(), clusterNode, v1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("error updating node %s label when add labels: %v", node.Hostname, err)
	}
	return nil
}

func (n *NodeJoinClusterOperate) workerNodeAddLabels(node *cluster.Node) error {
	defaultLabel := cluster.Labels{
		Name:  constants.DefaultWorkerLabelName,
		Value: constants.DefaultWorkerLabelValue,
	}

	labels := node.Labels
	labels = append(labels, defaultLabel)

	return n.addLabels(node, labels)
}

func (n *NodeJoinClusterOperate) masterNodeAddLabels(node *cluster.Node) error {
	defaultLabel := cluster.Labels{
		Name:  constants.DefaultMasterLabelName,
		Value: constants.DefaultMasterLabelValue,
	}

	labels := node.Labels
	labels = append(labels, defaultLabel)
	return n.addLabels(node, labels)
}
