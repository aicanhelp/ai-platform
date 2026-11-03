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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"installer/pkg/constants"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/zlog"
)

// 移除污点
func (n *NodeJoinClusterOperate) removeTaint(node *fuyaov1beta1.Node) error {
	if node.RemoveTaint == false {
		return nil
	}

	clusterNode, err := n.k8sClient.Kubernetes().CoreV1().Nodes().Get(context.TODO(), node.Hostname, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("error getting node %s when remove taint: %v", node.Hostname, err)
	}

	var newTaints []corev1.Taint
	for _, taint := range clusterNode.Spec.Taints {
		if taint.Key == constants.DefaultMasterLabelName && taint.Effect == constants.NoSchedule {
			continue
		}
		newTaints = append(newTaints, taint)
	}
	clusterNode.Spec.Taints = newTaints

	_, err = n.k8sClient.Kubernetes().CoreV1().Nodes().Update(context.TODO(), clusterNode, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("error updating node %s label when remove taint: %v", node.Hostname, err)
	}

	zlog.Info("remove taint successfully from node %s\n", node.Hostname)
	return nil
}
