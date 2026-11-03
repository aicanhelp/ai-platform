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

package kube

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func (c *Client) ListNodes(option *metav1.ListOptions) (*corev1.NodeList, error) {
	if option == nil {
		option = &metav1.ListOptions{}
	}

	list, err := c.ClientSet.CoreV1().Nodes().List(c.Ctx, *option)
	if err != nil {
		return nil, err
	}
	return list, nil
}

func (c *Client) GetPod(namespace, name string) (*corev1.Pod, error) {
	pod, err := c.ClientSet.CoreV1().Pods(namespace).Get(c.Ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return pod, nil
}
