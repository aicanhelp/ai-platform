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

package kube

import (
	"sync"

	corev1 "k8s.io/api/core/v1"
)

// healthCheckCache 是单次健康检查内的轻量缓存。
// 它复用本轮已获取的 Node/Pod 列表，减少并行组件检查时对同一 namespace 的重复 List 调用。
// 这里没有启动 Informer，避免每轮健康检查临时创建 Informer 带来的同步等待和生命周期管理问题。
type healthCheckCache struct {
	client *Client
	mu     sync.Mutex
	nodes  *corev1.NodeList
	pods   map[string][]corev1.Pod
}

func newHealthCheckCache(client *Client) *healthCheckCache {
	return &healthCheckCache{
		client: client,
		pods:   make(map[string][]corev1.Pod),
	}
}

func (c *healthCheckCache) GetNodes() (*corev1.NodeList, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.nodes != nil {
		return c.nodes, nil
	}
	nodes, err := c.client.ListNodes(nil)
	if err != nil {
		return nil, err
	}
	c.nodes = nodes
	return nodes, nil
}

func (c *healthCheckCache) GetPods(namespace string) ([]corev1.Pod, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if pods, ok := c.pods[namespace]; ok {
		return pods, nil
	}
	pods, err := c.client.getPods(namespace)
	if err != nil {
		return nil, err
	}
	c.pods[namespace] = pods
	return pods, nil
}
