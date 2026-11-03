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
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

// ApplyThrottlingConfig applies the centralized Kubernetes client rate limit.
func ApplyThrottlingConfig(cfg *rest.Config) *rest.Config {
	if cfg == nil {
		return nil
	}
	cfg.QPS = config.ClientQPS
	cfg.Burst = config.ClientBurst
	return cfg
}

/*
*

		这个创建的是 普通 Kubernetes ClientSet。它主要访问内置资源，比如：
	  Pod
	  Node
	  Service
	  ConfigMap
	  Secret
	  Deployment
	  DaemonSet
*/
func newKubernetesClientForConfig(cfg *rest.Config) (*kubernetes.Clientset, error) {
	return kubernetes.NewForConfig(ApplyThrottlingConfig(cfg))
}

// 这个创建的是 DynamicClient。发现某个 Addon YAML 里的任意 Kind
func newDynamicClientForConfig(cfg *rest.Config) (dynamic.Interface, error) {
	return dynamic.NewForConfig(ApplyThrottlingConfig(cfg))
}
