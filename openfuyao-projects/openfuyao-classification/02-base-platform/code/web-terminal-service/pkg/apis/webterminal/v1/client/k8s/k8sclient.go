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

// Package k8s create kubernetes config
package k8s

import (
	"errors"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// Client kubernetes client
type Client interface {
	K8s() kubernetes.Interface
	Cfg() *rest.Config
}

type kubernetesClient struct {
	k8s    kubernetes.Interface
	config *rest.Config
}

// NewKubernetesClient initializes a new client for interacting with Kubernetes.
// This client bundles access to various Kubernetes APIs such as the core Kubernetes API,
// snapshot operations, and API extensions.
func NewKubernetesClient(cfg *KubernetesCfg) (Client, error) {
	if cfg.KubeConfig == nil {
		return nil, errors.New("kubernetes configuration is missing")
	}
	cfg.KubeConfig.QPS = cfg.QPS
	cfg.KubeConfig.Burst = cfg.Burst
	k8sInterface, err := kubernetes.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}
	return &kubernetesClient{
		k8s:    k8sInterface,
		config: cfg.KubeConfig,
	}, nil
}

func (client *kubernetesClient) K8s() kubernetes.Interface {
	return client.k8s
}

func (client *kubernetesClient) Cfg() *rest.Config {
	return client.config
}
