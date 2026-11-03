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

package k8s

import (
	"errors"

	"github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// Client 定义Kubernetes客户端的接口
type Client interface {
	Kubernetes() kubernetes.Interface
	Snapshot() versioned.Interface
	ApiExtensions() clientset.Interface
	Config() *rest.Config
}

type kubernetesClient struct {
	k8s           kubernetes.Interface
	snapshot      versioned.Interface
	apiExtensions clientset.Interface
	config        *rest.Config
}

// NewKubernetesClient 用于创建Kubernetes客户端
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

	snapshotInterface, err := versioned.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	apiExtensionsInterface, err := clientset.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	return &kubernetesClient{
		k8s:           k8sInterface,
		snapshot:      snapshotInterface,
		apiExtensions: apiExtensionsInterface,
		config:        cfg.KubeConfig,
	}, nil
}

func (kubernetes *kubernetesClient) Kubernetes() kubernetes.Interface { return kubernetes.k8s }

func (kubernetes *kubernetesClient) ApiExtensions() clientset.Interface {
	return kubernetes.apiExtensions
}

func (kubernetes *kubernetesClient) Config() *rest.Config { return kubernetes.config }

func (kubernetes *kubernetesClient) Snapshot() versioned.Interface { return kubernetes.snapshot }
