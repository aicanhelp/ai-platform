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

	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// Client kubernetes client
type Client interface {
	ApiExtensions() apiextensionsclient.Interface
	Config() *rest.Config
	Kubernetes() kubernetes.Interface
	Snapshot() snapshotclient.Interface
}

type kubernetesClient struct {
	apiExtensions apiextensionsclient.Interface
	config        *rest.Config
	k8s           kubernetes.Interface
	snapshot      snapshotclient.Interface
}

// NewKubernetesClient initializes a new client for interacting with Kubernetes.
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

	snapshotInterface, err := snapshotclient.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	apiExtensionsInterface, err := apiextensionsclient.NewForConfig(cfg.KubeConfig)
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

func (kubernetes *kubernetesClient) Kubernetes() kubernetes.Interface {
	return kubernetes.k8s
}

func (kubernetes *kubernetesClient) Snapshot() snapshotclient.Interface {
	return kubernetes.snapshot
}

func (kubernetes *kubernetesClient) ApiExtensions() apiextensionsclient.Interface {
	return kubernetes.apiExtensions
}

func (kubernetes *kubernetesClient) Config() *rest.Config {
	return kubernetes.config
}
