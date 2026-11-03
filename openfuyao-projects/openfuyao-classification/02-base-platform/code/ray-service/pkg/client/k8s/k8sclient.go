/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

// Client interface
type Client interface {
	K8s() kubernetes.Interface
	Snapshot() versioned.Interface
	Api() clientset.Interface
	Cfg() *rest.Config
}

type kubernetesClient struct {
	k8s      kubernetes.Interface
	snapshot versioned.Interface
	api      clientset.Interface
	cfg      *rest.Config
}

// NewKubernetesClient initializes a new client for interacting with Kubernetes.
// This client bundles access to various Kubernetes APIs such as the core Kubernetes API,
// snapshot operations, and API extensions.
func NewKubernetesClient(cfg *KubernetesCfg) (Client, error) {
	// Check if the provided configuration object has a nil kubeconfig,
	// which is necessary to connect to the Kubernetes API.
	if cfg.KubeConfig == nil {
		return nil, errors.New("kubernetes configuration is missing")
	}

	// Set the QPS and Burst values on the kubeconfig to control the rate limit of the client.
	// QPS is the number of queries per second the client can make to the Kubernetes API,
	// and Burst is the maximum number of queries it can make in a short time frame.
	cfg.KubeConfig.QPS = cfg.QPS
	cfg.KubeConfig.Burst = cfg.Burst

	// Initialize the Kubernetes core API client using the kubeconfig.
	// This client is used for most Kubernetes operations such as creating pods, services, etc.
	k8sInterface, err := kubernetes.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Initialize the snapshot client using the same kubeconfig.
	// This client is specifically used for managing volume snapshots in Kubernetes.
	snapshotInterface, err := versioned.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Initialize the API extensions client using the same kubeconfig.
	// This client is used for interacting with API extensions that are not part of the core Kubernetes API.
	apiExtensionsInterface, err := clientset.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Return the initialized kubernetesClient struct which implements the Client interface.
	// This struct provides access to the initialized Kubernetes core, snapshot, and API extensions clients.
	return &kubernetesClient{
		k8s:      k8sInterface,
		snapshot: snapshotInterface,
		api:      apiExtensionsInterface,
		cfg:      cfg.KubeConfig,
	}, nil
}

func (client *kubernetesClient) K8s() kubernetes.Interface {
	return client.k8s
}

func (client *kubernetesClient) Snapshot() versioned.Interface {
	return client.snapshot
}

func (client *kubernetesClient) Api() clientset.Interface {
	return client.api
}

func (client *kubernetesClient) Cfg() *rest.Config {
	return client.cfg
}
