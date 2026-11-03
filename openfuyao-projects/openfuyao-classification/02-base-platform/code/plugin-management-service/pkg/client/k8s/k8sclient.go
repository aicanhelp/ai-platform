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

// BaseClient kubernetes client
type BaseClient interface {
	KubernetesClient() kubernetes.Interface
	SnapshotClient() snapshotclient.Interface
	ApiExtensionsClient() apiextensionsclient.Interface
	ConfigClient() *rest.Config
}

type kubernetesClient struct {
	k8s           kubernetes.Interface
	snapshot      snapshotclient.Interface
	apiExtensions apiextensionsclient.Interface
	config        *rest.Config
}

// NewKubernetesClient initializes a new client for interacting with KubernetesClient.
// This client bundles access to various KubernetesClient APIs such as the core KubernetesClient API,
// snapshot operations, and API extensions.
func NewKubernetesClient(cfg *KubernetesCfg) (BaseClient, error) {
	// Check if the provided configuration object has a nil kubeconfig,
	// which is necessary to connect to the KubernetesClient API.
	if cfg.KubeConfig == nil {
		return nil, errors.New("kubernetes configuration is missing")
	}

	// Set the QPS and Burst values on the kubeconfig to control the rate limit of the client.
	// QPS is the number of queries per second the client can make to the KubernetesClient API,
	// and Burst is the maximum number of queries it can make in a short time frame.
	cfg.KubeConfig.QPS = cfg.QPS
	cfg.KubeConfig.Burst = cfg.Burst

	// Initialize the KubernetesClient core API client using the kubeconfig.
	// This client is used for most KubernetesClient operations such as creating pods, services, etc.
	k8sInterface, err := kubernetes.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Initialize the snapshot client using the same kubeconfig.
	// This client is specifically used for managing volume snapshots in KubernetesClient.
	snapshotInterface, err := snapshotclient.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Initialize the API extensions client using the same kubeconfig.
	// This client is used for interacting with API extensions that are not part of the core KubernetesClient API.
	apiExtensionsInterface, err := apiextensionsclient.NewForConfig(cfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	// Return the initialized kubernetesClient struct which implements the BaseClient interface.
	// This struct provides access to the initialized KubernetesClient core, snapshot, and API extensions clients.
	return &kubernetesClient{
		k8s:           k8sInterface,
		snapshot:      snapshotInterface,
		apiExtensions: apiExtensionsInterface,
		config:        cfg.KubeConfig,
	}, nil
}

func (client *kubernetesClient) KubernetesClient() kubernetes.Interface {
	return client.k8s
}

func (client *kubernetesClient) SnapshotClient() snapshotclient.Interface {
	return client.snapshot
}

func (client *kubernetesClient) ApiExtensionsClient() apiextensionsclient.Interface {
	return client.apiExtensions
}

func (client *kubernetesClient) ConfigClient() *rest.Config {
	return client.config
}
