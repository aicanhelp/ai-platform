/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
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
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// kubernetesClients
type kubernetesClients struct {
	kubernetes   kubernetes.Interface
	kubeConfig   *rest.Config
	snapshot     versioned.Interface
	apiExtension clientset.Interface
	dynamic      dynamic.Interface
}

// NewKubernetesClient is new k8s client
func NewKubernetesClient(k8sCfg *KubernetesCfg) (Clients, error) {
	if k8sCfg.KubeConfig == nil {
		return nil, errors.New("k8s config get nil")
	}
	var err error = nil
	k8sCfg.KubeConfig.QPS = k8sCfg.QPS
	k8sCfg.KubeConfig.Burst = k8sCfg.Burst

	var k8sClient kubernetesClients
	k8sClient.kubernetes, err = kubernetes.NewForConfig(k8sCfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	k8sClient.snapshot, err = versioned.NewForConfig(k8sCfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	k8sClient.apiExtension, err = clientset.NewForConfig(k8sCfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	k8sClient.dynamic, err = dynamic.NewForConfig(k8sCfg.KubeConfig)
	if err != nil {
		return nil, err
	}

	k8sClient.kubeConfig = k8sCfg.KubeConfig

	return &k8sClient, nil
}

func (k *kubernetesClients) Kubernetes() kubernetes.Interface {
	return k.kubernetes
}

func (k *kubernetesClients) ApiExtension() clientset.Interface {
	return k.apiExtension
}

func (k *kubernetesClients) Snapshot() versioned.Interface {
	return k.snapshot
}

func (k *kubernetesClients) Dynamic() dynamic.Interface {
	return k.dynamic
}

func (k *kubernetesClients) KubeConfig() *rest.Config {
	return k.kubeConfig
}

// Clients is k8s client interface
type Clients interface {
	Kubernetes() kubernetes.Interface
	KubeConfig() *rest.Config
	Snapshot() versioned.Interface
	ApiExtension() clientset.Interface
	Dynamic() dynamic.Interface
}
