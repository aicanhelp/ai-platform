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

// Package openfuyao defines the openfuyao provider configs
package openfuyao

import (
	"flag"

	"k8s.io/client-go/kubernetes"
)

// KubeClientOptions defines the k8s client option
type KubeClientOptions struct {
	// RemoteKubeConfigFile is the file to use to connect to a "normal" kube API server
	RemoteKubeConfigFile string
}

// NewKubeClientOptions init KubeClientOptions
func NewKubeClientOptions() *KubeClientOptions {
	return &KubeClientOptions{}
}

// Validate validates kubeClientOption
func (o *KubeClientOptions) Validate() []error {
	return []error{}
}

// AddFlags add necessary configs for openFuyao kubeClient
func (o *KubeClientOptions) AddFlags(fs *flag.FlagSet) {
	fs.StringVar(&o.RemoteKubeConfigFile, "kubeconfig", o.RemoteKubeConfigFile, ""+
		"kubeconfig file pointing at the 'core' OpenFuyao server that has the oauth-server running on it")
}

// ToKubeClientConfig generate the kubeconfig
func (o *KubeClientOptions) ToKubeClientConfig() (*kubernetes.Clientset, error) {
	clientConfig, err := GetClientConfig(o.RemoteKubeConfigFile)
	if err != nil {
		return nil, err
	}

	kubeClient, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	return kubeClient, nil
}
