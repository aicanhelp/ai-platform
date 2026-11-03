/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kubeclient

import (
	"sync"

	"github.com/pkg/errors"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
)

type Client struct {
	ClientSet     *kubernetes.Clientset
	DynamicClient dynamic.Interface
}

var addToScheme sync.Once

// NewClient creates a new kubernetes client from a kubeconfig file
func NewClient(path string) (*Client, error) {
	// Load kubeconfig from file
	config, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load kubeconfig")
	}

	// Set client config overrides with 10s timeout
	overrides := clientcmd.ConfigOverrides{Timeout: "10s"}
	// Create REST config from kubeconfig
	restConfig, err := clientcmd.NewDefaultClientConfig(*config, &overrides).ClientConfig()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create REST config")
	}

	// Create kubernetes clientset from REST config
	clientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create kubernetes clientset")
	}

	// Create dynamic client from REST config (for handling arbitrary kubernetes resources)
	dynamicClient, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create dynamic client")
	}

	// support CRD v1 || v1beta1
	// 使用 sync.Once 但避免 panic，而是返回错误
	var addToSchemeErr error
	addToScheme.Do(func() {
		if err := apiextv1.AddToScheme(scheme.Scheme); err != nil {
			addToSchemeErr = errors.Wrap(err, "failed to add apiextensions v1 to scheme")
			return
		}
		if err := apiextv1beta1.AddToScheme(scheme.Scheme); err != nil {
			addToSchemeErr = errors.Wrap(err, "failed to add apiextensions v1beta1 to scheme")
			return
		}
	})

	if addToSchemeErr != nil {
		return nil, addToSchemeErr
	}

	return &Client{
		ClientSet:     clientSet,
		DynamicClient: dynamicClient,
	}, nil
}
