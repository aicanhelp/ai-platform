/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clientutil

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/kubeclient"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

// ClientSetFromManagerClusterSecret returns a ready-to-use client from a kubeconfig secret
func ClientSetFromManagerClusterSecret(nsName ...string) (*kubernetes.Clientset, error) {
	path := pkiutil.GetDefaultKubeConfigPath()

	// if ha we need use the ha kubeconfig, to get the clientset
	if len(nsName) == utils.NamespaceAndNameLen {
		//step 1 get manager cluster clientset
		managerClusterConfig := fmt.Sprintf("%s/%s", utils.Workspace, "config")
		config, err := clientcmd.BuildConfigFromFlags("", managerClusterConfig)
		if err != nil {
			return nil, err
		}
		if err != nil {
			return nil, fmt.Errorf("failed to get manger cluster rest config: %w", err)
		}
		clientSet, err := kubernetes.NewForConfig(config)
		if err != nil {
			return nil, err
		}
		// step 2 get ha kubeconfig secret
		secretName := fmt.Sprintf("%s-kubeconfig", nsName[1])

		secret, err := clientSet.CoreV1().Secrets(nsName[0]).Get(context.Background(), secretName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get ha kubeconfig secret: %w", err)
		}

		if err := pkiutil.StoreClusterAPICert(secret, os.TempDir()); err != nil {
			return nil, errors.Errorf("failed to store ha kubeconfig secret to %s, err: %v", os.TempDir(), err)
		}
		path = filepath.Join(os.TempDir(), pkiutil.BKEAdminKubeConfig().BaseName+".conf")
	}

	kubeConfig, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "admin kubeconfig load failed")
	}
	overrides := clientcmd.ConfigOverrides{Timeout: "10s"}
	clientConfig, err := clientcmd.NewDefaultClientConfig(*kubeConfig, &overrides).ClientConfig()
	if err != nil {
		return nil, errors.Wrap(err, "API client configuration create failed")
	}

	c, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, errors.Wrap(err, "API client create failed")
	}
	return c, nil
}

// Client exports kubeclient.Client for backward compatibility
type Client = kubeclient.Client

// NewKubernetesClient creates a new kubernetes client from a kubeconfig file
// This function wraps kubeclient.NewClient for backward compatibility
func NewKubernetesClient(path string) (*Client, error) {
	return kubeclient.NewClient(path)
}
