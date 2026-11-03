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

// Package config configure the oauth2-server defined in go-oauth2
package config

import (
	"os"
	"os/user"
	"path"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"openfuyao/oauth-server/pkg/zlog"
)

// KubernetesConfig specifies the configuration for k8s client
type KubernetesConfig struct {
	// KubeConfigFile defines the path to fetch kubeconfig
	KubeConfigFile string `json:"KubeConfigFile" yaml:"KubeConfigFile"`

	// QPS is kubernetes clientset qps
	QPS float32 `json:"QPS,omitempty" yaml:"QPS,omitempty"`

	// Burst is kubernetes clientset burst
	Burst int `json:"Burst,omitempty" yaml:"Burst,omitempty"`
}

// NewKubernetesConfig 返回默认的 k8s 相关配置（如KubeConfig）
func NewKubernetesConfig() *KubernetesConfig {
	return &KubernetesConfig{
		KubeConfigFile: getDefaultKubeConfigFile(),
		QPS:            1e6,
		Burst:          1e6,
	}
}

// Validate validate kubernetesConfig
func (k *KubernetesConfig) Validate() []error {
	var errs []error
	// since we have rest.InclusterConfig, the KubeConfigFile is allowed to be empty here, no validation is set
	return errs
}

func getDefaultKubeConfigFile() string {
	kubeConfigFile := ""
	homePath := homedir.HomeDir()
	if homePath == "" {
		if u, err := user.Current(); err == nil {
			homePath = u.HomeDir
		}
	}

	userHomeConfig := path.Join(homePath, ".kube/config")
	if _, err := os.Stat(userHomeConfig); err == nil {
		kubeConfigFile = userHomeConfig
	}

	return kubeConfigFile
}

// GetKubeConfigOrInClusterConfig loads in-cluster config if kubeConfigFile is empty or the file if not,
// then applies overrides.
func GetKubeConfigOrInClusterConfig(k8sConfig *KubernetesConfig) (clientConfig *rest.Config) {
	var err error
	if k8sConfig != nil && len(k8sConfig.KubeConfigFile) > 0 {
		clientConfig, err = clientcmd.BuildConfigFromFlags("", k8sConfig.KubeConfigFile)
		if err == nil {
			return clientConfig
		}
	}

	if k8sConfig == nil {
		// make sure we have qps and burst fields
		k8sConfig = NewKubernetesConfig()
	}

	clientConfig, err = rest.InClusterConfig()
	if err != nil {
		zlog.LogWarn("Get KubeConfig In Cluster Config error, Attempting to obtain from the default config file")
		kubeConfigFile := getDefaultKubeConfigFile()
		if kubeConfigFile == "" {
			zlog.LogFatalf("Error creating in-cluster config: %v", err)
		}
		if _, err = os.Stat(kubeConfigFile); err != nil {
			zlog.LogFatalf("Error creating in-filePath config: %v", err)
		}
		clientConfig, err = clientcmd.BuildConfigFromFlags("", kubeConfigFile)
		if err != nil {
			zlog.LogFatalf("Error creating k8s config")
		}
	}

	// override burst and qps
	if k8sConfig.QPS != 0 {
		clientConfig.QPS = k8sConfig.QPS
	}
	if k8sConfig.Burst != 0 {
		clientConfig.Burst = k8sConfig.Burst
	}

	return clientConfig
}

// GetKubernetesClient returns the k8sClient with k8sConfig
func GetKubernetesClient(k8sConfig *KubernetesConfig) kubernetes.Interface {
	kubeConfig := GetKubeConfigOrInClusterConfig(k8sConfig)
	k8sClient, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		// Here use LogFatalf by design. We MUST exit if k8sclient is not available.
		zlog.LogFatalf("Error converting k8s config to k8sclient")
	}

	return k8sClient
}

// GetDynamicClient returns the dynamicClient with k8sConfig
func GetDynamicClient(k8sConfig *KubernetesConfig) dynamic.Interface {
	kubeConfig := GetKubeConfigOrInClusterConfig(k8sConfig)
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		zlog.LogFatalf("Error converting k8s config to dynamicClient")
	}

	return dynamicClient
}
