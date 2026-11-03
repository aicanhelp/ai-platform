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

// Package utils contains a collection of utility functions and helpers that provide
// common functionalities useful across the entire engineering project.
package utils

import (
	"context"
	"encoding/json"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"github.com/karmada-io/karmada/pkg/karmadactl/util/apiclient"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// CreateHostClient creates and returns host cluster kubernetes clientset and apiConfig.
func CreateHostClient() (*kubernetes.Clientset, *api.Config, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		zlog.LogErrorf("Error Creating Hostcluster restconfig: %v", err)
		return nil, nil, err
	}

	apiConfig := api.NewConfig()

	// transfer restconfig to apiconfig
	err = restConfig2ApiConfig(config, apiConfig)
	if err != nil {
		zlog.LogErrorf("Error transferring restconfig to apiconfig: %v", err)
		return nil, nil, err
	}

	clusterClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.LogErrorf("Error creating cluster client: %v", err)
		return nil, nil, err
	}
	zlog.LogInfof("Join Hostcluster config. endpoint: %s", config.Host)
	return clusterClient, apiConfig, nil
}

// CreateClusterClient creates and returns a new kubernetes client based on the provided configuration.
func CreateClusterClient(config *api.Config) (*kubernetes.Clientset, error) {
	clientConfig := clientcmd.NewNonInteractiveClientConfig(*config, config.CurrentContext,
		&clientcmd.ConfigOverrides{}, nil)

	restConfig, err := clientConfig.ClientConfig()
	if err != nil {
		zlog.LogErrorf("Error creating cluster config: %v", err)
		return nil, err
	}

	// kubernetes.NewForConfigDie 错误会终止执行
	clusterClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		zlog.LogErrorf("Error creating cluster client: %v", err)
		return nil, err
	}
	zlog.LogInfof("Join cluster config. endpoint: %s", restConfig.Host)
	return clusterClient, nil
}

// CreateClient creates and returns a kubernetes client and kubeconfig based on the provied config path.
func CreateClient(path string) (*kubernetes.Clientset, *api.Config, error) {
	clientConfig, err := clientcmd.LoadFromFile(path)
	if err != nil {
		zlog.LogErrorf("Error Creating kubernetes Config : %v", err)
		return nil, nil, err
	}

	config, err := clientcmd.NewNonInteractiveClientConfig(*clientConfig, "",
		&clientcmd.ConfigOverrides{}, nil).ClientConfig()

	clusterClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.LogErrorf("Error creating client from path: %v", err)
		return nil, nil, err
	}
	zlog.LogInfof("Get cluster endpoint: %s", config.Host)
	return clusterClient, clientConfig, nil

}

// CreateKarmadaClusterClient creates and returns kubernetes client and karmada clietnset based on
// the karmada config path.
func CreateKarmadaClusterClient() (*kubernetes.Clientset, *versioned.Clientset, error) {
	path := KarmadaConfigPath
	restConfig, err := apiclient.RestConfig("", path)
	if err != nil {
		zlog.LogErrorf("Error creating karmada cluster config: %v", err)
		return nil, nil, err
	}
	kClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		zlog.LogErrorf("Error creating karmada kubernetes client: %v", err)
		return nil, nil, err
	}

	kVersionedClient, err := versioned.NewForConfig(restConfig)
	if err != nil {
		zlog.LogErrorf("Error creating karmada server client: %v", err)
		return nil, nil, err
	}

	return kClient, kVersionedClient, nil
}

// RetriveKubeconfigClient retrieves the kubeconfig data for the specified cluster object and returns
// a new kubernetes client configured with this kubeconfig.
func RetriveKubeconfigClient(mgrClient client.Client, clusterTag string) (*kubernetes.Clientset, error) {
	if clusterTag == "host" {
		kClient, _, err := CreateHostClient()
		if err != nil {
			zlog.LogErrorf("Error Creating host cluster client: %v", err)
			return nil, err
		}
		return kClient, nil
	}

	clusterObj := v1beta1.Cluster{}
	if err := mgrClient.Get(context.TODO(), client.ObjectKey{Name: clusterTag}, &clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object: %v", err)
		return nil, err
	}

	kubeData := clusterObj.Spec.KubeConfig.Raw
	var kubeconfig api.Config
	err := json.Unmarshal(kubeData, &kubeconfig)
	if err != nil {
		zlog.LogErrorf("Errorf kubeconfig data unmarshal failed.")
		return nil, err
	}

	kClient, err := CreateClusterClient(&kubeconfig)
	if err != nil {
		zlog.LogErrorf("Error Creating cluster client from json: %v", err)
		return nil, err
	}

	zlog.LogInfof("Creating cluster success!")
	return kClient, nil
}

func restConfig2ApiConfig(restCfg *rest.Config, apiCfg *api.Config) error {
	cluster := api.NewCluster()
	cluster.Server = restCfg.Host
	cluster.CertificateAuthority = restCfg.CAFile
	cluster.CertificateAuthorityData = restCfg.CAData

	context := api.NewContext()
	context.Cluster = "kubernetes"
	context.AuthInfo = "kubernetes-admin"

	authInfo := api.NewAuthInfo()
	authInfo.Token = restCfg.BearerToken
	authInfo.ClientCertificateData = restCfg.CertData
	authInfo.ClientCertificate = restCfg.CAFile
	authInfo.ClientKey = restCfg.KeyFile
	authInfo.ClientKeyData = restCfg.KeyData

	apiCfg.Clusters["kubernetes"] = cluster
	apiCfg.Contexts["kubernetes-admin@kubernetes"] = context
	apiCfg.AuthInfos["kubernetes-admin"] = authInfo
	apiCfg.CurrentContext = "kubernetes-admin@kubernetes"

	return nil
}

// LoadKubeconfigRawData takes a raw JSON message containing kubeconfig data and attempts to unmarshal
// it into an api.Config struct.
func LoadKubeconfigRawData(kubeconfigData json.RawMessage, apiKubeconfig *api.Config) error {
	var rawConfig struct {
		APIVersion string `json:"apiVersion"`
		Kind       string `json:"kind"`
		Clusters   []struct {
			Name    string      `json:"name"`
			Cluster api.Cluster `json:"cluster"`
		} `json:"clusters"`
		Contexts []struct {
			Name    string      `json:"name"`
			Context api.Context `json:"context"`
		} `json:"contexts"`
		CurrentContext string          `json:"current-context"`
		Preferences    api.Preferences `json:"preferences"`
		Users          []struct {
			Name string       `json:"name"`
			User api.AuthInfo `json:"user"`
		} `json:"users"`
	}

	err := json.Unmarshal(kubeconfigData, &rawConfig)
	if err != nil {
		zlog.LogErrorf("Error Unmarshal cluster kubeconfig: %v", err)
		return err
	}

	apiKubeconfig.APIVersion = rawConfig.APIVersion
	apiKubeconfig.Kind = rawConfig.Kind
	apiKubeconfig.CurrentContext = rawConfig.CurrentContext
	apiKubeconfig.Preferences = rawConfig.Preferences
	apiKubeconfig.Clusters = make(map[string]*api.Cluster)
	apiKubeconfig.Contexts = make(map[string]*api.Context)
	apiKubeconfig.AuthInfos = make(map[string]*api.AuthInfo)

	for _, cluster := range rawConfig.Clusters {
		apiKubeconfig.Clusters[cluster.Name] = &cluster.Cluster
	}

	for _, cluster := range rawConfig.Users {
		apiKubeconfig.AuthInfos[cluster.Name] = &cluster.User
	}

	for _, cluster := range rawConfig.Contexts {
		apiKubeconfig.Contexts[cluster.Name] = &cluster.Context
	}

	return nil
}
