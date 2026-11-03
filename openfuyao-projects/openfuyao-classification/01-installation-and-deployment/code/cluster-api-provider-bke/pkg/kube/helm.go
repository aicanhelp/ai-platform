/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package kube

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	cliopt "k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	cmd "k8s.io/client-go/tools/clientcmd"
)

// RestClientConfig is an implementation of the genericclioptions.RESTClientGetter
type RestClientConfig struct {
	namespace  string
	restConfig *rest.Config
}

// NewRESTClientConfig return new rest client for helm release usage
func NewRESTClientConfig(namespace string, restConfig *rest.Config) cliopt.RESTClientGetter {
	return &RestClientConfig{
		namespace:  namespace,
		restConfig: restConfig,
	}
}

// ToRawKubeConfigLoader 返回 kubeconfig 加载器
func (r *RestClientConfig) ToRawKubeConfigLoader() cmd.ClientConfig {
	overrides := &cmd.ConfigOverrides{ClusterDefaults: cmd.ClusterDefaults}
	overrides.Context.Namespace = r.namespace
	rules := cmd.NewDefaultClientConfigLoadingRules()

	return cmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides)
}

// ToRESTConfig returns restconfig
func (r *RestClientConfig) ToRESTConfig() (*rest.Config, error) {
	return r.restConfig, nil
}

// ToRESTMapper returns a restmapper
func (r *RestClientConfig) ToRESTMapper() (meta.RESTMapper, error) {
	mapperCache, err := GetDynamicRESTMapper(r.restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get dynamic REST mapper: %w", err)
	}
	se := restmapper.NewShortcutExpander(mapperCache.RESTMapper(), mapperCache.DiscoveryClient())
	return se, nil
}

// ToDiscoveryClient returns discovery client
func (r *RestClientConfig) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	mapperCache, err := GetDynamicRESTMapper(r.restConfig)
	if err != nil {
		return nil, err
	}
	return mapperCache.DiscoveryClient(), nil
}
