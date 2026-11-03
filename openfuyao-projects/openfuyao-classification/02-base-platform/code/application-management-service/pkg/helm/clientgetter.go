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

// Package helm
package helm

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	cliopt "k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	cmd "k8s.io/client-go/tools/clientcmd"

	"application-management-service/pkg/zlog"
)

// RestClientConfig is an implementation of the genericclioptions.RESTClientGetter,
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

// ToRESTConfig returns restconfig
func (r *RestClientConfig) ToRESTConfig() (*rest.Config, error) {
	return r.restConfig, nil
}

// ToDiscoveryClient returns discovery client
func (r *RestClientConfig) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	clientForConfig, _ := discovery.NewDiscoveryClientForConfig(r.restConfig)
	return memory.NewMemCacheClient(clientForConfig), nil
}

// ToRESTMapper returns a restmapper
func (r *RestClientConfig) ToRESTMapper() (meta.RESTMapper, error) {
	c, err := r.ToDiscoveryClient()
	if err != nil {
		return nil, fmt.Errorf("couldn't create discovery client: %s", err)
	}
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(c)
	se := restmapper.NewShortcutExpander(restMapper, c, func(warning string) {
		zlog.Warnf("Kubernetes API warning: %s", warning)
	})
	return se, nil
}

// ToRawKubeConfigLoader 返回 kubeconfig 加载器
func (r *RestClientConfig) ToRawKubeConfigLoader() cmd.ClientConfig {
	overrides := &cmd.ConfigOverrides{ClusterDefaults: cmd.ClusterDefaults}
	overrides.Context.Namespace = r.namespace
	rules := cmd.NewDefaultClientConfigLoadingRules()

	return cmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides)
}
