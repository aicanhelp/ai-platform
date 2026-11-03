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

package kube

import (
	"context"
	"time"

	"helm.sh/helm/v3/pkg/kube"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// Timeout defines a 30-second duration for operation timeout
// Operations exceeding this limit will be terminated and return a timeout error
const Timeout = 30 * time.Second

// waiter handles waiting for Kubernetes resources to become ready.
type waiter struct {
	// The object to wait for.
	unstructuredObj *unstructured.Unstructured
	namespace       string
	name            string
	block           bool
	timeout         time.Duration
	interval        time.Duration
	checker         *readyChecker
	ctx             context.Context
}

// readyChecker checks if Kubernetes resources are ready.
type readyChecker struct {
	client        *kubernetes.Clientset
	dynamicClient dynamic.Interface
	log           *log.Logger
	bkeLog        *bkev1beta1.BKELogger
	pausedAsReady bool
	fullComplete  bool

	// helmClient holds the reference to Helm client for waiting operations.
	helmClient *kube.Client
}

// waitWithHelm waits for the specified unstructured Kubernetes object
// using Helm's wait mechanism via the checker's helmClient.
// It wraps the object into a kube.ResourceList and delegates the waiting logic.
func (w *waiter) waitWithHelm(obj *unstructured.Unstructured, task *Task) error {
	resources := kube.ResourceList{
		{
			Name:      obj.GetName(),
			Namespace: obj.GetNamespace(),
			Object:    obj,
		},
	}
	// Use the helmClient from checker directly
	return w.checker.helmClient.Wait(resources, task.Timeout)
}

// NewClient creates a new Helm kube.Client instance using existing clients.
func NewClient(config *rest.Config, clientset kubernetes.Interface, dynamicClient dynamic.Interface) *kube.Client {
	if config == nil {
		return nil
	}

	// Create factory with existing clients
	factory := &kubeFactory{
		config:        config,
		client:        clientset,
		dynamicClient: dynamicClient,
	}

	return kube.New(factory)
}

// Wait waits for a Kubernetes object to become ready using Helm's wait functionality.
// For custom resources (like Paas), it uses our own implementation.
func (c *Client) Wait(obj *unstructured.Unstructured, task *Task) error {
	waiter := newWaiterFromUnstructured(obj).
		setPoller(task).
		setChecker(c)

	if obj.GetObjectKind().GroupVersionKind().Kind == "CustomResourceDefinition" {
		return nil
	}

	return waiter.waitWithHelm(obj, task)
}

// setPoller configures the polling parameters for the waiter.
func (w *waiter) setPoller(task *Task) *waiter {
	w.timeout = task.Timeout
	w.interval = task.Interval
	w.block = task.Block
	return w
}

// setChecker configures the readiness checker for the waiter.
func (w *waiter) setChecker(c *Client) *waiter {
	w.checker = newCheckerFromKubeClient(c)
	w.checker.fullComplete = w.block
	w.ctx = c.Ctx
	return w
}

// newCheckerFromKubeClient creates a new readyChecker from a kube Client.
func newCheckerFromKubeClient(c *Client) *readyChecker {
	return &readyChecker{
		client:        c.ClientSet,
		dynamicClient: c.DynamicClient,
		log:           c.Log,
		bkeLog:        c.BKELog,
		pausedAsReady: true,
		helmClient:    NewClient(c.RestConfig, c.ClientSet, c.DynamicClient),
	}
}

// newWaiterFromUnstructured creates a new waiter from an unstructured Kubernetes object.
func newWaiterFromUnstructured(obj *unstructured.Unstructured) *waiter {
	return &waiter{
		unstructuredObj: obj,
		namespace:       obj.GetNamespace(),
		name:            obj.GetName(),
	}
}

// ProductStatus defines the status information of a product.
type ProductStatus struct {
	Name           string            `json:"name"`
	StartTime      *metav1.Time      `json:"startTime,omitempty"`
	UpdateTime     *metav1.Time      `json:"updateTime,omitempty"`
	CompletionTime *metav1.Time      `json:"completionTime,omitempty"`
	Health         bool              `json:"health"`
	Component      []ComponentStatus `json:"component,omitempty"`
	Reason         string            `json:"reason"`
}

// ComponentStatus defines the status information of a component within a product.
type ComponentStatus struct {
	Name     string `json:"name"`
	Resource string `json:"resource"`
	Health   bool   `json:"componentHealth"`
	Message  string `json:"message"`
}

// paasProductComponentStatus stores the health status of PaaS product components.
var paasProductComponentStatus map[string]map[string]bool

/* ----------------------------
   kubeFactory implements genericclioptions.RESTClientGetter
----------------------------- */

// kubeFactory implements kube.Factory interface using existing clients
type kubeFactory struct {
	config        *rest.Config
	client        kubernetes.Interface
	dynamicClient dynamic.Interface
}

// ToRESTConfig returns a cleaned REST config
func (f *kubeFactory) ToRESTConfig() (*rest.Config, error) {
	return f.config, nil
}

// ToRESTMapper returns a REST mapper
func (f *kubeFactory) ToRESTMapper() (meta.RESTMapper, error) {
	mapperCache, err := GetDynamicRESTMapper(f.config)
	if err != nil {
		return nil, err
	}
	expander := restmapper.NewShortcutExpander(mapperCache.RESTMapper(), mapperCache.DiscoveryClient())
	return expander, nil
}

// ToDiscoveryClient returns a discovery client
func (f *kubeFactory) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	config, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	config.Timeout = Timeout

	mapperCache, err := GetDynamicRESTMapper(config)
	if err != nil {
		return nil, err
	}
	return mapperCache.DiscoveryClient(), nil
}

// ToRawKubeConfigLoader returns a clientcmd client config
func (f *kubeFactory) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	overrides := &clientcmd.ConfigOverrides{}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, overrides)
}
