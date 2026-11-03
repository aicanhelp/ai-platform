/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package k8s

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"text/template"

	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type KubernetesClient interface {
	GetClient() kubernetes.Interface
	GetDynamicClient() dynamic.Interface
	InstallYaml(filename string, variable map[string]string, ns string) error
	PatchYaml(filename string, variable map[string]string) error
	UninstallYaml(filename string, ns string) error
	WatchEventByAnnotation(namespace string)
	CreateNamespace(namespace *corev1.Namespace) error
	CreateSecret(secret *corev1.Secret) error
	GetNamespace(filename string) (string, error)
}

type Client struct {
	ClientSet     *kubernetes.Clientset
	DynamicClient dynamic.Interface

	restMapper *restmapper.DeferredDiscoveryRESTMapper
	// mu guards the lazy initialization of restMapper in getRESTMapper. The
	// lock covers clients constructed directly (e.g. in tests) and any future
	// concurrent use. See sig-installation#282.
	mu sync.Mutex
}

const (
	// yamlDecoderBufferSize is the buffer size for YAML decoder
	yamlDecoderBufferSize = 4096

	// clientQPS and clientBurst override client-go's conservative defaults
	// (discovery defaults to QPS=5, Burst=300). bkeadm is a one-shot
	// install/scale CLI rather than a long-running service, so a higher rate
	// limit has no downside and prevents client-side throttling against the
	// management apiserver when installing large clusters. See
	// sig-installation#282.
	clientQPS   = 100
	clientBurst = 200
)

func NewKubernetesClient(kubeConfig string) (KubernetesClient, error) {
	var config *rest.Config
	var err error

	if kubeConfig == "" {
		if home := homedir.HomeDir(); home != "" {
			kubeConfig = filepath.Join(home, ".kube", "config")
		}
	}
	if utils.Exists(kubeConfig) {
		config, err = clientcmd.BuildConfigFromFlags("", kubeConfig)
	}
	if err != nil {
		return nil, err
	}
	if config == nil {
		return nil, errors.New("The kube config configuration file does not exist. ")
	}

	// bkeadm is a one-shot CLI; raise the client-side rate limit so install/scale
	// of large clusters does not hit client-side throttling. See sig-installation#282.
	config.QPS = clientQPS
	config.Burst = clientBurst

	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return &Client{
		ClientSet:     clientSet,
		DynamicClient: dynamicClient,
	}, nil
}

// getRESTMapper returns the cached deferred RESTMapper, building it on first
// use. The deferred mapper performs API discovery once and reuses the result
// instead of re-discovering on every processYamlResources call: without this
// cache, a 64-node install made 65 full discovery passes against the
// management apiserver, exhausting client-go's discovery rate limit (QPS=5)
// for ~9 minutes. See sig-installation#282.
func (c *Client) getRESTMapper() *restmapper.DeferredDiscoveryRESTMapper {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.restMapper == nil {
		cachedDiscovery := memory.NewMemCacheClient(c.ClientSet.Discovery())
		c.restMapper = restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	}
	return c.restMapper
}

func (c *Client) GetClient() kubernetes.Interface {
	return c.ClientSet
}

func (c *Client) GetDynamicClient() dynamic.Interface {
	return c.DynamicClient
}

// yamlResourceHandler processes decoded YAML resources
type yamlResourceHandler func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error

// renderTemplateToTempFile renders a template file with variables and returns the temp file path
func renderTemplateToTempFile(filename string, variable map[string]string) (string, func(), error) {
	tmpl, err := template.ParseFiles(filename)
	if err != nil {
		return "", nil, err
	}
	tmpFile := fmt.Sprintf("/tmp/bke_%s.yaml", uuid.NewUUID())
	file, err := os.OpenFile(tmpFile, os.O_CREATE|os.O_WRONLY, utils.DefaultFilePermission)
	if err != nil {
		return "", nil, err
	}

	cleanup := func() {
		if removeErr := os.Remove(tmpFile); removeErr != nil && !os.IsNotExist(removeErr) {
			log.Debugf("remove temp yaml %s failed: %v", tmpFile, removeErr)
		}
	}

	if err = tmpl.Execute(file, variable); err != nil {
		if closeErr := file.Close(); closeErr != nil {
			log.Debugf("close temp yaml %s failed: %v", tmpFile, closeErr)
		}
		cleanup()
		return "", nil, err
	}

	if closeErr := file.Close(); closeErr != nil {
		cleanup()
		return "", nil, closeErr
	}

	return tmpFile, cleanup, nil
}

// processYamlResources processes each resource in a YAML file with the given handler
func (c *Client) processYamlResources(filepath string, handler yamlResourceHandler) error {
	f, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := f.Close(); closeErr != nil {
			log.Debugf("close yaml file %s failed: %v", filepath, closeErr)
		}
	}()

	decoder := yamlutil.NewYAMLOrJSONDecoder(f, yamlDecoderBufferSize)
	mapper := c.getRESTMapper()

	for {
		var rawObj runtime.RawExtension
		if err = decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
		if err != nil {
			return err
		}
		mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			// A CRD applied earlier in this run may not yet be reflected in the
			// cached discovery; invalidate the cache, re-discover once, then retry.
			if meta.IsNoMatchError(err) {
				log.Debugf("no REST mapping for %s/%s, refreshing discovery cache", gvk.Group, gvk.Kind)
				// mapper.Reset() invalidates the cached discovery client and
				// drops the deferred delegate, so the retry re-discovers.
				mapper.Reset()
				mapping, err = mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
			}
			if err != nil {
				return err
			}
		}

		unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			return err
		}
		unstruct := &unstructured.Unstructured{Object: unstructuredObj}

		if err = handler(unstruct, mapping); err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) InstallYaml(filename string, variable map[string]string, ns string) error {
	tmpFile, cleanup, err := renderTemplateToTempFile(filename, variable)
	if err != nil {
		return err
	}
	defer cleanup()

	return c.processYamlResources(tmpFile, func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
		return c.createResource(unstruct, mapping, ns)
	})
}

// createResource creates a single resource in the cluster
func (c *Client) createResource(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping, ns string) error {
	var obj *unstructured.Unstructured
	var err error

	targetNs := c.determineNamespace(unstruct, ns)
	if targetNs == "" {
		obj, err = c.DynamicClient.Resource(mapping.Resource).Create(context.Background(), unstruct, metav1.CreateOptions{})
	} else {
		obj, err = c.DynamicClient.Resource(mapping.Resource).Namespace(targetNs).Create(
			context.Background(), unstruct, metav1.CreateOptions{})
	}

	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			log.Debugf("%s/%s already exists", unstruct.GetKind(), unstruct.GetName())
			return nil
		}
		return err
	}
	if obj != nil {
		log.Debugf("%s/%s created", obj.GetKind(), obj.GetName())
	}
	return nil
}

// determineNamespace returns the namespace to use for a resource
func (c *Client) determineNamespace(unstruct *unstructured.Unstructured, ns string) string {
	if ns != "" {
		return ns
	}
	return unstruct.GetNamespace()
}

func (c *Client) PatchYaml(filename string, variable map[string]string) error {
	tmpFile, cleanup, err := renderTemplateToTempFile(filename, variable)
	if err != nil {
		return err
	}
	defer cleanup()

	return c.processYamlResources(tmpFile, func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
		jsonData, err := unstruct.MarshalJSON()
		if err != nil {
			return err
		}
		_, err = c.DynamicClient.Resource(mapping.Resource).Namespace(unstruct.GetNamespace()).Patch(
			context.Background(), unstruct.GetName(), types.MergePatchType, jsonData, metav1.PatchOptions{})
		if err != nil {
			return err
		}
		return nil
	})
}

func (c *Client) UninstallYaml(filename string, ns string) error {
	return c.processYamlResources(filename, func(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping) error {
		return c.deleteResource(unstruct, mapping, ns)
	})
}

// deleteResource deletes a single resource from the cluster
func (c *Client) deleteResource(unstruct *unstructured.Unstructured, mapping *meta.RESTMapping, ns string) error {
	targetNs := c.determineNamespace(unstruct, ns)
	return c.DynamicClient.Resource(mapping.Resource).Namespace(targetNs).Delete(
		context.Background(), unstruct.GetName(), metav1.DeleteOptions{})
}

// Listen for cluster deployment events

func (c *Client) WatchEventByAnnotation(namespace string) {
	stop := make(chan struct{})
	defer func() {
		_, isOpen := <-stop
		if isOpen {
			close(stop)
		}
	}()

	clientSet := c.GetClient()
	watchList := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			ls, err := clientSet.CoreV1().Events(namespace).List(context.Background(), options)
			return ls, err
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			w, err := clientSet.CoreV1().Events(namespace).Watch(context.Background(), options)
			return w, err
		},
	}
	eventHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e, ok := obj.(*corev1.Event)
			if ok {
				if e.Annotations == nil {
					return
				}
				if _, ok = e.Annotations[bkecommon.BKEFinishEventAnnotationKey]; ok {
					log.SteppedInfof(e.Type, "Reason: %s, Message: %s", e.Reason, e.Message)
					if !utils.IsChanClosed(stop) {
						close(stop)
					}
					return
				}
				if _, ok = e.Annotations[bkecommon.BKEEventAnnotationKey]; ok {
					log.SteppedInfof(e.Type, "Reason: %s, Message: %s", e.Reason, e.Message)
				}
			}
		},
		DeleteFunc: func(obj interface{}) {
			// Not needed: 只监听新事件，删除事件与部署进度无关
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			// Not needed: 事件更新与部署进度跟踪无关
		},
	}

	_, controller := cache.NewInformer(
		watchList,
		&corev1.Event{},
		0,
		eventHandler,
	)
	controller.Run(stop)
}

func (c *Client) CreateNamespace(namespace *corev1.Namespace) error {
	_, err := c.ClientSet.CoreV1().Namespaces().Get(context.Background(), namespace.Name, metav1.GetOptions{})
	if err != nil {
		_, err = c.ClientSet.CoreV1().Namespaces().Create(context.Background(), namespace, metav1.CreateOptions{})
	}
	return err
}

func (c *Client) CreateSecret(secret *corev1.Secret) error {
	_, err := c.ClientSet.CoreV1().Secrets(secret.Namespace).Get(context.Background(), secret.Name, metav1.GetOptions{})
	if err != nil {
		_, err = c.ClientSet.CoreV1().Secrets(secret.Namespace).Create(context.Background(), secret, metav1.CreateOptions{})
	}
	return err
}

// CreateNamespace 创建 Kubernetes namespace
// 该函数封装了创建 namespace 的通用逻辑，避免代码重复
// 返回错误供调用方处理
func CreateNamespace(k8sClient KubernetesClient, namespaceName string) error {
	namespace := corev1.Namespace{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Namespace",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: namespaceName,
		},
		Spec:   corev1.NamespaceSpec{},
		Status: corev1.NamespaceStatus{},
	}
	return k8sClient.CreateNamespace(&namespace)
}

func (c *Client) GetNamespace(filename string) (string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer f.Close()
	decoder := yamlutil.NewYAMLOrJSONDecoder(f, yamlDecoderBufferSize)
	for {
		rawObj := runtime.RawExtension{}
		if err := decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			} else {
				return "", err
			}
		}

		// runtime.Object
		obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
		if err != nil {
			return "", err
		}
		if gvk != nil && gvk.Version == "v1" && gvk.Kind == "Namespace" {
			// runtime.Object convert to unstructured
			unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
			if err != nil {
				return "", err
			}
			tmpMetadata, ok := unstructuredObj["metadata"].(map[string]interface{})
			if !ok {
				panic("failed to get metadata from unstructured object")
			}
			name, ok := tmpMetadata["name"].(string)
			if !ok {
				panic("failed to get name from metadata")
			}
			return name, nil
		}
	}
	return "", nil
}
