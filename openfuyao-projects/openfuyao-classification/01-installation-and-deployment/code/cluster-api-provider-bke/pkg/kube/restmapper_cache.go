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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"

	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiextensionsinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const restMapperCacheKeySeparator = "|ca="

type perClusterRESTMapper struct {
	mapper    *restmapper.DeferredDiscoveryRESTMapper
	discovery discovery.CachedDiscoveryInterface
	stopCh    chan struct{}
	stopOnce  sync.Once
}

func (m *perClusterRESTMapper) RESTMapper() meta.RESTMapper {
	return m.mapper
}

func (m *perClusterRESTMapper) DiscoveryClient() discovery.CachedDiscoveryInterface {
	return m.discovery
}

func (m *perClusterRESTMapper) Invalidate() {
	log.Infof("invalidating RESTMapper discovery cache")
	m.discovery.Invalidate()
	m.mapper.Reset()
}

func (m *perClusterRESTMapper) Stop() {
	m.stopOnce.Do(func() {
		close(m.stopCh)
	})
}

var restMapperCache sync.Map

// GetDynamicRESTMapper returns a RESTMapper and discovery cache scoped by API server host and CA fingerprint.
// The CA fingerprint is part of the cache key because e2e and reinstall flows can recreate a different
// target cluster on the same host:port with a new CA certificate.
func GetDynamicRESTMapper(cfg *rest.Config) (*perClusterRESTMapper, error) {
	key, err := restMapperCacheKey(cfg)
	if err != nil {
		return nil, err
	}

	if cached, ok := restMapperCache.Load(key); ok {
		log.Infof("RESTMapper cache hit for host %s", cfg.Host)
		return cached.(*perClusterRESTMapper), nil
	}

	mapper, err := newPerClusterRESTMapper(cfg)
	if err != nil {
		return nil, err
	}
	actual, loaded := restMapperCache.LoadOrStore(key, mapper)
	if loaded {
		mapper.Stop()
		log.Debugf("RESTMapper cache already created by another goroutine for host %s", cfg.Host)
		return actual.(*perClusterRESTMapper), nil
	}
	log.Infof("RESTMapper cache created for host %s", cfg.Host)
	return mapper, nil
}

// DeleteRESTMapperCacheByHost removes all cached RESTMappers for the API server host.
// A host can have multiple cached entries when the same host:port is reused with different CAs.
func DeleteRESTMapperCacheByHost(host string) {
	if strings.TrimSpace(host) == "" {
		return
	}
	prefix := host + restMapperCacheKeySeparator
	restMapperCache.Range(func(key, value interface{}) bool {
		cacheKey, ok := key.(string)
		if !ok || !strings.HasPrefix(cacheKey, prefix) {
			return true
		}
		restMapperCache.Delete(cacheKey)
		if mapper, ok := value.(*perClusterRESTMapper); ok && mapper != nil {
			mapper.Stop()
		}
		log.Infof("RESTMapper cache deleted for host %s", host)
		return true
	})
}

func restMapperCacheKey(cfg *rest.Config) (string, error) {
	if cfg == nil || cfg.Host == "" {
		return "", fmt.Errorf("rest config host is empty")
	}
	fingerprint, err := restConfigCAFingerprint(cfg)
	if err != nil {
		return "", err
	}
	return cfg.Host + restMapperCacheKeySeparator + fingerprint, nil
}

func restConfigCAFingerprint(cfg *rest.Config) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("rest config is nil")
	}
	caData := cfg.TLSClientConfig.CAData
	if len(caData) == 0 && cfg.TLSClientConfig.CAFile != "" {
		data, err := os.ReadFile(cfg.TLSClientConfig.CAFile)
		if err != nil {
			return "", fmt.Errorf("read rest config CA file %q failed: %w", cfg.TLSClientConfig.CAFile, err)
		}
		caData = data
	}
	sum := sha256.Sum256(caData)
	return hex.EncodeToString(sum[:]), nil
}

func newPerClusterRESTMapper(cfg *rest.Config) (*perClusterRESTMapper, error) {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(ApplyThrottlingConfig(cfg))
	if err != nil {
		return nil, err
	}
	cachedDiscovery := memory.NewMemCacheClient(discoveryClient)
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)

	crdInformer, err := newCRDInformer(cfg)
	if err != nil {
		return nil, err
	}

	m := &perClusterRESTMapper{
		mapper:    mapper,
		discovery: cachedDiscovery,
		stopCh:    make(chan struct{}),
	}

	if _, err := crdInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			logRESTMapperCRDEvent("add", obj)
			m.Invalidate()
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			logRESTMapperCRDEvent("update", newObj)
			m.Invalidate()
		},
		DeleteFunc: func(obj interface{}) {
			logRESTMapperCRDEvent("delete", obj)
			m.Invalidate()
		},
	}); err != nil {
		return nil, err
	}

	go crdInformer.Run(m.stopCh)

	return m, nil
}

func newCRDInformer(cfg *rest.Config) (cache.SharedIndexInformer, error) {
	clientset, err := apiextensionsclient.NewForConfig(ApplyThrottlingConfig(cfg))
	if err != nil {
		return nil, err
	}
	factory := apiextensionsinformers.NewSharedInformerFactory(clientset, 0)
	return factory.Apiextensions().V1().CustomResourceDefinitions().Informer(), nil
}
func logRESTMapperCRDEvent(action string, obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil || key == "" {
		log.Infof("CRD %s event received, invalidating RESTMapper discovery cache", action)
		return
	}
	log.Infof("CRD %s event received for %s, invalidating RESTMapper discovery cache", action, key)
}
