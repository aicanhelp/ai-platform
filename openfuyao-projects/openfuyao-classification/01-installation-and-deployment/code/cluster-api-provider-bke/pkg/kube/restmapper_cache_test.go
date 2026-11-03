package kube

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/discovery/cached/memory"
	fakekubernetes "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

func TestGetDynamicRESTMapperCacheHit(t *testing.T) {
	host := "https://cached.example:6443"
	cfg := &rest.Config{
		Host: host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-a"),
		},
	}
	cacheKey, err := restMapperCacheKey(cfg)
	if err != nil {
		t.Fatalf("restMapperCacheKey() error = %v", err)
	}
	cachedMapper := newTestCachedRESTMapper()
	restMapperCache.Store(cacheKey, cachedMapper)
	defer restMapperCache.Delete(cacheKey)
	defer cachedMapper.Stop()
	cachedMapper.Invalidate()

	got, err := GetDynamicRESTMapper(cfg)
	if err != nil {
		t.Fatalf("GetDynamicRESTMapper() error = %v", err)
	}
	if got != cachedMapper {
		t.Fatal("GetDynamicRESTMapper() should return cached mapper")
	}
}

func TestRESTMapperCacheKeyIncludesCAFingerprint(t *testing.T) {
	host := "https://same.example:6443"
	keyA, err := restMapperCacheKey(&rest.Config{
		Host: host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-a"),
		},
	})
	if err != nil {
		t.Fatalf("restMapperCacheKey() error = %v", err)
	}
	keyB, err := restMapperCacheKey(&rest.Config{
		Host: host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-b"),
		},
	})
	if err != nil {
		t.Fatalf("restMapperCacheKey() error = %v", err)
	}
	if keyA == keyB {
		t.Fatal("same host with different CA data should use different RESTMapper cache keys")
	}
	if !strings.HasPrefix(keyA, host+restMapperCacheKeySeparator) {
		t.Fatalf("RESTMapper cache key %q should keep host prefix", keyA)
	}
}

func TestDeleteRESTMapperCacheByHost(t *testing.T) {
	host := "https://delete.example:6443"
	otherHost := "https://other.example:6443"
	keyA := mustRESTMapperCacheKey(t, &rest.Config{
		Host: host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-a"),
		},
	})
	keyB := mustRESTMapperCacheKey(t, &rest.Config{
		Host: host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-b"),
		},
	})
	otherKey := mustRESTMapperCacheKey(t, &rest.Config{
		Host: otherHost,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("ca-a"),
		},
	})

	mapperA := newTestCachedRESTMapper()
	mapperB := newTestCachedRESTMapper()
	otherMapper := newTestCachedRESTMapper()
	restMapperCache.Store(keyA, mapperA)
	restMapperCache.Store(keyB, mapperB)
	restMapperCache.Store(otherKey, otherMapper)
	defer restMapperCache.Delete(keyA)
	defer restMapperCache.Delete(keyB)
	defer restMapperCache.Delete(otherKey)
	defer otherMapper.Stop()

	DeleteRESTMapperCacheByHost(host)

	if _, ok := restMapperCache.Load(keyA); ok {
		t.Fatalf("RESTMapper cache key %q should be deleted", keyA)
	}
	if _, ok := restMapperCache.Load(keyB); ok {
		t.Fatalf("RESTMapper cache key %q should be deleted", keyB)
	}
	if _, ok := restMapperCache.Load(otherKey); !ok {
		t.Fatalf("RESTMapper cache key %q should not be deleted", otherKey)
	}
	assertMapperStopped(t, mapperA)
	assertMapperStopped(t, mapperB)
}

func TestPerClusterRESTMapperStopIsIdempotent(t *testing.T) {
	mapper := &perClusterRESTMapper{stopCh: make(chan struct{})}
	mapper.Stop()
	mapper.Stop()

	assertMapperStopped(t, mapper)
}

func TestLogRESTMapperCRDEvent(t *testing.T) {
	logRESTMapperCRDEvent("add", &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "apiextensions.k8s.io/v1",
		"kind":       "CustomResourceDefinition",
		"metadata": map[string]interface{}{
			"name": "widgets.example.com",
		},
	}})
	logRESTMapperCRDEvent("delete", &metav1.Status{})
}

func newTestCachedRESTMapper() *perClusterRESTMapper {
	cachedDiscovery := memory.NewMemCacheClient(fakekubernetes.NewSimpleClientset().Discovery())
	return &perClusterRESTMapper{
		mapper:    restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery),
		discovery: cachedDiscovery,
		stopCh:    make(chan struct{}),
	}
}

func mustRESTMapperCacheKey(t *testing.T, cfg *rest.Config) string {
	t.Helper()
	key, err := restMapperCacheKey(cfg)
	if err != nil {
		t.Fatalf("restMapperCacheKey() error = %v", err)
	}
	return key
}

func assertMapperStopped(t *testing.T, mapper *perClusterRESTMapper) {
	t.Helper()
	select {
	case <-mapper.stopCh:
	default:
		t.Fatal("RESTMapper should be stopped")
	}
}
