/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

/*
 * 模拟客户端
 *
 */
package testutils

import (
	"context"
	"net/http"
	"net/http/httptest"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	fake1 "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	fake3 "sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

// TestGetRuntimeFakeClient Obtain controller-runtime to simulate access to k8s client
func TestGetRuntimeFakeClient(addToScheme []func(*runtime.Scheme) error,
	initObjs ...client.Object) (client.Client, *runtime.Scheme) {
	runtimeScheme := runtime.NewScheme()
	corev1.AddToScheme(runtimeScheme)

	if addToScheme != nil && len(addToScheme) > 0 {
		for i := range addToScheme {
			addToScheme[i](runtimeScheme)
		}
	}
	var fakeClient client.Client
	if len(initObjs) > 0 {
		fakeClient = fake.NewClientBuilder().
			WithScheme(runtimeScheme).
			WithObjects(initObjs...).
			WithStatusSubresource(initObjs...).Build()
	} else {
		fakeClient = fake.NewClientBuilder().WithScheme(runtimeScheme).Build()
	}
	return fakeClient, runtimeScheme
}

// TestGetClientGoFakeClient for http server
func TestGetClientGoFakeClient(mapObjs map[string]interface{}) (*kubernetes.Clientset, *rest.Config, *httptest.Server) {
	initRconfig, initTServer := TestGetK8sServerHttp(mapObjs)

	initClientset, _ := kubernetes.NewForConfig(initRconfig)
	return initClientset, initRconfig, initTServer
}

// TestGetSimpleDynamicClientWithCustomListKinds Obtain client-go dynamic to simulate access to k8s client
func TestGetSimpleDynamicClientWithCustomListKinds(
	addToScheme func(*runtime.Scheme) error,
	listKinds map[schema.GroupVersionResource]string,
	initObjs ...runtime.Object) (dynamic.Interface, *runtime.Scheme) {
	dynamicSscheme := runtime.NewScheme()
	corev1.AddToScheme(dynamicSscheme)
	if addToScheme != nil {
		if err := addToScheme(dynamicSscheme); err != nil {
		}
	}
	var fakeClient dynamic.Interface

	if listKinds != nil && len(listKinds) > 0 && len(initObjs) > 0 {
		fakeClient = fake1.NewSimpleDynamicClientWithCustomListKinds(dynamicSscheme, listKinds, initObjs...)
	} else {
		fakeClient = fake1.NewSimpleDynamicClientWithCustomListKinds(dynamicSscheme, listKinds)
	}

	return fakeClient, dynamicSscheme
}

// TestGetManagerClient Get custom manager to simulate accessing k8s client
func TestGetManagerClient(addToScheme []func(*runtime.Scheme) error,
	o ...client.Object) (*BocloudFakeManager, *runtime.Scheme) {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	if addToScheme != nil && len(addToScheme) > 0 {
		for i := range addToScheme {
			addToScheme[i](scheme)
		}
	}
	var fakeClient client.Client
	if len(o) > 0 {
		fakeClient = fake.NewClientBuilder().WithScheme(scheme).WithObjects(o...).WithStatusSubresource(o...).Build()
	} else {
		fakeClient = fake.NewClientBuilder().WithScheme(scheme).Build()
	}
	mgr := &BocloudFakeManager{
		client: fakeClient,
		scheme: scheme,
		name:   "",
	}
	return mgr, scheme
}

// BocloudFakeManager Custom fackmanager structure
type BocloudFakeManager struct {
	client client.Client
	name   string
	scheme *runtime.Scheme
}

// GetHTTPClient Implement interface
func (f *BocloudFakeManager) GetHTTPClient() *http.Client { return nil }

// GetFieldIndexer Implement interface
func (f *BocloudFakeManager) GetFieldIndexer() client.FieldIndexer { return nil }

// GetEventRecorderFor Implement interface
func (f *BocloudFakeManager) GetEventRecorderFor(name string) record.EventRecorder {
	// 1. 创建事件广播器
	eventBroadcaster := record.NewBroadcaster()

	// 3. 返回对应资源的事件记录器
	return eventBroadcaster.NewRecorder(
		f.scheme,
		corev1.EventSource{Component: name},
	)
}

// GetRESTMapper Implement interface
func (f *BocloudFakeManager) GetRESTMapper() meta.RESTMapper { return nil }

// GetAPIReader Implement interface
func (f *BocloudFakeManager) GetAPIReader() client.Reader { return f.client }

// Add Implement interface
func (f *BocloudFakeManager) Add(runnable fake3.Runnable) error { return nil }

// Elected Implement interface
func (f *BocloudFakeManager) Elected() <-chan struct{} { return nil }

// AddMetricsExtraHandler Implement interface
func (f *BocloudFakeManager) AddMetricsExtraHandler(path string, handler http.Handler) error {
	return nil
}

// GetLogger Implement interface
func (f *BocloudFakeManager) GetLogger() logr.Logger {
	return klog.NewKlogr()
}

// GetClient Implement interface
func (f *BocloudFakeManager) GetClient() client.Client {
	return f.client
}

// GetScheme Implement interface
func (f *BocloudFakeManager) GetScheme() *runtime.Scheme {
	return f.scheme
}

// GetCache Implement interface
func (f *BocloudFakeManager) GetCache() cache.Cache {
	return nil
}

// Start Implement interface
func (f *BocloudFakeManager) Start(ctx context.Context) error {
	return nil
}

// GetConfig Implement interface
func (f *BocloudFakeManager) GetConfig() *rest.Config {
	return &rest.Config{}
}

// GetControllerOptions Implement interface
func (f *BocloudFakeManager) GetControllerOptions() config.Controller {
	return config.Controller{}
}

// AddHealthzCheck Implement interface
func (f *BocloudFakeManager) AddHealthzCheck(name string, check healthz.Checker) error {
	return nil
}

// AddReadyzCheck Implement interface
func (f *BocloudFakeManager) AddReadyzCheck(name string, check healthz.Checker) error {
	return nil
}

// GetWebhookServer Implement interface
func (f *BocloudFakeManager) GetWebhookServer() webhook.Server {
	return nil
}

// GetName Implement interface
func (f *BocloudFakeManager) GetName() string {
	return f.name
}
