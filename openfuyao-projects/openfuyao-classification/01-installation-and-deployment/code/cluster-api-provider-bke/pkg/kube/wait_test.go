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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func TestTimeoutConstant(t *testing.T) {
	expected := 30 * time.Second
	if Timeout != expected {
		t.Errorf("Expected Timeout to be %v, got %v", expected, Timeout)
	}
}

func TestWaiterStruct(t *testing.T) {
	w := &waiter{
		unstructuredObj: nil,
		namespace:       "default",
		name:            "test",
		block:           true,
		timeout:         10 * time.Second,
		interval:        1 * time.Second,
	}

	if w.namespace != "default" {
		t.Errorf("Expected namespace to be default, got %s", w.namespace)
	}
	if w.name != "test" {
		t.Errorf("Expected name to be test, got %s", w.name)
	}
	if !w.block {
		t.Error("Expected block to be true")
	}
	if w.timeout != 10*time.Second {
		t.Errorf("Expected timeout to be 10s, got %v", w.timeout)
	}
	if w.interval != 1*time.Second {
		t.Errorf("Expected interval to be 1s, got %v", w.interval)
	}
}

func TestReadyCheckerStruct(t *testing.T) {
	rc := &readyChecker{
		pausedAsReady: true,
		fullComplete:  false,
	}

	if !rc.pausedAsReady {
		t.Error("Expected pausedAsReady to be true")
	}
	if rc.fullComplete {
		t.Error("Expected fullComplete to be false")
	}
}

func TestNewWaiterFromUnstructured(t *testing.T) {
	obj := &unstructured.Unstructured{}
	obj.SetName("test-pod")
	obj.SetNamespace("default")

	w := newWaiterFromUnstructured(obj)

	if w.name != "test-pod" {
		t.Errorf("Expected name to be test-pod, got %s", w.name)
	}
	if w.namespace != "default" {
		t.Errorf("Expected namespace to be default, got %s", w.namespace)
	}
	if w.unstructuredObj != obj {
		t.Error("Expected unstructuredObj to match input")
	}
}

func TestProductStatusStruct(t *testing.T) {
	now := metav1.Time{Time: time.Now()}
	ps := &ProductStatus{
		Name:           "test-product",
		StartTime:      &now,
		UpdateTime:     &now,
		CompletionTime: &now,
		Health:         true,
		Component: []ComponentStatus{
			{
				Name:     "component1",
				Resource: "deployment",
				Health:   true,
				Message:  "healthy",
			},
		},
		Reason: "running",
	}

	if ps.Name != "test-product" {
		t.Errorf("Expected Name to be test-product, got %s", ps.Name)
	}
	if !ps.Health {
		t.Error("Expected Health to be true")
	}
	if len(ps.Component) != 1 {
		t.Errorf("Expected 1 component, got %d", len(ps.Component))
	}
	if ps.Component[0].Name != "component1" {
		t.Errorf("Expected component name to be component1, got %s", ps.Component[0].Name)
	}
}

func TestComponentStatusStruct(t *testing.T) {
	cs := ComponentStatus{
		Name:     "test-component",
		Resource: "pod",
		Health:   true,
		Message:  "OK",
	}

	if cs.Name != "test-component" {
		t.Errorf("Expected Name to be test-component, got %s", cs.Name)
	}
	if cs.Resource != "pod" {
		t.Errorf("Expected Resource to be pod, got %s", cs.Resource)
	}
	if !cs.Health {
		t.Error("Expected Health to be true")
	}
	if cs.Message != "OK" {
		t.Errorf("Expected Message to be OK, got %s", cs.Message)
	}
}

func TestWaiter_SetPoller(t *testing.T) {
	w := &waiter{}
	task := &Task{
		Timeout:  5 * time.Second,
		Interval: 1 * time.Second,
		Block:    true,
	}

	result := w.setPoller(task)

	assert.Equal(t, 5*time.Second, result.timeout)
	assert.Equal(t, 1*time.Second, result.interval)
	assert.True(t, result.block)
	assert.Equal(t, w, result)
}

func TestWaiter_SetChecker(t *testing.T) {
	var logger *log.Logger = nil
	bkeLog := &bkev1beta1.BKELogger{}

	c := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: &dynamic.DynamicClient{},
		RestConfig:    &rest.Config{},
		Log:           logger,
		BKELog:        bkeLog,
		Ctx:           context.Background(),
	}

	w := &waiter{block: true}
	result := w.setChecker(c)

	assert.NotNil(t, result.checker)
	assert.Equal(t, c.ClientSet, result.checker.client)
	assert.Equal(t, c.DynamicClient, result.checker.dynamicClient)
	assert.Equal(t, logger, result.checker.log)
	assert.Equal(t, bkeLog, result.checker.bkeLog)
	assert.True(t, result.checker.pausedAsReady)
	assert.True(t, result.checker.fullComplete)
	assert.Equal(t, c.Ctx, result.ctx)
}

func TestNewCheckerFromKubeClient(t *testing.T) {
	var logger *log.Logger = nil
	bkeLog := &bkev1beta1.BKELogger{}

	c := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: &dynamic.DynamicClient{},
		RestConfig:    &rest.Config{},
		Log:           logger,
		BKELog:        bkeLog,
	}

	checker := newCheckerFromKubeClient(c)

	assert.NotNil(t, checker)
	assert.Equal(t, c.ClientSet, checker.client)
	assert.Equal(t, c.DynamicClient, checker.dynamicClient)
	assert.Equal(t, logger, checker.log)
	assert.Equal(t, bkeLog, checker.bkeLog)
	assert.True(t, checker.pausedAsReady)
	assert.NotNil(t, checker.helmClient)
}

func TestNewClient(t *testing.T) {
	t.Run("with valid config", func(t *testing.T) {
		config := &rest.Config{}
		clientset := &kubernetes.Clientset{}
		dynamicClient := &dynamic.DynamicClient{}

		client := NewClient(config, clientset, dynamicClient)
		assert.NotNil(t, client)
	})

	t.Run("with nil config", func(t *testing.T) {
		client := NewClient(nil, &kubernetes.Clientset{}, &dynamic.DynamicClient{})
		assert.Nil(t, client)
	})
}

func TestClient_Wait(t *testing.T) {
	var logger *log.Logger = nil
	c := &Client{
		ClientSet:     &kubernetes.Clientset{},
		DynamicClient: &dynamic.DynamicClient{},
		RestConfig:    &rest.Config{},
		Log:           logger,
		BKELog:        &bkev1beta1.BKELogger{},
		Ctx:           context.Background(),
	}

	t.Run("CRD returns nil", func(t *testing.T) {
		obj := &unstructured.Unstructured{}
		obj.SetGroupVersionKind(schema.GroupVersionKind{
			Kind: "CustomResourceDefinition",
		})

		task := &Task{
			Timeout:  5 * time.Second,
			Interval: 1 * time.Second,
		}

		err := c.Wait(obj, task)
		assert.NoError(t, err)
	})
}

func TestKubeFactory_ToRESTConfig(t *testing.T) {
	config := &rest.Config{Host: "https://test"}
	factory := &kubeFactory{config: config}

	result, err := factory.ToRESTConfig()
	assert.NoError(t, err)
	assert.Equal(t, config, result)
}

func TestKubeFactory_ToRawKubeConfigLoader(t *testing.T) {
	factory := &kubeFactory{}
	loader := factory.ToRawKubeConfigLoader()
	assert.NotNil(t, loader)
}
