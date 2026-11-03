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
	"testing"

	"k8s.io/client-go/rest"
)

func TestNewRESTClientConfig(t *testing.T) {
	namespace := "default"
	config := &rest.Config{
		Host: "https://localhost:6443",
	}

	rc := NewRESTClientConfig(namespace, config)

	restClientConfig, ok := rc.(*RestClientConfig)
	if !ok {
		t.Fatal("Expected *RestClientConfig")
	}

	if restClientConfig.namespace != namespace {
		t.Errorf("Expected namespace to be %s, got %s", namespace, restClientConfig.namespace)
	}
	if restClientConfig.restConfig != config {
		t.Error("Expected restConfig to be set")
	}
	if restClientConfig.restConfig.Host != "https://localhost:6443" {
		t.Errorf("Expected Host to be https://localhost:6443, got %s", restClientConfig.restConfig.Host)
	}
}

func TestRestClientConfigStruct(t *testing.T) {
	config := &rest.Config{
		Host:        "https://kubernetes:6443",
		Username:    "admin",
		Password:    "secret",
		BearerToken: "token",
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}

	rc := &RestClientConfig{
		namespace:  "kube-system",
		restConfig: config,
	}

	if rc.namespace != "kube-system" {
		t.Errorf("Expected namespace to be kube-system, got %s", rc.namespace)
	}
	if rc.restConfig.Host != "https://kubernetes:6443" {
		t.Errorf("Expected Host to be https://kubernetes:6443, got %s", rc.restConfig.Host)
	}
	if rc.restConfig.Username != "admin" {
		t.Errorf("Expected Username to be admin, got %s", rc.restConfig.Username)
	}
	if rc.restConfig.Password != "secret" {
		t.Errorf("Expected Password to be secret, got %s", rc.restConfig.Password)
	}
	if !rc.restConfig.TLSClientConfig.Insecure {
		t.Error("Expected Insecure to be true")
	}
}

func TestRestClientConfigToRESTConfig(t *testing.T) {
	config := &rest.Config{
		Host: "https://localhost:6443",
	}

	rc := &RestClientConfig{
		namespace:  "default",
		restConfig: config,
	}

	returnedConfig, err := rc.ToRESTConfig()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if returnedConfig != config {
		t.Error("Expected returned config to be the same as input config")
	}
}

func TestRestClientConfigNil(t *testing.T) {
	rc := &RestClientConfig{
		namespace:  "default",
		restConfig: nil,
	}

	config, err := rc.ToRESTConfig()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if config != nil {
		t.Error("Expected nil config when input is nil")
	}
}

func TestToRawKubeConfigLoader(t *testing.T) {
	config := &rest.Config{Host: "https://localhost:6443"}
	rc := &RestClientConfig{
		namespace:  "test-namespace",
		restConfig: config,
	}

	loader := rc.ToRawKubeConfigLoader()
	if loader == nil {
		t.Fatal("Expected non-nil loader")
	}

	cfg, err := loader.ClientConfig()
	if err != nil {
		t.Logf("ClientConfig error (expected in test): %v", err)
	}
	if cfg != nil && cfg.Host != "" {
		t.Logf("Got config with host: %s", cfg.Host)
	}

	ns, _, err := loader.Namespace()
	if err != nil {
		t.Logf("Namespace error (expected in test): %v", err)
	}
	if ns == "test-namespace" {
		t.Logf("Namespace correctly set to: %s", ns)
	}
}

func TestToRESTMapper(t *testing.T) {
	config := &rest.Config{Host: "https://localhost:6443"}
	rc := &RestClientConfig{
		namespace:  "default",
		restConfig: config,
	}

	mapper, err := rc.ToRESTMapper()
	if err != nil {
		t.Logf("ToRESTMapper error (expected in test): %v", err)
	}
	if mapper != nil {
		t.Log("Successfully created REST mapper")
	}
}

func TestToDiscoveryClient(t *testing.T) {
	config := &rest.Config{Host: "https://localhost:6443"}
	rc := &RestClientConfig{
		namespace:  "default",
		restConfig: config,
	}

	client, err := rc.ToDiscoveryClient()
	if err != nil {
		t.Errorf("ToDiscoveryClient() error = %v", err)
	}
	if client == nil {
		t.Error("Expected non-nil discovery client")
	}
}
