/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package server

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/client/k8s"
	"volcano-config-service/pkg/cmd/config"
	"volcano-config-service/pkg/server/runtime"
)

func TestNewServer_Insecure(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "0.0.0.0",
			InsecurePort: 9202,
			SecurePort:   0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
			QPS:        100,
			Burst:      200,
		},
	}
	server, err := NewServer(cfg, context.Background())
	assert.NoError(t, err)
	assert.NotNil(t, server)
	assert.NotNil(t, server.Server)
	assert.NotNil(t, server.KubernetesClient)
}

func TestNewServer_NilKubeConfig(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 9202,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: nil,
		},
	}
	_, err := NewServer(cfg, context.Background())
	assert.Error(t, err)
}

func TestResource(t *testing.T) {
	gr := resource("clusters")
	assert.Equal(t, "clusters", gr.Resource)
	assert.Equal(t, "resources.fuyao.io", gr.Group)
}

func TestNewServer_SecurePort_BadCert(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   9443,
			CertFile:     "/nonexistent/cert.pem",
			PrivateKey:   "/nonexistent/key.pem",
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
		},
	}
	_, err := NewServer(cfg, context.Background())
	assert.Error(t, err)
}

func TestNewServer_RegisterAPI(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "0.0.0.0",
			InsecurePort: 9202,
			SecurePort:   0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
			QPS:        100,
			Burst:      200,
		},
	}
	server, err := NewServer(cfg, context.Background())
	assert.NoError(t, err)

	t.Setenv("VCS_API_DISABLED", "true")
	server.registerAPI()
}

func TestNewServer_BuildHandlerChain(t *testing.T) {
	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "0.0.0.0",
			InsecurePort: 9202,
			SecurePort:   0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "https://example.com"},
			QPS:        100,
			Burst:      200,
		},
	}
	server, err := NewServer(cfg, context.Background())
	assert.NoError(t, err)

	server.container = nil
	server.Server.Handler = nil
	server.buildHandlerChain()
	assert.NotNil(t, server.Server.Handler)
}
