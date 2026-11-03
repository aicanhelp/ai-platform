package config

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"

	"marketplace-service/pkg/client/k8s"
	"marketplace-service/pkg/server/runtime"
)

func TestNewRunConfig(t *testing.T) {
	patches := gomonkey.ApplyFunc(runtime.NewServerConfig, func() *runtime.ServerConfig {
		return &runtime.ServerConfig{InsecurePort: 9037}
	}).ApplyFunc(k8s.NewKubernetesCfg, func() *k8s.KubernetesCfg {
		return &k8s.KubernetesCfg{KubeConfig: &rest.Config{}}
	})
	defer patches.Reset()

	cfg := NewRunConfig()
	assert.NotNil(t, cfg)
	assert.NotNil(t, cfg.Server)
	assert.NotNil(t, cfg.KubernetesCfg)
}

func TestRunConfigValidate(t *testing.T) {
	cfg := &RunConfig{
		Server: &runtime.ServerConfig{
			SecurePort:   0,
			InsecurePort: 0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfigFile: "C:/_definitely_not_exists_/kubeconfig",
			KubeConfig:     nil,
		},
	}

	errs := cfg.Validate()
	assert.GreaterOrEqual(t, len(errs), 2)

	cfg.KubernetesCfg = &k8s.KubernetesCfg{
		KubeConfigFile: "",
		KubeConfig:     &rest.Config{},
	}
	errs = cfg.Validate()
	assert.Len(t, errs, 1)
}
