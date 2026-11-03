/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package k8s

import (
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	"github.com/stretchr/testify/assert"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient(t *testing.T) {

	t.Run("missing config", func(t *testing.T) {
		_, err := NewKubernetesClient(&KubernetesCfg{})
		assert.Error(t, err)
	})

	t.Run("snapshot client error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kubernetes.NewForConfig, func(*rest.Config) (kubernetes.Interface, error) {
			return &kubernetes.Clientset{}, nil
		})
		patches.ApplyFunc(snapshotclient.NewForConfig, func(*rest.Config) (snapshotclient.Interface, error) {
			return nil, errors.New("snapshot error")
		})

		_, err := NewKubernetesClient(&KubernetesCfg{KubeConfig: &rest.Config{}})
		assert.Error(t, err)
	})

	t.Run("api extensions error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(kubernetes.NewForConfig, func(*rest.Config) (kubernetes.Interface, error) {
			return &kubernetes.Clientset{}, nil
		})
		patches.ApplyFunc(snapshotclient.NewForConfig, func(*rest.Config) (snapshotclient.Interface, error) {
			return &snapshotclient.Clientset{}, nil
		})
		patches.ApplyFunc(apiextensionsclient.NewForConfig, func(*rest.Config) (apiextensionsclient.Interface, error) {
			return nil, errors.New("api extensions error")
		})

		_, err := NewKubernetesClient(&KubernetesCfg{KubeConfig: &rest.Config{}})
		assert.Error(t, err)
	})
}

func TestKubernetesClientMethods(t *testing.T) {
	kc := &kubernetesClient{
		k8s:           &kubernetes.Clientset{},
		snapshot:      &snapshotclient.Clientset{},
		apiExtensions: &apiextensionsclient.Clientset{},
		config:        &rest.Config{},
	}

	assert.NotNil(t, kc.Kubernetes())
	assert.NotNil(t, kc.Snapshot())
	assert.NotNil(t, kc.ApiExtensions())
	assert.NotNil(t, kc.Config())
}
