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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/testing"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type MockK8sClient struct {
	ConfigMap *corev1.ConfigMap
	GetErr    error
}

func (m *MockK8sClient) GetClient() kubernetes.Interface {
	fakeClient := fake.NewSimpleClientset()
	if m.GetErr == nil && m.ConfigMap != nil {
		fakeClient.PrependReactor("get", "configmaps", func(action testing.Action) (handled bool, ret runtime.Object, err error) {
			getAction, ok := action.(testing.GetAction)
			if !ok {
				return false, nil, nil
			}
			if getAction.GetNamespace() == "openfuyao-patch" && getAction.GetName() == m.ConfigMap.Name {
				return true, m.ConfigMap, nil
			}
			return false, nil, nil
		})
		return fakeClient
	}
	if m.GetErr != nil {
		fakeClient.PrependReactor("get", "configmaps", func(action testing.Action) (handled bool, ret runtime.Object, err error) {
			return true, nil, m.GetErr
		})
	}
	return fakeClient
}

func (m *MockK8sClient) GetDynamicClient() dynamic.Interface {
	return dynamicfake.NewSimpleDynamicClient(runtime.NewScheme())
}

func (m *MockK8sClient) InstallYaml(_ string, _ map[string]string, _ string) error { return nil }
func (m *MockK8sClient) PatchYaml(_ string, _ map[string]string) error             { return nil }
func (m *MockK8sClient) UninstallYaml(_ string, _ string) error                    { return nil }
func (m *MockK8sClient) WatchEventByAnnotation(_ string)                           { log.Debugf("WatchEventByAnnotation") }
func (m *MockK8sClient) CreateNamespace(_ *corev1.Namespace) error                 { return nil }
func (m *MockK8sClient) CreateSecret(_ *corev1.Secret) error                       { return nil }
func (m *MockK8sClient) GetNamespace(_ string) (string, error)                     { return "", nil }
