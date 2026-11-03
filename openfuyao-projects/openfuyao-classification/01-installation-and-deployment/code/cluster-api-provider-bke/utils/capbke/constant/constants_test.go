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

package constant

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

func TestGetLocalKubeConfigObjectKey(t *testing.T) {
	key := GetLocalKubeConfigObjectKey()
	if key.Namespace != metav1.NamespaceSystem {
		t.Errorf("expected namespace %s, got %s", metav1.NamespaceSystem, key.Namespace)
	}
	if key.Name != LocalKubeConfigName {
		t.Errorf("expected name %s, got %s", LocalKubeConfigName, key.Name)
	}
}

func TestGetLocalConfigMapObjectKey(t *testing.T) {
	key := GetLocalConfigMapObjectKey()
	if key.Namespace != "cluster-system" {
		t.Errorf("expected namespace cluster-system, got %s", key.Namespace)
	}
	if key.Name != common.BKEClusterConfigFileName {
		t.Errorf("expected name %s, got %s", common.BKEClusterConfigFileName, key.Name)
	}
}

type mockObject struct {
	client.Object
	annotations map[string]string
}

func (m *mockObject) GetAnnotations() map[string]string {
	return m.annotations
}

func (m *mockObject) SetAnnotations(annotations map[string]string) {
	m.annotations = annotations
}

func TestGetLocalKubeConfigObjectKey_Value(t *testing.T) {
	key := GetLocalKubeConfigObjectKey()
	expected := client.ObjectKey{
		Namespace: metav1.NamespaceSystem,
		Name:      LocalKubeConfigName,
	}
	if key != expected {
		t.Errorf("expected %v, got %v", expected, key)
	}
}

func TestGetLocalConfigMapObjectKey_Value(t *testing.T) {
	key := GetLocalConfigMapObjectKey()
	expected := client.ObjectKey{
		Namespace: "cluster-system",
		Name:      common.BKEClusterConfigFileName,
	}
	if key != expected {
		t.Errorf("expected %v, got %v", expected, key)
	}
}
