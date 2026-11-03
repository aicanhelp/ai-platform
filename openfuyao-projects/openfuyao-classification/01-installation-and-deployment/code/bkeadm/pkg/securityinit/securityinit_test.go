/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package securityinit

import (
	"context"
	"encoding/base64"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"

	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

var bkenodeGVR = schema.GroupVersionResource{
	Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes",
}

// setupMockK8s wires fake clientset/dynamic clients into global.K8s via
// gomonkey patches, following the pattern of setupCertgenMockK8s.
func setupMockK8s(t *testing.T, clientset kubernetes.Interface, dyn dynamic.Interface) {
	t.Helper()
	resetForTesting()
	origK8s := global.K8s
	t.Cleanup(func() {
		global.K8s = origK8s
		resetForTesting()
	})

	patches := gomonkey.NewPatches()
	t.Cleanup(patches.Reset)
	patches.ApplyFunc((*k8sexec.MockK8sClient).GetClient, func(*k8sexec.MockK8sClient) kubernetes.Interface {
		return clientset
	})
	patches.ApplyFunc((*k8sexec.MockK8sClient).GetDynamicClient, func(*k8sexec.MockK8sClient) dynamic.Interface {
		return dyn
	})
	global.K8s = &k8sexec.MockK8sClient{}
}

func newDynamicClient(objects ...runtime.Object) *dynamicfake.FakeDynamicClient {
	return dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		runtime.NewScheme(),
		map[schema.GroupVersionResource]string{bkenodeGVR: "BKENodeList"},
		objects...)
}

func validKeySecret(key []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      security.DefaultSecretName,
			Namespace: security.DefaultSecretNamespace,
		},
		Data: map[string][]byte{
			security.SecretDataKey: []byte(base64.StdEncoding.EncodeToString(key)),
		},
	}
}

func bkenodeWithPassword(name, password string) *unstructured.Unstructured {
	node := &unstructured.Unstructured{}
	node.SetGroupVersionKind(bkenodeGVR.GroupVersion().WithKind("BKENode"))
	node.SetNamespace(security.DefaultSecretNamespace)
	node.SetName(name)
	_ = unstructured.SetNestedField(node.Object, password, "spec", "password")
	return node
}

func TestInitK8sNotInitialized(t *testing.T) {
	resetForTesting()
	origK8s := global.K8s
	global.K8s = nil
	t.Cleanup(func() { global.K8s = origK8s })

	err := Init()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubernetes client is not initialized")
}

func TestInitExistingSecretReused(t *testing.T) {
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	clientset := fake.NewSimpleClientset(validKeySecret(key))
	setupMockK8s(t, clientset, newDynamicClient())

	require.NoError(t, Init())

	encoded, err := security.CurrentKeyBase64(context.Background())
	require.NoError(t, err)
	assert.Equal(t, base64.StdEncoding.EncodeToString(key), encoded)

	// Idempotent: a second call succeeds without side effects.
	require.NoError(t, Init())
}

func TestInitGeneratesWhenMissing(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	setupMockK8s(t, clientset, newDynamicClient())

	require.NoError(t, Init())

	// The Secret was created with a valid base64 32-byte key.
	sec, err := clientset.CoreV1().Secrets(security.DefaultSecretNamespace).
		Get(context.Background(), security.DefaultSecretName, metav1.GetOptions{})
	require.NoError(t, err)
	raw, err := base64.StdEncoding.DecodeString(string(sec.Data[security.SecretDataKey]))
	require.NoError(t, err)
	assert.Len(t, raw, 32)
}

func TestInitGeneratesWithKeyLossWarning(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	// A BKENode whose password looks like new-format ciphertext (OFMK prefix).
	ciphertext := base64.StdEncoding.EncodeToString(append([]byte("OFMK"), make([]byte, 48)...))
	dyn := newDynamicClient(bkenodeWithPassword("node-1", ciphertext))
	setupMockK8s(t, clientset, dyn)

	// Key loss is reported but generation still proceeds.
	require.NoError(t, Init())

	sec, err := clientset.CoreV1().Secrets(security.DefaultSecretNamespace).
		Get(context.Background(), security.DefaultSecretName, metav1.GetOptions{})
	require.NoError(t, err)
	assert.NotEmpty(t, sec.Data[security.SecretDataKey])
}

func TestInitGetSecretError(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	clientset.PrependReactor("get", "secrets", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("apiserver down")
	})
	setupMockK8s(t, clientset, newDynamicClient())

	err := Init()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "apiserver down")
}

// TestInitCrdMissing reproduces the first-install timeline: the BKENode CRD
// (deployed later with 002-cluster-api-bke.yaml) does not exist yet, so the
// guard must skip detection and key generation must proceed.
func TestInitCrdMissing(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	dyn := newDynamicClient()
	dyn.PrependReactor("list", "bkenodes", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewNotFound(bkenodeGVR.GroupResource(), "")
	})
	setupMockK8s(t, clientset, dyn)

	require.NoError(t, Init())

	sec, err := clientset.CoreV1().Secrets(security.DefaultSecretNamespace).
		Get(context.Background(), security.DefaultSecretName, metav1.GetOptions{})
	require.NoError(t, err)
	assert.NotEmpty(t, sec.Data[security.SecretDataKey])
}
