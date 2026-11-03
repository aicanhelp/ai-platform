/*
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package security

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic/fake"
	kubefake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ───────────────────────── Secret IO adapters ─────────────────────────

func TestCtrlRuntimeSecretIO(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	t.Run("not found maps to nil nil", func(t *testing.T) {
		c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
		get, _ := CtrlRuntimeSecretIO(c, c, DefaultSecretNamespace, DefaultSecretName)
		sec, err := get(context.Background())
		require.NoError(t, err)
		assert.Nil(t, sec)
	})

	t.Run("existing secret returned", func(t *testing.T) {
		sec := validSecret(newTestKey(t))
		c := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(sec).Build()
		get, _ := CtrlRuntimeSecretIO(c, c, DefaultSecretNamespace, DefaultSecretName)
		got, err := get(context.Background())
		require.NoError(t, err)
		require.NotNil(t, got)
		assert.Equal(t, sec.Data[SecretDataKey], got.Data[SecretDataKey])
	})

	t.Run("create and already-exists", func(t *testing.T) {
		sec := validSecret(newTestKey(t))
		c := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(sec).Build()
		_, create := CtrlRuntimeSecretIO(c, c, DefaultSecretNamespace, DefaultSecretName)

		created, err := create(context.Background(), validSecret(newTestKey(t)))
		require.NoError(t, err)
		assert.False(t, created, "AlreadyExists must map to (false, nil)")

		created, err = create(context.Background(), &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "another-key", Namespace: DefaultSecretNamespace},
		})
		require.NoError(t, err)
		assert.True(t, created)
	})
}

func TestClientGoSecretIO(t *testing.T) {
	t.Run("not found maps to nil nil", func(t *testing.T) {
		cs := kubefake.NewSimpleClientset()
		get, _ := ClientGoSecretIO(cs, DefaultSecretNamespace, DefaultSecretName)
		sec, err := get(context.Background())
		require.NoError(t, err)
		assert.Nil(t, sec)
	})

	t.Run("get error propagates", func(t *testing.T) {
		cs := kubefake.NewSimpleClientset()
		cs.PrependReactor("get", "secrets", func(action k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, errors.New("apiserver down")
		})
		get, _ := ClientGoSecretIO(cs, DefaultSecretNamespace, DefaultSecretName)
		_, err := get(context.Background())
		assert.Error(t, err)
	})

	t.Run("create and already-exists", func(t *testing.T) {
		sec := validSecret(newTestKey(t))
		cs := kubefake.NewSimpleClientset()
		_, create := ClientGoSecretIO(cs, DefaultSecretNamespace, DefaultSecretName)

		created, err := create(context.Background(), sec)
		require.NoError(t, err)
		assert.True(t, created)

		created, err = create(context.Background(), validSecret(newTestKey(t)))
		require.NoError(t, err)
		assert.False(t, created, "AlreadyExists must map to (false, nil)")
	})
}

// ───────────────────────── BKENode guards ─────────────────────────

func newFormatPassword(t *testing.T) string {
	t.Helper()
	setTestKey(t)
	enc, err := AesEncrypt("guard-test-password")
	require.NoError(t, err)
	return enc
}

func ctrlFakeClientWithNodes(t *testing.T, nodes ...*confv1beta1.BKENode) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	// Register BKENode types directly (importing api/capbke/v1beta1 here would
	// create a test-only import cycle: capbke -> common/cluster/node -> security).
	gv := confv1beta1.BKENodeGVK.GroupVersion()
	metav1.AddToGroupVersion(scheme, gv)
	scheme.AddKnownTypes(gv, &confv1beta1.BKENode{}, &confv1beta1.BKENodeList{})
	objs := make([]client.Object, 0, len(nodes))
	for _, n := range nodes {
		objs = append(objs, n)
	}
	return ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func TestCtrlRuntimeBKENodeGuard(t *testing.T) {
	ctx := context.Background()

	t.Run("affected node: warn via event and allow", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "node-1", Namespace: DefaultSecretNamespace},
			Spec:       confv1beta1.BKENodeSpec{Password: newFormatPassword(t)},
		}
		c := ctrlFakeClientWithNodes(t, node)
		recorder := record.NewFakeRecorder(10)

		err := CtrlRuntimeBKENodeGuard(c, recorder)(ctx)
		require.NoError(t, err, "guard must not block generation")

		select {
		case ev := <-recorder.Events:
			assert.Contains(t, ev, "PasswordKeyLost")
			assert.Contains(t, ev, "Warning")
		default:
			t.Fatal("expected a Warning event for the affected node")
		}
	})

	t.Run("legacy-only or no nodes: silent allow", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "node-legacy", Namespace: DefaultSecretNamespace},
			Spec:       confv1beta1.BKENodeSpec{Password: "legacy-plain-or-ciphertext"},
		}
		c := ctrlFakeClientWithNodes(t, node)
		recorder := record.NewFakeRecorder(10)

		err := CtrlRuntimeBKENodeGuard(c, recorder)(ctx)
		require.NoError(t, err)
		select {
		case <-recorder.Events:
			t.Fatal("no event expected when no node is affected")
		default:
		}
	})

	t.Run("list error defers generation", func(t *testing.T) {
		guard := CtrlRuntimeBKENodeGuard(&erroringReader{}, nil)
		err := guard(ctx)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "deferring new password encryption key generation")
	})

	t.Run("crd missing (not found): skip detection and allow", func(t *testing.T) {
		guard := CtrlRuntimeBKENodeGuard(&erroringReader{err: apierrors.NewNotFound(
			schema.GroupResource{Group: "bke.bocloud.com", Resource: "bkenodes"}, "")}, nil)
		err := guard(ctx)
		require.NoError(t, err, "a missing CRD means no BKENodes exist, generation must proceed")
	})

	t.Run("nil recorder: log only, still allow", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "node-2", Namespace: DefaultSecretNamespace},
			Spec:       confv1beta1.BKENodeSpec{Password: newFormatPassword(t)},
		}
		c := ctrlFakeClientWithNodes(t, node)
		err := CtrlRuntimeBKENodeGuard(c, nil)(ctx)
		require.NoError(t, err)
	})
}

// erroringReader simulates an API failure on List.
type erroringReader struct {
	// err is returned by Get/List; defaults to a generic API error.
	err error
}

func (r *erroringReader) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if r.err != nil {
		return r.err
	}
	return errors.New("apiserver down")
}

func (r *erroringReader) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if r.err != nil {
		return r.err
	}
	return errors.New("apiserver down")
}

func TestDynamicBKENodeGuard(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	listKinds := map[schema.GroupVersionResource]string{bkenodeGVR: "BKENodeList"}

	t.Run("affected node: console warn and allow", func(t *testing.T) {
		node := &unstructured.Unstructured{}
		node.SetGroupVersionKind(bkenodeGVR.GroupVersion().WithKind("BKENode"))
		node.SetNamespace(DefaultSecretNamespace)
		node.SetName("dyn-node-1")
		require.NoError(t, unstructured.SetNestedField(node.Object, newFormatPassword(t), "spec", "password"))

		dyn := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, node)
		err := DynamicBKENodeGuard(dyn)(ctx)
		require.NoError(t, err, "guard must not block generation")
	})

	t.Run("no nodes: allow", func(t *testing.T) {
		dyn := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds)
		err := DynamicBKENodeGuard(dyn)(ctx)
		require.NoError(t, err)
	})

	t.Run("list error defers generation", func(t *testing.T) {
		dyn := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds)
		dyn.PrependReactor("list", "bkenodes", func(action k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, apierrors.NewInternalError(errors.New("apiserver down"))
		})
		err := DynamicBKENodeGuard(dyn)(ctx)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "deferring new password encryption key generation")
	})

	t.Run("crd missing (not found): skip detection and allow", func(t *testing.T) {
		dyn := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds)
		dyn.PrependReactor("list", "bkenodes", func(action k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, apierrors.NewNotFound(bkenodeGVR.GroupResource(), "")
		})
		err := DynamicBKENodeGuard(dyn)(ctx)
		require.NoError(t, err, "a missing CRD means no BKENodes exist, generation must proceed")
	})
}

// TestBKENodeGVRSanity keeps the GVR definition in sync with the CRD.
func TestBKENodeGVRSanity(t *testing.T) {
	assert.Equal(t, schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkenodes",
	}, bkenodeGVR)
}

// TestSecretKeyProviderWithCtrlRuntimeIO is an integration-style test: the
// provider drives the controller-runtime adapter end to end.
func TestSecretKeyProviderWithCtrlRuntimeIO(t *testing.T) {
	t.Run("generate on empty cluster, then reuse", func(t *testing.T) {
		scheme := runtime.NewScheme()
		require.NoError(t, corev1.AddToScheme(scheme))
		c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
		get, create := CtrlRuntimeSecretIO(c, c, DefaultSecretNamespace, DefaultSecretName)
		p, err := NewSecretKeyProvider(SecretKeyProviderConfig{GetSecret: get, CreateSecret: create})
		require.NoError(t, err)

		key, err := p.Key(context.Background())
		require.NoError(t, err)
		assert.Len(t, key, 32)

		// The Secret now exists in the fake cluster with the right format.
		sec := &corev1.Secret{}
		require.NoError(t, c.Get(context.Background(),
			client.ObjectKey{Namespace: DefaultSecretNamespace, Name: DefaultSecretName}, sec))
		decoded, err := base64.StdEncoding.DecodeString(string(sec.Data[SecretDataKey]))
		require.NoError(t, err)
		assert.Equal(t, key, decoded)
	})
}

// TestSecretWatchRunnable drives the watcher end to end with a fake
// clientset: an external rotation refreshes the provider cache, and an
// external deletion self-heals from the cached key.
func TestSecretWatchRunnable(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	origKey := newTestKey(t)
	clientset := kubefake.NewSimpleClientset(validSecret(origKey))
	get, create := ClientGoSecretIO(clientset, DefaultSecretNamespace, DefaultSecretName)
	provider, err := NewSecretKeyProvider(SecretKeyProviderConfig{GetSecret: get, CreateSecret: create})
	require.NoError(t, err)
	_, err = provider.Key(context.Background())
	require.NoError(t, err)

	w := NewSecretWatchRunnable(provider, clientset)
	require.NoError(t, w.Start(ctx))

	t.Run("external rotation refreshes the cache", func(t *testing.T) {
		newKey := newTestKey(t)
		assert.Eventually(t, func() bool {
			// fake clientset 的 watch 没有 list RV 断点续传：informer 在 Start
			// 后异步建立（List→Watch 存在窗口），窗口内的 Update 事件会丢失
			// （真实集群 watch 从 list RV 续传，无此问题）。幂等重复 Update
			// 保证 watch 建立后仍能收到 Modified 事件。
			_, err := clientset.CoreV1().Secrets(DefaultSecretNamespace).
				Update(ctx, validSecret(newKey), metav1.UpdateOptions{})
			require.NoError(t, err)
			k, err := provider.Key(context.Background())
			return err == nil && bytes.Equal(k, newKey)
		}, 5*time.Second, 50*time.Millisecond, "the cache must follow the external rotation")
	})

	t.Run("external deletion self-heals from the cached key", func(t *testing.T) {
		currentKey, err := provider.Key(context.Background())
		require.NoError(t, err)
		expected := base64.StdEncoding.EncodeToString(currentKey)

		require.NoError(t, clientset.CoreV1().Secrets(DefaultSecretNamespace).
			Delete(ctx, DefaultSecretName, metav1.DeleteOptions{}))

		assert.Eventually(t, func() bool {
			s, err := clientset.CoreV1().Secrets(DefaultSecretNamespace).
				Get(ctx, DefaultSecretName, metav1.GetOptions{})
			return err == nil && string(s.Data[SecretDataKey]) == expected
		}, 5*time.Second, 50*time.Millisecond, "the Secret must be recreated with the in-memory key")
	})
}
