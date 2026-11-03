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

package v1_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	fakerest "k8s.io/client-go/rest/fake"

	apisv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	"openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
	clientscheme "openfuyao.com/colocation-service/pkg/client/clientset/versioned/scheme"
	typedv1 "openfuyao.com/colocation-service/pkg/client/clientset/versioned/typed/colocation/v1"
)

func newFakeTypedClient() *typedv1.ColocationV1Client {
	rc := &fakerest.RESTClient{
		NegotiatedSerializer: clientscheme.Codecs.WithoutConversion(),
		GroupVersion:         apisv1.SchemeGroupVersion,
		VersionedAPIPath:     "/apis",
	}
	return typedv1.New(rc)
}

func TestTypedColocationWorkloadMatchRules_Get(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").Get(context.TODO(), "rule1", metav1.GetOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_List(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").List(context.TODO(), metav1.ListOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_Create(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").Create(context.TODO(),
		&apisv1.ColocationWorkloadMatchRule{ObjectMeta: metav1.ObjectMeta{Name: "rule1"}},
		metav1.CreateOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_Update(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").Update(context.TODO(),
		&apisv1.ColocationWorkloadMatchRule{ObjectMeta: metav1.ObjectMeta{Name: "rule1"}},
		metav1.UpdateOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_UpdateStatus(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").UpdateStatus(context.TODO(),
		&apisv1.ColocationWorkloadMatchRule{ObjectMeta: metav1.ObjectMeta{Name: "rule1"}},
		metav1.UpdateOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_Delete(t *testing.T) {
	c := newFakeTypedClient()
	err := c.ColocationWorkloadMatchRules("default").Delete(context.TODO(), "rule1", metav1.DeleteOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_DeleteCollection(t *testing.T) {
	c := newFakeTypedClient()
	err := c.ColocationWorkloadMatchRules("default").DeleteCollection(context.TODO(),
		metav1.DeleteOptions{}, metav1.ListOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_Patch(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").Patch(context.TODO(), "rule1",
		types.MergePatchType, []byte(`{}`), metav1.PatchOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_Watch(t *testing.T) {
	c := newFakeTypedClient()
	_, err := c.ColocationWorkloadMatchRules("default").Watch(context.TODO(), metav1.ListOptions{})
	_ = err
}

func TestTypedColocationWorkloadMatchRules_List_WithTimeout(t *testing.T) {
	c := newFakeTypedClient()
	timeout := int64(5)
	_, err := c.ColocationWorkloadMatchRules("default").List(context.TODO(),
		metav1.ListOptions{TimeoutSeconds: &timeout})
	_ = err
}

func TestColocationWorkloadMatchRules_CRUD(t *testing.T) {
	cs := fake.NewSimpleClientset()
	c := cs.ColocationV1().ColocationWorkloadMatchRules("default")

	created, err := c.Create(context.TODO(), &apisv1.ColocationWorkloadMatchRule{
		ObjectMeta: metav1.ObjectMeta{Name: "rule1", Namespace: "default"},
		Spec: apisv1.ColocationWorkloadMatchRuleSpec{
			Colocation: apisv1.ColocationOnline,
		},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	assert.Equal(t, "rule1", created.Name)

	got, err := c.Get(context.TODO(), "rule1", metav1.GetOptions{})
	require.NoError(t, err)
	assert.Equal(t, "rule1", got.Name)
	assert.Equal(t, apisv1.ColocationOnline, got.Spec.Colocation)

	list, err := c.List(context.TODO(), metav1.ListOptions{})
	require.NoError(t, err)
	assert.Equal(t, 1, len(list.Items))

	got.Spec.Colocation = apisv1.ColocationOffline
	updated, err := c.Update(context.TODO(), got, metav1.UpdateOptions{})
	require.NoError(t, err)
	assert.Equal(t, apisv1.ColocationOffline, updated.Spec.Colocation)

	_, err = c.UpdateStatus(context.TODO(), got, metav1.UpdateOptions{})
	require.NoError(t, err)

	patched, err := c.Patch(context.TODO(), "rule1", types.MergePatchType,
		[]byte(`{"spec":{"colocation":"online"}}`), metav1.PatchOptions{})
	require.NoError(t, err)
	assert.Equal(t, apisv1.ColocationOnline, patched.Spec.Colocation)

	err = c.Delete(context.TODO(), "rule1", metav1.DeleteOptions{})
	require.NoError(t, err)

	_, err = c.Get(context.TODO(), "rule1", metav1.GetOptions{})
	assert.Error(t, err)
}

func TestColocationWorkloadMatchRules_DeleteCollection(t *testing.T) {
	cs := fake.NewSimpleClientset()
	c := cs.ColocationV1().ColocationWorkloadMatchRules("default")

	for i := 0; i < 3; i++ {
		_, err := c.Create(context.TODO(), &apisv1.ColocationWorkloadMatchRule{
			ObjectMeta: metav1.ObjectMeta{Name: "rule" + string(rune('1'+i)), Namespace: "default"},
		}, metav1.CreateOptions{})
		require.NoError(t, err)
	}

	err := c.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	require.NoError(t, err)
}

func TestColocationWorkloadMatchRules_List_WithTimeout(t *testing.T) {
	cs := fake.NewSimpleClientset()
	c := cs.ColocationV1().ColocationWorkloadMatchRules("default")

	timeout := int64(5)
	list, err := c.List(context.TODO(), metav1.ListOptions{TimeoutSeconds: &timeout})
	require.NoError(t, err)
	assert.Equal(t, 0, len(list.Items))
}

func TestColocationWorkloadMatchRules_Watch(t *testing.T) {
	cs := fake.NewSimpleClientset()
	c := cs.ColocationV1().ColocationWorkloadMatchRules("default")

	timeout := int64(5)
	w, err := c.Watch(context.TODO(), metav1.ListOptions{TimeoutSeconds: &timeout})
	require.NoError(t, err)
	assert.NotNil(t, w)
	w.Stop()
}
