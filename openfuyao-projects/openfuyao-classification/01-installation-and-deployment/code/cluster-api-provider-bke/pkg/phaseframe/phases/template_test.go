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

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

func TestXXXConstants(t *testing.T) {
	assert.Equal(t, "XXX", string(XXXName))
}

func TestNewXXX(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	phase := NewXXX(ctx)
	assert.NotNil(t, phase)

	xxx, ok := phase.(*XXX)
	assert.True(t, ok)
	assert.NotNil(t, xxx)
}

func TestXXX_Execute(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	phase := NewXXX(ctx).(*XXX)
	result, err := phase.Execute()
	assert.Error(t, err)
	assert.Equal(t, "reconcileXXX not implemented", err.Error())
	assert.Empty(t, result)
}

func TestXXX_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, old, new *bkev1beta1.BKECluster) bool {
		return false
	})

	phase := NewXXX(ctx).(*XXX)
	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestXXX_NeedExecute_DefaultNeedExecuteTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, old, new *bkev1beta1.BKECluster) bool {
		return true
	})

	phase := NewXXX(ctx).(*XXX)
	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestXXX_ReconcileXXX(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Context:    context.Background(),
		Log:        createTestLogger(),
	}

	phase := NewXXX(ctx).(*XXX)
	err := phase.reconcileXXX()
	assert.Error(t, err)
	assert.Equal(t, "reconcileXXX not implemented", err.Error())
}
