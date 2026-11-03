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
	ctrl "sigs.k8s.io/controller-runtime"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

func TestEnsureCertsConstants(t *testing.T) {
	assert.Equal(t, "EnsureCerts", string(EnsureCertsName))
}

func TestNewEnsureCerts(t *testing.T) {
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
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureCerts(ctx)
	assert.NotNil(t, phase)

	ec, ok := phase.(*EnsureCerts)
	assert.True(t, ok)
	assert.NotNil(t, ec.certsGenerator)
}

func TestEnsureCerts_Execute_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(ec.certsGenerator, "LookUpOrGenerate", func(_ *certs.BKEKubernetesCertGenerator) error {
		return nil
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return false, nil
	})

	result, err := ec.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureCerts_Execute_LookUpOrGenerateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(ec.certsGenerator, "LookUpOrGenerate", func(_ *certs.BKEKubernetesCertGenerator) error {
		return assert.AnError
	})

	result, err := ec.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureCerts_Execute_NeedGenerateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(ec.certsGenerator, "LookUpOrGenerate", func(_ *certs.BKEKubernetesCertGenerator) error {
		return nil
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return false, assert.AnError
	})

	result, err := ec.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureCerts_Execute_StillNeedGenerate(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(ec.certsGenerator, "LookUpOrGenerate", func(_ *certs.BKEKubernetesCertGenerator) error {
		return nil
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return true, nil
	})

	result, err := ec.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureCerts_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	oldCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	newCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(newCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: newCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, newCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(&ec.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return false
	})

	result := ec.NeedExecute(oldCluster, newCluster)
	assert.False(t, result)
}

func TestEnsureCerts_NeedExecute_NeedGenerateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	oldCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	newCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(newCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: newCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, newCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(&ec.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return false, assert.AnError
	})

	result := ec.NeedExecute(oldCluster, newCluster)
	assert.False(t, result)
}

func TestEnsureCerts_NeedExecute_NoNeedGenerate(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	oldCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	newCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(newCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: newCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, newCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(&ec.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return false, nil
	})

	result := ec.NeedExecute(oldCluster, newCluster)
	assert.False(t, result)
}

func TestEnsureCerts_NeedExecute_NeedGenerateTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	oldCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	newCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(newCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: newCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, newCluster),
	}

	phase := NewEnsureCerts(ctx)
	ec := phase.(*EnsureCerts)

	patches.ApplyMethod(&ec.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})
	patches.ApplyMethod(ec.certsGenerator, "NeedGenerate", func(_ *certs.BKEKubernetesCertGenerator) (bool, error) {
		return true, nil
	})

	result := ec.NeedExecute(oldCluster, newCluster)
	assert.True(t, result)
}
