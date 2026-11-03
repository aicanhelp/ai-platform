/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for the more details.
 ******************************************************************/

package capbke

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func newUpgradePathWebhook(objs ...client.Object) *UpgradePath {
	scheme := runtime.NewScheme()
	_ = confv1alpha1.AddToScheme(scheme)

	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > 0 {
		builder = builder.WithObjects(objs...)
	}

	c := builder.Build()
	return &UpgradePath{
		Client:    c,
		APIReader: c,
	}
}

func TestUpgradePathValidateCreateAllowed(t *testing.T) {
	webhook := newUpgradePathWebhook()

	up := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "default"},
		Spec: confv1alpha1.UpgradePathSpec{
			Paths: []confv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
			},
		},
	}

	_, err := webhook.ValidateCreate(context.Background(), up)
	assert.NoError(t, err)
}

func TestUpgradePathValidateCreateRejectedWhenExisting(t *testing.T) {
	existing := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "existing-path"},
		Spec: confv1alpha1.UpgradePathSpec{
			Paths: []confv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
			},
		},
	}

	webhook := newUpgradePathWebhook(existing)

	newUp := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "another-path"},
		Spec: confv1alpha1.UpgradePathSpec{
			Paths: []confv1alpha1.UpgradePathRule{
				{From: "v2.0.0", To: "v2.1.0"},
			},
		},
	}

	_, err := webhook.ValidateCreate(context.Background(), newUp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only one UpgradePath CR is allowed per cluster")
	assert.Contains(t, err.Error(), "existing-path")
}

func TestUpgradePathValidateUpdateAllowed(t *testing.T) {
	webhook := newUpgradePathWebhook()

	oldUp := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "default"},
		Spec: confv1alpha1.UpgradePathSpec{
			Paths: []confv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
			},
		},
	}

	newUp := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "default"},
		Spec: confv1alpha1.UpgradePathSpec{
			Paths: []confv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.1.0", To: "v1.2.0"},
			},
		},
	}

	_, err := webhook.ValidateUpdate(context.Background(), oldUp, newUp)
	assert.NoError(t, err)
}

func TestUpgradePathValidateDeleteAllowed(t *testing.T) {
	webhook := newUpgradePathWebhook()

	up := &confv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{Name: "default"},
	}

	_, err := webhook.ValidateDelete(context.Background(), up)
	assert.NoError(t, err)
}
