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
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

func TestEnsureFinalizer_Execute(t *testing.T) {
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

	phase := NewEnsureFinalizer(ctx).(*EnsureFinalizer)
	result, err := phase.Execute()
	assert.NoError(t, err)
	assert.Equal(t, false, result.Requeue)
}

func TestEnsureFinalizer_NeedExecute(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	tests := []struct {
		name    string
		cluster *bkev1beta1.BKECluster
		want    bool
	}{
		{
			name: "without finalizer",
			cluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
			},
			want: true,
		},
		{
			name: "with finalizer",
			cluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test",
					Namespace:  "default",
					Finalizers: []string{bkev1beta1.ClusterFinalizer},
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(tt.cluster).Build()
			ctx := &phaseframe.PhaseContext{
				BKECluster: tt.cluster,
				Client:     c,
				Scheme:     scheme,
				Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, tt.cluster),
			}
			phase := NewEnsureFinalizer(ctx).(*EnsureFinalizer)
			result := phase.NeedExecute(nil, tt.cluster)
			assert.Equal(t, tt.want, result)
		})
	}
}
