// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testResourceName = "testresource"
)

func fakeClient() client.Client {
	return fake.NewClientBuilder().Build()
}

func TestNPUClusterPolicyReconciler_getInstance(t *testing.T) {
	type args struct {
		ctx  context.Context
		name string
	}
	tests := []struct {
		name    string
		args    args
		mockErr error
	}{
		{
			name: "test",
			args: args{
				ctx:  context.Background(),
				name: testResourceName,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, r := getManagerAndReconciler(nil, getInstance())
			err := r.getInstance(tt.args.ctx, tt.args.name)
			assert.ErrorContains(t, err, "Failed to get ClusterPolicy object")
			assert.True(t, apierrors.IsNotFound(err))
		})
	}
}
