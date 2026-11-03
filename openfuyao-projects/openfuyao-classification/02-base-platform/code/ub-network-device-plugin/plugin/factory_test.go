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

package plugin

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
)

var res, _ = resource.InitDefaultResources()

func TestNew(t *testing.T) {
	type args struct {
		ctx  context.Context
		opts []Option
	}
	tests := []struct {
		name    string
		args    args
		want    []Interface
		wantErr bool
	}{
		{
			name: "resources empty",
			args: args{
				ctx:  context.Background(),
				opts: []Option{},
			},
			want:    nil,
			wantErr: false,
		},
		{
			name: "get plugins",
			args: args{
				ctx:  context.Background(),
				opts: []Option{WithResources(res)},
			},
			want:    []Interface{&urmaDevicePlugin{}},
			wantErr: false,
		},
	}

	mock := urma.MockUrma{}
	defer mock.Reset()

	for _, tt := range tests {
		mock.MockUbsGetSharedDevice(tt.name)

		t.Run(tt.name, func(t *testing.T) {
			got, err := New(tt.args.ctx, tt.args.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.want != nil {
				assert.NotNil(t, got)
			} else {
				assert.Nil(t, got)
			}
		})
	}
}

func TestNew_GetResourceManagersError(t *testing.T) {
	mock := urma.MockUrma{}
	defer mock.Reset()

	mock.MockUbsGetSharedDevice("UbsGetSharedDevice err")

	ctx := context.Background()
	opts := []Option{WithResources(res)}

	got, err := New(ctx, opts...)
	assert.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "failed to construct resource managers")
}

func TestNew_EmptyDeviceMap(t *testing.T) {
	mock := urma.MockUrma{}
	defer mock.Reset()

	mock.MockUbsGetSharedDevice("empty")

	ctx := context.Background()
	opts := []Option{WithResources(res)}

	got, err := New(ctx, opts...)
	assert.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "failed to construct resource managers")
}

func TestGetResourceManagers(t *testing.T) {
	mock := urma.MockUrma{}
	defer mock.Reset()

	mock.MockUbsGetSharedDevice("test")

	o := &options{resources: res}
	managers, err := o.getResourceManagers()

	assert.NoError(t, err)
	assert.Len(t, managers, 1)
}

func TestGetResourceManagers_Error(t *testing.T) {
	mock := urma.MockUrma{}
	defer mock.Reset()

	mock.MockUbsGetSharedDevice("UbsGetSharedDevice err")

	o := &options{resources: res}
	managers, err := o.getResourceManagers()

	assert.Error(t, err)
	assert.Nil(t, managers)
}
