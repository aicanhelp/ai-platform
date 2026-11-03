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

package urma

import (
	"reflect"
	"testing"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

func TestAllocateDevice(t *testing.T) {
	type args struct {
		name string
	}
	tests := []struct {
		name    string
		args    args
		want    urmasdk.DeviceInfo
		wantErr bool
	}{
		{
			name: "UbsAllocateDevice err",
			args: args{
				name: "vfe-0",
			},
			want:    urmasdk.DeviceInfo{},
			wantErr: true,
		},
		{
			name: "UbsAllocateDevice success",
			args: args{
				name: "vfe-0",
			},
			want: urmasdk.DeviceInfo{
				VfePaths:    []string{"/dev/uburma/urma1", "/dev/uburma/urma2"},
				BondingPath: "/dev/uburma/bonding_dev_1",
				BondingEid:  "0000:0000:0000:0ae0:0010:0000",
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		mock := MockUrma{}
		defer mock.Reset()

		mock.MockUbsAllocateDevice(tt.name)
		t.Run(tt.name, func(t *testing.T) {
			got, err := AllocateDevice(tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("AllocateDevice() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("AllocateDevice() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSharedDevice(t *testing.T) {
	tests := []struct {
		name    string
		want    []urmasdk.Device
		wantErr bool
	}{
		{
			name:    "UbsGetSharedDevice err",
			want:    nil,
			wantErr: true,
		},
		{
			name: "UbsGetSharedDevice success",
			want: []urmasdk.Device{
				{Name: "shared-1", Healthy: true, HwResId: 1},
				{Name: "shared-2", Healthy: false, HwResId: 1},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		mock := MockUrma{}
		defer mock.Reset()

		mock.MockUbsGetSharedDevice(tt.name)
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSharedDevice()
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSharedDevice() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetSharedDevice() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertDevices(t *testing.T) {
	tests := []struct {
		name            string
		originalDevices []urmasdk.Device
		expectedDevices []pluginapi.Device
	}{
		{
			name:            "empty slice",
			originalDevices: []urmasdk.Device{},
			expectedDevices: []pluginapi.Device{},
		},
		{
			name:            "nil slice",
			originalDevices: nil,
			expectedDevices: []pluginapi.Device{},
		},
		{
			name: "single healthy device",
			originalDevices: []urmasdk.Device{
				{Name: "device-1", Healthy: true, HwResId: 1},
			},
			expectedDevices: []pluginapi.Device{
				{ID: "device-1", Health: pluginapi.Healthy},
			},
		},
		{
			name: "single unhealthy device",
			originalDevices: []urmasdk.Device{
				{Name: "device-2", Healthy: false, HwResId: 2},
			},
			expectedDevices: []pluginapi.Device{
				{ID: "device-2", Health: pluginapi.Unhealthy},
			},
		},
		{
			name: "multiple devices mixed health",
			originalDevices: []urmasdk.Device{
				{Name: "device-1", Healthy: true, HwResId: 1},
				{Name: "device-2", Healthy: false, HwResId: 2},
				{Name: "device-3", Healthy: true, HwResId: 3},
			},
			expectedDevices: []pluginapi.Device{
				{ID: "device-1", Health: pluginapi.Healthy},
				{ID: "device-2", Health: pluginapi.Unhealthy},
				{ID: "device-3", Health: pluginapi.Healthy},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertDevices(tt.originalDevices)
			if !reflect.DeepEqual(got, tt.expectedDevices) {
				t.Errorf("convertDevices() = %v, want %v", got, tt.expectedDevices)
			}
		})
	}
}
