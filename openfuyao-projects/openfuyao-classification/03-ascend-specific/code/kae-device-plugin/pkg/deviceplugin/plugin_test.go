/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package deviceplugin

import (
	"context"
	"fmt"
	"io/ioutil"
	"path"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

type recordingListAndWatchServer struct {
	grpc.ServerStream
	responses []*v1beta1.ListAndWatchResponse
	err       error
}

func (s *recordingListAndWatchServer) Send(response *v1beta1.ListAndWatchResponse) error {
	s.responses = append(s.responses, response)
	return s.err
}

// TestGetDevicePluginOptions 测试 GetDevicePluginOptions
func TestGetDevicePluginOptions(t *testing.T) {
	s := &pluginService{kaeManager: NewKAEManager()}
	e := new(v1beta1.Empty)
	options, err := s.GetDevicePluginOptions(context.TODO(), e)
	assert.NotNil(t, options)
	assert.Nil(t, err)
}

func TestSendDeviceUpdatesSendsChangedDevices(t *testing.T) {
	manager := NewKAEManager()
	manager.grpcServer = grpc.NewServer()
	manager.hasDeviceChange = true
	manager.devices["hisi_hpre0"] = v1beta1.Device{ID: "hisi_hpre0", Health: v1beta1.Healthy}
	service := &pluginService{kaeManager: manager}
	stream := &recordingListAndWatchServer{}

	if err := service.sendDeviceUpdates(stream); err != nil {
		t.Fatalf("send device update: %v", err)
	}
	if len(stream.responses) != 1 || len(stream.responses[0].Devices) != 1 {
		t.Fatalf("unexpected device updates: %#v", stream.responses)
	}
	if stream.responses[0].Devices[0].ID != "hisi_hpre0" {
		t.Fatalf("unexpected device update: %#v", stream.responses[0].Devices[0])
	}
}

func TestSendDeviceUpdatesReturnsStreamError(t *testing.T) {
	manager := NewKAEManager()
	manager.grpcServer = grpc.NewServer()
	manager.hasDeviceChange = true
	manager.devices["hisi_hpre0"] = v1beta1.Device{ID: "hisi_hpre0", Health: v1beta1.Healthy}
	service := &pluginService{kaeManager: manager}
	stream := &recordingListAndWatchServer{err: fmt.Errorf("stream closed")}

	if err := service.sendDeviceUpdates(stream); err == nil {
		t.Fatal("sendDeviceUpdates succeeded after the stream failed")
	}
}

func TestListAndWatchStopsWhenManagerStops(t *testing.T) {
	manager := NewKAEManager()
	manager.stopMonitor = make(chan struct{}, 1)
	manager.stopMonitor <- struct{}{}
	service := &pluginService{kaeManager: manager}

	if err := service.ListAndWatch(&v1beta1.Empty{}, &recordingListAndWatchServer{}); err != nil {
		t.Fatalf("ListAndWatch returned an error while stopping: %v", err)
	}
}

func TestListAndWatchReturnsStreamError(t *testing.T) {
	manager := NewKAEManager()
	manager.devDirectory = t.TempDir()
	manager.grpcServer = grpc.NewServer()
	manager.stopMonitor = make(chan struct{})
	manager.hasDeviceChange = true
	manager.devices["hisi_hpre0"] = v1beta1.Device{ID: "hisi_hpre0", Health: v1beta1.Healthy}
	service := &pluginService{kaeManager: manager}
	stream := &recordingListAndWatchServer{err: fmt.Errorf("stream closed")}

	if err := service.ListAndWatch(&v1beta1.Empty{}, stream); err == nil {
		t.Fatal("ListAndWatch succeeded after the stream failed")
	}
}

// TestAllocate 测试Allocate
func TestAllocate(t *testing.T) {
	type fields struct {
		manager *KAEManager
	}

	type args struct {
		ctx      context.Context
		requests *v1beta1.AllocateRequest
	}

	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *v1beta1.AllocateResponse
		wantErr bool
	}{
		{
			name: "test_devices_empty",
			fields: fields{
				manager: &KAEManager{},
			},
			args: args{
				ctx: context.TODO(),
				requests: &v1beta1.AllocateRequest{
					ContainerRequests: []*v1beta1.ContainerAllocateRequest{
						{
							DevicesIDs: []string{"devID1"},
						},
					},
				},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "test_unhealthy_empty",
			fields: fields{
				manager: &KAEManager{
					devices: map[string]v1beta1.Device{"devID1": {Health: v1beta1.Unhealthy}},
				},
			},
			args: args{
				ctx: context.TODO(),
				requests: &v1beta1.AllocateRequest{
					ContainerRequests: []*v1beta1.ContainerAllocateRequest{
						{
							DevicesIDs: []string{"devID1"},
						},
					},
				},
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(
			tt.name, func(t *testing.T) {
				s := &pluginService{
					kaeManager: tt.fields.manager,
				}
				got, err := s.Allocate(tt.args.ctx, tt.args.requests)
				if (err != nil) != tt.wantErr {
					t.Errorf("allocate() error = %v, wanErr %v", err, tt.wantErr)
					return
				}
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("allocate() got = %v, want %v", got, tt.want)
				}
			},
		)
	}
}

// TestAllocateSuccess 成功分配设备
func TestAllocateSuccess(t *testing.T) {
	s := &pluginService{
		kaeManager: &KAEManager{
			devices:        map[string]v1beta1.Device{"devID1": {Health: v1beta1.Healthy}},
			defaultDevices: []string{"defaultDev"},
		},
	}
	got, err := s.Allocate(context.TODO(), &v1beta1.AllocateRequest{
		ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{
				DevicesIDs: []string{"devID1"},
			},
		},
	})
	assert.Nil(t, err)
	assert.NotNil(t, got)
}

// TestPreStartContainer 测试 PreStartContainer
func TestPreStartContainer(t *testing.T) {
	s := &pluginService{}
	request := &v1beta1.PreStartContainerRequest{}
	_, err := s.PreStartContainer(context.TODO(), request)
	assert.Nil(t, err)
}

// TestGetPreferredAllocation 测试GetPreferredAllocation
func TestGetPreferredAllocation(t *testing.T) {
	s := &pluginService{}
	_, err := s.GetPreferredAllocation(context.TODO(), &v1beta1.PreferredAllocationRequest{})
	assert.Nil(t, err)
}

// TestRegisterWithV1Beta1Kubelet 测试RegisterWithV1Beta1Kubelet
func TestRegisterWithV1Beta1Kubelet(t *testing.T) {
	testDir, err := ioutil.TempDir("", "/dev")
	err = RegisterWithV1Beta1Kubelet(path.Join(testDir, "plugin.sock"), resourceName)
	assert.NotNil(t, err)

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(
		grpc.Dial, func(_ string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
			return new(grpc.ClientConn), fmt.Errorf("break")
		},
	)

	err = RegisterWithV1Beta1Kubelet(path.Join(testDir, "plugin.sock"), resourceName)
	assert.NotNil(t, err)
}

func TestCreateUnixDialerReturnsConnectionErrorForMissingSocket(t *testing.T) {
	dialer := createUnixDialer()
	_, err := dialer(context.Background(), path.Join(t.TempDir(), "missing.sock"))
	assert.Error(t, err)
}

func TestCloseConnectionClosesClientConnection(t *testing.T) {
	connection, err := grpc.Dial("passthrough:///unused", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("create client connection: %v", err)
	}

	closeConnection(connection)
}
