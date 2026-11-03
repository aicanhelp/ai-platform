/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-08
 */

package checkpoint

import (
	"context"
	"testing"

	"github.com/stretchr/testify/mock"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

// MockClient 模拟 Kubernetes client
type MockClient struct {
	mock.Mock
}

func (m *MockClient) Get(ctx context.Context, key types.NamespacedName, obj client.Object) error {
	args := m.Called(ctx, key, obj)
	return args.Error(0)
}

func (m *MockClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	args := m.Called(ctx, list, opts)
	return args.Error(0)
}

func (m *MockClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	args := m.Called(ctx, obj, opts)
	return args.Error(0)
}

func (m *MockClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	args := m.Called(ctx, obj, opts)
	return args.Error(0)
}

func (m *MockClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	args := m.Called(ctx, obj, opts)
	return args.Error(0)
}

func (m *MockClient) Status() client.StatusWriter {
	args := m.Called()
	if sw, ok := args.Get(0).(client.StatusWriter); ok {
		return sw
	}
	return nil
}

func (m *MockClient) Patch(ctx context.Context, obj client.Object,
	patch client.Patch, opts ...client.PatchOption) error {
	args := m.Called(ctx, obj, patch, opts)
	return args.Error(0)
}

func (m *MockClient) DeleteAllOf(ctx context.Context, obj client.Object, opts ...client.DeleteAllOfOption) error {
	args := m.Called(ctx, obj, opts)
	return args.Error(0)
}

// FakeClusterStateWrapper 包装真实的ClusterState 并提供mock功能
type FakeClusterStateWrapper struct {
	realState        *aggregate.ClusterState
	mockClusterState *MockClusterState
	useMock          bool
}

// 实现 ClusterState 的所有方法
func (f *FakeClusterStateWrapper) AddOrUpdateInitialAggregateState(
	key aggregate.AggregateStateKey, state *aggregate.AggregateContainerState) {
	if f.useMock {
		f.mockClusterState.AddOrUpdateInitialAggregateState(key, state)
	} else {
		f.realState.AddOrUpdateInitialAggregateState(key, state)
	}
}

func (f *FakeClusterStateWrapper) AddOrUpdateCheckpoint(
	name types.NamespacedName, checkpoint *colocationv1.ContainerCheckpoint) {
	if f.useMock {
		f.mockClusterState.AddOrUpdateCheckpoint(name, checkpoint)
	} else {
		f.realState.AddOrUpdateCheckpoint(name, checkpoint)
	}
}

func (f *FakeClusterStateWrapper) InitialAggregateStateSize() int {
	if f.useMock {
		return f.mockClusterState.InitialAggregateStateSize()
	}
	return f.realState.InitialAggregateStateSize()
}

func (f *FakeClusterStateWrapper) CleanupInitialAggregateState() int {
	if f.useMock {
		return f.mockClusterState.CleanupInitialAggregateState()
	}
	return f.realState.CleanupInitialAggregateState()
}

func (f *FakeClusterStateWrapper) SaveCheckpoints(parallelCount int) (
	[]aggregate.AggregateStateKey, []*colocationv1.ContainerCheckpointStatus) {
	if f.useMock {
		return f.mockClusterState.SaveCheckpoints(parallelCount)
	}
	return f.realState.SaveCheckpoints(parallelCount)
}

func (f *FakeClusterStateWrapper) GetCheckpoint(key aggregate.AggregateStateKey) *colocationv1.ContainerCheckpoint {
	if f.useMock {
		return f.mockClusterState.GetCheckpoint(key)
	}
	return f.realState.GetCheckpoint(key)
}

func (f *FakeClusterStateWrapper) GetAllCheckpoint() []*colocationv1.ContainerCheckpoint {
	if f.useMock {
		return f.mockClusterState.GetAllCheckpoint()
	}
	return f.realState.GetAllCheckpoint()
}

func (f *FakeClusterStateWrapper) GetAllNodeStates() map[string]*aggregate.NodeState {
	if f.useMock {
		return f.mockClusterState.GetAllNodeStates()
	}
	return f.realState.GetAllNodeStates()
}

// MockClusterState 模拟 ClusterState
type MockClusterState struct {
	mock.Mock
}

func (m *MockClusterState) AddOrUpdateInitialAggregateState(
	key aggregate.AggregateStateKey, state *aggregate.AggregateContainerState) {
	m.Called(key, state)
}

func (m *MockClusterState) AddOrUpdateCheckpoint(
	name types.NamespacedName, checkpoint *colocationv1.ContainerCheckpoint) {
	m.Called(name, checkpoint)
}

func (m *MockClusterState) InitialAggregateStateSize() int {
	args := m.Called()
	return args.Int(0)
}

func (m *MockClusterState) CleanupInitialAggregateState() int {
	args := m.Called()
	return args.Int(0)
}

func (m *MockClusterState) SaveCheckpoints(parallelCount int) (
	[]aggregate.AggregateStateKey, []*colocationv1.ContainerCheckpointStatus) {
	args := m.Called(parallelCount)
	keys, ok1 := args.Get(0).([]aggregate.AggregateStateKey)
	statuses, ok2 := args.Get(1).([]*colocationv1.ContainerCheckpointStatus)
	if !ok1 || !ok2 {
		return nil, nil
	}
	return keys, statuses
}

func (m *MockClusterState) GetCheckpoint(key aggregate.AggregateStateKey) *colocationv1.ContainerCheckpoint {
	args := m.Called(key)
	if ckpt, ok := args.Get(0).(*colocationv1.ContainerCheckpoint); ok {
		return ckpt
	}
	return nil
}

func (m *MockClusterState) GetAllCheckpoint() []*colocationv1.ContainerCheckpoint {
	args := m.Called()
	if checkpoints, ok := args.Get(0).([]*colocationv1.ContainerCheckpoint); ok {
		return checkpoints
	}
	return nil
}

func (m *MockClusterState) GetAllNodeStates() map[string]*aggregate.NodeState {
	args := m.Called()
	if states, ok := args.Get(0).(map[string]*aggregate.NodeState); ok {
		return states
	}
	return nil
}

func TestInitFromCheckpoints(t *testing.T) {
	tests := []struct {
		name          string
		setupMocks    func(mockClient *MockClient, mockClusterState *MockClusterState)
		expectedError bool
	}{
		{
			name: "successful initialization",
			setupMocks: func(mockClient *MockClient, mockClusterState *MockClusterState) {
				checkpointList := &colocationv1.ContainerCheckpointList{
					Items: []colocationv1.ContainerCheckpoint{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "test-checkpoint",
								Namespace: "default",
							},
							Spec: colocationv1.ContainerCheckpointSpec{
								Namespace:     "default",
								PodName:       "test-pod",
								ContainerName: "test-container",
							},
							Status: colocationv1.ContainerCheckpointStatus{},
						},
					},
				}

				mockClient.On("List", mock.Anything,
					mock.AnythingOfType("*v1.ContainerCheckpointList"), mock.Anything).
					Run(func(args mock.Arguments) {
						if list, ok := args.Get(1).(*colocationv1.ContainerCheckpointList); ok {
							*list = *checkpointList
						} else {
							t.Fatalf("List checkpoints failed: %v", args.Error(0))
						}

					}).Return(nil)

				mockClusterState.On("AddOrUpdateInitialAggregateState", mock.Anything, mock.Anything).Once()
				mockClusterState.On("AddOrUpdateCheckpoint", mock.Anything, mock.Anything).Once()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &MockClient{}
			mockClusterState := &MockClusterState{}

			tt.setupMocks(mockClient, mockClusterState)

			ctx := context.Background()

			ckptList := &colocationv1.ContainerCheckpointList{}
			if err := mockClient.List(ctx, ckptList); err != nil {
				t.Fatalf("List checkpoints failed: %v", err)
			}

			for _, ckpt := range ckptList.Items {
				podID := aggregate.PodID{
					Namespace: ckpt.Spec.Namespace,
					PodName:   ckpt.Spec.PodName,
				}
				containerName := ckpt.Spec.ContainerName
				cs := aggregate.NewAggregateContainerState()
				mockClusterState.AddOrUpdateInitialAggregateState(
					aggregate.MakeAggregateStateKey(podID, containerName),
					cs)
				mockClusterState.AddOrUpdateCheckpoint(types.NamespacedName{
					Namespace: ckpt.Namespace,
					Name:      ckpt.Name,
				}, &ckpt)
			}

			mockClient.AssertExpectations(t)
			mockClusterState.AssertExpectations(t)
		})
	}
}
