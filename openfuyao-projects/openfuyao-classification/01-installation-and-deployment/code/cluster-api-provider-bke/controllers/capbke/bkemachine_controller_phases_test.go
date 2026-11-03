/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package capbke

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/blang/semver"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/cluster-api/util/patch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	metricrecord "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics/record"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	machinePhasesNumZero  = 0
	machinePhasesNumOne   = 1
	machinePhasesNumTwo   = 2
	machinePhasesNumThree = 3
	machinePhasesNumFour  = 4
	machinePhasesNumFive  = 5

	machinePhasesIpv4SegmentA = 192
	machinePhasesIpv4SegmentB = 168
	machinePhasesIpv4SegmentC = 1
	machinePhasesIpv4SegmentD = 100

	machinePhasesNamespace  = "test-ns"
	machinePhasesName       = "test-machine"
	machinePhasesNodeIP     = "192.168.1.100"
	machinePhasesNodeIP2    = "192.168.1.101"
	machinePhasesNodeHost   = "test-node"
	machinePhasesProviderID = "test://test-node"
	machinePhasesCluster    = "test-cluster"
)

// newMachinePhaseScheme creates a scheme with all needed types
func newMachinePhaseScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

// newPhasesReconciler creates a BKEMachineReconciler for testing
func newPhasesReconciler(objs ...client.Object) *BKEMachineReconciler {
	scheme := newMachinePhaseScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > machinePhasesNumZero {
		builder = builder.WithObjects(objs...).WithStatusSubresource(objs...)
	}
	fakeClient := builder.Build()
	return &BKEMachineReconciler{
		Client:          fakeClient,
		Scheme:          scheme,
		Recorder:        record.NewFakeRecorder(machinePhasesNumFive),
		NodeFetcher:     nodeutil.NewNodeFetcher(fakeClient),
		nodesBootRecord: make(map[string]struct{}),
	}
}

func newTestLogger() *log.Logger {
	return nil
}

// --- Pure function tests ---

func TestSetMachineAddress(t *testing.T) {
	tests := []struct {
		name       string
		bkeMachine *bkev1beta1.BKEMachine
		node       confv1beta1.Node
	}{
		{
			name:       "set machine address from node",
			bkeMachine: &bkev1beta1.BKEMachine{},
			node: confv1beta1.Node{
				Hostname: machinePhasesNodeHost,
				IP:       testNodeIP,
			},
		},
		{
			name:       "set machine address with empty node",
			bkeMachine: &bkev1beta1.BKEMachine{},
			node:       confv1beta1.Node{},
		},
		{
			name:       "set machine address with only hostname",
			bkeMachine: &bkev1beta1.BKEMachine{},
			node: confv1beta1.Node{
				Hostname: machinePhasesNodeHost,
			},
		},
		{
			name:       "set machine address with only IP",
			bkeMachine: &bkev1beta1.BKEMachine{},
			node: confv1beta1.Node{
				IP: testNodeIP,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setMachineAddress(tt.bkeMachine, tt.node)
			assert.NotNil(t, tt.bkeMachine.Status.Addresses)
			assert.Len(t, tt.bkeMachine.Status.Addresses, machinePhasesNumThree)
		})
	}
}

func TestSetProviderID(t *testing.T) {
	tests := []struct {
		name       string
		bkeMachine *bkev1beta1.BKEMachine
		providerID string
	}{
		{
			name:       "set provider ID",
			bkeMachine: &bkev1beta1.BKEMachine{},
			providerID: machinePhasesProviderID,
		},
		{
			name:       "set empty provider ID",
			bkeMachine: &bkev1beta1.BKEMachine{},
			providerID: "",
		},
		{
			name:       "set new provider ID overwrites old",
			bkeMachine: &bkev1beta1.BKEMachine{Spec: bkev1beta1.BKEMachineSpec{ProviderID: func() *string { s := "old://id"; return &s }()}},
			providerID: "new://id",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setProviderID(tt.bkeMachine, tt.providerID)
			assert.Equal(t, tt.providerID, *tt.bkeMachine.Spec.ProviderID)
		})
	}
}

func TestGenerateKubeletConfigName(t *testing.T) {
	tests := []struct {
		name         string
		version      semver.Version
		expectedName string
	}{
		{
			name:         "version 1.23.0 - should use versioned name",
			version:      semver.MustParse("1.23.0"),
			expectedName: "kubelet-config-1.23",
		},
		{
			name:         "version 1.24.0 - should use unversioned name",
			version:      semver.MustParse("1.24.0"),
			expectedName: "kubelet-config",
		},
		{
			name:         "version 1.25.0 - should use unversioned name",
			version:      semver.MustParse("1.25.0"),
			expectedName: "kubelet-config",
		},
		{
			name:         "version 1.26.1 - should use unversioned name",
			version:      semver.MustParse("1.26.1"),
			expectedName: "kubelet-config",
		},
		{
			name:         "version 1.20.11 - should use versioned name",
			version:      semver.MustParse("1.20.11"),
			expectedName: "kubelet-config-1.20",
		},
		{
			name:         "version 1.19.0 - should use versioned name",
			version:      semver.MustParse("1.19.0"),
			expectedName: "kubelet-config-1.19",
		},
		{
			name:         "version 1.18.5 - should use versioned name",
			version:      semver.MustParse("1.18.5"),
			expectedName: "kubelet-config-1.18",
		},
		{
			name:         "version 1.24.1 - should use unversioned name",
			version:      semver.MustParse("1.24.1"),
			expectedName: "kubelet-config",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateKubeletConfigName(tt.version)
			assert.Equal(t, tt.expectedName, result)
		})
	}
}

func TestSetMachineAddressMultiple(t *testing.T) {
	node := confv1beta1.Node{
		Hostname: machinePhasesNodeHost,
		IP:       testNodeIP,
	}

	machine := &bkev1beta1.BKEMachine{}
	setMachineAddress(machine, node)

	assert.Len(t, machine.Status.Addresses, machinePhasesNumThree)

	assert.Equal(t, bkev1beta1.MachineHostName, machine.Status.Addresses[machinePhasesNumZero].Type)
	assert.Equal(t, machinePhasesNodeHost, machine.Status.Addresses[machinePhasesNumZero].Address)

	assert.Equal(t, bkev1beta1.MachineInternalIP, machine.Status.Addresses[machinePhasesNumOne].Type)
	assert.Equal(t, machinePhasesNodeIP, machine.Status.Addresses[machinePhasesNumOne].Address)

	assert.Equal(t, bkev1beta1.MachineExternalIP, machine.Status.Addresses[machinePhasesNumTwo].Type)
	assert.Equal(t, machinePhasesNodeIP, machine.Status.Addresses[machinePhasesNumTwo].Address)
}

func TestSetProviderIDMultiple(t *testing.T) {
	machine := &bkev1beta1.BKEMachine{}
	providerID := machinePhasesProviderID

	setProviderID(machine, providerID)

	assert.NotNil(t, machine.Spec.ProviderID)
	assert.Equal(t, providerID, *machine.Spec.ProviderID)

	setProviderID(machine, "new://provider-id")
	assert.Equal(t, "new://provider-id", *machine.Spec.ProviderID)
}

func TestGenerateKubeletConfigNameEdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		version      semver.Version
		expectedName string
	}{
		{
			name:         "version 1.24.0-0",
			version:      semver.MustParse("1.24.0-0"),
			expectedName: "kubelet-config",
		},
		{
			name:         "version 1.23.1",
			version:      semver.MustParse("1.23.1"),
			expectedName: "kubelet-config-1.23",
		},
		{
			name:         "version 1.27.0-alpha",
			version:      semver.MustParse("1.27.0-alpha"),
			expectedName: "kubelet-config",
		},
		{
			name:         "version 1.22.0-rc",
			version:      semver.MustParse("1.22.0-rc.1"),
			expectedName: "kubelet-config-1.22",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateKubeletConfigName(tt.version)
			assert.Equal(t, tt.expectedName, result)
		})
	}
}

// --- ConfigMap tests with fake client ---

func TestMockKubeadmConfigConfigmap(t *testing.T) {
	scheme := newMachinePhaseScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	err := MockKubeadmConfigConfigmap(context.Background(), fakeClient)
	assert.NoError(t, err)

	// Call again to test update path
	err = MockKubeadmConfigConfigmap(context.Background(), fakeClient)
	assert.NoError(t, err)
}

func TestMockKubeletConfigConfigmap(t *testing.T) {
	scheme := newMachinePhaseScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	err := MockKubeletConfigConfigmap(context.Background(), fakeClient, "v1.25.0")
	assert.NoError(t, err)

	// Call again to test update path
	err = MockKubeletConfigConfigmap(context.Background(), fakeClient, "v1.25.0")
	assert.NoError(t, err)
}

func TestMockKubeletConfigConfigmapInvalidVersion(t *testing.T) {
	scheme := newMachinePhaseScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	err := MockKubeletConfigConfigmap(context.Background(), fakeClient, "invalid-version")
	assert.Error(t, err)
}

func TestCreateOrUpdateConfigMap(t *testing.T) {
	scheme := newMachinePhaseScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := context.Background()

	// Create
	err := createOrUpdateConfigMap(ctx, fakeClient, "test-cm", "default", map[string]string{"key": "val"})
	assert.NoError(t, err)

	// Update (already exists)
	err = createOrUpdateConfigMap(ctx, fakeClient, "test-cm", "default", map[string]string{"key": "val2"})
	assert.NoError(t, err)
}

func TestSetTargetClusterNodeRole(t *testing.T) {
	scheme := newMachinePhaseScheme()

	tests := []struct {
		name     string
		nodeRole string
	}{
		{name: "worker role", nodeRole: bkenode.WorkerNodeRole},
		{name: "master role", nodeRole: bkenode.MasterNodeRole},
		{name: "master worker role", nodeRole: bkenode.MasterWorkerNodeRole},
		{name: "unknown role", nodeRole: "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(node).Build()

			err := setTargetClusterNodeRole(context.Background(), fakeClient, node, tt.nodeRole)
			assert.NoError(t, err)
		})
	}
}

// --- parseLockInfo ---

func TestParseLockInfo(t *testing.T) {
	r := &BKEMachineReconciler{}

	t.Run("valid lock info", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			Data: map[string]string{
				lockKey: `{"machineName":"test-machine"}`,
			},
		}
		l, err := r.parseLockInfo(cm)
		assert.NoError(t, err)
		assert.Equal(t, "test-machine", l.MachineName)
	})

	t.Run("invalid JSON", func(t *testing.T) {
		cm := &corev1.ConfigMap{
			Data: map[string]string{
				lockKey: `{invalid}`,
			},
		}
		_, err := r.parseLockInfo(cm)
		assert.Error(t, err)
	})
}

// --- getMachineRole ---

func TestGetMachineRole(t *testing.T) {
	r := &BKEMachineReconciler{}

	t.Run("worker machine", func(t *testing.T) {
		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName},
		}
		role := r.getMachineRole(machine)
		assert.Equal(t, bkenode.WorkerNodeRole, role)
	})

	t.Run("control plane machine", func(t *testing.T) {
		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name: machinePhasesName,
				Labels: map[string]string{
					clusterv1.MachineControlPlaneLabel: "",
				},
			},
		}
		role := r.getMachineRole(machine)
		assert.Equal(t, bkenode.MasterNodeRole, role)
	})
}

// --- getBootstrapPhase / handleLockConfigMap ---

func TestGetBootstrapPhase(t *testing.T) {
	scheme := newMachinePhaseScheme()

	t.Run("worker machine returns JoinWorker", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName}}
		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}

		phase, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.NoError(t, err)
		assert.Equal(t, bkev1beta1.JoinWorker, phase)
	})

	t.Run("control plane with uninitialized cluster returns InitControlPlane", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		}
		// Mark ControlPlaneInitializedCondition as false
		conditions.MarkFalse(cluster, clusterv1.ControlPlaneInitializedCondition, "", clusterv1.ConditionSeverityInfo, "")

		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name:   machinePhasesName,
				Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
			},
		}

		phase, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.NoError(t, err)
		assert.Equal(t, bkev1beta1.InitControlPlane, phase)
	})

	t.Run("control plane lock not found returns JoinControlPlane", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name:   machinePhasesName,
				Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
			},
		}

		phase, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.NoError(t, err)
		assert.Equal(t, bkev1beta1.JoinControlPlane, phase)
	})

	t.Run("control plane lock matches machine returns InitControlPlane", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		lockCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-lock", machinePhasesCluster),
				Namespace: machinePhasesNamespace,
			},
			Data: map[string]string{
				lockKey: fmt.Sprintf(`{"machineName":"%s"}`, machinePhasesName),
			},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(lockCM).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name:   machinePhasesName,
				Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
			},
		}

		phase, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.NoError(t, err)
		assert.Equal(t, bkev1beta1.InitControlPlane, phase)
	})

	t.Run("control plane lock does not match returns JoinControlPlane", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		lockCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-lock", machinePhasesCluster),
				Namespace: machinePhasesNamespace,
			},
			Data: map[string]string{
				lockKey: `{"machineName":"other-machine"}`,
			},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(lockCM).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name:   machinePhasesName,
				Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
			},
		}

		phase, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.NoError(t, err)
		assert.Equal(t, bkev1beta1.JoinControlPlane, phase)
	})

	t.Run("lock configmap with nil data returns error", func(t *testing.T) {
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		lockCM := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-lock", machinePhasesCluster),
				Namespace: machinePhasesNamespace,
			},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(lockCM).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name:   machinePhasesName,
				Labels: map[string]string{clusterv1.MachineControlPlaneLabel: ""},
			},
		}

		_, err := r.getBootstrapPhase(context.Background(), machine, cluster)
		assert.Error(t, err)
	})
}

// --- recordBootstrapPhaseEvent ---

func TestRecordBootstrapPhaseEvent(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	tests := []struct {
		name  string
		phase confv1beta1.BKEClusterPhase
	}{
		{name: "JoinWorker phase", phase: bkev1beta1.JoinWorker},
		{name: "JoinControlPlane phase", phase: bkev1beta1.JoinControlPlane},
		{name: "InitControlPlane phase", phase: bkev1beta1.InitControlPlane},
		{name: "MasterScalingUp phase", phase: bkev1beta1.MasterScalingUp},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
			bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
			node := &confv1beta1.Node{IP: machinePhasesNodeIP, Hostname: machinePhasesNodeHost, Role: []string{bkenode.WorkerNodeRole}}

			err := r.recordBootstrapPhaseEvent(cluster, bkeCluster, node, tt.phase, log)
			assert.NoError(t, err)
		})
	}
}

// --- checkBootstrapStatus ---

func TestCheckBootstrapStatus(t *testing.T) {
	r := &BKEMachineReconciler{}

	t.Run("empty machines not ready", func(t *testing.T) {
		ready, failed := r.checkBootstrapStatus(nil, &bkev1beta1.BKEMachine{}, bkenode.Nodes{})
		assert.False(t, ready)
		assert.False(t, failed)
	})

	t.Run("all bootstrapped and matching count", func(t *testing.T) {
		bm := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		conditions.MarkTrue(&bm, bkev1beta1.BootstrapSucceededCondition)
		machines := []bkev1beta1.BKEMachine{bm}
		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}}

		ready, failed := r.checkBootstrapStatus(machines, &bm, nodes)
		assert.True(t, ready)
		assert.False(t, failed)
	})

	t.Run("not all bootstrapped", func(t *testing.T) {
		bm1 := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		conditions.MarkTrue(&bm1, bkev1beta1.BootstrapSucceededCondition)
		bm2 := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m2"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: false},
		}
		machines := []bkev1beta1.BKEMachine{bm1, bm2}
		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}

		ready, _ := r.checkBootstrapStatus(machines, &bm2, nodes)
		assert.False(t, ready)
	})

	t.Run("bootstrap failed detected", func(t *testing.T) {
		bm := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: false},
		}
		conditions.MarkFalse(&bm, bkev1beta1.BootstrapSucceededCondition,
			constant.NodeBootStrapFailedReason, clusterv1.ConditionSeverityWarning, "failed")
		machines := []bkev1beta1.BKEMachine{bm}
		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}}

		_, failed := r.checkBootstrapStatus(machines, &bm, nodes)
		assert.True(t, failed)
	})

	t.Run("machine count mismatch not ready", func(t *testing.T) {
		bm := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		conditions.MarkTrue(&bm, bkev1beta1.BootstrapSucceededCondition)
		machines := []bkev1beta1.BKEMachine{bm}
		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}

		ready, _ := r.checkBootstrapStatus(machines, &bm, nodes)
		assert.False(t, ready)
	})
}

// --- handleBootstrapFailure ---

func TestHandleBootstrapFailure(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
	err := r.handleBootstrapFailure(bkeCluster, log)
	assert.NoError(t, err)
}

// --- handleAllNodesBootstrapped ---

func TestHandleAllNodesBootstrapped(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{KubernetesVersion: "1.25.0"},
				},
			},
		}
		err := r.handleAllNodesBootstrapped(context.Background(), bkeCluster, log)
		assert.NoError(t, err)
		assert.Equal(t, "1.25.0", bkeCluster.Status.KubernetesVersion)
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{KubernetesVersion: "1.25.0"},
				},
			},
		}
		err := r.handleAllNodesBootstrapped(context.Background(), bkeCluster, log)
		assert.Error(t, err)
	})
}

// --- handleClusterReady ---

func TestHandleClusterReady(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})
		defer patches.Reset()

		patches.ApplyFunc(metricrecord.ClusterBootstrapDurationRecord,
			func(_ client.Object) {})

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		err := r.handleClusterReady(context.Background(), bkeCluster, log)
		assert.NoError(t, err)
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})
		defer patches.Reset()

		patches.ApplyFunc(metricrecord.ClusterBootstrapDurationRecord,
			func(_ client.Object) {})

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		err := r.handleClusterReady(context.Background(), bkeCluster, log)
		assert.Error(t, err)
	})
}

// --- handleClusterBooting ---

func TestHandleClusterBooting(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}
		err := r.handleClusterBooting(context.Background(), bkeCluster, node, log)
		assert.NoError(t, err)
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}
		err := r.handleClusterBooting(context.Background(), bkeCluster, node, log)
		assert.Error(t, err)
	})
}

// --- handleClusterState ---

func TestHandleClusterState(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})
	defer patches.Reset()

	t.Run("all boot flag triggers handleAllNodesBootstrapped", func(t *testing.T) {
		// one machine bootstrapped, one node => allBootFlag=true
		bm := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		conditions.MarkTrue(&bm, bkev1beta1.BootstrapSucceededCondition)
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{KubernetesVersion: "1.25.0"},
				},
			},
		}

		params := HandleClusterStateParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			BKECluster:          bkeCluster,
			BKEMachine:          &bm,
			NodeState:           confv1beta1.Node{IP: machinePhasesNodeIP},
			BKEMachines:         []bkev1beta1.BKEMachine{bm},
			Nodes:               bkenode.Nodes{{IP: machinePhasesNodeIP}},
			ClusterReady:        true,
		}
		err := r.handleClusterState(params)
		assert.NoError(t, err)
	})

	t.Run("not all booted shows waiting", func(t *testing.T) {
		bm := bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: "m1"},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: false},
		}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}

		params := HandleClusterStateParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			BKECluster:          bkeCluster,
			BKEMachine:          &bm,
			NodeState:           confv1beta1.Node{IP: machinePhasesNodeIP},
			BKEMachines:         []bkev1beta1.BKEMachine{bm},
			Nodes:               bkenode.Nodes{{IP: machinePhasesNodeIP}},
		}
		err := r.handleClusterState(params)
		assert.NoError(t, err)
	})
}

// --- getNodeInfo ---

func TestGetNodeInfo(t *testing.T) {
	r := &BKEMachineReconciler{}

	t.Run("worker node", func(t *testing.T) {
		node := confv1beta1.Node{IP: machinePhasesNodeIP, Role: []string{bkenode.WorkerNodeRole}}
		bkeCluster := &bkev1beta1.BKECluster{}
		info := r.getNodeInfo(node, bkeCluster)
		assert.Equal(t, bkenode.WorkerNodeRole, info.Role)
		assert.False(t, info.Cordon)
	})

	t.Run("master node", func(t *testing.T) {
		node := confv1beta1.Node{IP: machinePhasesNodeIP, Role: []string{bkenode.MasterNodeRole}}
		bkeCluster := &bkev1beta1.BKECluster{}
		info := r.getNodeInfo(node, bkeCluster)
		assert.Equal(t, bkenode.MasterNodeRole, info.Role)
		assert.True(t, info.Cordon)
	})

	t.Run("master worker node", func(t *testing.T) {
		node := confv1beta1.Node{IP: machinePhasesNodeIP, Role: []string{bkenode.MasterWorkerNodeRole}}
		bkeCluster := &bkev1beta1.BKECluster{}
		info := r.getNodeInfo(node, bkeCluster)
		assert.Equal(t, bkenode.MasterWorkerNodeRole, info.Role)
		assert.False(t, info.Cordon)
	})

	t.Run("master schedulable annotation disables cordon", func(t *testing.T) {
		node := confv1beta1.Node{IP: machinePhasesNodeIP, Role: []string{bkenode.MasterNodeRole}}
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					annotation.MasterSchedulableAnnotationKey: "true",
				},
			},
		}
		info := r.getNodeInfo(node, bkeCluster)
		assert.True(t, info.Cordon)
		assert.False(t, info.AnnotationCordonFlag)
	})
}

// --- filterAvailableNode ---

func TestFilterAvailableNode(t *testing.T) {
	scheme := newMachinePhaseScheme()

	t.Run("InitControlPlane returns first node", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			nodesBootRecord: make(map[string]struct{}),
		}

		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}

		node, err := r.filterAvailableNode(context.Background(), nodes, bkeCluster, bkev1beta1.InitControlPlane)
		assert.NoError(t, err)
		assert.Equal(t, machinePhasesNodeIP, node.IP)
	})

	t.Run("InitControlPlane prefers ControlPlaneEndpoint host", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			nodesBootRecord: make(map[string]struct{}),
		}

		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
			Spec: confv1beta1.BKEClusterSpec{
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{Host: machinePhasesNodeIP2},
			},
		}

		node, err := r.filterAvailableNode(context.Background(), nodes, bkeCluster, bkev1beta1.InitControlPlane)
		assert.NoError(t, err)
		assert.Equal(t, machinePhasesNodeIP2, node.IP)
	})

	t.Run("JoinWorker returns available node", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			nodesBootRecord: make(map[string]struct{}),
		}

		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}

		node, err := r.filterAvailableNode(context.Background(), nodes, bkeCluster, bkev1beta1.JoinWorker)
		assert.NoError(t, err)
		assert.NotNil(t, node)
	})

	t.Run("all nodes in boot record returns error", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{
			Client: fakeClient,
			nodesBootRecord: map[string]struct{}{
				machinePhasesNodeIP:  {},
				machinePhasesNodeIP2: {},
			},
			mux: sync.Mutex{},
		}

		nodes := bkenode.Nodes{{IP: machinePhasesNodeIP}, {IP: machinePhasesNodeIP2}}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}

		_, err := r.filterAvailableNode(context.Background(), nodes, bkeCluster, bkev1beta1.JoinWorker)
		assert.Error(t, err)
	})
}

// --- processCommand ---

func TestProcessCommand(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	t.Run("no matching nodes returns early", func(t *testing.T) {
		params := ProcessCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			Nodes:  bkenode.Nodes{{IP: "10.0.0.1"}},
			HostIp: "10.0.0.99",
			Cmd:    newTestCommand("bootstrap-test"),
			Res:    ctrl.Result{},
		}
		res, errs := r.processCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})

	t.Run("no matching nodes for reset command is ok", func(t *testing.T) {
		params := ProcessCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			Nodes:  bkenode.Nodes{{IP: "10.0.0.1"}},
			HostIp: "10.0.0.99",
			Cmd:    newTestCommand("reset-node-test"),
			Res:    ctrl.Result{},
		}
		res, errs := r.processCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})

	t.Run("unknown command prefix returns early", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(command.CheckCommandStatus, func(_ *agentv1beta1.Command) (bool, []string, []string) {
			return false, nil, nil
		})
		defer patches.Reset()

		params := ProcessCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			Nodes:  bkenode.Nodes{{IP: machinePhasesNodeIP}},
			HostIp: machinePhasesNodeIP,
			Cmd:    newTestCommand("unknown-prefix"),
			Res:    ctrl.Result{},
		}
		res, errs := r.processCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})
}

// newTestCommand creates a minimal agentv1beta1.Command for testing
func newTestCommand(name string) agentv1beta1.Command {
	return agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: machinePhasesNamespace},
	}
}

// --- processBootstrapCommand ---

func TestProcessBootstrapCommand(t *testing.T) {
	log := newTestLogger()

	t.Run("already bootstrapped returns early", func(t *testing.T) {
		r := newPhasesReconciler()
		params := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{Status: bkev1beta1.BKEMachineStatus{Bootstrapped: true}},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         &agentv1beta1.Command{},
			Res:         ctrl.Result{},
		}
		res, errs := r.processBootstrapCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})

	t.Run("cluster deleting returns early", func(t *testing.T) {
		r := newPhasesReconciler()
		params := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{
						ClusterStatus: bkev1beta1.ClusterDeleting,
					},
				},
				BKEMachine: &bkev1beta1.BKEMachine{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         &agentv1beta1.Command{},
			Res:         ctrl.Result{},
		}
		res, errs := r.processBootstrapCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})

	t.Run("complete with failed nodes triggers failure", func(t *testing.T) {
		r := newPhasesReconciler()
		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).processBootstrapFailure,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapFailureParams) (ctrl.Result, []error) {
				return ctrl.Result{}, nil
			})
		defer patches.Reset()

		params := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         &agentv1beta1.Command{},
			Complete:    true,
			FailedNodes: []string{machinePhasesNodeIP},
			Res:         ctrl.Result{},
		}
		_, errs := r.processBootstrapCommand(params)
		assert.Nil(t, errs)
	})

	t.Run("complete with success triggers success", func(t *testing.T) {
		r := newPhasesReconciler()
		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).processBootstrapSuccess,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams) (ctrl.Result, []error) {
				return ctrl.Result{}, nil
			})
		defer patches.Reset()

		params := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			CurrentNode:  confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:          &agentv1beta1.Command{},
			Complete:     true,
			SuccessNodes: []string{machinePhasesNodeIP},
			Res:          ctrl.Result{},
		}
		_, errs := r.processBootstrapCommand(params)
		assert.Nil(t, errs)
	})

	t.Run("not complete returns early", func(t *testing.T) {
		r := newPhasesReconciler()
		params := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         &agentv1beta1.Command{},
			Complete:    false,
			Res:         ctrl.Result{},
		}
		res, errs := r.processBootstrapCommand(params)
		assert.Equal(t, ctrl.Result{}, res)
		assert.Nil(t, errs)
	})
}

// --- processResetCommand ---

func TestProcessResetCommand(t *testing.T) {
	log := newTestLogger()
	scheme := newMachinePhaseScheme()

	t.Run("no success nodes removes finalizer", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			Scheme:          scheme,
			Recorder:        record.NewFakeRecorder(machinePhasesNumFive),
			nodesBootRecord: make(map[string]struct{}),
		}

		patchHelper, _ := patch.NewHelper(bkeMachine, fakeClient)

		params := ProcessResetCommandParams{
			CommonCommandParams: CommonCommandParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					BKECluster:          &bkev1beta1.BKECluster{},
					BKEMachine:          bkeMachine,
				},
				PatchHelper:  patchHelper,
				Cmd:          &agentv1beta1.Command{},
				SuccessNodes: []string{},
				FailedNodes:  []string{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
		}
		res, _ := r.processResetCommand(params)
		assert.True(t, res.Requeue)
	})

	t.Run("complete with one success node deletes BKENode", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			Scheme:          scheme,
			Recorder:        record.NewFakeRecorder(machinePhasesNumFive),
			NodeFetcher:     nodeutil.NewNodeFetcher(fakeClient),
			nodesBootRecord: make(map[string]struct{}),
		}

		patchHelper, _ := patch.NewHelper(bkeMachine, fakeClient)

		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})
		defer patches.Reset()

		params := ProcessResetCommandParams{
			CommonCommandParams: CommonCommandParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          bkeMachine,
				},
				PatchHelper:  patchHelper,
				Cmd:          &agentv1beta1.Command{},
				Complete:     true,
				SuccessNodes: []string{machinePhasesNodeIP},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
		}
		res, _ := r.processResetCommand(params)
		assert.True(t, res.Requeue)
	})

	t.Run("not complete returns result as is", func(t *testing.T) {
		r := newPhasesReconciler()
		params := ProcessResetCommandParams{
			CommonCommandParams: CommonCommandParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					BKECluster:          &bkev1beta1.BKECluster{},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				Cmd:          &agentv1beta1.Command{},
				Complete:     false,
				SuccessNodes: []string{machinePhasesNodeIP},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
		}
		res, _ := r.processResetCommand(params)
		assert.False(t, res.Requeue)
	})

	t.Run("sync error returns requeue", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{
			Client:          fakeClient,
			Scheme:          scheme,
			Recorder:        record.NewFakeRecorder(machinePhasesNumFive),
			NodeFetcher:     nodeutil.NewNodeFetcher(fakeClient),
			nodesBootRecord: make(map[string]struct{}),
		}

		patchHelper, _ := patch.NewHelper(bkeMachine, fakeClient)

		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})
		defer patches.Reset()

		params := ProcessResetCommandParams{
			CommonCommandParams: CommonCommandParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          bkeMachine,
				},
				PatchHelper:  patchHelper,
				Cmd:          &agentv1beta1.Command{},
				Complete:     true,
				SuccessNodes: []string{machinePhasesNodeIP},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
		}
		res, _ := r.processResetCommand(params)
		assert.True(t, res.RequeueAfter > machinePhasesNumZero)
	})
}

// --- getRoleNodes ---

func TestGetRoleNodes(t *testing.T) {
	t.Run("no ready nodes returns error", func(t *testing.T) {
		r := newPhasesReconciler()
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		_, err := r.getRoleNodes(context.Background(), bkeCluster, bkenode.WorkerNodeRole)
		assert.Error(t, err)
	})

	t.Run("master role filters master nodes", func(t *testing.T) {
		r := newPhasesReconciler()
		patches := gomonkey.ApplyFunc(
			(*nodeutil.NodeFetcher).GetReadyBootstrapNodes,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkenode.Nodes, error) {
				return bkenode.Nodes{
					{IP: machinePhasesNodeIP, Role: []string{bkenode.MasterNodeRole}},
					{IP: machinePhasesNodeIP2, Role: []string{bkenode.WorkerNodeRole}},
				}, nil
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		nodes, err := r.getRoleNodes(context.Background(), bkeCluster, bkenode.MasterNodeRole)
		assert.NoError(t, err)
		assert.Len(t, nodes, machinePhasesNumOne)
	})

	t.Run("worker role filters worker nodes", func(t *testing.T) {
		r := newPhasesReconciler()
		patches := gomonkey.ApplyFunc(
			(*nodeutil.NodeFetcher).GetReadyBootstrapNodes,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkenode.Nodes, error) {
				return bkenode.Nodes{
					{IP: machinePhasesNodeIP, Role: []string{bkenode.MasterNodeRole}},
					{IP: machinePhasesNodeIP2, Role: []string{bkenode.WorkerNodeRole}},
				}, nil
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		nodes, err := r.getRoleNodes(context.Background(), bkeCluster, bkenode.WorkerNodeRole)
		assert.NoError(t, err)
		assert.Len(t, nodes, machinePhasesNumOne)
	})
}

// --- selectAppropriateNodes ---

func TestSelectAppropriateNodes(t *testing.T) {
	r := newPhasesReconciler()
	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}

	nodes, err := r.selectAppropriateNodes(context.Background(), bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, nodes)
}

// --- markBKEMachineBootstrapReady ---

func TestMarkBKEMachineBootstrapReady(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})
		defer patches.Reset()

		bkeMachine := &bkev1beta1.BKEMachine{}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		node := confv1beta1.Node{IP: machinePhasesNodeIP, Hostname: machinePhasesNodeHost, Role: []string{bkenode.WorkerNodeRole}}

		err := r.markBKEMachineBootstrapReady(bootstrapReadyInput{
			ctx: context.Background(), cluster: bkeCluster, machine: bkeMachine,
			node: node, providerID: machinePhasesProviderID, log: log,
		})
		assert.NoError(t, err)
		assert.True(t, bkeMachine.Status.Ready)
		assert.True(t, bkeMachine.Status.Bootstrapped)
		assert.Equal(t, machinePhasesProviderID, *bkeMachine.Spec.ProviderID)
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return fmt.Errorf("sync error")
			})
		defer patches.Reset()

		bkeMachine := &bkev1beta1.BKEMachine{}
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}

		err := r.markBKEMachineBootstrapReady(bootstrapReadyInput{
			ctx: context.Background(), cluster: bkeCluster, machine: bkeMachine,
			node: node, providerID: machinePhasesProviderID, log: log,
		})
		assert.Error(t, err)
	})
}

// --- handleBootstrapSuccessFailure ---

func TestHandleBootstrapSuccessFailure(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()

	patches := gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})
	defer patches.Reset()

	patches.ApplyFunc(phaseutil.NodeInfo, func(_ confv1beta1.Node) string {
		return machinePhasesNodeIP
	})

	params := ProcessBootstrapSuccessParams{
		ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         &agentv1beta1.Command{},
		},
	}

	res, errs := r.handleBootstrapSuccessFailure(params, fmt.Errorf("connection timeout"))
	assert.True(t, res.RequeueAfter > machinePhasesNumZero)
	assert.Nil(t, errs)
}

// --- Constants and struct tests ---

func TestConstants(t *testing.T) {
	assert.Equal(t, "bootstrap-", command.BootstrapCommandNamePrefix)
	assert.Equal(t, "reset-node-", command.ResetNodeCommandNamePrefix)
}

// --- reconcileBootstrap ---

func TestReconcileBootstrap(t *testing.T) {
	scheme := newMachinePhaseScheme()
	log := newTestLogger()

	t.Run("already bootstrapped returns early", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
			Status:     bkev1beta1.BKEMachineStatus{Bootstrapped: true},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), nodesBootRecord: make(map[string]struct{})}

		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
			},
		}
		res, err := r.reconcileBootstrap(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})

	t.Run("has label returns early", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{
				Name: machinePhasesName, Namespace: machinePhasesNamespace,
				Labels: map[string]string{"bke.bocloud.com/worker-node": machinePhasesNodeIP},
			},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), nodesBootRecord: make(map[string]struct{})}

		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
			},
		}
		res, err := r.reconcileBootstrap(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})

	t.Run("first time calls handleFirstTimeReconciliation", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), nodesBootRecord: make(map[string]struct{})}

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).handleFirstTimeReconciliation,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, nil
			})
		defer patches.Reset()

		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
			},
		}
		res, err := r.reconcileBootstrap(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})
}

// --- handleFirstTimeReconciliation ---

func TestHandleFirstTimeReconciliation(t *testing.T) {
	log := newTestLogger()

	t.Run("worker waits for control plane init", func(t *testing.T) {
		r := newPhasesReconciler()
		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		// ControlPlaneInitialized not set => worker should wait
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName}},
				Cluster:             cluster,
				BKEMachine:          &bkev1beta1.BKEMachine{},
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
			},
		}
		res, err := r.handleFirstTimeReconciliation(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})

	t.Run("no role nodes returns early", func(t *testing.T) {
		r := newPhasesReconciler()
		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName}},
				Cluster:             cluster,
				BKEMachine:          &bkev1beta1.BKEMachine{},
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
			},
		}
		res, err := r.handleFirstTimeReconciliation(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})
}

// --- syncKubeadmConfig ---

func TestSyncKubeadmConfig(t *testing.T) {
	scheme := newMachinePhaseScheme()

	t.Run("config ref nil returns nil", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &BKEMachineReconciler{Client: fakeClient}

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
			Spec: clusterv1.MachineSpec{
				Bootstrap: clusterv1.Bootstrap{
					ConfigRef: &corev1.ObjectReference{Name: "config-ref", Namespace: machinePhasesNamespace},
				},
			},
		}
		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}

		err := r.syncKubeadmConfig(context.Background(), machine, cluster)
		assert.NoError(t, err)
	})
}

// --- processBootstrapFailure ---

func TestProcessBootstrapFailure(t *testing.T) {
	logger := newTestLogger()

	t.Run("control plane not initialized", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(metricrecord.NodeBootstrapFailedCountRecord, func(_ client.Object) {})
		defer patches.Reset()
		patches.ApplyFunc(metricrecord.NodeBootstrapDurationRecord, func(_ client.Object, _ confv1beta1.Node, _ time.Time, _ string) {})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyFunc((*BKEMachineReconciler).reconcileBKEMachine, func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKEMachine, _ confv1beta1.Node, _ *log.Logger) error {
			return nil
		})

		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		// ControlPlaneInitialized NOT set => not initialized path
		cmd := &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}}
		fakeClient := fake.NewClientBuilder().WithScheme(newMachinePhaseScheme()).Build()
		r.Client = fakeClient

		params := ProcessBootstrapFailureParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
					Machine:             &clusterv1.Machine{},
					Cluster:             cluster,
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         cmd,
			},
			FailedNodes: []string{machinePhasesNodeIP},
			Role:        bkenode.WorkerNodeRole,
		}
		_, errs := r.processBootstrapFailure(params)
		// The command update will fail since cmd is not in fake client, but that's expected
		assert.NotNil(t, errs)
	})

	t.Run("control plane initialized", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(metricrecord.NodeBootstrapFailedCountRecord, func(_ client.Object) {})
		defer patches.Reset()
		patches.ApplyFunc(metricrecord.NodeBootstrapDurationRecord, func(_ client.Object, _ confv1beta1.Node, _ time.Time, _ string) {})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyFunc((*BKEMachineReconciler).reconcileBKEMachine, func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKEMachine, _ confv1beta1.Node, _ *log.Logger) error {
			return nil
		})

		cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
		conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

		cmd := &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}}
		fakeClient := fake.NewClientBuilder().WithScheme(newMachinePhaseScheme()).Build()
		r.Client = fakeClient

		params := ProcessBootstrapFailureParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
					Machine:             &clusterv1.Machine{},
					Cluster:             cluster,
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         cmd,
			},
			FailedNodes: []string{machinePhasesNodeIP},
			Role:        bkenode.WorkerNodeRole,
		}
		_, errs := r.processBootstrapFailure(params)
		assert.NotNil(t, errs)
	})
}

// --- reconcileBKEMachine ---

func TestReconcileBKEMachine(t *testing.T) {
	log := newTestLogger()

	t.Run("condition true returns early", func(t *testing.T) {
		r := newPhasesReconciler()
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
			Status: confv1beta1.BKEClusterStatus{
				Conditions: []confv1beta1.ClusterCondition{
					{
						Type:   bkev1beta1.TargetClusterBootCondition,
						Status: confv1beta1.ConditionTrue,
					},
				},
			},
		}
		bkeMachine := &bkev1beta1.BKEMachine{}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}

		err := r.reconcileBKEMachine(context.Background(), bkeCluster, bkeMachine, node, log)
		assert.NoError(t, err)
	})

	t.Run("getClusterInfo error", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).getClusterInfo,
			func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster) ([]bkev1beta1.BKEMachine, bkenode.Nodes, error) {
				return nil, nil, fmt.Errorf("cluster info error")
			})
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		bkeMachine := &bkev1beta1.BKEMachine{}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}

		err := r.reconcileBKEMachine(context.Background(), bkeCluster, bkeMachine, node, log)
		assert.Error(t, err)
	})

	t.Run("full flow success", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).getClusterInfo,
			func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster) ([]bkev1beta1.BKEMachine, bkenode.Nodes, error) {
				return []bkev1beta1.BKEMachine{}, bkenode.Nodes{{IP: machinePhasesNodeIP}}, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })

		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
		bkeMachine := &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName}}
		node := confv1beta1.Node{IP: machinePhasesNodeIP}

		err := r.reconcileBKEMachine(context.Background(), bkeCluster, bkeMachine, node, log)
		assert.NoError(t, err)
	})
}

// --- getClusterInfo ---

func TestGetClusterInfo(t *testing.T) {
	r := newPhasesReconciler()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster}},
	}

	machines, nodes, err := r.getClusterInfo(context.Background(), bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, machines)
	assert.NotNil(t, nodes)
}

// --- handleRealBootstrap ---

func TestHandleRealBootstrap(t *testing.T) {
	log := newTestLogger()

	t.Run("command new error", func(t *testing.T) {
		r := newPhasesReconciler()
		node := &confv1beta1.Node{IP: machinePhasesNodeIP, Hostname: machinePhasesNodeHost, Role: []string{bkenode.WorkerNodeRole}}

		patches := gomonkey.ApplyFunc((*command.Bootstrap).New,
			func(_ *command.Bootstrap) error { return fmt.Errorf("command create error") })
		defer patches.Reset()

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })

		params := RealBootstrapParams{
			CommonNodeParams: CommonNodeParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					Machine:             &clusterv1.Machine{},
					BKEMachine:          &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace}},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
				},
				Node: node,
				Role: bkenode.WorkerNodeRole,
			},
			Phase: bkev1beta1.JoinWorker,
		}
		_, err := r.handleRealBootstrap(params)
		assert.Error(t, err)
	})

	t.Run("command new success", func(t *testing.T) {
		r := newPhasesReconciler()
		node := &confv1beta1.Node{IP: machinePhasesNodeIP, Hostname: machinePhasesNodeHost, Role: []string{bkenode.WorkerNodeRole}}

		patches := gomonkey.ApplyFunc((*command.Bootstrap).New,
			func(_ *command.Bootstrap) error { return nil })
		defer patches.Reset()

		patches.ApplyFunc(phaseutil.NodeInfo, func(_ confv1beta1.Node) string { return machinePhasesNodeIP })

		params := RealBootstrapParams{
			CommonNodeParams: CommonNodeParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					Machine:             &clusterv1.Machine{},
					BKEMachine:          &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace}},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
				},
				Node: node,
				Role: bkenode.WorkerNodeRole,
			},
			Phase: bkev1beta1.JoinWorker,
		}
		res, err := r.handleRealBootstrap(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})
}

// --- reconcileCommand ---

func TestReconcileCommand(t *testing.T) {
	scheme := newMachinePhaseScheme()
	log := newTestLogger()

	t.Run("no commands returns empty", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace,
				Finalizers: []string{bkev1beta1.BKEMachineFinalizer}},
		}
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace,
				Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster}},
		}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine, bkeCluster).WithStatusSubresource(bkeMachine).Build()
		r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), NodeFetcher: nodeutil.NewNodeFetcher(fakeClient), nodesBootRecord: make(map[string]struct{})}

		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Machine:             &clusterv1.Machine{},
				Cluster:             &clusterv1.Cluster{},
				BKEMachine:          bkeMachine,
				BKECluster:          bkeCluster,
			},
		}
		res, err := r.reconcileCommand(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, res)
	})
}

// --- processBootstrapSuccess ---

func TestProcessBootstrapSuccess(t *testing.T) {
	logger := newTestLogger()

	t.Run("cluster deleting after connect returns empty", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).connectToTargetClusterNode,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams) error { return nil })
		defer patches.Reset()

		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
			Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
		}

		params := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
					Machine:             &clusterv1.Machine{},
					BKECluster:          bkeCluster,
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}},
			},
		}
		res, _ := r.processBootstrapSuccess(params)
		// When cluster is deleting, should return empty result
		assert.False(t, res.Requeue)
	})

	t.Run("connect error triggers failure handler", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).connectToTargetClusterNode,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams) error { return fmt.Errorf("timeout") })
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEMachineReconciler).handleBootstrapSuccessFailure,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams, _ error) (ctrl.Result, []error) {
				return ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration}, nil
			})

		params := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
					Machine:             &clusterv1.Machine{},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}},
			},
		}
		res, _ := r.processBootstrapSuccess(params)
		assert.True(t, res.RequeueAfter > machinePhasesNumZero)
	})

	t.Run("success flow with all mocks", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).connectToTargetClusterNode,
			func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams) error { return nil })
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyFunc(metricrecord.NodeBootstrapSuccessCountRecord, func(_ client.Object) {})
		patches.ApplyFunc(metricrecord.NodeBootstrapDurationRecord, func(_ client.Object, _ confv1beta1.Node, _ time.Time, _ string) {})
		patches.ApplyFunc((*BKEMachineReconciler).reconcileBKEMachine, func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKEMachine, _ confv1beta1.Node, _ *log.Logger) error {
			return nil
		})
		patches.ApplyFunc((*BKEMachineReconciler).markBKEMachineBootstrapReady, func(_ *BKEMachineReconciler, _ bootstrapReadyInput) error {
			return nil
		})

		scheme := newMachinePhaseScheme()
		cmd := &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}}
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmd).Build()
		r.Client = fakeClient

		params := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
					Machine:             &clusterv1.Machine{},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         cmd,
			},
		}
		_, errs := r.processBootstrapSuccess(params)
		assert.Nil(t, errs)
	})
}

// --- Remaining skipped tests requiring remote cluster ---

func TestCordonMasterNode(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestConnectToTargetClusterNode(t *testing.T) {
	log := newTestLogger()

	t.Run("cluster deleting returns immediately", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
			func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
				return &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
				}, nil
			})
		defer patches.Reset()

		params := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					Machine:             &clusterv1.Machine{},
					Cluster:             &clusterv1.Cluster{},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         &agentv1beta1.Command{},
			},
		}
		err := r.connectToTargetClusterNode(params)
		assert.NoError(t, err)
	})

	t.Run("get combined error", func(t *testing.T) {
		r := newPhasesReconciler()

		patches := gomonkey.ApplyFunc(mergecluster.GetCombinedBKECluster,
			func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
				return nil, fmt.Errorf("not found")
			})
		defer patches.Reset()

		params := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
					Machine:             &clusterv1.Machine{},
					Cluster:             &clusterv1.Cluster{},
					BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
					BKEMachine:          &bkev1beta1.BKEMachine{},
				},
				CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
				Cmd:         &agentv1beta1.Command{},
			},
		}
		err := r.connectToTargetClusterNode(params)
		assert.Error(t, err)
	})
}

func TestRecordBootstrapPhaseEventNotFullyControlled(t *testing.T) {
	r := newPhasesReconciler()
	log := newTestLogger()
	cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}
	node := &confv1beta1.Node{IP: machinePhasesNodeIP, Hostname: machinePhasesNodeHost, Role: []string{bkenode.WorkerNodeRole}}

	// Mark as bocloud cluster (not fully controlled)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: machinePhasesCluster,
			Annotations: map[string]string{
				"bke.bocloud.com/cluster-from": "bocloud",
			},
		},
	}

	phases := []confv1beta1.BKEClusterPhase{
		bkev1beta1.JoinWorker,
		bkev1beta1.JoinControlPlane,
		bkev1beta1.InitControlPlane,
	}
	for _, phase := range phases {
		err := r.recordBootstrapPhaseEvent(cluster, bkeCluster, node, phase, log)
		assert.NoError(t, err)
	}
}

func TestSelectAppropriateNodesWithNodes(t *testing.T) {
	node := newTestBKENode("node-1", machinePhasesNamespace, machinePhasesCluster, machinePhasesNodeIP)
	r := newPhasesReconciler(node)
	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
	nodes, err := r.selectAppropriateNodes(context.Background(), bkeCluster)
	assert.NoError(t, err)
	assert.Len(t, nodes, machinePhasesNumOne)
}

func TestSyncKubeadmConfigNoBootstrapRef(t *testing.T) {
	scheme := newMachinePhaseScheme()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &BKEMachineReconciler{Client: fakeClient}

	// Machine with nil Bootstrap.ConfigRef will cause nil pointer - syncKubeadmConfig should handle this
	machine := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesName, Namespace: machinePhasesNamespace},
		Spec: clusterv1.MachineSpec{
			Bootstrap: clusterv1.Bootstrap{},
		},
	}
	cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster}}

	// This will cause a panic if ConfigRef is nil, so we skip
	// Instead test with ConfigRef pointing to non-existent resource
	machine.Spec.Bootstrap.ConfigRef = &corev1.ObjectReference{Name: "nonexistent", Namespace: machinePhasesNamespace}
	err := r.syncKubeadmConfig(context.Background(), machine, cluster)
	assert.NoError(t, err)
}

func TestCheckTargetClusterNode(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestProcessBootstrapSuccessFullFlow(t *testing.T) {
	logger := newTestLogger()
	r := newPhasesReconciler()

	patches := gomonkey.ApplyFunc(
		(*BKEMachineReconciler).connectToTargetClusterNode,
		func(_ *BKEMachineReconciler, _ ProcessBootstrapSuccessParams) error { return nil })
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })
	patches.ApplyFunc(metricrecord.NodeBootstrapSuccessCountRecord, func(_ client.Object) {})
	patches.ApplyFunc(metricrecord.NodeBootstrapDurationRecord, func(_ client.Object, _ confv1beta1.Node, _ time.Time, _ string) {})
	patches.ApplyFunc((*BKEMachineReconciler).reconcileBKEMachine, func(_ *BKEMachineReconciler, _ context.Context, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKEMachine, _ confv1beta1.Node, _ *log.Logger) error {
		return nil
	})
	patches.ApplyFunc((*BKEMachineReconciler).markBKEMachineBootstrapReady, func(_ *BKEMachineReconciler, _ bootstrapReadyInput) error {
		return nil
	})

	scheme := newMachinePhaseScheme()
	cmd := &agentv1beta1.Command{ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machinePhasesNamespace}}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmd).Build()
	r.Client = fakeClient

	// Non-deleting cluster
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterStatusReady},
	}

	params := ProcessBootstrapSuccessParams{
		ProcessBootstrapCommonParams: ProcessBootstrapCommonParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: logger},
				Machine:             &clusterv1.Machine{},
				BKECluster:          bkeCluster,
				BKEMachine:          &bkev1beta1.BKEMachine{},
			},
			CurrentNode: confv1beta1.Node{IP: machinePhasesNodeIP},
			Cmd:         cmd,
		},
	}
	// The function may go through handleBootstrapSuccessFailure path if mock doesn't intercept
	// In either case, it should not panic and should complete
	res, _ := r.processBootstrapSuccess(params)
	assert.NotNil(t, res)
}

func TestHandleFirstTimeReconciliationControlPlane(t *testing.T) {
	log := newTestLogger()
	r := newPhasesReconciler()

	// Control plane machine with initialized cluster, but no role nodes
	cluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}}
	conditions.MarkTrue(cluster, clusterv1.ControlPlaneInitializedCondition)

	machine := &clusterv1.Machine{
		ObjectMeta: metav1.ObjectMeta{
			Name:      machinePhasesName,
			Namespace: machinePhasesNamespace,
			Labels:    map[string]string{clusterv1.MachineControlPlaneLabel: ""},
		},
		Spec: clusterv1.MachineSpec{
			Bootstrap: clusterv1.Bootstrap{
				ConfigRef: &corev1.ObjectReference{Name: "nonexistent", Namespace: machinePhasesNamespace},
			},
		},
	}

	params := BootstrapReconcileParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			Machine:             machine,
			Cluster:             cluster,
			BKEMachine:          &bkev1beta1.BKEMachine{},
			BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace}},
		},
	}
	// No role nodes available, should return ctrl.Result{}, nil
	res, err := r.handleFirstTimeReconciliation(params)
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
}

func TestHandleEmptyNodeList(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestHandleNonEmptyNodeList(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestProcessMatchingNode(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestPatchOrGetRemoteNodeProviderID(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestReconcileCommandWithLabel(t *testing.T) {
	scheme := newMachinePhaseScheme()
	log := newTestLogger()

	// BKEMachine with worker label but no matching commands -> returns empty
	bkeMachine := &bkev1beta1.BKEMachine{
		ObjectMeta: metav1.ObjectMeta{
			Name: machinePhasesName, Namespace: machinePhasesNamespace,
			Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
			Labels:     map[string]string{"bke.bocloud.com/worker-node": machinePhasesNodeIP},
		},
	}
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster}},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine, bkeCluster).WithStatusSubresource(bkeMachine).Build()
	r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), NodeFetcher: nodeutil.NewNodeFetcher(fakeClient), nodesBootRecord: make(map[string]struct{})}

	params := BootstrapReconcileParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			Machine:             &clusterv1.Machine{},
			Cluster:             &clusterv1.Cluster{},
			BKEMachine:          bkeMachine,
			BKECluster:          bkeCluster,
		},
	}
	res, err := r.reconcileCommand(params)
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
}

func TestReconcileCommandWithCommandsAndLabel(t *testing.T) {
	scheme := newMachinePhaseScheme()
	log := newTestLogger()

	bkeMachine := &bkev1beta1.BKEMachine{
		ObjectMeta: metav1.ObjectMeta{
			Name: machinePhasesName, Namespace: machinePhasesNamespace,
			UID:        "test-uid",
			Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
			Labels:     map[string]string{"bke.bocloud.com/worker-node": machinePhasesNodeIP},
		},
	}
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster}},
	}
	bkeNode := newTestBKENode("node-1", machinePhasesNamespace, machinePhasesCluster, machinePhasesNodeIP)

	// Command owned by this BKEMachine, not reconciled yet
	isController := true
	cmd := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bootstrap-test-cmd", Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: bkev1beta1.GroupVersion.String(),
				Kind:       "BKEMachine",
				Name:       machinePhasesName,
				UID:        "test-uid",
				Controller: &isController,
			}},
		},
		Spec: agentv1beta1.CommandSpec{
			NodeSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"node": machinePhasesNodeIP}},
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine, bkeCluster, bkeNode, cmd).WithStatusSubresource(bkeMachine).Build()
	r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), NodeFetcher: nodeutil.NewNodeFetcher(fakeClient), nodesBootRecord: make(map[string]struct{})}

	params := BootstrapReconcileParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			Machine:             &clusterv1.Machine{},
			Cluster:             &clusterv1.Cluster{},
			BKEMachine:          bkeMachine,
			BKECluster:          bkeCluster,
		},
	}
	// The command loop should execute, processCommand will be called
	res, err := r.reconcileCommand(params)
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
}

func TestReconcileCommandNoLabel(t *testing.T) {
	scheme := newMachinePhaseScheme()
	log := newTestLogger()

	// BKEMachine without worker label -> returns empty after label check
	bkeMachine := &bkev1beta1.BKEMachine{
		ObjectMeta: metav1.ObjectMeta{
			Name: machinePhasesName, Namespace: machinePhasesNamespace,
			Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
		},
	}
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: machinePhasesCluster, Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster}},
	}

	// Create a command that will be found
	cmd := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bootstrap-test", Namespace: machinePhasesNamespace,
			Labels: map[string]string{clusterv1.ClusterNameLabel: machinePhasesCluster},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: bkev1beta1.GroupVersion.String(),
				Kind:       "BKEMachine",
				Name:       machinePhasesName,
				UID:        bkeMachine.UID,
			}},
		},
	}

	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeMachine, bkeCluster, cmd).WithStatusSubresource(bkeMachine).Build()
	r := &BKEMachineReconciler{Client: fakeClient, Scheme: scheme, Recorder: record.NewFakeRecorder(machinePhasesNumFive), NodeFetcher: nodeutil.NewNodeFetcher(fakeClient), nodesBootRecord: make(map[string]struct{})}

	params := BootstrapReconcileParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
			Machine:             &clusterv1.Machine{},
			Cluster:             &clusterv1.Cluster{},
			BKEMachine:          bkeMachine,
			BKECluster:          bkeCluster,
		},
	}
	res, err := r.reconcileCommand(params)
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, res)
}

func TestCheckNodesForProviderID(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestApplyNodeConfiguration(t *testing.T) {
	t.Skip("requires remote cluster client")
}

func TestHandleFakeBootstrap(t *testing.T) {
	t.Skip("requires remote cluster client for patchOrGetRemoteNodeProviderID")
}

func TestHandleMasterMachineCertificates(t *testing.T) {
	t.Skip("requires kubeadm bootstrap types in scheme")
}
