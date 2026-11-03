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
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	machineNumZero  = 0
	machineNumOne   = 1
	machineNumTwo   = 2
	machineNumThree = 3
	machineNumFive  = 5
	machineNumTen   = 10

	machineTestNamespace = "test-ns"
	machineTestName      = "test-machine"
	machineTestCluster   = "test-cluster"
	machineTestNodeIP    = "192.168.1.100"
	machineTestNodeIP2   = "192.168.1.101"

	// testNodeIP is kept for backward compatibility with bkemachine_controller_phases_test.go
	testNodeIP = machineTestNodeIP
)

// newMachineScheme creates a scheme with all needed types
func newMachineScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	_ = clusterv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

// newMachineReconciler creates a BKEMachineReconciler for testing
func newMachineReconciler(objs ...client.Object) *BKEMachineReconciler {
	scheme := newMachineScheme()
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > machineNumZero {
		builder = builder.WithObjects(objs...).WithStatusSubresource(objs...)
	}
	fakeClient := builder.Build()
	return &BKEMachineReconciler{
		Client:          fakeClient,
		Scheme:          scheme,
		Recorder:        record.NewFakeRecorder(machineNumTen),
		NodeFetcher:     nodeutil.NewNodeFetcher(fakeClient),
		nodesBootRecord: make(map[string]struct{}),
	}
}

// newMachineLogger creates a logger for testing
func newMachineLogger() *log.Logger {
	return nil
}

// --- isMarkDeletion ---

func TestIsMarkDeletion(t *testing.T) {
	tests := []struct {
		name         string
		bkeMachine   *bkev1beta1.BKEMachine
		expectResult bool
	}{
		{
			name: "machine with delete annotation",
			bkeMachine: &bkev1beta1.BKEMachine{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{clusterv1.DeleteMachineAnnotation: ""},
				},
			},
			expectResult: true,
		},
		{
			name: "machine without delete annotation",
			bkeMachine: &bkev1beta1.BKEMachine{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
			},
			expectResult: false,
		},
		{
			name:         "machine with nil annotations",
			bkeMachine:   &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{}},
			expectResult: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectResult, isMarkDeletion(tt.bkeMachine))
		})
	}
}

// --- bkeMachineToNode ---

func TestBkeMachineToNode(t *testing.T) {
	tests := []struct {
		name         string
		bkeMachine   *bkev1beta1.BKEMachine
		bkeNodes     bkenode.Nodes
		expectError  bool
		expectNodeIP string
	}{
		{
			name: "machine with status node",
			bkeMachine: &bkev1beta1.BKEMachine{
				Status: bkev1beta1.BKEMachineStatus{Node: &confv1beta1.Node{IP: machineTestNodeIP}},
			},
			bkeNodes:     bkenode.Nodes{},
			expectNodeIP: machineTestNodeIP,
		},
		{
			name: "machine with label and node found",
			bkeMachine: &bkev1beta1.BKEMachine{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{label.WorkerNodeHost: machineTestNodeIP}},
			},
			bkeNodes:     bkenode.Nodes{{IP: machineTestNodeIP}},
			expectNodeIP: machineTestNodeIP,
		},
		{
			name:        "machine without label returns error",
			bkeMachine:  &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Name: machineTestName}},
			bkeNodes:    bkenode.Nodes{},
			expectError: true,
		},
		{
			name: "machine with label but node not found in list",
			bkeMachine: &bkev1beta1.BKEMachine{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{label.WorkerNodeHost: "10.0.0.1"}},
			},
			bkeNodes:     bkenode.Nodes{},
			expectNodeIP: "10.0.0.1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := bkeMachineToNode(tt.bkeMachine, tt.bkeNodes)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.expectNodeIP != "" {
					assert.Equal(t, tt.expectNodeIP, node.IP)
				}
			}
		})
	}
}

// --- handlePauseAndFinalizer ---

func TestHandlePauseAndFinalizer(t *testing.T) {
	t.Run("not paused", func(t *testing.T) {
		r := newMachineReconciler()
		objects := &RequiredObjects{
			Cluster:    &clusterv1.Cluster{},
			BKEMachine: &bkev1beta1.BKEMachine{},
		}
		result, stopped := r.handlePauseAndFinalizer(objects, nil)
		assert.False(t, stopped)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("paused via spec", func(t *testing.T) {
		r := newMachineReconciler()
		log := newMachineLogger()
		paused := true
		objects := &RequiredObjects{
			Cluster: &clusterv1.Cluster{
				Spec: clusterv1.ClusterSpec{Paused: paused},
			},
			BKEMachine: &bkev1beta1.BKEMachine{},
		}
		result, stopped := r.handlePauseAndFinalizer(objects, log)
		assert.True(t, stopped)
		assert.Equal(t, ctrl.Result{}, result)
	})
}

// --- fetchRequiredObjects ---

func TestFetchRequiredObjects(t *testing.T) {
	t.Run("BKEMachine not found returns nil", func(t *testing.T) {
		r := newMachineReconciler()
		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: "non-existent", Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, nil)
		assert.NoError(t, err)
		assert.Nil(t, objects)
	})

	t.Run("machine without owner returns nil", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		log := newMachineLogger()

		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return nil, nil
			})
		defer patches.Reset()

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, log)
		assert.NoError(t, err)
		assert.Nil(t, objects)
	})

	t.Run("GetOwnerMachine returns nil machine and error returns nil nil", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		log := newMachineLogger()

		// When machine==nil, code returns (nil,nil) regardless of error
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return nil, fmt.Errorf("owner error")
			})
		defer patches.Reset()

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, log)
		assert.NoError(t, err)
		assert.Nil(t, objects)
	})

	t.Run("GetOwnerMachine returns non-nil machine with error", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		log := newMachineLogger()

		machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: machineTestName}}
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return machine, fmt.Errorf("owner error")
			})
		defer patches.Reset()

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, log)
		assert.Error(t, err)
		assert.Nil(t, objects)
	})

	t.Run("cluster not found returns nil", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		log := newMachineLogger()

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
			Spec:       clusterv1.MachineSpec{ClusterName: machineTestCluster},
		}
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return machine, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(util.GetClusterFromMetadata,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return nil, nil
			})

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, log)
		assert.NoError(t, err)
		assert.Nil(t, objects)
	})

	t.Run("GetClusterFromMetadata returns error", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		log := newMachineLogger()

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
			Spec:       clusterv1.MachineSpec{ClusterName: machineTestCluster},
		}
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return machine, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(util.GetClusterFromMetadata,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return nil, fmt.Errorf("cluster error")
			})

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, log)
		assert.Error(t, err)
		assert.Nil(t, objects)
	})

	t.Run("all objects found returns RequiredObjects", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)

		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		cluster := &clusterv1.Cluster{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster},
		}
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return machine, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(util.GetClusterFromMetadata,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return cluster, nil
			})

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		objects, err := r.fetchRequiredObjects(context.Background(), req, nil)
		assert.NoError(t, err)
		assert.NotNil(t, objects)
		assert.Equal(t, machineTestName, objects.BKEMachine.Name)
	})
}

// --- log methods ---

func TestLogMethods(t *testing.T) {
	log := newMachineLogger()
	bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}

	t.Run("logInfoAndEvent with logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logInfoAndEvent(log, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logInfoAndEvent without logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logInfoAndEvent(nil, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logErrorAndEvent with logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logErrorAndEvent(log, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logErrorAndEvent without logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logErrorAndEvent(nil, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logWarningAndEvent with logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logWarningAndEvent(log, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logWarningAndEvent without logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logWarningAndEvent(nil, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logFinishAndEvent with logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logFinishAndEvent(log, bkeCluster, "reason", "msg %s", "arg")
	})

	t.Run("logFinishAndEvent without logger", func(t *testing.T) {
		r := newMachineReconciler()
		r.logFinishAndEvent(nil, bkeCluster, "reason", "msg %s", "arg")
	})
}

// --- reconcile ---

func TestMachineReconcile(t *testing.T) {
	log := newMachineLogger()

	t.Run("cluster deleting returns early", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
				},
			},
		}
		result, err := r.reconcile(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("reconcileCommand error", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterManaging},
				},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).reconcileCommand,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, fmt.Errorf("command error")
			})
		defer patches.Reset()

		_, err := r.reconcile(params)
		assert.Error(t, err)
	})

	t.Run("reconcileBootstrap error", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterManaging},
				},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).reconcileCommand,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEMachineReconciler).reconcileBootstrap,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, fmt.Errorf("bootstrap error")
			})

		_, err := r.reconcile(params)
		assert.Error(t, err)
	})

	t.Run("reconcile success", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterManaging},
				},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).reconcileCommand,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEMachineReconciler).reconcileBootstrap,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{}, nil
			})

		result, err := r.reconcile(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})
}

// --- handleMainReconcile ---

func TestHandleMainReconcile(t *testing.T) {
	log := newMachineLogger()

	t.Run("infrastructure not ready", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Cluster:             &clusterv1.Cluster{Status: clusterv1.ClusterStatus{InfrastructureReady: false}},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{},
			},
		}
		result, err := r.handleMainReconcile(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("infrastructure ready with failed node flag", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Cluster:             &clusterv1.Cluster{Status: clusterv1.ClusterStatus{InfrastructureReady: true}},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(label.CheckBKEMachineLabel,
			func(_ interface{}) (string, bool) { return machineTestNodeIP, true })
		defer patches.Reset()

		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).GetNodeStateFlagForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) (bool, error) {
				return true, nil
			})

		result, err := r.handleMainReconcile(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("infrastructure ready calls reconcile", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				Cluster:             &clusterv1.Cluster{Status: clusterv1.ClusterStatus{InfrastructureReady: true}},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(label.CheckBKEMachineLabel,
			func(_ interface{}) (string, bool) { return "", false })
		defer patches.Reset()

		patches.ApplyFunc(
			(*BKEMachineReconciler).reconcile,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{Requeue: true}, nil
			})

		result, err := r.handleMainReconcile(params)
		assert.NoError(t, err)
		assert.True(t, result.Requeue)
	})
}

// --- handlePreDeletionCleanup ---

func TestHandlePreDeletionCleanup(t *testing.T) {
	log := newMachineLogger()

	t.Run("machine with finalizer does nothing", func(t *testing.T) {
		r := newMachineReconciler()
		r.nodesBootRecord[machineTestNodeIP] = struct{}{}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKEMachine: &bkev1beta1.BKEMachine{
					ObjectMeta: metav1.ObjectMeta{Finalizers: []string{bkev1beta1.BKEMachineFinalizer}},
				},
				BKECluster: &bkev1beta1.BKECluster{},
			},
		}
		r.handlePreDeletionCleanup(params)
		_, ok := r.nodesBootRecord[machineTestNodeIP]
		assert.True(t, ok, "node should NOT be removed when finalizer exists")
	})

	t.Run("no finalizer no label does nothing", func(t *testing.T) {
		r := newMachineReconciler()
		r.nodesBootRecord[machineTestNodeIP] = struct{}{}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKEMachine: &bkev1beta1.BKEMachine{
					ObjectMeta: metav1.ObjectMeta{Finalizers: []string{}},
				},
				BKECluster: &bkev1beta1.BKECluster{},
			},
		}

		patches := gomonkey.ApplyFunc(label.CheckBKEMachineLabel,
			func(_ interface{}) (string, bool) { return "", false })
		defer patches.Reset()

		r.handlePreDeletionCleanup(params)
		_, ok := r.nodesBootRecord[machineTestNodeIP]
		assert.True(t, ok)
	})

	t.Run("no finalizer with label and IP cleans up", func(t *testing.T) {
		r := newMachineReconciler()
		r.nodesBootRecord[machineTestNodeIP] = struct{}{}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine: &bkev1beta1.BKEMachine{
					ObjectMeta: metav1.ObjectMeta{Finalizers: []string{}},
				},
				BKECluster: &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(label.CheckBKEMachineLabel,
			func(_ interface{}) (string, bool) { return machineTestNodeIP, true })
		defer patches.Reset()

		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).DeleteBKENodeForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string) error {
				return nil
			})

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})

		r.handlePreDeletionCleanup(params)
		_, ok := r.nodesBootRecord[machineTestNodeIP]
		assert.False(t, ok, "node should be removed")
	})
}

// --- handleAlreadyMarkedDeletion ---

func TestHandleAlreadyMarkedDeletion(t *testing.T) {
	log := newMachineLogger()

	t.Run("without finalizer returns early", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKEMachine:          &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Finalizers: []string{}}},
				BKECluster:          &bkev1beta1.BKECluster{},
			},
		}
		result, err := r.handleAlreadyMarkedDeletion(params)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("with finalizer calls reconcileCommand", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKEMachine: &bkev1beta1.BKEMachine{
					ObjectMeta: metav1.ObjectMeta{Finalizers: []string{bkev1beta1.BKEMachineFinalizer}},
				},
				BKECluster: &bkev1beta1.BKECluster{},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*BKEMachineReconciler).reconcileCommand,
			func(_ *BKEMachineReconciler, _ BootstrapReconcileParams) (ctrl.Result, error) {
				return ctrl.Result{Requeue: true}, nil
			})
		defer patches.Reset()

		result, err := r.handleAlreadyMarkedDeletion(params)
		assert.NoError(t, err)
		assert.True(t, result.Requeue)
	})
}

// --- shouldSkipDeletion ---

func TestShouldSkipDeletion(t *testing.T) {
	log := newMachineLogger()
	node := &confv1beta1.Node{IP: machineTestNodeIP}

	t.Run("agent switched returns skip", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{
						Conditions: confv1beta1.ClusterConditions{
							{Type: bkev1beta1.SwitchBKEAgentCondition, Status: confv1beta1.ConditionTrue},
						},
					},
				},
			},
		}
		assert.True(t, r.shouldSkipDeletion(params, node))
	})

	t.Run("cluster deleting without annotation skips", func(t *testing.T) {
		r := newMachineReconciler()
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKECluster: &bkev1beta1.BKECluster{
					Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
				},
			},
		}
		assert.True(t, r.shouldSkipDeletion(params, node))
	})

	t.Run("agent not ready returns skip", func(t *testing.T) {
		bkeNode := newTestBKENode("node-1", machineTestNamespace, machineTestCluster, machineTestNodeIP)
		bkeNode.Status.StateCode = machineNumZero // no agent ready flag
		r := newMachineReconciler(bkeNode)
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKECluster:          bkeCluster,
			},
		}
		assert.True(t, r.shouldSkipDeletion(params, node))
	})

	t.Run("agent ready does not skip", func(t *testing.T) {
		bkeNode := newTestBKENode("node-1", machineTestNamespace, machineTestCluster, machineTestNodeIP)
		bkeNode.Status.StateCode = bkev1beta1.NodeAgentReadyFlag
		r := newMachineReconciler(bkeNode)
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKECluster:          bkeCluster,
			},
		}
		assert.False(t, r.shouldSkipDeletion(params, node))
	})

	t.Run("no BKENode found returns false (will try reset)", func(t *testing.T) {
		r := newMachineReconciler() // no BKENode objects
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKECluster:          bkeCluster,
			},
		}
		assert.False(t, r.shouldSkipDeletion(params, node))
	})
}

// --- getNodeForDeletion ---

func TestGetNodeForDeletion(t *testing.T) {
	log := newMachineLogger()

	t.Run("get nodes error removes finalizer", func(t *testing.T) {
		r := newMachineReconciler()
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{
				Name: machineTestName, Namespace: machineTestNamespace,
				Finalizers: []string{bkev1beta1.BKEMachineFinalizer},
			},
		}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*nodeutil.NodeFetcher).GetNodesForBKECluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, fmt.Errorf("get nodes error")
			})
		defer patches.Reset()

		_, err := r.getNodeForDeletion(params)
		assert.Error(t, err)
	})

	t.Run("node found via status", func(t *testing.T) {
		r := newMachineReconciler()
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
			Status:     bkev1beta1.BKEMachineStatus{Node: &confv1beta1.Node{IP: machineTestNodeIP}},
		}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*nodeutil.NodeFetcher).GetNodesForBKECluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return bkenode.Nodes{}, nil
			})
		defer patches.Reset()

		node, err := r.getNodeForDeletion(params)
		assert.NoError(t, err)
		assert.Equal(t, machineTestNodeIP, node.IP)
	})
}

// --- handlePostNodeSetupCleanup ---

func TestHandlePostNodeSetupCleanup(t *testing.T) {
	log := newMachineLogger()
	node := &confv1beta1.Node{IP: machineTestNodeIP}

	t.Run("with finalizer does nothing", func(t *testing.T) {
		r := newMachineReconciler()
		r.nodesBootRecord[machineTestNodeIP] = struct{}{}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Log: log},
				BKEMachine: &bkev1beta1.BKEMachine{
					ObjectMeta: metav1.ObjectMeta{Finalizers: []string{bkev1beta1.BKEMachineFinalizer}},
				},
				BKECluster: &bkev1beta1.BKECluster{},
			},
		}
		r.handlePostNodeSetupCleanup(params, node)
		_, ok := r.nodesBootRecord[machineTestNodeIP]
		assert.True(t, ok)
	})

	t.Run("without finalizer cleans up", func(t *testing.T) {
		r := newMachineReconciler()
		r.nodesBootRecord[machineTestNodeIP] = struct{}{}
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Finalizers: []string{}}},
				BKECluster:          &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}},
			},
		}

		patches := gomonkey.ApplyFunc(
			(*nodeutil.NodeFetcher).DeleteBKENodeForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string) error {
				return nil
			})
		defer patches.Reset()

		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
				return nil
			})

		r.handlePostNodeSetupCleanup(params, node)
		_, ok := r.nodesBootRecord[machineTestNodeIP]
		assert.False(t, ok)
	})
}

// --- setupDeletionProcess ---

func TestSetupDeletionProcess(t *testing.T) {
	log := newMachineLogger()

	t.Run("succeeds", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)
		params := BootstrapReconcileParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{Ctx: context.Background(), Log: log},
				BKEMachine:          bkeMachine,
				BKECluster:          &bkev1beta1.BKECluster{},
			},
		}
		patchHelper, err := r.setupDeletionProcess(params)
		assert.NoError(t, err)
		assert.NotNil(t, patchHelper)
		assert.Contains(t, bkeMachine.GetAnnotations(), clusterv1.DeleteMachineAnnotation)
	})
}

// --- BKEClusterToBKEMachines ---

func TestBKEClusterToBKEMachines(t *testing.T) {
	t.Run("cluster not found returns empty", func(t *testing.T) {
		r := newMachineReconciler()
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace},
		}

		patches := gomonkey.ApplyFunc(util.GetOwnerCluster,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return nil, nil
			})
		defer patches.Reset()

		result := r.BKEClusterToBKEMachines(context.Background(), bkeCluster)
		assert.Empty(t, result)
	})

	t.Run("returns requests for unready machines", func(t *testing.T) {
		ownerCluster := &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster}}
		machine := &clusterv1.Machine{
			ObjectMeta: metav1.ObjectMeta{
				Name: machineTestName, Namespace: machineTestNamespace,
				Labels: map[string]string{clusterv1.ClusterNameLabel: machineTestCluster},
			},
			Spec: clusterv1.MachineSpec{
				ClusterName:       machineTestCluster,
				InfrastructureRef: corev1.ObjectReference{Name: "bke-machine-1", Namespace: machineTestNamespace},
			},
			Status: clusterv1.MachineStatus{BootstrapReady: false},
		}
		r := newMachineReconciler(ownerCluster, machine)
		bkeCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace},
		}

		patches := gomonkey.ApplyFunc(util.GetOwnerCluster,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return ownerCluster, nil
			})
		defer patches.Reset()

		result := r.BKEClusterToBKEMachines(context.Background(), bkeCluster)
		assert.Len(t, result, machineNumOne)
		assert.Equal(t, "bke-machine-1", result[machineNumZero].Name)
	})
}

// --- LogCommandFailed ---

func TestLogCommandFailed(t *testing.T) {
	t.Run("no failed nodes returns empty", func(t *testing.T) {
		r := newMachineReconciler()
		cmd := agentv1beta1.Command{Status: map[string]*agentv1beta1.CommandStatus{}}
		result := r.LogCommandFailed(cmd, nil, []string{}, nil, "reason")
		assert.Equal(t, "", result)
	})

	t.Run("failed node with stderr returns message", func(t *testing.T) {
		r := newMachineReconciler()
		log := newMachineLogger()
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}
		cmd := agentv1beta1.Command{
			ObjectMeta: metav1.ObjectMeta{Name: "cmd-1", Namespace: machineTestNamespace},
			Status: map[string]*agentv1beta1.CommandStatus{
				machineTestNodeIP: {
					Conditions: []*agentv1beta1.Condition{
						{ID: "step-1", Status: metav1.ConditionFalse, StdErr: []string{"error msg"}},
					},
				},
			},
		}
		result := r.LogCommandFailed(cmd, bkeCluster, []string{machineTestNodeIP}, log, "reason")
		assert.Equal(t, "error msg", result)
	})
}

// --- getBKEMachineAssociateCommands ---

func TestGetBKEMachineAssociateCommands(t *testing.T) {
	t.Run("no commands returns empty", func(t *testing.T) {
		scheme := newMachineScheme()
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		bkeCluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: machineTestCluster, Namespace: machineTestNamespace}}
		bkeMachine := &bkev1beta1.BKEMachine{ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace}}

		commands, err := getBKEMachineAssociateCommands(context.Background(), fakeClient, bkeCluster, bkeMachine)
		assert.NoError(t, err)
		assert.Empty(t, commands)
	})
}

// --- Reconcile (entry point) ---

func TestReconcileEntry(t *testing.T) {
	t.Run("BKEMachine not found returns nil", func(t *testing.T) {
		r := newMachineReconciler()
		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: "non-existent", Namespace: machineTestNamespace}}
		_, err := r.Reconcile(context.Background(), req)
		assert.NoError(t, err)
	})

	t.Run("owner machine nil returns nil", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)

		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return nil, nil
			})
		defer patches.Reset()

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		result, err := r.Reconcile(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("cluster nil returns nil", func(t *testing.T) {
		bkeMachine := &bkev1beta1.BKEMachine{
			ObjectMeta: metav1.ObjectMeta{Name: machineTestName, Namespace: machineTestNamespace},
		}
		r := newMachineReconciler(bkeMachine)

		machine := &clusterv1.Machine{ObjectMeta: metav1.ObjectMeta{Name: machineTestName}}
		patches := gomonkey.ApplyFunc(util.GetOwnerMachine,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Machine, error) {
				return machine, nil
			})
		defer patches.Reset()

		patches.ApplyFunc(util.GetClusterFromMetadata,
			func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
				return nil, nil
			})

		req := ctrl.Request{NamespacedName: client.ObjectKey{Name: machineTestName, Namespace: machineTestNamespace}}
		result, err := r.Reconcile(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})
}

// --- markBKEMachineDeletion ---

func TestMarkBKEMachineDeletion(t *testing.T) {
	t.Run("requires real object in cluster", func(t *testing.T) {
		// markBKEMachineDeletion needs a real patched object, tested via setupDeletionProcess
		t.Skip("covered by TestSetupDeletionProcess")
	})
}

// --- patchBKEMachine ---

func TestPatchBKEMachine(t *testing.T) {
	t.Run("requires real patch helper", func(t *testing.T) {
		t.Skip("covered by TestSetupDeletionProcess and TestHandleMainReconcile")
	})
}

// --- reconcileDelete ---

func TestReconcileDelete(t *testing.T) {
	t.Run("already marked calls handleAlreadyMarkedDeletion", func(t *testing.T) {
		t.Skip("requires deep integration with patch.Helper")
	})
}

// --- executeResetCommand ---

func TestBuildResetExtraCleanupArgs(t *testing.T) {
	rootKubeConfigPath := filepath.Join("/root", ".kube", "config")
	userKubeConfigPath := filepath.Join("/home", "workeruser", ".kube", "config")

	t.Run("non-root user appends both user and root kubeconfig paths", func(t *testing.T) {
		extra := []string{"10.0.0.1", "10.0.0.2"}
		node := &confv1beta1.Node{Username: "workeruser"}

		got := buildResetExtraCleanupArgsForNode(node, extra)

		assert.Equal(t, []string{"10.0.0.1", "10.0.0.2", rootKubeConfigPath, userKubeConfigPath}, got)
		assert.Equal(t, []string{"10.0.0.1", "10.0.0.2"}, extra)
	})

	t.Run("root user only appends root kubeconfig path", func(t *testing.T) {
		node := &confv1beta1.Node{Username: "root"}

		got := buildResetExtraCleanupArgsForNode(node, nil)

		assert.Equal(t, []string{rootKubeConfigPath}, got)
	})

	t.Run("legacy helper still appends base cleanup paths", func(t *testing.T) {
		got := buildResetExtraCleanupArgs(nil)

		assert.Equal(t, []string{rootKubeConfigPath}, got)
	})
}

func TestExecuteResetCommand(t *testing.T) {
	t.Run("requires command.Reset integration", func(t *testing.T) {
		t.Skip("requires command.Reset which needs cluster access")
	})
}

// --- SetupWithManager ---

func TestSetupWithManager(t *testing.T) {
	t.Run("requires real manager", func(t *testing.T) {
		t.Skip("requires real controller manager")
	})
}
