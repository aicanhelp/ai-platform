// Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"fmt"
	"testing"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func TestNodeDSnapshotCommandTransition(t *testing.T) {
	for _, tt := range []struct {
		name    string
		command v1.CommandSpec
	}{
		{name: "default"},
		{name: "command only", command: v1.CommandSpec{Command: []string{"/usr/local/bin/noded"}}},
		{name: "args only", command: v1.CommandSpec{Args: []string{"/usr/local/bin/noded -logLevel=1"}}},
		{name: "command and args", command: v1.CommandSpec{Command: []string{"/usr/local/bin/noded"}, Args: []string{"-logLevel=0"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			p := snapshotTestPolicy(t, true)
			p.Spec.NodeD.Command = tt.command
			_, r := getManagerAndReconciler(nil, p)
			r.hasNPUNodes = true
			base := nodeDTestTemplate(t)
			res := resourceCreators[gvkDaemonSet](base)
			for _, enabled := range []bool{false, true, false, true, false} {
				r.instance.Spec.NodeD.ContainerSnapshot.Enabled = enabled
				_, err := res.clone().reconcile(ctx, &r, true)
				require.NoError(t, err)
				ds := &appsv1.DaemonSet{}
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), ds))
				c := ds.Spec.Template.Spec.Containers[0]
				wantCommand := tt.command.Command
				if len(wantCommand) == 0 {
					wantCommand = []string{"/bin/bash", "-c", "--"}
				}
				assert.Equal(t, wantCommand, c.Command)
				if tt.name == "default" {
					assert.Equal(t, []string{"/usr/local/bin/noded -logFile=/var/log/mindx-dl/noded/noded.log -logLevel=0"}, c.Args)
				} else {
					assert.Equal(t, tt.command.Args, c.Args)
				}
			}
			// Removing explicit overrides restores the normal startup command.
			r.instance.Spec.NodeD.Command = v1.CommandSpec{}
			_, err := res.clone().reconcile(ctx, &r, true)
			require.NoError(t, err)
			ds := &appsv1.DaemonSet{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), ds))
			assert.Equal(t, []string{"/bin/bash", "-c", "--"}, ds.Spec.Template.Spec.Containers[0].Command)
			assert.Contains(t, ds.Spec.Template.Spec.Containers[0].Args[0], "/usr/local/bin/noded")
		})
	}
}

func TestInvalidInferDoesNotBlockRDMADisable(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.InferOperator = v1.InferOperatorSpec{ComponentWorkloadSpec: v1.ComponentWorkloadSpec{Managed: true, Image: v1.ImageSpec{Repository: "infer-operator:v26.1.0"}}}
	p.Spec.RDMADevicePlugin = v1.RDMADevicePluginSpec{ComponentWorkloadSpec: v1.ComponentWorkloadSpec{Managed: true, Image: v1.ImageSpec{Repository: "k8s-rdma-shared-dp:v26.1.0"}}}
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes, r.hasNFDLabels = true, true
	r.components = []component{loadTestComponent(t, &r, "infer-operator"), loadTestComponent(t, &r, "rdma-device-plugin")}
	_, err := r.reconcileComponents(ctx)
	require.NoError(t, err)
	r.instance.Spec.InferOperator.Config = map[string]string{"fault_handling": "invalid"}
	r.instance.Spec.RDMADevicePlugin.Managed = false
	require.NoError(t, r.Update(ctx, r.instance))
	for i := 0; i < 2; i++ {
		_, err = r.reconcileComponents(ctx)
		require.ErrorContains(t, err, "fault_handling")
	}
	for _, res := range r.components[1].resources {
		ref := res.ref(r.namespace)
		obj, err := r.Scheme.New(ref.gvk)
		require.NoError(t, err)
		err = r.Get(ctx, client.ObjectKey{Name: ref.name, Namespace: ref.namespace}, obj.(client.Object))
		assert.True(t, apierrors.IsNotFound(err), "disabled RDMA resource %s remains: %v", ref.name, err)
	}
	assert.Equal(t, v1.ComponentUnmanaged, r.instance.Status.GetComponentStatus(rdmaDevicePluginComponent).State.Phase)
}

func TestCleanupContinuesAndAggregatesErrors(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.InferOperator.Managed = true
	p.Spec.InferOperator.Config = map[string]string{"fault_handling": "invalid"}
	p.Spec.Exporter.Managed = false
	_, r := getManagerAndReconciler(nil, p)
	role := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: "rdma-role", Namespace: "kube-system"}}
	role.SetGroupVersionKind(gvkRole)
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "exporter-config", Namespace: "mindx-dl"}}
	cm.SetGroupVersionKind(gvkConfigMap)
	roleRes, cmRes := resourceCreators[gvkRole](role), resourceCreators[gvkConfigMap](cm)
	for _, res := range []componentResource{roleRes, cmRes} {
		_, err := res.clone().reconcile(ctx, &r, true)
		require.NoError(t, err)
	}
	inferConfig := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: inferOperatorConfigMapName, Namespace: "mindx-dl"}}
	inferConfig.SetGroupVersionKind(gvkConfigMap)
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "later-managed-component", Namespace: "mindx-dl"}}
	sa.SetGroupVersionKind(gvkServiceAccount)
	r.components = []component{
		{name: inferOperatorComponent, resources: []componentResource{resourceCreators[gvkConfigMap](inferConfig)}},
		{name: clusterdComponent, resources: []componentResource{resourceCreators[gvkServiceAccount](sa)}},
		{name: rdmaDevicePluginComponent, resources: []componentResource{roleRes}},
		{name: exporterComponent, resources: []componentResource{cmRes}},
	}
	original := r.Client
	r.Client = roleGetFailureClient{Client: original}
	_, err := r.reconcileComponents(ctx)
	require.ErrorContains(t, err, "fault_handling")
	require.ErrorContains(t, err, "injected API read failure")
	assert.True(t, apierrors.IsNotFound(original.Get(ctx, client.ObjectKeyFromObject(cm), &corev1.ConfigMap{})))
	assert.True(t, apierrors.IsNotFound(original.Get(ctx, client.ObjectKeyFromObject(sa), &corev1.ServiceAccount{})), "later managed components must retain their dependency ordering")
}

type roleGetFailureClient struct{ client.Client }

func (c roleGetFailureClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if _, ok := obj.(*rbacv1.Role); ok {
		return fmt.Errorf("injected API read failure")
	}
	return c.Client.Get(ctx, key, obj, opts...)
}

func TestComponentDisableRetriesReadFailure(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.RDMADevicePlugin.Managed = false
	_, r := getManagerAndReconciler(nil, p)
	r.hasNPUNodes, r.hasNFDLabels = true, true
	role := &rbacv1.Role{TypeMeta: metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"}, ObjectMeta: metav1.ObjectMeta{Name: "rdma-shared-dp-cm-role", Namespace: "kube-system"}}
	res := resourceCreators[gvkRole](role)
	_, err := res.clone().reconcile(ctx, &r, true)
	require.NoError(t, err)
	r.components = []component{{name: rdmaDevicePluginComponent, resources: []componentResource{res}}}
	original := r.Client
	r.Client = roleGetFailureClient{Client: original}
	result, err := r.reconcileComponents(ctx)
	t.Logf("cleanup result: error=%v requeue=%v status=%+v", err, result.RequeueAfter, r.instance.Status.ComponentStatuses)
	require.NoError(t, original.Get(ctx, client.ObjectKeyFromObject(role), &rbacv1.Role{}), "role remains in storage")
	require.ErrorContains(t, err, "injected API read failure")
	assert.NotEqual(t, v1.ComponentUnmanaged, r.instance.Status.GetComponentStatus(rdmaDevicePluginComponent).State.Phase)
	// The next reconcile must finish cleanup after the API recovers.
	r.Client = original
	_, err = r.reconcileComponents(ctx)
	require.NoError(t, err)
	assert.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(role), &rbacv1.Role{})))
	assert.Equal(t, v1.ComponentUnmanaged, r.instance.Status.GetComponentStatus(rdmaDevicePluginComponent).State.Phase)
}

type deletionReadErrorClient struct {
	client.Client
	err error
}

func (c deletionReadErrorClient) Get(context.Context, client.ObjectKey, client.Object, ...client.GetOption) error {
	return c.err
}

func TestDisabledResourceReadErrors(t *testing.T) {
	resource := schema.GroupResource{Group: "monitoring.coreos.com", Resource: "servicemonitors"}
	for _, tt := range []struct {
		name      string
		err       error
		wantError bool
	}{
		{name: "not found", err: apierrors.NewNotFound(resource, "test")},
		{name: "optional API absent", err: &meta.NoKindMatchError{GroupKind: schema.GroupKind{Group: resource.Group, Kind: "ServiceMonitor"}}},
		{name: "forbidden", err: apierrors.NewForbidden(resource, "test", fmt.Errorf("denied")), wantError: true},
		{name: "timeout", err: apierrors.NewTimeoutError("API timeout", 1), wantError: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, r := getManagerAndReconciler(nil, getInstance())
			r.Client = deletionReadErrorClient{Client: r.Client, err: tt.err}
			obj := &monitoringv1.ServiceMonitor{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}}
			obj.SetGroupVersionKind(gvkServiceMonitor)
			_, err := resourceCreators[gvkServiceMonitor](obj).reconcile(context.Background(), &r, false)
			if tt.wantError {
				require.ErrorIs(t, err, tt.err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
