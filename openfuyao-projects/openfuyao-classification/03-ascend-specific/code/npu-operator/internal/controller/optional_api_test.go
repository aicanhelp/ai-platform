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
	"encoding/json"
	"fmt"
	"testing"

	monitoringv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type optionalAPIErrorClient struct {
	client.Client
	err error
}

func (c optionalAPIErrorClient) Get(context.Context, client.ObjectKey, client.Object, ...client.GetOption) error {
	return c.err
}
func (c optionalAPIErrorClient) List(context.Context, client.ObjectList, ...client.ListOption) error {
	return c.err
}

func TestOptionalAPIAbsentRejectsNilDiscoveryError(t *testing.T) {
	var discoveryErr *discovery.ErrGroupDiscoveryFailed
	require.False(t, optionalAPIAbsent(discoveryErr, gvkServiceMonitor.GroupVersion()))
}

func TestOptionalAPICleanup(t *testing.T) {
	for _, target := range []struct {
		name string
		gvk  schema.GroupVersionKind
	}{
		{"ServiceMonitor", gvkServiceMonitor}, {"DeviceClass", gvkDeviceClass}, {"ResourceSlice", gvkResourceSlice},
	} {
		t.Run(target.name, func(t *testing.T) {
			gv := target.gvk.GroupVersion()
			missing := apierrors.NewNotFound(schema.GroupResource{Group: gv.Group, Resource: "test"}, "")
			forbidden := apierrors.NewForbidden(schema.GroupResource{Group: gv.Group, Resource: "test"}, "", fmt.Errorf("denied"))
			for _, tt := range []struct {
				name      string
				err       error
				wantError bool
			}{
				{"no match", &meta.NoKindMatchError{GroupKind: target.gvk.GroupKind(), SearchedVersions: []string{gv.Version}}, false},
				{"not found", missing, false},
				{"discovery absent", fmt.Errorf("mapping: %w", &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{gv: missing}}), false},
				{"discovery forbidden", &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{gv: forbidden}}, true},
				{"discovery timeout", &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{gv: apierrors.NewTimeoutError("timeout", 1)}}, true},
				{"other group", &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{{Group: "other", Version: "v1"}: missing}}, true},
				{"mixed failure", &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{gv: missing, {Group: "other", Version: "v1"}: forbidden}}, true},
				{"forbidden", forbidden, true},
				{"timeout", apierrors.NewTimeoutError("timeout", 1), true},
			} {
				t.Run(tt.name, func(t *testing.T) {
					p := getInstance()
					p.Spec.DRA.Managed = false
					_, r := getManagerAndReconciler(nil, p)
					r.Client = optionalAPIErrorClient{Client: r.Client, err: tt.err}
					var err error
					switch target.name {
					case "ServiceMonitor":
						obj := &monitoringv1.ServiceMonitor{ObjectMeta: metav1.ObjectMeta{Name: "test"}}
						obj.SetGroupVersionKind(gvkServiceMonitor)
						_, err = resourceCreators[gvkServiceMonitor](obj).reconcile(context.Background(), &r, false)
					case "DeviceClass":
						err = reconcileDRADeviceClassGarbage(context.Background(), &r)
					case "ResourceSlice":
						obj := &unstructured.Unstructured{Object: map[string]any{"spec": map[string]any{"selectors": []any{map[string]any{"cel": map[string]any{"expression": `device.driver == "test.example.com"`}}}}}}
						err = deleteDRAResourceSlicesForDeviceClass(context.Background(), &r, obj)
					default:
						t.Fatalf("unsupported cleanup target %q", target.name)
					}
					if tt.wantError {
						require.Error(t, err)
					} else {
						require.NoError(t, err)
					}
				})
			}
		})
	}
}

func TestExporterServiceMonitorSwitch(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	_, r := getManagerAndReconciler(nil, p)
	comp := loadTestComponent(t, &r, "npu-exporter")
	// Exercise real Service and ServiceMonitor assets without waiting for Pods.
	var resources []componentResource
	var monitorRef, serviceRef componentResourceRef
	for _, resource := range comp.resources {
		ref := resource.ref(r.namespace)
		switch ref.gvk {
		case gvkServiceMonitor:
			monitorRef = ref
			resources = append(resources, resource)
		case gvkService:
			serviceRef = ref
			resources = append(resources, resource)
		default:
			continue
		}
	}
	require.Len(t, resources, 2)
	comp.resources = resources
	r.components = []component{comp}
	obj := &monitoringv1.ServiceMonitor{}
	key := client.ObjectKey{Name: "npu-exporter-servicemonitor", Namespace: "monitoring"}
	for _, step := range []struct{ managed, enabled, exists bool }{
		{true, false, false}, {true, true, true}, {true, false, false}, {true, true, true}, {false, true, false},
	} {
		require.NoError(t, json.Unmarshal([]byte(fmt.Sprintf(`{"managed":%t,"serviceMonitor":{"enabled":%t}}`, step.managed, step.enabled)), &r.instance.Spec.Exporter))
		require.NoError(t, r.Update(ctx, r.instance))
		refs, err := r.activeManagedResourceRefs()
		require.NoError(t, err)
		require.Equal(t, step.exists, refs[monitorRef])
		require.Equal(t, step.managed, refs[serviceRef])
		_, err = comp.reconcile(ctx, &r)
		require.NoError(t, err)
		service := &corev1.Service{}
		err = r.Get(ctx, client.ObjectKey{Name: serviceRef.name, Namespace: serviceRef.namespace}, service)
		if step.managed {
			require.NoError(t, err)
		} else {
			require.True(t, apierrors.IsNotFound(err))
		}
		err = r.Get(ctx, key, obj)
		if step.exists {
			require.NoError(t, err)
		} else {
			require.True(t, apierrors.IsNotFound(err))
		}
	}
	// Disabled management must not delete a ServiceMonitor belonging to someone else.
	external := &monitoringv1.ServiceMonitor{ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}}
	require.NoError(t, r.Create(ctx, external))
	_, err := comp.reconcile(ctx, &r)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key, obj))
}

func TestManagedDRAReportsMissingAPI(t *testing.T) {
	p := getInstance()
	p.Spec.DRA.Managed = true
	p.Spec.DRA.DeviceClasses.Managed = true
	_, r := getManagerAndReconciler(nil, p)
	missing := &discovery.ErrGroupDiscoveryFailed{Groups: map[schema.GroupVersion]error{
		gvkDeviceClass.GroupVersion(): apierrors.NewNotFound(schema.GroupResource{Group: gvkDeviceClass.Group, Resource: "deviceclasses"}, ""),
	}}
	r.Client = optionalAPIErrorClient{Client: r.Client, err: missing}
	require.ErrorIs(t, reconcileDRADeviceClassGarbage(context.Background(), &r), missing)
}
