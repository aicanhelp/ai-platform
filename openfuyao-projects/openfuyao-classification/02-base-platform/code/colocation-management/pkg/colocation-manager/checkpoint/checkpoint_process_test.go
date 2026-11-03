/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package checkpoint

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

func ckptScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(s); err != nil {
		t.Fatalf("clientgo scheme: %v", err)
	}
	if err := colocationv1.AddToScheme(s); err != nil {
		t.Fatalf("colocation scheme: %v", err)
	}
	return s
}

func newCkpt(name, podName string) *colocationv1.ContainerCheckpoint {
	return &colocationv1.ContainerCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns"},
		Spec: colocationv1.ContainerCheckpointSpec{
			Namespace:     "ns",
			PodName:       podName,
			ContainerName: "c1",
		},
	}
}

func TestProcessCheckpoint(t *testing.T) {
	ctx := context.Background()

	t.Run("pod not found, checkpoint present -> delete succeeds", func(t *testing.T) {
		ckpt := newCkpt("ckpt-a", "pod-a")
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(ckpt).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		assert.Equal(t, 1, lw.processCheckpoint(ckpt))
	})

	t.Run("pod not found, checkpoint absent -> delete fails", func(t *testing.T) {
		ckpt := newCkpt("ckpt-b", "pod-b")
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).Build() // no objects
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		assert.Equal(t, 0, lw.processCheckpoint(ckpt))
	})

	t.Run("pod present non-contributive (Succeeded) -> delete succeeds", func(t *testing.T) {
		ckpt := newCkpt("ckpt-c", "pod-c")
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-c", Namespace: "ns"}, Status: corev1.PodStatus{Phase: corev1.PodSucceeded}}
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(ckpt, pod).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		assert.Equal(t, 1, lw.processCheckpoint(ckpt))
	})

	t.Run("pod present non-contributive (Failed), checkpoint absent -> delete fails", func(t *testing.T) {
		ckpt := newCkpt("ckpt-d", "pod-d")
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-d", Namespace: "ns"}, Status: corev1.PodStatus{Phase: corev1.PodFailed}}
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(pod).Build() // ckpt absent
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		assert.Equal(t, 0, lw.processCheckpoint(ckpt))
	})

	t.Run("pod present contributive (Running) -> keep", func(t *testing.T) {
		ckpt := newCkpt("ckpt-e", "pod-e")
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-e", Namespace: "ns"}, Status: corev1.PodStatus{Phase: corev1.PodRunning}}
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(ckpt, pod).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		assert.Equal(t, 0, lw.processCheckpoint(ckpt))
	})
}

func TestGarbageCollectCheckpoints(t *testing.T) {
	ctx := context.Background()
	t.Run("empty -> no deletions", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		lw.GarbageCollectCheckpoints(metav1.Now().Time) // must not panic
	})
	t.Run("with checkpoint whose pod is gone -> deleted", func(t *testing.T) {
		ckpt := newCkpt("ckpt-x", "pod-x")
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(ckpt).Build()
		cs := aggregate.NewClusterState(ctx, &apps.Configuration{})
		cs.AddOrUpdateCheckpoint(types.NamespacedName{Namespace: "ns", Name: "ckpt-x"}, ckpt)
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: cs}
		lw.GarbageCollectCheckpoints(metav1.Now().Time)
	})
}

func TestInitFromCheckpointsCases(t *testing.T) {
	ctx := context.Background()
	t.Run("empty list -> no-op", func(t *testing.T) {
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		lw.InitFromCheckpoints()
	})
	t.Run("list with checkpoints", func(t *testing.T) {
		ckpt := newCkpt("ckpt-i", "pod-i")
		ckpt.Status = colocationv1.ContainerCheckpointStatus{Version: "1.0"}
		cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).WithObjects(ckpt).Build()
		lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
		lw.InitFromCheckpoints()
	})
}

func TestMaintainCheckpointsEmpty(t *testing.T) {
	ctx := context.Background()
	cl := fake.NewClientBuilder().WithScheme(ckptScheme(t)).Build()
	lw := &LoaderWriter{ctx: ctx, Client: cl, clusterState: aggregate.NewClusterState(ctx, &apps.Configuration{})}
	lw.MaintainCheckpoints(metav1.Now().Time) // empty clusterState -> no create/update
}
