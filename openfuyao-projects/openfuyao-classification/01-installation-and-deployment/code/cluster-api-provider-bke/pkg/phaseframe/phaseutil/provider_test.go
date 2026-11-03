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

package phaseutil

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/pointer"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// 创建测试 Deployment（不指定副本数）
func createTestDeployment(name, namespace, containerName, image string) *appsv1.Deployment {
	deploy := createTestDeploymentWithReplicas(name, namespace, containerName, image, 0)
	deploy.Spec.Replicas = nil // 不设置副本数
	return deploy
}

// 创建带副本数的 Deployment
func createTestDeploymentWithReplicas(name, namespace, containerName, image string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: pointer.Int32(replicas),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: containerName, Image: image},
					},
				},
			},
		},
	}
}

// 创建 fake client
func createFakeClient(scheme *runtime.Scheme, objs ...client.Object) client.Client {
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func TestGetDeploymentImage(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = appsv1.AddToScheme(scheme)

	tests := []struct {
		name       string
		deployment *appsv1.Deployment
		target     DeploymentTarget
		want       string
		wantErr    bool
	}{
		{
			name:       "成功获取镜像",
			deployment: createTestDeployment("test-deploy", "test-ns", "manager", "test/image:v1.0.0"),
			target:     DeploymentTarget{Namespace: "test-ns", Name: "test-deploy", Container: "manager"},
			want:       "test/image:v1.0.0",
			wantErr:    false,
		},
		{
			name:       "容器不存在",
			deployment: createTestDeployment("test-deploy", "test-ns", "other", "test/image:v1.0.0"),
			target:     DeploymentTarget{Namespace: "test-ns", Name: "test-deploy", Container: "manager"},
			wantErr:    true,
		},
		{
			name:       "Deployment不存在",
			deployment: nil,
			target:     DeploymentTarget{Namespace: "test-ns", Name: "not-exist", Container: "manager"},
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.deployment != nil {
				objs = append(objs, tt.deployment)
			}

			fakeClient := createFakeClient(scheme, objs...)
			got, err := GetDeploymentImage(context.Background(), fakeClient, tt.target)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestPatchDeploymentImage(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = appsv1.AddToScheme(scheme)

	tests := []struct {
		name       string
		deployment *appsv1.Deployment
		target     DeploymentTarget
		newImage   string
		wantErr    bool
	}{
		{
			name:       "成功更新镜像",
			deployment: createTestDeploymentWithReplicas("test-deploy", "test-ns", "manager", "test/image:v1.0.0", 1),
			target:     DeploymentTarget{Namespace: "test-ns", Name: "test-deploy", Container: "manager"},
			newImage:   "test/image:v2.0.0",
			wantErr:    false,
		},
		{
			name:       "容器不存在",
			deployment: createTestDeployment("test-deploy", "test-ns", "other", "test/image:v1.0.0"),
			target:     DeploymentTarget{Namespace: "test-ns", Name: "test-deploy", Container: "manager"},
			newImage:   "test/image:v2.0.0",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := createFakeClient(scheme, tt.deployment)
			err := PatchDeploymentImage(context.Background(), fakeClient, tt.target, tt.newImage)

			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			verifyDeploymentImageUpdated(t, fakeClient, tt.target, tt.newImage)
		})
	}
}

// 验证 Deployment 镜像已更新
func verifyDeploymentImageUpdated(t *testing.T, c client.Client, target DeploymentTarget, expectedImage string) {
	var deploy appsv1.Deployment
	err := c.Get(context.Background(), client.ObjectKey{
		Namespace: target.Namespace,
		Name:      target.Name,
	}, &deploy)
	require.NoError(t, err)

	found := false
	for _, container := range deploy.Spec.Template.Spec.Containers {
		if container.Name == target.Container {
			assert.Equal(t, expectedImage, container.Image)
			found = true
			break
		}
	}
	assert.True(t, found, "容器未找到")
	assert.NotEmpty(t, deploy.Spec.Template.Annotations["bke.openfuyao.cn/restartedAt"])
}

func TestPodHasImage(t *testing.T) {
	tests := []struct {
		name  string
		pod   corev1.Pod
		image string
		want  bool
	}{
		{
			name:  "Pod包含指定镜像",
			pod:   corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "test/image:v1.0.0"}}}},
			image: "test/image:v1.0.0",
			want:  true,
		},
		{
			name:  "Pod不包含指定镜像",
			pod:   corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{Image: "test/image:v1.0.0"}}}},
			image: "test/image:v2.0.0",
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := podHasImage(tt.pod, tt.image)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestPodIsReady(t *testing.T) {
	tests := []struct {
		name string
		pod  corev1.Pod
		want bool
	}{
		{
			name: "Pod就绪",
			pod:  corev1.Pod{Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}},
			want: true,
		},
		{
			name: "Pod未就绪",
			pod:  corev1.Pod{Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}}},
			want: false,
		},
		{
			name: "Pod无Ready条件",
			pod:  corev1.Pod{Status: corev1.PodStatus{Conditions: []corev1.PodCondition{}}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PodIsReady(tt.pod)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestWaitDeploymentReady_ContextCanceled(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = appsv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	deploy := createTestDeploymentForWait()
	fakeClient := createFakeClient(scheme, deploy)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	target := DeploymentTarget{Namespace: "test-ns", Name: "test-deploy", Container: "manager"}
	err := WaitDeploymentReady(ctx, fakeClient, target, "test/image:v1.0.0", 1*time.Second)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "context canceled")
}

// 创建用于等待测试的 Deployment
func createTestDeploymentForWait() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-deploy", Namespace: "test-ns"},
		Spec: appsv1.DeploymentSpec{
			Replicas: pointer.Int32(1),
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "test"}},
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "manager", Image: "test/image:v1.0.0"}},
				},
			},
		},
		Status: appsv1.DeploymentStatus{Replicas: 0, UpdatedReplicas: 0, AvailableReplicas: 0},
	}
}
