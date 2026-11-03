package webterminal

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
)

func TestCreateInitContainers(t *testing.T) {
	containers := createInitContainers("img")
	if assert.Len(t, containers, 1) {
		assert.Equal(t, "init-kubeconfig", containers[0].Name)
		assert.Equal(t, "img", containers[0].Image)
		if assert.NotNil(t, containers[0].SecurityContext) {
			assert.NotNil(t, containers[0].SecurityContext.RunAsUser)
		}
	}
}

func TestGetInitContainerVolumeMounts(t *testing.T) {
	mounts := getInitContainerVolumeMounts()
	if assert.Len(t, mounts, 2) {
		assert.Equal(t, "kubeconfig", mounts[0].Name)
		assert.Equal(t, "/mnt/.kube", mounts[0].MountPath)
		assert.Equal(t, "config-source", mounts[1].Name)
		assert.Equal(t, "/etc/kubernetes", mounts[1].MountPath)
	}
}

func TestCreateMainContainers(t *testing.T) {
	containers := createMainContainers("img", 1000)
	if assert.Len(t, containers, 1) {
		assert.Equal(t, UserContainerName, containers[0].Name)
		assert.Equal(t, "img", containers[0].Image)
		if assert.NotNil(t, containers[0].SecurityContext) {
			if assert.NotNil(t, containers[0].SecurityContext.RunAsUser) {
				assert.Equal(t, int64(1000), *containers[0].SecurityContext.RunAsUser)
			}
		}
		if assert.Len(t, containers[0].Env, 1) {
			assert.Equal(t, "KUBECONFIG", containers[0].Env[0].Name)
		}
	}
}

func TestGetMainContainerVolumeMounts(t *testing.T) {
	mounts := getMainContainerVolumeMounts()
	if assert.Len(t, mounts, 1) {
		assert.Equal(t, "kubeconfig", mounts[0].Name)
		assert.Equal(t, "/mnt/.kube", mounts[0].MountPath)
	}
}

func TestCreateVolumes(t *testing.T) {
	vols := createVolumes()
	if assert.Len(t, vols, 2) {
		assert.Equal(t, "kubeconfig", vols[0].Name)
		assert.NotNil(t, vols[0].EmptyDir)
		assert.Equal(t, "config-source", vols[1].Name)
		if assert.NotNil(t, vols[1].HostPath) {
			assert.Equal(t, RootPath, vols[1].HostPath.Path)
		}
	}
}

func TestCreatePodTemplate(t *testing.T) {
	pod := createPodTemplate("user", "img")
	assert.Equal(t, "user", pod.ObjectMeta.Name)
	assert.Equal(t, UserPodNamespace, pod.ObjectMeta.Namespace)
	assert.Len(t, pod.Spec.InitContainers, 1)
	assert.Len(t, pod.Spec.Containers, 1)
	assert.Equal(t, v1.PullIfNotPresent, pod.Spec.Containers[0].ImagePullPolicy)
	assert.Len(t, pod.Spec.Volumes, 2)
}
