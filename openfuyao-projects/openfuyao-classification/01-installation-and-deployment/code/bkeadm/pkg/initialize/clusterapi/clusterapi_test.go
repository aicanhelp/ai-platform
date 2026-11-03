/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package clusterapi

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

const (
	testZeroValue = 0
	testOneValue  = 1
	testTwoValue  = 2
	testThreeVal  = 3
)

func TestEnsureK8sClientWithExistingClient(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	err := ensureK8sClient()
	assert.NoError(t, err)
}

func TestEnsureK8sClientNewClientSuccess(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	global.K8s = nil

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesClient, func(string) (k8s.KubernetesClient, error) {
		return &k8s.MockK8sClient{}, nil
	})

	err := ensureK8sClient()
	assert.NoError(t, err)
}

func TestEnsureK8sClientNewClientError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	global.K8s = nil

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesClient, func(string) (k8s.KubernetesClient, error) {
		return nil, assert.AnError
	})

	err := ensureK8sClient()
	assert.Error(t, err)
}

func TestWriteClusterAPITemplatesSuccess(t *testing.T) {
	tempDir := t.TempDir()
	tmplDir := filepath.Join(tempDir, "tmpl")

	err := writeClusterAPITemplates(tmplDir)
	assert.NoError(t, err)
	assert.DirExists(t, tmplDir)
}

func TestWriteClusterAPITemplatesMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return assert.AnError
	})

	err := writeClusterAPITemplates("/invalid/path/tmpl")
	assert.Error(t, err)
}

func TestWriteClusterAPITemplatesWriteFileError(t *testing.T) {
	tempDir := t.TempDir()

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return assert.AnError
	})

	err := writeClusterAPITemplates(tempDir)
	assert.Error(t, err)
}

func TestWriteClusterAPITemplatesMultipleFilesError(t *testing.T) {
	tempDir := t.TempDir()
	writeCount := 0

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		writeCount++
		if writeCount == testTwoValue {
			return assert.AnError
		}
		return nil
	})

	err := writeClusterAPITemplates(tempDir)
	assert.Error(t, err)
}

func podReadyCondition(status corev1.ConditionStatus) []corev1.PodCondition {
	return []corev1.PodCondition{
		{Type: corev1.PodReady, Status: status},
	}
}

func TestAreAllPodsReadyAllReady(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: podReadyCondition(corev1.ConditionTrue)}},
		{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: podReadyCondition(corev1.ConditionTrue)}},
	}

	result := AreAllPodsReady(pods)
	assert.True(t, result)
}

func TestAreAllPodsReadyRunningButNotReady(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: podReadyCondition(corev1.ConditionFalse)}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyRunningNoConditions(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodRunning}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyNotAllRunning(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: podReadyCondition(corev1.ConditionTrue)}},
		{Status: corev1.PodStatus{Phase: corev1.PodPending}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyWithContainerWaiting(t *testing.T) {
	pods := []corev1.Pod{
		{
			Status: corev1.PodStatus{
				Phase: corev1.PodPending,
				ContainerStatuses: []corev1.ContainerStatus{
					{
						Name: "manager",
						State: corev1.ContainerState{
							Waiting: &corev1.ContainerStateWaiting{
								Reason: "ImagePullBackOff",
							},
						},
					},
				},
			},
		},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyEmpty(t *testing.T) {
	pods := []corev1.Pod{}
	result := AreAllPodsReady(pods)
	assert.True(t, result)
}

func TestAreAllPodsReadyWithSucceededPhase(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodSucceeded}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyWithFailedPhase(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodFailed}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyWithUnknownPhase(t *testing.T) {
	pods := []corev1.Pod{
		{Status: corev1.PodStatus{Phase: corev1.PodUnknown}},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestAreAllPodsReadyWithMultipleContainers(t *testing.T) {
	pods := []corev1.Pod{
		{
			Status: corev1.PodStatus{
				Phase:      corev1.PodRunning,
				Conditions: podReadyCondition(corev1.ConditionTrue),
				ContainerStatuses: []corev1.ContainerStatus{
					{
						Name:  "capi-manager",
						Ready: true,
						State: corev1.ContainerState{
							Running: &corev1.ContainerStateRunning{},
						},
					},
					{
						Name:  "kbp-manager",
						Ready: true,
						State: corev1.ContainerState{
							Running: &corev1.ContainerStateRunning{},
						},
					},
				},
			},
		},
		{
			Status: corev1.PodStatus{
				Phase:      corev1.PodRunning,
				Conditions: podReadyCondition(corev1.ConditionFalse),
				ContainerStatuses: []corev1.ContainerStatus{
					{
						Name:  "manager",
						Ready: false,
						State: corev1.ContainerState{
							Waiting: &corev1.ContainerStateWaiting{
								Reason: "CrashLoopBackOff",
							},
						},
					},
				},
			},
		},
	}

	result := AreAllPodsReady(pods)
	assert.False(t, result)
}

func TestIsPodReadyTrue(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodScheduled, Status: corev1.ConditionTrue},
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
	assert.True(t, IsPodReady(pod))
}

func TestIsPodReadyFalse(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionFalse},
			},
		},
	}
	assert.False(t, IsPodReady(pod))
}

func TestIsPodReadyNoConditions(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{},
	}
	assert.False(t, IsPodReady(pod))
}

func TestNotReadyContainers(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "c1", Ready: true},
				{Name: "c2", Ready: false},
				{Name: "c3", Ready: false},
			},
		},
	}
	names := notReadyContainers(pod)
	assert.Equal(t, []string{"c2", "c3"}, names)
}

func TestNotReadyContainersAllReady(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "c1", Ready: true},
			},
		},
	}
	names := notReadyContainers(pod)
	assert.Empty(t, names)
}

func TestDeployClusterAPISuccess(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		return nil
	})

	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.NoError(t, err)
}

func TestDeployClusterAPIEnsureClientError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	global.K8s = nil

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesClient, func(string) (k8s.KubernetesClient, error) {
		return nil, assert.AnError
	})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestDeployClusterAPIWriteTemplatesError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return assert.AnError
	})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestDeployClusterAPIEnsureWebhookSecretsError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "CreateSecret", func(*k8s.MockK8sClient, *corev1.Secret) error {
		return assert.AnError
	})

	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return nil
	})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestDeployClusterAPIInstallClusterAPIError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		return assert.AnError
	})

	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	patches.ApplyFunc(installClusterAPIWithRetry, func(string, string, string) error {
		return assert.AnError
	})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestDeployClusterAPIInstallClusterAPIBKEError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	installCount := 0
	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		installCount++
		if installCount == testTwoValue {
			return assert.AnError
		}
		return nil
	})

	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestDeployClusterAPIWaitPodsError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(os.WriteFile, func(string, []byte, os.FileMode) error {
		return nil
	})

	patches.ApplyMethod(mockK8sClient, "InstallYaml", func(*k8s.MockK8sClient, string, map[string]string, string) error {
		return nil
	})

	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return assert.AnError
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := DeployClusterAPI("http://repo.example.com", "v1.0.0", "v1.0.0")
	assert.Error(t, err)
}

func TestInstallClusterAPIWithRetrySuccess(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyMethodFunc(mockK8sClient, "InstallYaml", func(string, map[string]string, string) error {
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := installClusterAPIWithRetry("/test/yaml/file.yaml", "http://repo.example.com", "caBundle")
	assert.NoError(t, err)
}

func TestInstallClusterAPIWithRetryMultipleRetriesSuccess(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	retryCount := 0
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethodFunc(mockK8sClient, "InstallYaml", func(string, map[string]string, string) error {
		retryCount++
		if retryCount < testThreeVal {
			return assert.AnError
		}
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := installClusterAPIWithRetry("/test/yaml/file.yaml", "http://repo.example.com", "caBundle")
	assert.NoError(t, err)
	assert.Equal(t, testThreeVal, retryCount)
}

func TestInstallClusterAPIWithRetryInstallError(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()

	mockK8sClient := &k8s.MockK8sClient{}
	global.K8s = mockK8sClient

	callCount := 0
	maxCalls := 2

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyMethodFunc(mockK8sClient, "InstallYaml", func(string, map[string]string, string) error {
		callCount++
		if callCount < maxCalls {
			return assert.AnError
		}
		return nil
	})

	patches.ApplyFunc(time.Sleep, func(time.Duration) {})

	err := installClusterAPIWithRetry("/test/yaml/file.yaml", "http://repo.example.com", "caBundle")
	assert.NoError(t, err)
	assert.Equal(t, maxCalls, callCount)
}

func TestWaitForClusterAPIPodsRunningSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return nil
	})

	err := waitForClusterAPIPodsRunning()
	assert.NoError(t, err)
}

func TestWaitForClusterAPIPodsRunningError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(waitForClusterAPIPodsRunning, func() error {
		return assert.AnError
	})

	err := waitForClusterAPIPodsRunning()
	assert.Error(t, err)
}
