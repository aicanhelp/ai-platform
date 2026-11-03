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

package constant

import (
	"testing"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

func TestGetValidCrdResources(t *testing.T) {
	resources := GetValidCrdResources()
	if len(resources) != 3 {
		t.Fatalf("GetValidCrdResources() should return 3 resources, got %d", len(resources))
	}
	for _, name := range []string{RayCluster, RayJob, RayService} {
		if _, ok := resources[name]; !ok {
			t.Errorf("GetValidCrdResources() should contain resource %q", name)
		}
	}
}

func TestGetJobDeploymentStatusMap(t *testing.T) {
	statusMap := GetJobDeploymentStatusMap()
	if len(statusMap) != 7 {
		t.Fatalf("GetJobDeploymentStatusMap() should return 7 statuses, got %d", len(statusMap))
	}
	expectedKeys := []rayv1.JobDeploymentStatus{
		rayv1.JobDeploymentStatusInitializing,
		rayv1.JobDeploymentStatusRunning,
		rayv1.JobDeploymentStatusComplete,
		rayv1.JobDeploymentStatusFailed,
		rayv1.JobDeploymentStatusSuspending,
		rayv1.JobDeploymentStatusSuspended,
	}
	for _, key := range expectedKeys {
		if _, ok := statusMap[key]; !ok {
			t.Errorf("GetJobDeploymentStatusMap() should contain status %q", key)
		}
	}
}

func TestGetServiceStatusMap(t *testing.T) {
	statusMap := GetServiceStatusMap()
	if len(statusMap) != 5 {
		t.Fatalf("GetServiceStatusMap() should return 5 statuses, got %d", len(statusMap))
	}
	expectedKeys := []rayv1.ServiceStatus{
		rayv1.WaitForServeDeploymentReady,
		rayv1.Running,
		rayv1.Restarting,
	}
	for _, key := range expectedKeys {
		if _, ok := statusMap[key]; !ok {
			t.Errorf("GetServiceStatusMap() should contain status %q", key)
		}
	}
}

func TestGetClusterStatusMap(t *testing.T) {
	statusMap := GetClusterStatusMap()
	if len(statusMap) != 4 {
		t.Fatalf("GetClusterStatusMap() should return 4 statuses, got %d", len(statusMap))
	}
	expectedKeys := []rayv1.ClusterState{
		rayv1.Ready,
		rayv1.Failed,
		rayv1.Unhealthy,
		rayv1.Suspended,
	}
	for _, key := range expectedKeys {
		if _, ok := statusMap[key]; !ok {
			t.Errorf("GetClusterStatusMap() should contain status %q", key)
		}
	}
}

func TestGetGlobalStatusMap(t *testing.T) {
	statusMap := GetGlobalStatusMap()
	if len(statusMap) != 3 {
		t.Fatalf("GetGlobalStatusMap() should return 3 statuses, got %d", len(statusMap))
	}
	for _, key := range []string{Running, Suspended, Failed} {
		if _, ok := statusMap[key]; !ok {
			t.Errorf("GetGlobalStatusMap() should contain status %q", key)
		}
	}
}

func TestGetServiceFailedMap(t *testing.T) {
	failedMap := GetServiceFailedMap()
	if len(failedMap) != 4 {
		t.Fatalf("GetServiceFailedMap() should return 4 statuses, got %d", len(failedMap))
	}
	expectedKeys := []rayv1.ServiceStatus{
		rayv1.FailedToGetServeDeploymentStatus,
		rayv1.FailedToGetOrCreateRayCluster,
		rayv1.FailedToUpdateServingPodLabel,
		rayv1.FailedToUpdateService,
	}
	for _, key := range expectedKeys {
		if _, ok := failedMap[key]; !ok {
			t.Errorf("GetServiceFailedMap() should contain status %q", key)
		}
	}
}
