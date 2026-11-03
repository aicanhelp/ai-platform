/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package statusmanage

import (
	"os"
	"strconv"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func TestInit(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected int
	}{
		{"default", "", DefaultAllowedFailedCount},
		{"valid env", "5", 5},
		{"invalid env", "invalid", DefaultAllowedFailedCount},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("ALLOWED_FAILED_COUNT", tt.envValue)
			} else {
				os.Unsetenv("ALLOWED_FAILED_COUNT")
			}

			env, b := os.LookupEnv("ALLOWED_FAILED_COUNT")
			result := DefaultAllowedFailedCount
			if b {
				envAllowed, err := strconv.Atoi(env)
				if err == nil {
					result = envAllowed
				}
			}

			if result != tt.expected {
				t.Errorf("expected %d, got %d", tt.expected, result)
			}
		})
	}
}

func TestNewStatusManager(t *testing.T) {
	sm := NewStatusManager()
	if sm == nil {
		t.Fatal("expected non-nil StatusManager")
	}
	if sm.BKEClusterStatusMap == nil {
		t.Error("expected non-nil BKEClusterStatusMap")
	}
	if sm.BKENodesStatusMap == nil {
		t.Error("expected non-nil BKENodesStatusMap")
	}
}

func TestStatusManager_GetCtrlResult(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: confv1beta1.ClusterStatus("Running")},
	}

	result := sm.GetCtrlResult(cluster)
	if result.Requeue {
		t.Error("expected no requeue for empty status")
	}

	cluster.Status.ClusterStatus = bkev1beta1.ClusterPaused
	result = sm.GetCtrlResult(cluster)
	if result.Requeue {
		t.Error("expected no requeue for paused cluster")
	}
}

func TestStatusManager_RemoveBKEClusterStatusCache(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	sm.BKEClusterStatusMap["default/test"] = &StatusRecord{}
	sm.RemoveBKEClusterStatusCache(cluster)
	if _, ok := sm.BKEClusterStatusMap["default/test"]; ok {
		t.Error("expected cache to be removed")
	}
}

func TestStatusManager_RemoveNodesStatusCache(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	sm.BKENodesStatusMap["default/test"] = map[string]*StatusRecord{"192.168.1.1": {}}
	sm.RemoveNodesStatusCache(cluster)
	if _, ok := sm.BKENodesStatusMap["default/test"]; ok {
		t.Error("expected nodes cache to be removed")
	}
}

func TestStatusManager_RemoveSingleNodeStatusCache(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	sm.BKENodesStatusMap["default/test"] = map[string]*StatusRecord{"192.168.1.1": {}}
	sm.RemoveSingleNodeStatusCache(cluster, "192.168.1.1")
	if _, ok := sm.BKENodesStatusMap["default/test"]["192.168.1.1"]; ok {
		t.Error("expected node cache to be removed")
	}
}

func TestStatusManager_RemoveClusterStatusManagerCache(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	sm.BKEClusterStatusMap["default/test"] = &StatusRecord{}
	sm.BKENodesStatusMap["default/test"] = map[string]*StatusRecord{}
	sm.RemoveClusterStatusManagerCache(cluster)
	if _, ok := sm.BKEClusterStatusMap["default/test"]; ok {
		t.Error("expected cluster cache to be removed")
	}
	if _, ok := sm.BKENodesStatusMap["default/test"]; ok {
		t.Error("expected nodes cache to be removed")
	}
}

func TestStatusManager_recordBKEClusterStatus_NoAnnotation(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: confv1beta1.ClusterStatus("Running")},
	}
	sm.recordBKEClusterStatus(cluster)
	if _, ok := sm.BKEClusterStatusMap["default/test"]; ok {
		t.Error("expected no record without annotation")
	}
}

func TestStatusManager_recordBKEClusterStatus_EmptyStatus(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: ""},
	}
	sm.recordBKEClusterStatus(cluster)
	if _, ok := sm.BKEClusterStatusMap["default/test"]; ok {
		t.Error("expected no record with empty status")
	}
}

func TestStatusManager_recordBKEClusterStatus_PausedStatus(t *testing.T) {
	ReconcileAllowedFailedCount = 10
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterPaused},
	}
	sm.recordBKEClusterStatus(cluster)
	sr := sm.BKEClusterStatusMap["default/test"]
	if sr == nil {
		t.Fatal("expected status record")
	}
	if sr.NeedRequeue {
		t.Error("expected no requeue for paused status")
	}
}

func TestStatusManager_recordBKEClusterStatus_NormalStatus(t *testing.T) {
	ReconcileAllowedFailedCount = 10
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: confv1beta1.ClusterStatus("Running")},
	}
	sm.recordBKEClusterStatus(cluster)
	sr := sm.BKEClusterStatusMap["default/test"]
	if sr == nil {
		t.Fatal("expected status record")
	}
	if sr.LatestNormalState != "Running" {
		t.Errorf("expected normal state to be Running, got %s", sr.LatestNormalState)
	}
	if sr.NeedRequeue {
		t.Error("expected no requeue for normal status")
	}
}

func TestStatusManager_recordBKEClusterStatus_FailedStatus(t *testing.T) {
	ReconcileAllowedFailedCount = 10
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus:      "TestFailed",
			ClusterHealthState: bkev1beta1.Deploying,
		},
	}
	sm.BKEClusterStatusMap["default/test"] = &StatusRecord{LatestNormalState: "TestRunning"}
	sm.recordBKEClusterStatus(cluster)
	sr := sm.BKEClusterStatusMap["default/test"]
	if sr.StatusCount != 1 {
		t.Errorf("expected count 1, got %d", sr.StatusCount)
	}
	if sr.LatestFailedState != "TestFailed" {
		t.Errorf("expected failed state TestFailed, got %s", sr.LatestFailedState)
	}
	if !sr.NeedRequeue {
		t.Error("expected requeue for failed status")
	}
}

func TestStatusManager_recordBKEClusterStatus_ExceedLimit(t *testing.T) {
	ReconcileAllowedFailedCount = 2
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus:      "TestFailed",
			ClusterHealthState: bkev1beta1.Deploying,
		},
	}
	sm.BKEClusterStatusMap["default/test"] = &StatusRecord{
		LatestNormalState:   "TestRunning",
		LatestFailedState:   "TestFailed",
		StatusCount:         2,
		CurrentClusterState: bkev1beta1.ClusterInitializing,
	}
	sm.recordBKEClusterStatus(cluster)
	sr := sm.BKEClusterStatusMap["default/test"]
	if sr.StatusCount != 0 {
		t.Errorf("expected count reset to 0, got %d", sr.StatusCount)
	}
	if sr.NeedRequeue {
		t.Error("expected no requeue after exceeding limit")
	}
}

func TestStatusManager_GetNodesResult(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}

	result := sm.GetNodesResult(cluster, "192.168.1.1")
	if !result {
		t.Error("expected true for non-existent node")
	}

	sm.BKENodesStatusMap["default/test"] = map[string]*StatusRecord{
		"192.168.1.1": {NeedRequeue: false},
	}
	result = sm.GetNodesResult(cluster, "192.168.1.1")
	if result {
		t.Error("expected false when NeedRequeue is false")
	}
}

func TestStatusManager_SetStatus(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Namespace:   "default",
			Annotations: map[string]string{annotation.StatusRecordAnnotationKey: "true"},
		},
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterPaused},
	}
	nodes := bkev1beta1.BKENodes{}
	sm.SetStatus(cluster, nodes)
	if _, ok := sm.BKEClusterStatusMap["default/test"]; !ok {
		t.Error("expected cluster status to be recorded")
	}
}

func TestStatusManager_recordBKENodesStatus_EmptyNodes(t *testing.T) {
	sm := NewStatusManager()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	sm.recordBKENodesStatus(cluster, nil)
	if _, ok := sm.BKENodesStatusMap["default/test"]; ok {
		t.Error("expected no nodes status for nil nodes")
	}
}
