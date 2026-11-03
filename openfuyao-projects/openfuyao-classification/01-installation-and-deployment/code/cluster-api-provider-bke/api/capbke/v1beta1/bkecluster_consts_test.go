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

package v1beta1

import (
	"testing"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

const (
	testReason = "TestReason"
	testMsg    = "Test message"
)

func TestNewBKELogger(t *testing.T) {
	tests := []struct {
		name      string
		logger    *log.Logger
		recorder  record.EventRecorder
		binder    runtime.Object
		expectNil bool
	}{
		{
			name:      "All parameters provided",
			logger:    nil,
			recorder:  &record.FakeRecorder{},
			binder:    nil,
			expectNil: false,
		},
		{
			name:      "Nil logger",
			logger:    nil,
			recorder:  &record.FakeRecorder{},
			binder:    nil,
			expectNil: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewBKELogger(tt.logger, tt.recorder, tt.binder)
			if tt.expectNil && result != nil {
				t.Error("expected nil result")
			}
			if !tt.expectNil && result == nil {
				t.Error("expected non-nil result")
			}
			if result != nil {
				if result.Recorder != tt.recorder {
					t.Errorf("expected recorder %v, got %v", tt.recorder, result.Recorder)
				}
			}
		})
	}
}

func TestBKELoggerInfo(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Info(testReason, testMsg)
}

func TestBKELoggerInfoWithArgs(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Info(testReason, testMsg, "key", "value")
}

func TestBKELoggerError(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Error(testReason, testMsg)
}

func TestBKELoggerErrorWithArgs(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Error(testReason, testMsg, "key", "value")
}

func TestBKELoggerWarn(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Warn(testReason, testMsg)
}

func TestBKELoggerWarnWithArgs(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Warn(testReason, testMsg, "key", "value")
}

func TestBKELoggerFinish(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Finish(testReason, testMsg)
}

func TestBKELoggerFinishWithArgs(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Finish(testReason, testMsg, "key", "value")
}

func TestBKELoggerDebug(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Debug(testMsg)
}

func TestBKELoggerDebugWithArgs(t *testing.T) {
	recorder := &record.FakeRecorder{}
	logger := NewBKELogger(nil, recorder, nil)

	logger.Debug(testMsg + " key=value")
}

func TestClusterConditionConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{"ControlPlaneEndPointSetCondition", "ControlPlaneEndPointSet"},
		{"TargetClusterReadyCondition", "TargetClusterReady"},
		{"TargetClusterBootCondition", "TargetClusterBoot"},
		{"ClusterAddonCondition", "Addon"},
		{"NodesInfoCondition", "NodesInfo"},
		{"BKEAgentCondition", "BKEAgent"},
		{"LoadBalancerCondition", "LoadBalancer"},
		{"NodesEnvCondition", "NodesEnv"},
		{"ClusterAPIObjCondition", "ClusterAPIObj"},
		{"SwitchBKEAgentCondition", "SwitchBKEAgent"},
		{"ControlPlaneInitializedCondition", "ControlPlaneInitialized"},
		{"BKEConfigCondition", "BKEConfig"},
		{"ClusterHealthyStateCondition", "ClusterHealthyState"},
		{"NodesPostProcessCondition", "NodesPostProcess"},
		{"BootstrapSucceededCondition", "BootstrapSucceeded"},
		{"BocloudClusterDataBackupCondition", "BocloudClusterDataBackup"},
		{"BocloudClusterMasterCertDistributionCondition", "BocloudClusterMasterCertDistribution"},
		{"BocloudClusterWorkerCertDistributionCondition", "BocloudClusterWorkerCertDistribution"},
		{"BocloudClusterEnvInitCondition", "BocloudClusterEnvInit"},
		{"TypeOfManagementClusterGuessCondition", "TypeOfManagementClusterGuess"},
		{"InternalSpecChangeCondition", "InternalSpecChange"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var actual string
			switch tt.name {
			case "ControlPlaneEndPointSetCondition":
				actual = string(ControlPlaneEndPointSetCondition)
			case "TargetClusterReadyCondition":
				actual = string(TargetClusterReadyCondition)
			case "TargetClusterBootCondition":
				actual = string(TargetClusterBootCondition)
			case "ClusterAddonCondition":
				actual = string(ClusterAddonCondition)
			case "NodesInfoCondition":
				actual = string(NodesInfoCondition)
			case "BKEAgentCondition":
				actual = string(BKEAgentCondition)
			case "LoadBalancerCondition":
				actual = string(LoadBalancerCondition)
			case "NodesEnvCondition":
				actual = string(NodesEnvCondition)
			case "ClusterAPIObjCondition":
				actual = string(ClusterAPIObjCondition)
			case "SwitchBKEAgentCondition":
				actual = string(SwitchBKEAgentCondition)
			case "ControlPlaneInitializedCondition":
				actual = string(ControlPlaneInitializedCondition)
			case "BKEConfigCondition":
				actual = string(BKEConfigCondition)
			case "ClusterHealthyStateCondition":
				actual = string(ClusterHealthyStateCondition)
			case "NodesPostProcessCondition":
				actual = string(NodesPostProcessCondition)
			case "BootstrapSucceededCondition":
				actual = BootstrapSucceededCondition
			case "BocloudClusterDataBackupCondition":
				actual = string(BocloudClusterDataBackupCondition)
			case "BocloudClusterMasterCertDistributionCondition":
				actual = string(BocloudClusterMasterCertDistributionCondition)
			case "BocloudClusterWorkerCertDistributionCondition":
				actual = string(BocloudClusterWorkerCertDistributionCondition)
			case "BocloudClusterEnvInitCondition":
				actual = string(BocloudClusterEnvInitCondition)
			case "TypeOfManagementClusterGuessCondition":
				actual = string(TypeOfManagementClusterGuessCondition)
			case "InternalSpecChangeCondition":
				actual = string(InternalSpecChangeCondition)
			}
			if actual != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, actual)
			}
		})
	}
}

func TestBKEClusterPhaseConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected confv1beta1.BKEClusterPhase
	}{
		{"InitControlPlane", InitControlPlane},
		{"JoinControlPlane", JoinControlPlane},
		{"JoinWorker", JoinWorker},
		{"FakeInitControlPlane", FakeInitControlPlane},
		{"FakeJoinControlPlane", FakeJoinControlPlane},
		{"FakeJoinWorker", FakeJoinWorker},
		{"FailedBootstrapNode", FailedBootstrapNode},
		{"UpgradeControlPlane", UpgradeControlPlane},
		{"UpgradeWorker", UpgradeWorker},
		{"UpgradeEtcd", UpgradeEtcd},
		{"ClusterReady", ClusterReady},
		{"MasterScalingUp", MasterScalingUp},
		{"MasterScalingDown", MasterScalingDown},
		{"WorkerScalingUp", WorkerScalingUp},
		{"WorkerScalingDown", WorkerScalingDown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected == "" {
				t.Error("phase should not be empty")
			}
		})
	}
}

func TestClusterStatusConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected confv1beta1.ClusterStatus
	}{
		{"ClusterStatusReady", ClusterStatusReady},
		{"ClusterUnhealthy", ClusterUnhealthy},
		{"ClusterUnknown", ClusterUnknown},
		{"ClusterChecking", ClusterChecking},
		{"ClusterPaused", ClusterPaused},
		{"ClusterPauseFailed", ClusterPauseFailed},
		{"ClusterDryRun", ClusterDryRun},
		{"ClusterDryRunFailed", ClusterDryRunFailed},
		{"ClusterInitializing", ClusterInitializing},
		{"ClusterInitializationFailed", ClusterInitializationFailed},
		{"ClusterUpgrading", ClusterUpgrading},
		{"ClusterUpgradeFailed", ClusterUpgradeFailed},
		{"ClusterMasterScalingUp", ClusterMasterScalingUp},
		{"ClusterMasterScalingDown", ClusterMasterScalingDown},
		{"ClusterWorkerScalingUp", ClusterWorkerScalingUp},
		{"ClusterWorkerScalingDown", ClusterWorkerScalingDown},
		{"ClusterMasterScalingUpFailed", ClusterMasterScalingUpFailed},
		{"ClusterMasterScalingDownFailed", ClusterMasterScalingDownFailed},
		{"ClusterWorkerScalingUpFailed", ClusterWorkerScalingUpFailed},
		{"ClusterWorkerScalingDownFailed", ClusterWorkerScalingDownFailed},
		{"ClusterDeployingAddon", ClusterDeployingAddon},
		{"ClusterDeployAddonFailed", ClusterDeployAddonFailed},
		{"ClusterManaging", ClusterManaging},
		{"ClusterManageFailed", ClusterManageFailed},
		{"ClusterDeleting", ClusterDeleting},
		{"ClusterDeleteFailed", ClusterDeleteFailed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected == "" {
				t.Error("status should not be empty")
			}
		})
	}
}

func TestPhaseStatusConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected confv1beta1.BKEClusterPhaseStatus
	}{
		{"PhaseSucceeded", PhaseSucceeded},
		{"PhaseFailed", PhaseFailed},
		{"PhaseUnknown", PhaseUnknown},
		{"PhaseWaiting", PhaseWaiting},
		{"PhaseRunning", PhaseRunning},
		{"PhaseSkipped", PhaseSkipped},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected == "" {
				t.Error("phase status should not be empty")
			}
		})
	}
}

func TestNodeStateConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected confv1beta1.NodeState
	}{
		{"NodeUnknown", NodeUnknown},
		{"NodeInitializing", NodeInitializing},
		{"NodeInitFailed", NodeInitFailed},
		{"NodeBootStrapping", NodeBootStrapping},
		{"NodeBootStrapFailed", NodeBootStrapFailed},
		{"NodeDeleting", NodeDeleting},
		{"NodeDeleteFailed", NodeDeleteFailed},
		{"NodeUpgrading", NodeUpgrading},
		{"NodeUpgradeFailed", NodeUpgradeFailed},
		{"NodeReady", NodeReady},
		{"NodeNotReady", NodeNotReady},
		{"NodeManaging", NodeManaging},
		{"NodeManageFailed", NodeManageFailed},
		{"EtcdUpgrading", EtcdUpgrading},
		{"EtcdUpgradeFailed", EtcdUpgradeFailed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected == "" {
				t.Error("node state should not be empty")
			}
		})
	}
}

func TestClusterHealthStateConstants(t *testing.T) {
	tests := []struct {
		name     string
		expected confv1beta1.ClusterHealthState
	}{
		{"Deploying", Deploying},
		{"DeployFailed", DeployFailed},
		{"Upgrading", Upgrading},
		{"UpgradeFailed", UpgradeFailed},
		{"Managing", Managing},
		{"ManageFailed", ManageFailed},
		{"Unhealthy", Unhealthy},
		{"Healthy", Healthy},
		{"Deleting", Deleting},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected == "" {
				t.Error("health state should not be empty")
			}
		})
	}
}

func TestNodeStateFlagConstants(t *testing.T) {
	tests := []struct {
		name  string
		value int
	}{
		{"NodeAgentPushedFlag", NodeAgentPushedFlag},
		{"NodeAgentReadyFlag", NodeAgentReadyFlag},
		{"NodeEnvFlag", NodeEnvFlag},
		{"NodeBootFlag", NodeBootFlag},
		{"NodeHAFlag", NodeHAFlag},
		{"MasterInitFlag", MasterInitFlag},
		{"NodeDeletingFlag", NodeDeletingFlag},
		{"NodeFailedFlag", NodeFailedFlag},
		{"NodeStateNeedRecord", NodeStateNeedRecord},
		{"NodePostProcessFlag", NodePostProcessFlag},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value == numZero {
				t.Errorf("%s should not be zero", tt.name)
			}
		})
	}
}

func TestClusterFinalizer(t *testing.T) {
	if ClusterFinalizer != "bkecluster.infrastructure.cluster.x-k8s.io" {
		t.Errorf("unexpected ClusterFinalizer value: %s", ClusterFinalizer)
	}
}

func TestExpectMinK8sVersion(t *testing.T) {
	if ExpectMinK8sVersion.Major == numZero && ExpectMinK8sVersion.Minor == numZero {
		t.Error("ExpectMinK8sVersion should have valid version")
	}
}
