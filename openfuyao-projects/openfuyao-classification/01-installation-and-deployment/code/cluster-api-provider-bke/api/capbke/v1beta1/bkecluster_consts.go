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

package v1beta1

import (
	"fmt"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/cluster-api/util/version"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	ClusterFinalizer = "bkecluster.infrastructure.cluster.x-k8s.io"
)

var (
	// ExpectMinK8sVersion is the minimum kubernetes version supported by the provider
	ExpectMinK8sVersion, _ = version.ParseMajorMinorPatch("v1.27.0")
)

// BKELogger is a wrapper of zap.SugaredLogger and record.EventRecorder.
// +kubebuilder:object:generate:=false
type BKELogger struct {
	NormalLogger *log.Logger
	Recorder     record.EventRecorder
	EventBinder  runtime.Object

	mu sync.RWMutex
}

// NewBKELogger creates a new BKE logger with the specified parameters
func NewBKELogger(log *log.Logger, recorder record.EventRecorder, binder runtime.Object) *BKELogger {
	return &BKELogger{
		NormalLogger: log,
		Recorder:     recorder,
		EventBinder:  binder,
	}
}

// SetNormalLogger safely updates the underlying normal logger.
func (logger *BKELogger) SetNormalLogger(l *log.Logger) {
	if logger == nil {
		return
	}
	logger.mu.Lock()
	defer logger.mu.Unlock()
	logger.NormalLogger = l
}

func (logger *BKELogger) getNormalLogger() *log.Logger {
	if logger == nil {
		return nil
	}
	logger.mu.RLock()
	defer logger.mu.RUnlock()
	return logger.NormalLogger
}

func (logger *BKELogger) Info(reason, msg string, args ...interface{}) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	logger.Recorder.AnnotatedEventf(logger.EventBinder, annotation.BKENormalEventAnnotation(), corev1.EventTypeNormal, reason, msg, args...)
	if nl := logger.getNormalLogger(); nl != nil {
		nl.Infof(msg, args...)
		return
	}
	log.Infof(msg, args...)
}

func (logger *BKELogger) Error(reason, msg string, args ...interface{}) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	logger.Recorder.AnnotatedEventf(logger.EventBinder, annotation.BKENormalEventAnnotation(), corev1.EventTypeWarning, reason, msg, args...)
	if nl := logger.getNormalLogger(); nl != nil {
		nl.Errorf(msg, args...)
		return
	}
	log.Errorf(msg, args...)
}

func (logger *BKELogger) Warn(reason, msg string, args ...interface{}) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	logger.Recorder.AnnotatedEventf(logger.EventBinder, annotation.BKENormalEventAnnotation(), corev1.EventTypeWarning, reason, msg, args...)
	if nl := logger.getNormalLogger(); nl != nil {
		nl.Warnf(msg, args...)
		return
	}
	log.Warnf(msg, args...)
}

func (logger *BKELogger) Finish(reason, msg string, args ...interface{}) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	logger.Recorder.AnnotatedEventf(logger.EventBinder, annotation.BKEFinishEventAnnotation(), corev1.EventTypeNormal, reason, msg, args...)
	if nl := logger.getNormalLogger(); nl != nil {
		nl.Infof(msg, args...)
		return
	}
	log.Infof(msg, args...)
}

func (logger *BKELogger) Debug(msg string, args ...interface{}) {
	if nl := logger.getNormalLogger(); nl != nil {
		nl.Debugf(msg, args...)
		return
	}
	log.Debugf(msg, args...)
}

// condition type constants
const (
	ControlPlaneEndPointSetCondition confv1beta1.ClusterConditionType = "ControlPlaneEndPointSet"
	TargetClusterReadyCondition      confv1beta1.ClusterConditionType = "TargetClusterReady"
	TargetClusterBootCondition       confv1beta1.ClusterConditionType = "TargetClusterBoot"
	ClusterAddonCondition            confv1beta1.ClusterConditionType = "Addon"
	NodesInfoCondition               confv1beta1.ClusterConditionType = "NodesInfo"
	BKEAgentCondition                confv1beta1.ClusterConditionType = "BKEAgent"
	LoadBalancerCondition            confv1beta1.ClusterConditionType = "LoadBalancer"
	NodesEnvCondition                confv1beta1.ClusterConditionType = "NodesEnv"
	ClusterAPIObjCondition           confv1beta1.ClusterConditionType = "ClusterAPIObj"
	SwitchBKEAgentCondition          confv1beta1.ClusterConditionType = "SwitchBKEAgent"
	ControlPlaneInitializedCondition confv1beta1.ClusterConditionType = "ControlPlaneInitialized"
	BKEConfigCondition               confv1beta1.ClusterConditionType = "BKEConfig"
	ClusterHealthyStateCondition     confv1beta1.ClusterConditionType = "ClusterHealthyState"
	NodesPostProcessCondition        confv1beta1.ClusterConditionType = "NodesPostProcess"

	BootstrapSucceededCondition = "BootstrapSucceeded"

	// for bocloud cluster
	BocloudClusterDataBackupCondition             confv1beta1.ClusterConditionType = "BocloudClusterDataBackup"
	BocloudClusterMasterCertDistributionCondition confv1beta1.ClusterConditionType = "BocloudClusterMasterCertDistribution"
	BocloudClusterWorkerCertDistributionCondition confv1beta1.ClusterConditionType = "BocloudClusterWorkerCertDistribution"
	BocloudClusterEnvInitCondition                confv1beta1.ClusterConditionType = "BocloudClusterEnvInit"
	TypeOfManagementClusterGuessCondition         confv1beta1.ClusterConditionType = "TypeOfManagementClusterGuess"

	InternalSpecChangeCondition confv1beta1.ClusterConditionType = "InternalSpecChange"

	// ManualInterventionRequiredCondition indicates that manual intervention is required
	// to resolve a failure that automatic retries cannot fix.
	// When True, the Message field contains a description of the issue and
	// actionable guidance for the operator.
	ManualInterventionRequiredCondition confv1beta1.ClusterConditionType = "ManualInterventionRequired"
)

const (
	InitControlPlane     confv1beta1.BKEClusterPhase = "InitControlPlane"
	JoinControlPlane     confv1beta1.BKEClusterPhase = "JoinControlPlane"
	JoinWorker           confv1beta1.BKEClusterPhase = "JoinWorker"
	FakeInitControlPlane confv1beta1.BKEClusterPhase = "FakeInitControlPlan"
	FakeJoinControlPlane confv1beta1.BKEClusterPhase = "FakeJoinControlPlan"
	FakeJoinWorker       confv1beta1.BKEClusterPhase = "FakeJoinWorker"
	FailedBootstrapNode  confv1beta1.BKEClusterPhase = "FailedBootstrapNode"
	UpgradeControlPlane  confv1beta1.BKEClusterPhase = "UpgradeControlPlane"
	UpgradeWorker        confv1beta1.BKEClusterPhase = "UpgradeWorker"
	UpgradeEtcd          confv1beta1.BKEClusterPhase = "UpgradeEtcd"
	ClusterReady         confv1beta1.BKEClusterPhase = "ClusterReady"
	MasterScalingUp      confv1beta1.BKEClusterPhase = "MasterScalingUp"
	MasterScalingDown    confv1beta1.BKEClusterPhase = "MasterScalingDown"
	WorkerScalingUp      confv1beta1.BKEClusterPhase = "WorkerScalingUp"
	WorkerScalingDown    confv1beta1.BKEClusterPhase = "WorkerScalingDown"

	UpgradeControlPlaneConfig confv1beta1.BKEClusterPhase = "UpgradeControlPlaneConfig"
	UpgradeEtcdConfig         confv1beta1.BKEClusterPhase = "UpgradeEtcdConfig"
)

const (
	ClusterStatusReady confv1beta1.ClusterStatus = "Ready"
	ClusterUnhealthy   confv1beta1.ClusterStatus = "Unhealthy"
	ClusterUnknown     confv1beta1.ClusterStatus = "Unknown"
	ClusterChecking    confv1beta1.ClusterStatus = "Checking"

	ClusterPaused      confv1beta1.ClusterStatus = "Paused"
	ClusterPauseFailed confv1beta1.ClusterStatus = "PauseFailed"

	ClusterDryRun       confv1beta1.ClusterStatus = "DryRun"
	ClusterDryRunFailed confv1beta1.ClusterStatus = "DryRunFailed"

	ClusterInitializing         confv1beta1.ClusterStatus = "Initializing"
	ClusterInitializationFailed confv1beta1.ClusterStatus = "InitializationFailed"

	ClusterUpgrading     confv1beta1.ClusterStatus = "Upgrading"
	ClusterUpgradeFailed confv1beta1.ClusterStatus = "UpgradeFailed"

	ClusterMasterScalingUp         confv1beta1.ClusterStatus = "ScalingMasterNodesUp"
	ClusterMasterScalingDown       confv1beta1.ClusterStatus = "ScalingMasterNodesDown"
	ClusterWorkerScalingUp         confv1beta1.ClusterStatus = "ScalingWorkerNodesUp"
	ClusterWorkerScalingDown       confv1beta1.ClusterStatus = "ScalingWorkerNodesDown"
	ClusterMasterScalingUpFailed   confv1beta1.ClusterStatus = "MasterScalingUpFailed"
	ClusterMasterScalingDownFailed confv1beta1.ClusterStatus = "MasterScalingDownFailed"
	ClusterWorkerScalingUpFailed   confv1beta1.ClusterStatus = "WorkerScalingUpFailed"
	ClusterWorkerScalingDownFailed confv1beta1.ClusterStatus = "WorkerScalingDownFailed"

	ClusterDeployingAddon    confv1beta1.ClusterStatus = "DeployingAddon"
	ClusterDeployAddonFailed confv1beta1.ClusterStatus = "DeployAddonFailed"

	ClusterManaging     confv1beta1.ClusterStatus = "Managing"
	ClusterManageFailed confv1beta1.ClusterStatus = "ManageFailed"

	ClusterDeleting     confv1beta1.ClusterStatus = "Deleting"
	ClusterDeleteFailed confv1beta1.ClusterStatus = "DeleteFailed"
)

const (
	PhaseSucceeded confv1beta1.BKEClusterPhaseStatus = "Succeeded"
	PhaseFailed    confv1beta1.BKEClusterPhaseStatus = "Failed"
	PhaseUnknown   confv1beta1.BKEClusterPhaseStatus = "Unknown"
	PhaseWaiting   confv1beta1.BKEClusterPhaseStatus = "Waiting"
	PhaseRunning   confv1beta1.BKEClusterPhaseStatus = "Running"
	PhaseSkipped   confv1beta1.BKEClusterPhaseStatus = "Skipped"
)

// Node state constants - now use NodeState type from bkecommon/cluster/api/v1beta1
// These match the NodeState type defined in bkenode_types.go
const (
	NodeUnknown confv1beta1.NodeState = "Unknown"

	NodeInitializing confv1beta1.NodeState = "Initializing"
	NodeInitFailed   confv1beta1.NodeState = "InitFailed"

	NodeBootStrapping   confv1beta1.NodeState = "BootStrapping"
	NodeBootStrapFailed confv1beta1.NodeState = "BootStrapFailed"

	NodeDeleting     confv1beta1.NodeState = "Deleting"
	NodeDeleteFailed confv1beta1.NodeState = "DeleteFailed"

	NodeUpgrading     confv1beta1.NodeState = "Upgrading"
	NodeUpgradeFailed confv1beta1.NodeState = "UpgradeFailed"

	NodeReady    confv1beta1.NodeState = "Ready"
	NodeNotReady confv1beta1.NodeState = "NotReady"

	NodeManaging     confv1beta1.NodeState = "Managing"
	NodeManageFailed confv1beta1.NodeState = "ManageFailed"

	EtcdUpgrading     confv1beta1.NodeState = "Upgrading"
	EtcdUpgradeFailed confv1beta1.NodeState = "UpgradeFailed"
)

const (
	Deploying     confv1beta1.ClusterHealthState = "Deploying"
	DeployFailed  confv1beta1.ClusterHealthState = "DeployFailed"
	Upgrading     confv1beta1.ClusterHealthState = "Upgrading"
	UpgradeFailed confv1beta1.ClusterHealthState = "UpgradeFailed"
	Managing      confv1beta1.ClusterHealthState = "Managing"
	ManageFailed  confv1beta1.ClusterHealthState = "ManageFailed"
	Unhealthy     confv1beta1.ClusterHealthState = "Unhealthy"
	Healthy       confv1beta1.ClusterHealthState = "Healthy"
	Deleting      confv1beta1.ClusterHealthState = "Deleting"
)

const (
	NodeAgentPushedFlag = 1 << iota
	NodeAgentReadyFlag
	NodeEnvFlag
	NodeBootFlag
	NodeHAFlag
	MasterInitFlag
	NodeDeletingFlag
	NodeFailedFlag
	//NodeStateNeedRecord 这个code只是标记这个节点的状态需要在更新时进行记录，无其他作用，会在被记录后由statusManager移除
	NodeStateNeedRecord
	// NodePostProcessFlag marks node postprocess execution done.
	NodePostProcessFlag
)
