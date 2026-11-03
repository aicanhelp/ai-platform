/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package intervention provides a reusable mechanism for notifying operators
// when manual intervention is required to resolve a failure that automatic
// retries cannot fix. It sets a Condition, emits a K8s Warning Event, logs the
// guidance, and persists the updated status.
package intervention

import (
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// Category represents the category of manual intervention required.
type Category string

const (
	// CategoryMasterInit indicates a master initialization failure requiring intervention.
	CategoryMasterInit Category = "MasterInit"
	// CategoryNodeFailure indicates a node-level permanent failure requiring intervention.
	CategoryNodeFailure Category = "NodeFailure"
	// CategoryRetryExhausted indicates that automatic retry limit has been exceeded.
	CategoryRetryExhausted Category = "RetryExhausted"
)

// Guidance encapsulates the information needed to notify operators
// about a manual intervention requirement.
type Guidance struct {
	// Category classifies the type of intervention.
	Category Category
	// Reason is the condition reason constant (from constant package).
	Reason string
	// Message is a human-readable description of the failure.
	Message string
	// Actions is a list of actionable steps for the operator.
	Actions []string
}

// Params holds all dependencies needed by Require().
type Params struct {
	// BKECluster is the target cluster object (conditions are set on it,
	// and K8s events are recorded against it).
	BKECluster *bkev1beta1.BKECluster
	// Client is used to persist status updates.
	// When nil, Require only sets the condition in memory and emits the event.
	Client client.Client
	// Recorder is the Kubernetes EventRecorder used to emit Warning events.
	// Obtained from controller manager (mgr.GetEventRecorderFor) or
	// from BKELogger.Recorder (exported field).
	Recorder record.EventRecorder
	// Guidance describes the intervention requirement.
	Guidance Guidance
}

// Require marks the BKECluster as requiring manual intervention by:
// 1. Setting ManualInterventionRequiredCondition = True (idempotent)
// 2. Emitting a K8s Warning Event with actionable guidance
// 3. Logging the guidance via ologger (utils/log)
// 4. Persisting the updated status (when Client is non-nil)
//
// It is safe to call repeatedly; duplicate calls with the same Reason
// will not produce duplicate events.
//
// When called from inside a Phase's Execute(), the condition is also set
// in memory on the BKECluster pointer, so the PhaseFlow PostHook's
// SyncStatusUntilComplete will persist it along with other status changes.
// When Client is provided, Require persists immediately as well, which is
// important for controller-level callers that are not inside a PhaseFlow.
func Require(params Params) error {
	if params.BKECluster == nil || params.Recorder == nil {
		return nil
	}

	guidance := params.Guidance
	fullMessage := buildMessage(guidance)

	// Idempotency: skip if already marked with the same reason
	if isAlreadyMarked(params.BKECluster, guidance.Reason) {
		return nil
	}

	// 1. Set condition (in memory on the BKECluster pointer)
	condition.ConditionMark(
		params.BKECluster,
		bkev1beta1.ManualInterventionRequiredCondition,
		confv1beta1.ConditionTrue,
		guidance.Reason,
		fullMessage,
	)

	// 2. Emit K8s Warning Event (standard EventRecorder interface)
	timestampedMsg := fmt.Sprintf("(%s) %s", time.Now().Format("2006-01-02 15:04:05 -0700 MST"), fullMessage)
	params.Recorder.AnnotatedEventf(
		params.BKECluster,
		annotation.BKENormalEventAnnotation(),
		corev1.EventTypeWarning,
		guidance.Reason,
		timestampedMsg,
	)

	// 3. Log via ologger (utils/log package-level function)
	log.Warnf("Manual intervention required: %s", fullMessage)

	// 4. Persist status (when Client is available)
	// This is important for controller-level callers that run outside a
	// PhaseFlow PostHook. For Phase-level callers, the PostHook's Report
	// will also call SyncStatusUntilComplete, but since the condition is
	// already on the in-memory BKECluster pointer, it will be included.
	if params.Client != nil {
		if err := mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster); err != nil {
			log.Warnf("Failed to sync status after marking manual intervention: %v", err)
			return err
		}
	}

	return nil
}

// Clear removes the manual intervention mark by setting the condition to False
// and persisting the status. Called when a phase starts executing again (e.g.,
// after the operator fixed the issue and modified the BKECluster spec).
func Clear(c client.Client, bkeCluster *bkev1beta1.BKECluster) error {
	if bkeCluster == nil {
		return nil
	}

	existing, ok := condition.HasCondition(bkev1beta1.ManualInterventionRequiredCondition, bkeCluster)
	if !ok || existing.Status != confv1beta1.ConditionTrue {
		return nil
	}

	condition.ConditionMark(
		bkeCluster,
		bkev1beta1.ManualInterventionRequiredCondition,
		confv1beta1.ConditionFalse,
		"",
		"Manual intervention resolved",
	)

	if c != nil {
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			log.Warnf("Failed to sync status after clearing manual intervention: %v", err)
			return err
		}
	}

	return nil
}

// ClearInMemory sets the manual intervention condition to False in memory
// without persisting to the API server. Use this when the caller will
// subsequently call SyncStatusUntilComplete itself (e.g., setupConditionAndRefresh),
// avoiding a redundant double-sync.
func ClearInMemory(bkeCluster *bkev1beta1.BKECluster) {
	if bkeCluster == nil {
		return
	}

	existing, ok := condition.HasCondition(bkev1beta1.ManualInterventionRequiredCondition, bkeCluster)
	if !ok || existing.Status != confv1beta1.ConditionTrue {
		return
	}

	condition.ConditionMark(
		bkeCluster,
		bkev1beta1.ManualInterventionRequiredCondition,
		confv1beta1.ConditionFalse,
		"",
		"Manual intervention resolved",
	)
}

// IsRequired checks whether the BKECluster currently has
// ManualInterventionRequiredCondition = True.
func IsRequired(bkeCluster *bkev1beta1.BKECluster) bool {
	return condition.HasConditionStatus(
		bkev1beta1.ManualInterventionRequiredCondition,
		bkeCluster,
		confv1beta1.ConditionTrue,
	)
}

// buildMessage formats the guidance into a single message string
// suitable for Condition.Message and K8s Event text.
func buildMessage(g Guidance) string {
	var sb strings.Builder
	sb.WriteString(g.Message)
	if len(g.Actions) > 0 {
		sb.WriteString("\nRecommended actions:")
		for i, action := range g.Actions {
			sb.WriteString(fmt.Sprintf("\n  %d. %s", i+1, action))
		}
	}
	return sb.String()
}

// isAlreadyMarked checks if the condition is already True with the same reason.
func isAlreadyMarked(bkeCluster *bkev1beta1.BKECluster, reason string) bool {
	existing, ok := condition.HasCondition(bkev1beta1.ManualInterventionRequiredCondition, bkeCluster)
	if !ok {
		return false
	}
	return existing.Status == confv1beta1.ConditionTrue && existing.Reason == reason
}
