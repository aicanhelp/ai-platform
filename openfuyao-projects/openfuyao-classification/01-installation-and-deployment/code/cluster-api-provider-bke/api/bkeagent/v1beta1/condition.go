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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// ConditionCountResult 封装 ConditionCount 函数的返回结果
type ConditionCountResult struct {
	// Succeeded 成功的条件数量
	Succeeded int
	// Failed 失败的条件数量
	Failed int
	// Status 条件状态
	Status metav1.ConditionStatus
	// Phase 命令阶段
	Phase CommandPhase
}

func RemoveCondition(cond []*Condition, con *Condition) []*Condition {
	if con == nil {
		return cond
	}
	c := GetCondition(cond, con)
	if c == nil {
		return cond
	}
	var newCond []*Condition
	for _, c1 := range cond {
		if c1.ID == con.ID {
			continue
		}
		newCond = append(newCond, c1)
	}
	return newCond
}

func GetCondition(cond []*Condition, con *Condition) *Condition {
	if con == nil || len(cond) == 0 {
		return nil
	}
	for _, c := range cond {
		if c.ID == con.ID {
			return c
		}
	}
	return nil
}

func ReplaceCondition(cond []*Condition, con *Condition) []*Condition {
	if con == nil {
		return cond
	}
	c := GetCondition(cond, con)
	if c == nil {
		cond = append(cond, con)
		return cond
	}
	var newCond []*Condition
	for _, c1 := range cond {
		if c1.ID == con.ID {
			newCond = append(newCond, con)
			continue
		}
		newCond = append(newCond, c1)
	}
	return newCond
}

// determineStatus 根据条件列表和命令总数确定状态
func determineStatus(latestCond *Condition, condLen, commandCount int) metav1.ConditionStatus {
	if condLen < commandCount {
		return metav1.ConditionUnknown
	}
	switch latestCond.Status {
	case metav1.ConditionTrue:
		return metav1.ConditionTrue
	case metav1.ConditionFalse:
		return metav1.ConditionFalse
	default:
		return metav1.ConditionUnknown
	}
}

// determinePhase 根据条件列表和命令总数确定阶段
func determinePhase(latestCond *Condition, condLen, commandCount int) CommandPhase {
	if latestCond.Phase == CommandFailed {
		return CommandFailed
	}
	if condLen < commandCount {
		if latestCond.Phase == CommandRunning || latestCond.Phase == CommandSkip || latestCond.Phase == CommandComplete {
			return CommandRunning
		}
	}
	if condLen == commandCount {
		if latestCond.Phase == CommandRunning {
			return CommandRunning
		}
		if latestCond.Phase == CommandSkip || latestCond.Phase == CommandComplete {
			return CommandComplete
		}
	}
	return CommandUnKnown
}

// countSucceededFailed 统计成功和失败的条件数量。
// Skip 视为成功（P6），即使历史对象 Status 曾为 False。
func countSucceededFailed(cond []*Condition) (succeeded, failed int) {
	for _, c := range cond {
		if c.Phase == CommandSkip || c.Status == metav1.ConditionTrue {
			succeeded++
			continue
		}
		if c.Status == metav1.ConditionFalse {
			failed++
			continue
		}
		succeeded++
	}
	return
}

func ConditionCount(cond []*Condition, commandCount int) ConditionCountResult {
	if len(cond) == 0 {
		return ConditionCountResult{
			Succeeded: 0,
			Failed:    0,
			Status:    metav1.ConditionUnknown,
			Phase:     CommandRunning,
		}
	}
	succeeded, failed := countSucceededFailed(cond)
	latestCond := cond[len(cond)-1]
	status := determineStatus(latestCond, len(cond), commandCount)
	phase := determinePhase(latestCond, len(cond), commandCount)
	// 当阶段为失败时，覆盖状态为 False
	if phase == CommandFailed {
		status = metav1.ConditionFalse
	}
	return ConditionCountResult{
		Succeeded: succeeded,
		Failed:    failed,
		Status:    status,
		Phase:     phase,
	}
}
