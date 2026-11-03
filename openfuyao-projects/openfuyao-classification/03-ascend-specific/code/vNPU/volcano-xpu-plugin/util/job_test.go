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

package util

import (
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"volcano.sh/volcano/pkg/scheduler/api"
)

func TestGetXPUTaskNumInJob(t *testing.T) {
	// 测试空ReqXPUName的XPUJob
	xJob := &XPUJob{}
	if xJob.GetXPUTaskNumInJob() != 0 {
		t.Errorf("Want 0, got %d", xJob.GetXPUTaskNumInJob())
	}

	// 测试非VNPUName的XPUJob
	xJob.ReqXPUName = "CPU"
	if xJob.GetXPUTaskNumInJob() != 0 {
		t.Errorf("Want 0, got %d", xJob.GetXPUTaskNumInJob())
	}

	// 测试有VNPUName的XPUJob
	xJob.ReqXPUName = VNPUName
	xJob.Tasks = map[api.TaskID]*XPUTask{
		"1": &XPUTask{TaskResource: &TaskResource{ReqXPUName: VNPUName}},
		"2": &XPUTask{TaskResource: &TaskResource{ReqXPUName: "aaa"}},
	}
	if xJob.GetXPUTaskNumInJob() != 1 {
		t.Errorf("Want 1, got %d", xJob.GetXPUTaskNumInJob())
	}
}

func TestGetSchedulingTaskNum(t *testing.T) {
	xJob := &XPUJob{}
	if xJob.GetSchedulingTaskNum() != 0 {
		t.Errorf("Want 0, got %d", xJob.GetSchedulingTaskNum())
	}

	xJob.ReqXPUName = "CPU"
	if xJob.GetSchedulingTaskNum() != 0 {
		t.Errorf("Want 0, got %d", xJob.GetSchedulingTaskNum())
	}

	xJob.ReqXPUName = VNPUName
	xJob.Tasks = map[api.TaskID]*XPUTask{
		"1": &XPUTask{NodeName: ""}, // scheduling task
		"2": &XPUTask{NodeName: "node1"},
	}
	if xJob.GetSchedulingTaskNum() != 1 {
		t.Errorf("Want 1, got %d", xJob.GetSchedulingTaskNum())
	}
}

func TestGetJobReferenceName(t *testing.T) {
	job := &api.JobInfo{}
	if GetJobReferenceName(job) != "" {
		t.Errorf("Want '', got %s", GetJobReferenceName(job))
	}

	job.PodGroup = &api.PodGroup{}
	if GetJobReferenceName(job) != "" {
		t.Errorf("Want '', got %s", GetJobReferenceName(job))
	}

	job.PodGroup.OwnerReferences = []v1.OwnerReference{}
	if GetJobReferenceName(job) != "" {
		t.Errorf("Want '', got %s", GetJobReferenceName(job))
	}

	job.PodGroup.OwnerReferences = []v1.OwnerReference{
		{Name: "job1"},
	}
	if GetJobReferenceName(job) != "job1" {
		t.Errorf("Want 'job1', got %s", GetJobReferenceName(job))
	}
}

func TestSelectorMatches(t *testing.T) {
	jobSelectors := map[string]string{}
	conf := map[string]string{"key1": "value1"}
	if !SelectorMatches(jobSelectors, conf) {
		t.Errorf("Want true, got %v", SelectorMatches(jobSelectors, conf))

		jobSelectors = map[string]string{"key2": "value2"}
		if SelectorMatches(jobSelectors, conf) {
			t.Errorf("Want false, got %v", SelectorMatches(jobSelectors, conf))
		}

		jobSelectors = map[string]string{"key1": "value2"}
		if SelectorMatches(jobSelectors, conf) {
			t.Errorf("Want false, got %v", SelectorMatches(jobSelectors, conf))
		}

		jobSelectors = map[string]string{"key1": "value1"}
		if !SelectorMatches(jobSelectors, conf) {
			t.Errorf("Want true, got %v", SelectorMatches(jobSelectors, conf))
		}
	}
}
