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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
)

func TestNewXpuScheduleHandler(t *testing.T) {
	if got := NewXpuScheduleHandler(); got == nil {
		t.Errorf("NewXpuScheduleHandler() return nil, want not nil")
	}
}

func TestOnSessionOpen(t *testing.T) {
	plugin := New(nil)
	assert.NotNil(t, plugin)
	assert.Equal(t, PluginName, plugin.Name())
}

func TestOnSessionClose(t *testing.T) {
	plugin := &volcanoXPUPlugin{Scheduler: scheduleHandler, Arguments: nil}
	ssn := &framework.Session{}
	plugin.Scheduler.DeleteJobInfos["100"] = &api.JobInfo{}
	plugin.OnSessionClose(ssn)
	assert.Equal(t, 1, len(ssn.Jobs))
}
