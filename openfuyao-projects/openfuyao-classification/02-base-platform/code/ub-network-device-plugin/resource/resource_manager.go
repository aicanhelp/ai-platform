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

package resource

import (
	"sync"
	"time"
)

type resourceManager struct {
	mu sync.RWMutex

	name      Name
	resources Resources
	devices   Devices
}

type Manager interface {
	Resource() Name
	Devices() Devices
	SyncDevices(time.Duration, <-chan interface{}, chan struct{}) error
	UpdateDevices(Devices)
	Recover() error
}

func (r *resourceManager) Resource() Name {
	return r.name
}

func (r *resourceManager) Devices() Devices {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.devices
}
