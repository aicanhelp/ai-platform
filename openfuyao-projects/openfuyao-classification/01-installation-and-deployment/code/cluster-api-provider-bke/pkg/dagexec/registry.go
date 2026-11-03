/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	"fmt"
	"sync"
)

// ExecutorRegistry maps ComponentType to a ComponentExecutor for pluggable dispatch.
type ExecutorRegistry struct {
	mu        sync.RWMutex
	executors map[ComponentType]ComponentExecutor
}

// NewExecutorRegistry creates an empty executor registry.
func NewExecutorRegistry() *ExecutorRegistry {
	return &ExecutorRegistry{
		executors: make(map[ComponentType]ComponentExecutor),
	}
}

// Register stores executor under executor.GetComponentType().
// A later Register for the same type overwrites the previous entry.
func (r *ExecutorRegistry) Register(executor ComponentExecutor) error {
	if r == nil {
		return fmt.Errorf("executor registry is nil")
	}
	if executor == nil {
		return fmt.Errorf("component executor is nil")
	}
	typ := executor.GetComponentType()
	if typ == "" {
		return fmt.Errorf("component type is empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.executors == nil {
		r.executors = make(map[ComponentType]ComponentExecutor)
	}
	r.executors[typ] = executor
	return nil
}

// Get returns the executor registered for typ.
func (r *ExecutorRegistry) Get(typ ComponentType) (ComponentExecutor, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	executor, ok := r.executors[typ]
	return executor, ok
}

// Has reports whether typ has a registered executor.
func (r *ExecutorRegistry) Has(typ ComponentType) bool {
	_, ok := r.Get(typ)
	return ok
}
