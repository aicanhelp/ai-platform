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

package componentfactory

import (
	"fmt"
	"sync"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

// PhaseFactory creates a Phase bound to the given PhaseContext.
type PhaseFactory func(ctx *phaseframe.PhaseContext) phaseframe.Phase

// ComponentInstance holds a registered inline phase factory.
type ComponentInstance struct {
	Name    string
	Version string
	Factory PhaseFactory
}

// ComponentFactory registers and resolves phase factories by name and version.
type ComponentFactory struct {
	mu       sync.RWMutex
	registry map[string]ComponentInstance
}

// NewComponentFactory creates an empty ComponentFactory.
func NewComponentFactory() *ComponentFactory {
	return &ComponentFactory{
		registry: make(map[string]ComponentInstance),
	}
}

// registryKey returns the map key for a component name and version.
func registryKey(name, version string) string {
	return fmt.Sprintf("%s@%s", name, version)
}

// Register stores a phase factory under name@version.
func (f *ComponentFactory) Register(name, version string, factory PhaseFactory) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.registry[registryKey(name, version)] = ComponentInstance{
		Name:    name,
		Version: version,
		Factory: factory,
	}
}

// Resolve looks up a registered factory and instantiates the phase.
func (f *ComponentFactory) Resolve(name, version string, ctx *phaseframe.PhaseContext) (phaseframe.Phase, error) {
	f.mu.RLock()
	inst, ok := f.registry[registryKey(name, version)]
	f.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("component %s@%s not found", name, version)
	}
	if inst.Factory == nil {
		return nil, fmt.Errorf("component %s@%s has nil factory", name, version)
	}
	if ctx == nil {
		return nil, fmt.Errorf("phase context is required for %s@%s", name, version)
	}
	return inst.Factory(ctx), nil
}

// Lookup returns the registered instance without instantiating.
func (f *ComponentFactory) Lookup(name, version string) (*ComponentInstance, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	inst, ok := f.registry[registryKey(name, version)]
	if !ok {
		return nil, fmt.Errorf("component %s@%s not found", name, version)
	}
	return &inst, nil
}
