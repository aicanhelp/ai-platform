/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package hooks

import (
	"fmt"
	"strings"
	"sync"

	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/utils"
)

// Actions 可以注册的Actions清单
const Actions = "PreRunPodSandbox,PreCreateContainer,PreStartContainer，" +
	"PostStartContainer,PreUpdateContainerResources,PostStopContainer,PostStopPodSandbox，PreRemoveRunPodSandbox"

var (
	rwlock      sync.RWMutex
	actionHooks = map[string][]*Hook{}
)

// HookFunc is the hook function
type HookFunc func(info core.Info) error

// Hook is the hook object
type Hook struct {
	name   string
	action string
	hfunc  HookFunc
}

func init() {
	// 实例化actionHooks
	actions := strings.Split(Actions, ",")
	for _, action := range actions {
		actionHooks[action] = make([]*Hook, 0)
	}
}

// RegisterFunc is to register a hook
func RegisterFunc(configPath string, action string, name string, hookfunc HookFunc) (*Hook, error) {
	// 单个hook独立配置path
	if configPath != "" {
		cfg, err := readHookActionConfigFile(configPath)
		if err != nil {
			klog.Errorf("read hookAction %v/%v config error: %v", action, name, err)
			return nil, err
		}
		allErr := validateHookActionConfigFile(cfg)
		if len(allErr) > 0 {
			return nil, fmt.Errorf("failed to validate credential provider config: %v", utils.AggregateErrors(allErr))
		}
	}

	newHook, err := AddHook(action, name)
	if err != nil {
		klog.Errorf("hook %s register action %v func error: %v", action, name, err)
		return nil, err
	}

	newHook.hfunc = hookfunc
	return newHook, nil
}

// AddHook is to add a Hook
func AddHook(action string, name string) (*Hook, error) {
	rwlock.Lock()
	defer rwlock.Unlock()
	if hooks, found := actionHooks[action]; found {
		for _, hook := range hooks {
			if hook.name == name {
				return nil, fmt.Errorf("hooks already exists")
			}
		}

		newHook := &Hook{
			name:   name,
			action: action,
		}

		actionHooks[action] = append(actionHooks[action], newHook)
		return newHook, nil
	} else {
		return nil, fmt.Errorf("action not found")
	}

}

// GetHook is to get the interface of object.
func GetHook(action string) ([]*Hook, error) {
	rwlock.RLock()
	defer rwlock.RUnlock()
	if hooks, found := actionHooks[action]; found {
		return hooks, nil
	} else {
		return nil, fmt.Errorf("action not found")
	}
}

// RunHook is to run the hook.
func RunHook(action string, info core.Info) error {
	hooks, err := GetHook(action)
	if err != nil {
		klog.Errorf("no hooks that need to be executed during the %v phase", action)
	}
	for _, h := range hooks {
		err = h.hfunc(info)
		if err != nil {
			klog.Errorf("failed to run %s/%s hook : %v", action, h.name, err)
		}
	}
	return nil
}

// ClearAllHooksSafe 线程安全的清除所有 hook
func ClearAllHooksSafe() {
	rwlock.Lock()
	defer rwlock.Unlock()

	if actionHooks == nil {
		return
	}
	for action := range actionHooks {
		actionHooks[action] = make([]*Hook, 0)
	}
}
