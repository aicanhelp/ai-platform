/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"

	"plugin-management-service/pkg/zlog"
)

// ConsolePluginManager contains a client to access ConsolePlugin resources
type ConsolePluginManager struct {
	Client dynamic.Interface
}

// NewConsolePluginManager returns a new ConsolePluginManager
func NewConsolePluginManager(config *rest.Config) (*ConsolePluginManager, error) {
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return &ConsolePluginManager{
		Client: client,
	}, nil
}

// ListConsolePlugins returns all the ConsolePlugin in the cluster
func (cm *ConsolePluginManager) ListConsolePlugins() ([]ConsolePlugin, error) {
	return ListConsolePlugins(cm.Client)
}

// GetConsolePlugin returns the ConsolePlugin with given name
func (cm *ConsolePluginManager) GetConsolePlugin(pluginName string) (*ConsolePlugin, error) {
	return GetConsolePlugin(cm.Client, pluginName)
}

// CheckPluginInstallment checks whether the ConsolePlugin with given name is installed
func (cm *ConsolePluginManager) CheckPluginInstallment(pluginName string) bool {
	_, err := GetConsolePlugin(cm.Client, pluginName)
	return err == nil
}

// CheckPluginEnablementIfInstalled checks the enablement of an installed ConsolePlugin
func (cm *ConsolePluginManager) CheckPluginEnablementIfInstalled(pluginName string) (bool, error) {
	cp, err := GetConsolePlugin(cm.Client, pluginName)
	if err != nil {
		return false, err
	}
	return cp.Spec.Enabled, nil
}

// SetPluginEnablementIfInstalled sets the enablement of the ConsolePlugin with given name
func (cm *ConsolePluginManager) SetPluginEnablementIfInstalled(pluginName string, newEnabled bool) error {
	cp, err := GetConsolePlugin(cm.Client, pluginName)
	if err != nil {
		return err
	}

	if cp.Spec.Enabled == newEnabled {
		zlog.Infof("ConsolePlugin enabled already satisfied: %t, skip patching", cp.Spec.Enabled)
		return nil
	}

	patch := []byte(fmt.Sprintf(`{"spec": {"enabled": %t}}`, newEnabled))
	err = PatchConsolePlugin(cm.Client, pluginName, patch)
	return err
}

const (
	consolePluginKind = "ConsolePlugin"
)

var (
	consolePluginGVR = schema.GroupVersionResource{
		Group:    "console.openfuyao.com",
		Version:  "v1beta1",
		Resource: "consoleplugins",
	}
)

// ListConsolePlugins returns all the ConsolePlugin in the cluster
func ListConsolePlugins(c dynamic.Interface) ([]ConsolePlugin, error) {
	cpList, err := c.Resource(consolePluginGVR).List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var consolePlugins []ConsolePlugin
	for _, cp := range cpList.Items {
		var consolePlugin ConsolePlugin
		err := runtime.DefaultUnstructuredConverter.FromUnstructured(cp.Object, &consolePlugin)
		if err != nil {
			zlog.Errorf("Error converting to %s: %s", consolePluginKind, cp.GetName())
			return nil, err
		}
		consolePlugins = append(consolePlugins, consolePlugin)
	}
	return consolePlugins, nil
}

// GetConsolePlugin returns the ConsolePlugin with given consoleplugin name
func GetConsolePlugin(c dynamic.Interface, name string) (*ConsolePlugin, error) {
	cp, err := c.Resource(consolePluginGVR).Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	var consolePlugin ConsolePlugin
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(cp.Object, &consolePlugin)
	if err != nil {
		zlog.Errorf("Error converting to %s: %s", consolePluginKind, name)
		return nil, err
	}
	return &consolePlugin, nil
}

// PatchConsolePlugin updates the ConsolePlugin with given patch data
func PatchConsolePlugin(c dynamic.Interface, name string, data []byte) error {
	_, err := c.Resource(consolePluginGVR).
		Patch(context.Background(), name, "application/merge-patch+json", data, metav1.PatchOptions{}, "")
	return err
}
