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

// Package utils
package utils

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
)

// LinuxContainerResources is the resource configuration for linux containers.
type LinuxContainerResources struct {
	CPUShares   *int64
	CFSQuota    *int64
	MemoryLimit *int64
	CPUSet      *string
	CPUSetMem   *string
}

// LinuxPodSandboxConfig is the linux specific pod sandbox configuration
type LinuxPodSandboxConfig struct {
	CgroupParent string
	Resources    LinuxContainerResources
}

// PodSandboxConfig is the configuration of a PodSandbox.
type PodSandboxConfig struct {
	Linux       *LinuxPodSandboxConfig
	Labels      map[string]string
	Annotations map[string]string
}

// ContainerConfig is the configuration of a container.
type ContainerConfig struct {
	Envs        map[string]string
	Labels      map[string]string
	Annotations map[string]string
}

// AggregateErrors aggregates multiple errors into one.
func AggregateErrors(errors []error) error {
	if len(errors) == 0 {
		return nil
	}
	var errorsFiltered []error
	for _, e := range errors {
		if e != nil {
			errorsFiltered = append(errorsFiltered, e)
		}
	}
	if len(errorsFiltered) == 0 {
		return nil
	}
	errorMessage := sets.NewString()
	for _, err := range errorsFiltered {
		msg := err.Error()
		if !errorMessage.Has(msg) {
			errorMessage.Insert(msg)
		}
	}
	return fmt.Errorf("%v", errorMessage.List())
}
