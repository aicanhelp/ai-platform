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

// Package file  provides a new health check with file probe
package file

import (
	"os"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/probe"
)

// Prober is health check interface
type Prober interface {
	Probe(path string, timeout int32) (probe.Result, string, error)
}

// New creat a file probe
func New() Prober {
	return fileProbe{}
}

type fileProbe struct{}

// Probe checks if the target file was modified within the specified validity period and returns the health check result
func (p fileProbe) Probe(path string, validity int32) (probe.Result, string, error) {
	// Retrieve file information
	fileInfo, err := os.Stat(path)
	if err != nil {
		klog.ErrorS(err, "Failed to get file information", "path", path)
		return probe.Failure, err.Error(), err // �ؼ��޸ģ����� err
	}

	// Get file modification time and current time
	modTime := fileInfo.ModTime()
	now := time.Now()
	duration := now.Sub(modTime)

	// Check if time difference is within a configured validity period
	validitySeconds := float64(validity)
	if duration.Seconds() <= validitySeconds {
		klog.V(2).Infof("File probe success! duration: %v, valid: %v", duration.Seconds(), validitySeconds)
		return probe.Success, "", nil
	}
	klog.V(2).Infof("File probe failed! duration: %v, valid: %v", duration.Seconds(), validitySeconds)

	return probe.Failure, "", nil
}
