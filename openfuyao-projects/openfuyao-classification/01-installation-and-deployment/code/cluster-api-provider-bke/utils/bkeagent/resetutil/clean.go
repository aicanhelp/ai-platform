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

package resetutil

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// LeastFieldNums unmount路径时最少的字段数
const LeastFieldNums = 2

// CleanDir removes everything in a directory, but not the directory itself
func CleanDir(dirPath string) error {
	// Verify the path exists and is a directory
	fileInfo, err := os.Stat(dirPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if !fileInfo.IsDir() {
		return errors.Errorf("path %s is not a directory", dirPath)
	}

	dir, err := os.Open(filepath.Clean(dirPath))
	if err != nil {
		return err
	}
	defer dir.Close()

	// Read directory contents
	entries, err := dir.Readdir(-1)
	if err != nil {
		return err
	}

	// Remove each entry
	for _, entry := range entries {
		fullPath := filepath.Join(dirPath, entry.Name())
		if err = os.RemoveAll(fullPath); err != nil {
			return err
		}
	}
	return nil
}

func CleanFile(filePath string) error {
	if s, err := os.Stat(filePath); os.IsNotExist(err) {
		return errors.Errorf("file path %s does not exist", filePath)
	} else if s.IsDir() {
		return errors.Errorf("path %s is a directory ", filePath)
	}
	return os.Remove(filePath)
}

// readMountPoints securely reads and parses system mount information
func readMountPoints() ([]string, error) {
	content, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return nil, err
	}

	var mountPoints []string
	for _, line := range strings.Split(string(content), "\n") {
		if line == "" {
			continue
		}

		// Handle escaped spaces safely
		line = strings.ReplaceAll(line, `\040`, " ")
		fields := strings.Fields(line)
		if len(fields) >= LeastFieldNums {
			mountPoints = append(mountPoints, fields[1])
		}
	}
	return mountPoints, nil
}

// shouldUnmount determines if a mount point should be unmounted
func shouldUnmount(mountPoint, target string) bool {
	if !strings.HasPrefix(mountPoint, target) {
		return false
	}

	return mountPoint != target && mountPoint != target[:len(target)-1]
}

// secureUnmount performs unmount with additional safety checks
func secureUnmount(mountPoint string) error {
	if _, err := os.Stat(mountPoint); err != nil {
		return err
	}
	return syscall.Unmount(mountPoint, 0)
}

// UnmountKubeletDirectory unmounts all paths that contain KubeletRunDirectory
func UnmountKubeletDirectory(target string) error {
	if !strings.HasSuffix(target, "/") {
		target += "/"
	}

	// Read mount info securely
	mounts, err := readMountPoints()
	if err != nil {
		return fmt.Errorf("failed to access mount information")
	}

	// Process mounts
	for _, mp := range mounts {
		if shouldUnmount(mp, target) {
			if err = secureUnmount(mp); err != nil {
				log.Warnf("unmount operation fail directory in %s: %s, err: %v", target, mp, err)
			}
		}
	}

	return nil
}
