/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package disk performs disk space check
package disk

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/shirou/gopsutil/v3/disk"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

const (
	numThousand = 1024
	gbBytes     = numThousand * numThousand * numThousand
)

// Checker disk checker
type Checker struct {
	cfg    *config.DiskCheckConfig
	logger logger.Logger
	roles  []string // Current node roles
}

// NewChecker creates a new disk checker
func NewChecker(cfg *config.DiskCheckConfig, log logger.Logger, roles []string) *Checker {
	return &Checker{
		cfg:    cfg,
		logger: log,
		roles:  roles,
	}
}

// Execute executes disk check
func (c *Checker) Execute() (*config.DiskCheckResult, error) {
	c.logger.Info("disk check start......")

	result := &config.DiskCheckResult{
		Timestamp: time.Now(),
		Spaces:    make([]config.DiskSpace, 0),
		Summary:   config.DiskSummary{},
	}

	for _, item := range c.cfg.CheckItems {
		if !c.shouldCheckForRole(item.Roles) {
			c.logger.Info(fmt.Sprintf("skip disk check for %s (role not match)", item.Path))
			continue
		}

		diskSpace := c.checkDiskSpace(item)
		result.Spaces = append(result.Spaces, diskSpace)
		result.Summary.TotalChecked++

		if diskSpace.IsSufficient {
			result.Summary.SufficientPath++
			c.logger.Info(fmt.Sprintf("disk space check passed for %s: free=%dGB, required=%dGB",
				diskSpace.Path, diskSpace.Free/gbBytes, diskSpace.MinFree/gbBytes))
		} else {
			result.Summary.InsufficientPath++
			c.logger.Warning(fmt.Sprintf("disk space check failed for %s: free=%dGB, required=%dGB",
				diskSpace.Path, diskSpace.Free/gbBytes, diskSpace.MinFree/gbBytes))
		}
	}

	c.logger.Info("disk check completed")
	return result, nil
}

func (c *Checker) shouldCheckForRole(itemRoles []string) bool {
	if len(itemRoles) == 0 {
		return true
	}

	for _, currentRole := range c.roles {
		for _, itemRole := range itemRoles {
			if currentRole == itemRole {
				return true
			}
		}
	}
	return false
}

func (c *Checker) checkDiskSpace(item config.DiskCheckItem) config.DiskSpace {
	space := config.DiskSpace{
		Path:    item.Path,
		MinFree: item.MinFreeGB * gbBytes,
	}

	checkPath := item.Path
	info, err := os.Stat(checkPath)
	if err != nil {
		if !os.IsNotExist(err) {
			space.Error = fmt.Sprintf("failed to stat path: %v", err)
			return space
		}
		parent := filepath.Dir(checkPath)
		if parent == checkPath {
			space.Error = fmt.Sprintf("path does not exist: %s", checkPath)
			return space
		}
		checkPath = parent
	} else if !info.IsDir() {
		checkPath = filepath.Dir(checkPath)
	} else {
		// path exists and is a directory, check it directly
	}

	usage, err := disk.Usage(checkPath)
	if err != nil {
		space.Error = fmt.Sprintf("failed to get disk usage: %v", err)
		return space
	}

	space.Path = checkPath
	space.Total = usage.Total
	space.Free = usage.Free
	space.Used = usage.Used
	space.UsedPercent = usage.UsedPercent
	space.Filesystem = usage.Fstype
	space.IsSufficient = space.Free >= space.MinFree

	return space
}
