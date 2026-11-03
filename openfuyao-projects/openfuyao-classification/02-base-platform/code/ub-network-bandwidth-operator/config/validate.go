/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package config

import (
	"fmt"
)

const (
	priority0      uint32 = 0
	priority1      uint32 = 1
	minBandwidth   uint32 = 1
	maxBandwidth   uint32 = 50
	maxBandwidthLo uint32 = 25
	minItems       int    = 1
	maxItems       int    = 2
)

// Validate checks the bandwidth configuration constraints.
func Validate(cfg *BandwidthConfig) error {
	if cfg == nil {
		return fmt.Errorf("config is nil")
	}

	if len(cfg.Bandwidths) < minItems || len(cfg.Bandwidths) > maxItems {
		return fmt.Errorf("bandwidths must contain %d to %d items, got %d", minItems, maxItems, len(cfg.Bandwidths))
	}

	seenPriorities := make(map[uint32]bool)
	for i, item := range cfg.Bandwidths {
		if item.Priority != priority0 && item.Priority != priority1 {
			return fmt.Errorf("bandwidths[%d].priority must be either %d or %d, got %d", i, priority0, priority1, item.Priority)
		}
		if seenPriorities[item.Priority] {
			return fmt.Errorf("bandwidths[%d].priority %d is duplicated", i, item.Priority)
		}
		seenPriorities[item.Priority] = true
	}

	if len(cfg.Bandwidths) == 1 {
		bw := cfg.Bandwidths[0].Bandwidth
		if bw < minBandwidth || bw > maxBandwidthLo {
			return fmt.Errorf("bandwidths[0].bandwidth must be between %d and %d, got %d", minBandwidth, maxBandwidthLo, bw)
		}
	} else {
		a, b := cfg.Bandwidths[0].Bandwidth, cfg.Bandwidths[1].Bandwidth
		if a < minBandwidth || b < minBandwidth {
			return fmt.Errorf("bandwidths[].bandwidth must be >= %d", minBandwidth)
		}
		ok := (a <= maxBandwidthLo && b <= maxBandwidth) || (b <= maxBandwidthLo && a <= maxBandwidth)
		if !ok {
			return fmt.Errorf("one bandwidth must be <= %d and the other <= %d, got %d and %d", maxBandwidthLo, maxBandwidth, a, b)
		}
	}

	return nil
}
