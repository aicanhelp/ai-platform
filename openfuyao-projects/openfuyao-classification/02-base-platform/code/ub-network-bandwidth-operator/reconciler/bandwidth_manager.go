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

// Package reconciler coordinates bandwidth configuration apply and reset operations.
package reconciler

import (
	"errors"
	"fmt"

	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/sdk/ubse"
)

// BandwidthManager applies and resets bandwidth configuration.
type BandwidthManager interface {
	BandwidthSet(items []config.BandwidthItem) error
	BandwidthReset() error
	BandwidthGet() ([]config.BandwidthItem, error)
}

type bandwidthManagerImpl struct{}

var errResetBeforeApplyRequired = errors.New("reset before apply required")

var (
	ubseBandwidthSet   = ubse.BandwidthSet
	ubseBandwidthGet   = ubse.BandwidthGet
	ubseBandwidthReset = ubse.BandwidthReset
)

// NewBandwidthManager creates the default UBSE-backed bandwidth manager.
func NewBandwidthManager() *bandwidthManagerImpl {
	return &bandwidthManagerImpl{}
}

const (
	priority0        uint32 = 0
	priority1        uint32 = 1
	defaultBandwidth uint32 = 50
)

func (bm *bandwidthManagerImpl) BandwidthSet(items []config.BandwidthItem) error {
	bandwidthItems := bm.buildSDKBandwidthItems(items)
	klog.Infof("Setting bandwidth with items: %+v", bandwidthItems)
	if err := ubseBandwidthSet(bandwidthItems); err != nil {
		err = wrapBandwidthSetError(err)
		if errors.Is(err, errResetBeforeApplyRequired) {
			klog.Warningf("Bandwidth config exists in UBSE: %v", err)
			if bm.bandwidthConfigExistsAndMatches(bandwidthItems) {
				klog.Info("Existing UBSE bandwidth configuration matches desired config, treating set as successful")
				return nil
			}
			return err
		}
		klog.Errorf("Set bandwidth operation failed: %v", err)
		return err
	}
	klog.Info("Set bandwidth operation completed successfully")
	return nil
}

func (bm *bandwidthManagerImpl) bandwidthConfigExistsAndMatches(desired []ubse.BandwidthItem) bool {
	current, err := ubseBandwidthGet()
	if err != nil {
		klog.Warningf("Query existing UBSE bandwidth configuration failed: %v", err)
		return false
	}
	if bandwidthItemsEqual(current, desired) {
		return true
	}
	klog.Warningf("Existing UBSE bandwidth configuration differs from desired config, current=%+v, desired=%+v",
		current, desired)
	return false
}

func wrapBandwidthSetError(err error) error {
	if ubse.IsDaemonErrorCode(err, ubse.BandwidthConfigExistsCode) {
		return fmt.Errorf("%w: %w", errResetBeforeApplyRequired, err)
	}
	return err
}

func (bm *bandwidthManagerImpl) BandwidthReset() error {
	klog.Info("Starting bandwidth reset operation to clear current QoS configuration")
	if err := ubseBandwidthReset(); err != nil {
		klog.Errorf("Bandwidth reset operation failed while clearing QoS configuration: %v", err)
		return err
	}
	klog.Info("Bandwidth reset operation completed, QoS configuration cleared")
	return nil
}

func (bm *bandwidthManagerImpl) BandwidthGet() ([]config.BandwidthItem, error) {
	klog.Info("Querying current UBSE bandwidth configuration")
	items, err := ubseBandwidthGet()
	if err != nil {
		if isBandwidthGetResetRequired(err) {
			klog.Warningf("Current UBSE bandwidth configuration needs reset before apply: %v", err)
			return nil, fmt.Errorf("%w: %w", errResetBeforeApplyRequired, err)
		}
		klog.Errorf("Query current UBSE bandwidth configuration failed: %v", err)
		return nil, err
	}

	configItems := make([]config.BandwidthItem, 0, len(items))
	for _, item := range items {
		configItems = append(configItems, config.BandwidthItem{
			Priority:  item.Priority,
			Bandwidth: item.Bandwidth,
		})
	}
	klog.Infof("Current UBSE bandwidth configuration: %+v", configItems)
	return configItems, nil
}

func isBandwidthGetResetRequired(err error) bool {
	return ubse.IsDaemonErrorCode(err, ubse.BandwidthGetOVSInterfaceMissingCode) ||
		ubse.IsDaemonErrorCode(err, ubse.BandwidthGetOVSInterfaceInvalidCode)
}

func (bm *bandwidthManagerImpl) buildSDKBandwidthItems(items []config.BandwidthItem) []ubse.BandwidthItem {
	bandwidthItems := make([]ubse.BandwidthItem, 0, len(items)+1)
	for _, item := range items {
		bandwidthItems = append(bandwidthItems, ubse.BandwidthItem{
			Priority:  item.Priority,
			Bandwidth: item.Bandwidth,
		})
	}
	return bm.appendDefaultBandwidth(items, bandwidthItems)
}

func (bm *bandwidthManagerImpl) appendDefaultBandwidth(
	items []config.BandwidthItem,
	bandwidthItems []ubse.BandwidthItem,
) []ubse.BandwidthItem {
	if len(bandwidthItems) != 1 {
		return bandwidthItems
	}
	if items[0].Priority == priority0 {
		return append(bandwidthItems, ubse.BandwidthItem{Priority: priority1, Bandwidth: defaultBandwidth})
	}
	if items[0].Priority == priority1 {
		return append(bandwidthItems, ubse.BandwidthItem{Priority: priority0, Bandwidth: defaultBandwidth})
	}
	return bandwidthItems
}

func bandwidthItemsEqual(a, b []ubse.BandwidthItem) bool {
	if len(a) != len(b) {
		return false
	}
	itemsByPriority := make(map[uint32]uint32, len(a))
	for _, item := range a {
		if _, ok := itemsByPriority[item.Priority]; ok {
			return false
		}
		itemsByPriority[item.Priority] = item.Bandwidth
	}
	seenPriorities := make(map[uint32]struct{}, len(b))
	for _, item := range b {
		if _, ok := seenPriorities[item.Priority]; ok {
			return false
		}
		seenPriorities[item.Priority] = struct{}{}
		bandwidth, ok := itemsByPriority[item.Priority]
		if !ok || bandwidth != item.Bandwidth {
			return false
		}
	}
	return true
}
