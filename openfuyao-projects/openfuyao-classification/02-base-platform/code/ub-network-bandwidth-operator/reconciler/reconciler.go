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
)

// Controller reconciles the current config file with UBSE bandwidth state.
type Controller struct {
	manager BandwidthManager
	config  string
}

// NewController creates a Controller for the given manager and config path.
func NewController(manager BandwidthManager, configPath string) *Controller {
	return &Controller{
		manager: manager,
		config:  configPath,
	}
}

// Reconcile applies the current config file state to the bandwidth manager.
func (r *Controller) Reconcile() error {
	klog.Infof("Starting bandwidth reconcile, configPath=%q", r.config)
	cfg, err := config.ParseConfig(r.config)
	if err != nil {
		if config.IsInvalidConfigError(err) {
			klog.Errorf("parse config: %v", err)
			// No retry for invalid user config content
			return nil
		}
		return fmt.Errorf("parse config: %w", err)
	}

	if cfg != nil {
		if err := config.Validate(cfg); err != nil {
			klog.Errorf("validate config: %v", err)
			// No retry for validation errors
			return nil
		}
	}

	currentCfg, err := r.getCurrentConfig()
	if err != nil {
		if errors.Is(err, errResetBeforeApplyRequired) {
			return r.resetAndApplyConfig(cfg, err)
		}
		return fmt.Errorf("query current UBSE config: %w", err)
	}

	if cfg == nil {
		return r.reconcileDeletedConfig(currentCfg)
	}

	return r.reconcileDesiredConfig(cfg, currentCfg)
}

func (r *Controller) getCurrentConfig() (*config.BandwidthConfig, error) {
	items, err := r.manager.BandwidthGet()
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		klog.Info("No existing UBSE bandwidth configuration found")
		return nil, nil
	}
	currentCfg := &config.BandwidthConfig{Bandwidths: append([]config.BandwidthItem(nil), items...)}
	klog.Infof("Current UBSE bandwidth configuration: %+v", currentCfg.Bandwidths)
	return currentCfg, nil
}

func (r *Controller) reconcileDeletedConfig(currentCfg *config.BandwidthConfig) error {
	if currentCfg == nil {
		klog.Info("Bandwidth config is absent and current UBSE config is empty, skipping reset")
		return nil
	}
	klog.Info("Bandwidth config is absent, clearing current UBSE bandwidth configuration")
	if err := r.manager.BandwidthReset(); err != nil {
		return fmt.Errorf("bandwidth reset: %w", err)
	}
	klog.Info("Current UBSE bandwidth configuration cleared because config is absent")
	return nil
}

func (r *Controller) reconcileDesiredConfig(cfg, currentCfg *config.BandwidthConfig) error {
	if bandwidthConfigsEqual(cfg, currentCfg) {
		klog.Info("Bandwidth configuration matches current UBSE config, skipping set/reset operation")
		return nil
	}

	if currentCfg != nil {
		klog.Info("Bandwidth configuration differs from current UBSE config, clearing current configuration before applying new one")
		if err := r.manager.BandwidthReset(); err != nil {
			return fmt.Errorf("bandwidth reset: %w", err)
		}
		klog.Info("Current UBSE bandwidth configuration cleared before applying new one")
	}

	return r.applyConfig(cfg)
}

func (r *Controller) applyConfig(cfg *config.BandwidthConfig) error {
	klog.Infof("Applying bandwidth configuration: %+v", cfg.Bandwidths)
	if err := r.manager.BandwidthSet(cfg.Bandwidths); err != nil {
		if errors.Is(err, errResetBeforeApplyRequired) {
			return r.resetAndRetryBandwidthSet(cfg, err)
		}
		return fmt.Errorf("bandwidth set: %w", err)
	}
	klog.Info("Bandwidth configuration applied successfully")
	return nil
}

func (r *Controller) resetAndApplyConfig(cfg *config.BandwidthConfig, err error) error {
	klog.Warningf("Current UBSE bandwidth configuration needs reset before apply: %v", err)
	if resetErr := r.manager.BandwidthReset(); resetErr != nil {
		return fmt.Errorf("bandwidth reset: %w", resetErr)
	}
	if cfg == nil {
		klog.Info("Current UBSE bandwidth configuration cleared because config is absent")
		return nil
	}
	klog.Info("Current UBSE bandwidth configuration cleared, applying desired config")
	return r.applyConfig(cfg)
}

func (r *Controller) resetAndRetryBandwidthSet(cfg *config.BandwidthConfig, err error) error {
	klog.Warningf(
		"Reset before apply is required, resetting and retrying set immediately: %v",
		err,
	)
	if resetErr := r.manager.BandwidthReset(); resetErr != nil {
		return fmt.Errorf("bandwidth reset: %w", resetErr)
	}
	klog.Info("Previous bandwidth configuration cleared before immediate retry")
	if retryErr := r.manager.BandwidthSet(cfg.Bandwidths); retryErr != nil {
		return fmt.Errorf("retry bandwidth set: %w", retryErr)
	}
	klog.Info("Bandwidth configuration applied successfully after immediate retry")
	return nil
}

func bandwidthConfigsEqual(a, b *config.BandwidthConfig) bool {
	if a == nil || b == nil {
		return a == b
	}
	return configItemsEqual(a.Bandwidths, b.Bandwidths)
}

func configItemsEqual(a, b []config.BandwidthItem) bool {
	normalizedA, ok := normalizeConfigItems(a)
	if !ok {
		return false
	}
	normalizedB, ok := normalizeConfigItems(b)
	if !ok {
		return false
	}
	if len(normalizedA) != len(normalizedB) {
		return false
	}
	for priority, bandwidthA := range normalizedA {
		bandwidthB, ok := normalizedB[priority]
		if !ok || bandwidthA != bandwidthB {
			return false
		}
	}
	return true
}

func normalizeConfigItems(items []config.BandwidthItem) (map[uint32]uint32, bool) {
	normalized := make(map[uint32]uint32, len(items)+1)
	for _, item := range items {
		if _, ok := normalized[item.Priority]; ok {
			return nil, false
		}
		normalized[item.Priority] = item.Bandwidth
	}
	if len(items) == 1 {
		if items[0].Priority == priority0 {
			normalized[priority1] = defaultBandwidth
		}
		if items[0].Priority == priority1 {
			normalized[priority0] = defaultBandwidth
		}
	}
	return normalized, true
}
