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

package resource

import (
	"context"
	"fmt"
	"time"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
)

type UrmaResourceManager struct {
	resourceManager
}

var _ Manager = (*UrmaResourceManager)(nil)

func NewUrmaResourceManagers(resources *Resources) ([]Manager, error) {
	deviceMap, err := NewDeviceMap(resources)
	if len(deviceMap) == 0 || err != nil {
		return nil, fmt.Errorf("error building device map: %v", err)
	}

	klog.Infof("deviceMap: %v", deviceMap)
	var rms []Manager
	for resourceName, devices := range deviceMap {
		if len(devices) == 0 {
			continue
		}
		r := &UrmaResourceManager{
			resourceManager: resourceManager{
				name:      resourceName,
				resources: *resources,
				devices:   devices,
			},
		}
		klog.Infof("UrmaResourceManager.resourceManager, r.name: %v, r.resources: %v, r.devices%v", r.name, r.resources, r.devices)
		rms = append(rms, r)
	}

	return rms, nil
}

func (r *UrmaResourceManager) SyncDevices(interval time.Duration, stop <-chan interface{}, updated chan struct{}) error {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	lastDevices := r.Devices()

	for {
		select {
		case <-stop:
			return nil
		case <-ticker.C:
			newDevices, err := r.scanAndUpdateDevices()
			if err != nil {
				klog.Errorf("Failed to scan devices: %v", err)
				continue
			}
			klog.Info("scan and update devices, will check if changed.")

			if !lastDevices.Equal(newDevices) {
				lastDevices = newDevices
				select {
				case updated <- struct{}{}:
					klog.Info("Device list changed, send update to kubelet.")
				default:
					klog.Info("Update channel busy, will send on next ListAndWatch.")
				}
			}
		}
	}
}

func (r *UrmaResourceManager) UpdateDevices(devices Devices) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.devices = devices
}

func (r *UrmaResourceManager) scanAndUpdateDevices() (Devices, error) {
	klog.Infof("Scanning devices for '%s'", r.Resource())
	var newDevices = make(Devices)
	switch r.Resource() {
	case utils.ResourceName:
		sharedDevs, err := urma.GetSharedDevice()
		if err != nil {
			return nil, fmt.Errorf("failed to get Shared devices: %v", err)
		}
		for _, sharedDev := range sharedDevs {
			newDevices[sharedDev.Name] = &sharedDev
		}
	default:
	}

	klog.Infof("Updated devices for '%s': %d", r.Resource(), len(newDevices))
	r.UpdateDevices(newDevices)
	return newDevices, nil
}

func (r *UrmaResourceManager) Recover() error {
	const (
		maxRetryRounds  = 3
		pollInterval    = 3 * time.Second
		pollTimeout     = 30 * time.Second
		retryRoundDelay = 5 * time.Second
	)

	for retryRound := 0; retryRound < maxRetryRounds; retryRound++ {
		if retryRound > 0 {
			klog.Infof("Starting recovery retry round: %d/%d", retryRound+1, maxRetryRounds)
			time.Sleep(retryRoundDelay)
		}

		deviceIds, err := utils.GetAllDeviceIdsFromKubeletCheckpoint(utils.KubeletCheckpoint)
		if err != nil {
			klog.Errorf("Failed to get device ids from kubelet checkpoint: %v", err)
			if retryRound == maxRetryRounds-1 {
				return fmt.Errorf("failed to get device ids after %d retries: %w", maxRetryRounds, err)
			}
			continue
		}
		if len(deviceIds) == 0 {
			klog.Infof("No device ids found in kubelet checkpoint")
			return nil
		}

		klog.Infof("Found %d devices to recover in round %d: %v", len(deviceIds), retryRound+1, deviceIds)

		successCount := 0
		for _, deviceId := range deviceIds {
			err := wait.PollUntilContextTimeout(context.Background(), pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
				_, err = urmasdk.UbsAllocateDevice(deviceId)
				if err != nil {
					klog.Errorf("failed to recover device %s from kubelet checkpoint, err: %v", deviceId, err)
					return false, nil
				}
				klog.Infof("Successfully recovered device %s", deviceId)
				return true, nil
			})

			if err == nil {
				successCount++
			}
		}

		if successCount == len(deviceIds) {
			klog.Infof("All %d devices recovered successfully in round %d", successCount, retryRound+1)
			return nil
		}

		klog.Warningf("Round %d recovery incomplete: %d/%d devices recovered, will retry", retryRound+1, successCount, len(deviceIds))
	}

	return fmt.Errorf("failed to recover devices after %d retries", maxRetryRounds)
}
