package utils

import "fmt"

// ValidateNPUMetricsInRange checks that NPU utilization metrics in the leader route endpoint
// are within [0, 100]. Returns an error if SelectedRoute or its leader NPU data is nil.
func ValidateNPUMetricsInRange(record CompletedRecord) error {
	if record.SelectedRoute == nil {
		return fmt.Errorf("selectedRoute is nil, no NPU data available")
	}
	npu := record.SelectedRoute.Leader.NPU
	if npu == nil {
		return fmt.Errorf("leader route endpoint NPU data is nil")
	}
	if npu.AICoreUtilizationAvg < 0 || npu.AICoreUtilizationAvg > 100 {
		return fmt.Errorf("npu aiCoreUtilizationAvg %f is out of range [0, 100]", npu.AICoreUtilizationAvg)
	}
	if npu.HBMUtilizationAvg < 0 || npu.HBMUtilizationAvg > 100 {
		return fmt.Errorf("npu hbmUtilizationAvg %f is out of range [0, 100]", npu.HBMUtilizationAvg)
	}
	if npu.HBMBandwidthUtilizationAvg < 0 || npu.HBMBandwidthUtilizationAvg > 100 {
		return fmt.Errorf("npu hbmBandwidthUtilizationAvg %f is out of range [0, 100]", npu.HBMBandwidthUtilizationAvg)
	}
	return nil
}

// ValidateNPUPresent checks that NPU data in the leader route endpoint is present
// and at least one device is reported. Returns an error if SelectedRoute or leader NPU is nil.
func ValidateNPUPresent(record CompletedRecord) error {
	if record.SelectedRoute == nil {
		return fmt.Errorf("selectedRoute is nil, no NPU data available")
	}
	npu := record.SelectedRoute.Leader.NPU
	if npu == nil {
		return fmt.Errorf("leader route endpoint NPU data is nil")
	}
	if npu.DataStatus != "Present" {
		return fmt.Errorf("npu data status is %s, expected Present", npu.DataStatus)
	}
	if npu.DeviceCount < 1 {
		return fmt.Errorf("npu device count %d is less than 1", npu.DeviceCount)
	}
	return nil
}
