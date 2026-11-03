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
package common

import (
	"testing"
	"time"
)

// constant.go only declares constants and one type (no executable statements),
// so this test mainly guarantees the package is exercised and compiles.
func TestConstantsAreExpected(t *testing.T) {
	if DefaultReportsBufCapacity != 5 {
		t.Fatalf("DefaultReportsBufCapacity = %d, want 5", DefaultReportsBufCapacity)
	}
	if FreshContainerMinMetricsResolution != 10*time.Second {
		t.Fatalf("unexpected FreshContainerMinMetricsResolution")
	}
	if DefaultSecureKubeletMetricsPort != 10250 {
		t.Fatalf("unexpected DefaultSecureKubeletMetricsPort")
	}
	if QosBe != QosLevel("BE") || QosLs != "LS" || QosHLS != "HLS" || QosNone != "None" {
		t.Fatalf("unexpected QoS level constants")
	}
	if CgroupDriverSystemd != "systemd" || CgroupDriverCgroupfs != "cgroupfs" {
		t.Fatalf("unexpected cgroup driver constants")
	}
}
