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

package plugin

import (
	"testing"
)

func TestEncodeContainerDevices_NPUOnly(t *testing.T) {
	tests := []struct {
		name, expected string
		input          ContainerDevices
	}{
		{"Single_NPU_Device_Shortening", "1,NPU-A,NPU,32768,100,1,vir01:",
			ContainerDevices{{1, "NPU-A", "NPU", 32768, 100, 1, "vir01", "soft"}}},
		{"Multiple_NPU_Devices_Mixed_Types", "2,NPU-A,NPU,1024,50,1,vir01:3,NPU-B,NPU,512,25,2,vir01:",
			ContainerDevices{{2, "NPU-A", "NPU", 1024, 50, 1, "vir01", "hard"}, {3, "NPU-B", "NPU", 512, 25, 2, "vir01", "hard"}}},
		{"Device_Type_Not_NPU_But_Index_Large", "100,NPU-X,NPU,99999999,1,12345,vir01:",
			ContainerDevices{{100, "NPU-X", "NPU", 99999999, 1, 12345, "vir01", "soft"}}},
		{"Empty_Input", "", ContainerDevices{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EncodeContainerDevices(tt.input)
			if result != tt.expected {
				t.Errorf("Test '%s' failed\nExpected: %v\n  Actual: %v", tt.name, tt.expected, result)
			}
		})
	}
}
