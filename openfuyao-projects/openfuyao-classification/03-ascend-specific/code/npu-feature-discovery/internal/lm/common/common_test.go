// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package common

import (
	"reflect"
	"testing"
)

func TestListLabels(t *testing.T) {
	tests := []struct {
		name     string
		labelers Labelers
		want     Labels
	}{
		{
			name: "test list labels",
			labelers: Labelers{
				Labels{"label1": "value1", "label2": "value2"},
				Labels{"label3": "value3", "label4": "value4"},
			},
			want: Labels{"label1": "value1",
				"label2": "value2",
				"label3": "value3",
				"label4": "value4"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.labelers.Labels(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Labels() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMergeSuccess(t *testing.T) {
	type args struct {
		labelers Labelers
	}
	tests := []struct {
		name string
		args args
		want Labeler
	}{
		{
			name: "test merge",
			args: args{
				labelers: Labelers{
					Labels{"label1": "value1"},
					Labels{"label2": "value2"},
				},
			},
			want: Labelers{Labels{"label1": "value1"}, Labels{"label2": "value2"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Merge(tt.args.labelers...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("merge() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNPUDeviceType(t *testing.T) {
	tests := []struct {
		name  string
		label string
		want  string
		ok    bool
	}{
		{name: "Ascend 310", label: NpuDeviceLabelPrefix + "d100.present", want: "huawei-Ascend310", ok: true},
		{name: "Ascend 310P", label: NpuDeviceLabelPrefix + "d500.present", want: "huawei-Ascend310P", ok: true},
		{name: "Ascend 910", label: NpuDeviceLabelPrefix + "d801.present", want: "huawei-Ascend910", ok: true},
		{name: "Ascend 910 d802", label: NpuDeviceLabelPrefix + "d802.present", want: "huawei-Ascend910", ok: true},
		{name: "Ascend 910 d803", label: NpuDeviceLabelPrefix + "d803.present", want: "huawei-Ascend910", ok: true},
		{name: "Ascend A5 d806", label: NpuDeviceLabelPrefix + "d806.present", want: "huawei-AscendA5", ok: true},
		{name: "unknown", label: "other-label", want: "", ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := NPUDeviceType(tt.label)
			if got != tt.want || ok != tt.ok {
				t.Errorf("NPUDeviceType(%q) = (%q, %t), want (%q, %t)", tt.label, got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestIsServerDeviceID(t *testing.T) {
	tests := []struct {
		name     string
		server   string
		deviceID string
		want     bool
	}{
		{name: "Atlas 800 device", server: "Atlas_800", deviceID: "0x27", want: true},
		{name: "Atlas 800 A2 device", server: "Atlas_800_A2", deviceID: "0x38", want: true},
		{name: "Atlas 900 A2 PoD device", server: "Atlas_900_A2_PoD", deviceID: "0x34", want: true},
		{name: "Atlas 200T A2 device", server: "Atlas_200T_A2_Box16", deviceID: "0x52", want: true},
		{name: "Atlas 300T device", server: "Atlas_300T", deviceID: "0x03", want: true},
		{name: "Atlas 300T A2 device", server: "Atlas_300T_A2", deviceID: "0x11", want: true},
		{name: "Atlas 200T A3 device", server: "Atlas_200T_A3_Box8", deviceID: "0xb1", want: true},
		{name: "Atlas 800I A3 device", server: "Atlas_800I_A3", deviceID: "0xb3", want: true},
		{name: "SuperPod device", server: "Atlas_900_A3_SuperPod", deviceID: "0xb2", want: true},
		{name: "wrong device", server: "Atlas_800", deviceID: "0xb2", want: false},
		{name: "unknown server", server: "Unknown", deviceID: "0x27", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsServerDeviceID(tt.server, tt.deviceID); got != tt.want {
				t.Errorf("IsServerDeviceID(%q, %q) = %t, want %t", tt.server, tt.deviceID, got, tt.want)
			}
		})
	}
}

func TestResolveCardServerName(t *testing.T) {
	tests := []struct {
		name     string
		identity PCIIdentity
		archKey  string
		want     string
	}{
		{
			name:     "d100 x86",
			identity: PCIIdentity{"0x19e5", "0xd100", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "A300-3010",
		},
		{
			name:     "d100 arm",
			identity: PCIIdentity{"0x19e5", "0xd100", "0x0200", "0x0100"},
			archKey:  "aarch64",
			want:     "A300-3000",
		},
		{
			name:     "A300T 9000",
			identity: PCIIdentity{"0x19e5", "0xd801", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "A300T-9000",
		},
		{
			name:     "A900T",
			identity: PCIIdentity{"0x19e5", "0xd802", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "A900T",
		},
		{
			name:     "A900T subsystem",
			identity: PCIIdentity{"0x19e5", "0xd802", "0x19e5", "0x3403"},
			archKey:  "aarch64",
			want:     "A900T",
		},
		{
			name:     "A300t A2",
			identity: PCIIdentity{"0x19e5", "0xd802", "0x19e5", "0x6000"},
			archKey:  "x86_64",
			want:     "A300t-a2",
		},
		{
			name:     "A300i pro",
			identity: PCIIdentity{"0x19e5", "0xd500", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "A300i-pro",
		},
		{
			name:     "A300i duo",
			identity: PCIIdentity{"0x19e5", "0xd500", "0x0200", "0x0110"},
			archKey:  "x86_64",
			want:     "A300i-duo",
		},
		{
			name:     "A300i A2",
			identity: PCIIdentity{"0x19e5", "0xd802", "0x19e5", "0x4000"},
			archKey:  "x86_64",
			want:     "A300i-a2",
		},
		{
			name:     "A200i A2 d105",
			identity: PCIIdentity{"0x19e5", "0xd105", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "A200i-a2",
		},
		{
			name:     "A200i A2 d107",
			identity: PCIIdentity{"0x19e5", "0xd107", "0x0000", "0x0000"},
			archKey:  "x86_64",
			want:     "A200i-a2",
		},
		{
			name:     "Atlas 800I A2",
			identity: PCIIdentity{"0x19e5", "0xd802", "0x19e5", "0x3005"},
			archKey:  "x86_64",
			want:     "Atlas 800I A2",
		},
		{
			name:     "SuperPod",
			identity: PCIIdentity{"0x19e5", "0xd803", "0x19e5", "0x3001"},
			archKey:  "x86_64",
			want:     "Atlas 900 A3 SuperPod",
		},
		{
			name:     "Atlas 800I A3",
			identity: PCIIdentity{"0x19e5", "0xd803", "0x19e5", "0x3003"},
			archKey:  "x86_64",
			want:     "Atlas 800I A3",
		},
		{
			name:     "unknown architecture",
			identity: PCIIdentity{"0x19e5", "0xd801", "0x0200", "0x0100"},
			archKey:  "UNKNOWN",
			want:     "",
		},
		{
			name:     "unknown identity",
			identity: PCIIdentity{"0x19e5", "0xffff", "0x0200", "0x0100"},
			archKey:  "x86_64",
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ResolveCardServerName(tt.identity, tt.archKey); got != tt.want {
				t.Errorf("ResolveCardServerName(%+v, %q) = %q, want %q", tt.identity, tt.archKey, got, tt.want)
			}
		})
	}
}

func TestResolveCardServerNameA900TPCIIdentities(t *testing.T) {
	identities := []PCIIdentity{
		{"0x19e5", "0xd802", "0x0200", "0x0100"},
		{"0x19e5", "0xd802", "0x19e5", "0x3000"},
		{"0x19e5", "0xd802", "0x19e5", "0x3001"},
		{"0x19e5", "0xd802", "0x19e5", "0x3003"},
		{"0x19e5", "0xd802", "0x19e5", "0x3400"},
		{"0x19e5", "0xd802", "0x19e5", "0x3401"},
		{"0x19e5", "0xd802", "0x19e5", "0x3402"},
		{"0x19e5", "0xd802", "0x19e5", "0x3403"},
	}

	for _, identity := range identities {
		t.Run(identity.SubsystemVendor+"/"+identity.SubsystemDevice, func(t *testing.T) {
			if got := ResolveCardServerName(identity, "x86_64"); got != "A900T" {
				t.Errorf("ResolveCardServerName(%+v) = %q, want A900T", identity, got)
			}
		})
	}
}

func TestServerRoCEBandwidth(t *testing.T) {
	tests := []struct {
		name   string
		server string
		want   string
		ok     bool
	}{
		{name: "Atlas 800I A2", server: "Atlas 800I A2", want: "200Gbps", ok: true},
		{name: "SuperPod", server: "Atlas 900 A3 SuperPod", want: "400Gbps", ok: true},
		{name: "Atlas 800I A3", server: "Atlas 800I A3", want: "400Gbps", ok: true},
		{name: "unknown", server: "Unknown", want: "", ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ServerRoCEBandwidth(tt.server)
			if got != tt.want || ok != tt.ok {
				t.Errorf("ServerRoCEBandwidth(%q) = (%q, %t), want (%q, %t)", tt.server, got, ok, tt.want, tt.ok)
			}
		})
	}
}
