/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

func TestSetStatus(t *testing.T) {
	policy := &kaev1.KAEPolicy{}
	SetStatus(policy, kaev1.Ready, "test-namespace")

	if policy.Status.State != kaev1.Ready {
		t.Errorf("Expected state %v, got %v", kaev1.Ready, policy.Status.State)
	}
	if policy.Status.Namespace != "test-namespace" {
		t.Errorf("Expected namespace %s, got %s", "test-namespace", policy.Status.Namespace)
	}
}

func TestImage(t *testing.T) {
	tests := []struct {
		name    string
		spec    interface{}
		envName string
		envVal  string
		want    string
		err     bool
	}{
		{
			name: "DriverSpec with repository and version",
			spec: &kaev1.DriverSpec{
				Repository: "repo.com",
				Image:      "driver",
				Version:    "v1.0.0",
			},
			want: "repo.com/driver:v1.0.0",
			err:  false,
		},
		{
			name: "DriverSpec with repository and version",
			spec: &kaev1.DriverSpec{
				Repository: "repo.com",
				Image:      "driver",
				Version:    "sha256:123456",
			},
			want: "repo.com/driver@sha256:123456",
			err:  false,
		},
		{
			name:    "DevicePluginSpec with environment variable",
			spec:    &kaev1.DevicePluginSpec{},
			envName: "KAE_DEVICE_PLUGIN_IMAGE",
			envVal:  "env/device-plugin:latest",
			want:    "env/device-plugin:latest",
			err:     false,
		},
		{
			name: "Unknown spec type",
			spec: struct{}{},
			want: "",
			err:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envName != "" {
				err := os.Setenv(tt.envName, tt.envVal)
				if err != nil {
					t.Errorf("failed to set environment variable: %v", err)
				}
				defer os.Unsetenv(tt.envName)
			}
			got, err := Image(tt.spec)
			if (err != nil) != tt.err {
				t.Errorf("Image() error = %v, wantErr %v", err, tt.err)
				return
			}
			if got != tt.want {
				t.Errorf("Image() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestImagePullPolicy(t *testing.T) {
	tests := []struct {
		name       string
		pullPolicy string
		want       corev1.PullPolicy
	}{
		{
			name:       "Always",
			pullPolicy: "Always",
			want:       corev1.PullAlways,
		},
		{
			name:       "Never",
			pullPolicy: "Never",
			want:       corev1.PullNever,
		},
		{
			name:       "IfNotPresent",
			pullPolicy: "IfNotPresent",
			want:       corev1.PullIfNotPresent,
		},
		{
			name:       "Default",
			pullPolicy: "",
			want:       corev1.PullIfNotPresent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ImagePullPolicy(tt.pullPolicy); got != tt.want {
				t.Errorf("ImagePullPolicy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsDriverEnabled(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		d    kaev1.DriverSpec
		want bool
	}{
		{
			name: "Enabled is true",
			d:    kaev1.DriverSpec{Enabled: &trueVal},
			want: true,
		},
		{
			name: "Enabled is false",
			d:    kaev1.DriverSpec{Enabled: &falseVal},
			want: false,
		},
		{
			name: "Enabled is nil",
			d:    kaev1.DriverSpec{Enabled: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDriverEnabled(tt.d); got != tt.want {
				t.Errorf("IsDriverEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsDriverEnabledCleanup(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		d    kaev1.DriverSpec
		want bool
	}{
		{
			name: "EnabledCleanup is true",
			d:    kaev1.DriverSpec{EnabledCleanup: &trueVal},
			want: true,
		},
		{
			name: "EnabledCleanup is false",
			d:    kaev1.DriverSpec{EnabledCleanup: &falseVal},
			want: false,
		},
		{
			name: "EnabledCleanup is nil",
			d:    kaev1.DriverSpec{EnabledCleanup: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDriverEnabledCleanup(tt.d); got != tt.want {
				t.Errorf("IsDriverEnabledCleanup() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsDevicePluginEnabledHPRE(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.DevicePluginSpec
		want bool
	}{
		{
			name: "EnabledHPRE is true",
			p:    kaev1.DevicePluginSpec{EnabledHPRE: &trueVal},
			want: true,
		},
		{
			name: "EnabledHPRE is false",
			p:    kaev1.DevicePluginSpec{EnabledHPRE: &falseVal},
			want: false,
		},
		{
			name: "EnabledHPRE is nil",
			p:    kaev1.DevicePluginSpec{EnabledHPRE: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDevicePluginEnabledHPRE(tt.p); got != tt.want {
				t.Errorf("IsDevicePluginEnabledHPRE() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsDevicePluginEnabledSEC(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.DevicePluginSpec
		want bool
	}{
		{
			name: "EnabledSEC is true",
			p:    kaev1.DevicePluginSpec{EnabledSEC: &trueVal},
			want: true,
		},
		{
			name: "EnabledSEC is false",
			p:    kaev1.DevicePluginSpec{EnabledSEC: &falseVal},
			want: false,
		},
		{
			name: "EnabledSEC is nil",
			p:    kaev1.DevicePluginSpec{EnabledSEC: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDevicePluginEnabledSEC(tt.p); got != tt.want {
				t.Errorf("IsDevicePluginEnabledSEC() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsDevicePluginEnabledZIP(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.DevicePluginSpec
		want bool
	}{
		{
			name: "EnabledZIP is true",
			p:    kaev1.DevicePluginSpec{EnabledZIP: &trueVal},
			want: true,
		},
		{
			name: "EnabledZIP is false",
			p:    kaev1.DevicePluginSpec{EnabledZIP: &falseVal},
			want: false,
		},
		{
			name: "EnabledZIP is nil",
			p:    kaev1.DevicePluginSpec{EnabledZIP: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDevicePluginEnabledZIP(tt.p); got != tt.want {
				t.Errorf("IsDevicePluginEnabledZIP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsPSAEnabled(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.PSASpec
		want bool
	}{
		{
			name: "Enabled is true",
			p:    kaev1.PSASpec{Enabled: &trueVal},
			want: true,
		},
		{
			name: "Enabled is false",
			p:    kaev1.PSASpec{Enabled: &falseVal},
			want: false,
		},
		{
			name: "Enabled is nil",
			p:    kaev1.PSASpec{Enabled: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsPSAEnabled(tt.p); got != tt.want {
				t.Errorf("IsPSAEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsOperatorEnabledHPRE(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.OperatorSpec
		want bool
	}{
		{
			name: "EnabledHPRE is true",
			p:    kaev1.OperatorSpec{EnabledHPRE: &trueVal},
			want: true,
		},
		{
			name: "EnabledHPRE is false",
			p:    kaev1.OperatorSpec{EnabledHPRE: &falseVal},
			want: false,
		},
		{
			name: "EnabledHPRE is nil",
			p:    kaev1.OperatorSpec{EnabledHPRE: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsOperatorEnabledHPRE(tt.p); got != tt.want {
				t.Errorf("IsOperatorEnabledHPRE() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsOperatorEnabledSEC(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.OperatorSpec
		want bool
	}{
		{
			name: "EnabledSEC is true",
			p:    kaev1.OperatorSpec{EnabledSEC: &trueVal},
			want: true,
		},
		{
			name: "EnabledSEC is false",
			p:    kaev1.OperatorSpec{EnabledSEC: &falseVal},
			want: false,
		},
		{
			name: "EnabledSEC is nil",
			p:    kaev1.OperatorSpec{EnabledSEC: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsOperatorEnabledSEC(tt.p); got != tt.want {
				t.Errorf("IsOperatorEnabledSEC() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsOperatorEnabledZIP(t *testing.T) {
	trueVal := true
	falseVal := false
	tests := []struct {
		name string
		p    kaev1.OperatorSpec
		want bool
	}{
		{
			name: "EnabledZIP is true",
			p:    kaev1.OperatorSpec{EnabledZIP: &trueVal},
			want: true,
		},
		{
			name: "EnabledZIP is false",
			p:    kaev1.OperatorSpec{EnabledZIP: &falseVal},
			want: false,
		},
		{
			name: "EnabledZIP is nil",
			p:    kaev1.OperatorSpec{EnabledZIP: nil},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsOperatorEnabledZIP(tt.p); got != tt.want {
				t.Errorf("IsOperatorEnabledZIP() = %v, want %v", got, tt.want)
			}
		})
	}
}
