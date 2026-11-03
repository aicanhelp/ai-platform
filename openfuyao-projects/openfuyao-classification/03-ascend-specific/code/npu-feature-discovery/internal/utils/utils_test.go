// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package utils

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/urfave/cli/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	kubernetes2 "openfuyao.com/npu-feature-discovery/internal/kubernetes"
	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestGetEnvVarEmpty(t *testing.T) {
	const envName = "TEST_ENV_EMPTY"
	patch := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == envName {
			return ""
		}
		return ""
	})
	defer patch.Reset()
	_, err := GetEnvVar(envName)
	if err == nil {
		t.Error("GetEnvVar() expected error when variable is empty")
	}
}

func TestGetEnvVarSet(t *testing.T) {
	const envName = "TEST_ENV_SET"
	const want = "some-value"
	patch := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == envName {
			return want
		}
		return ""
	})
	defer patch.Reset()
	got, err := GetEnvVar(envName)
	if err != nil {
		t.Fatalf("GetEnvVar() err = %v", err)
	}
	if got != want {
		t.Errorf("GetEnvVar() = %q, want %q", got, want)
	}
}

func TestGetNodeNoK8s(t *testing.T) {
	patch := gomonkey.ApplyFunc(kubernetes2.GetKubernetesClient,
		func() (kubernetes.Interface, error) { return nil, errors.New(`kubernetes does not exist`) })
	defer patch.Reset()
	type args struct {
		c *cli.Context
	}
	tests := []struct {
		name                string
		argsGetNodeNoK8s    args
		want0GetNodeNoK8s   kubernetes.Interface
		want1GetNodeNoK8s   *corev1.Node
		wantErrGetNodeNoK8s bool
	}{
		{
			name:                "no k8s error",
			argsGetNodeNoK8s:    args{},
			want0GetNodeNoK8s:   nil,
			want1GetNodeNoK8s:   nil,
			wantErrGetNodeNoK8s: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := GetNode(tt.argsGetNodeNoK8s.c)
			if (err != nil) != tt.wantErrGetNodeNoK8s {
				t.Errorf("getNode() error = %v, wantErr %v", err, tt.wantErrGetNodeNoK8s)
				return
			}
			if !reflect.DeepEqual(got, tt.want0GetNodeNoK8s) {
				t.Errorf("getNode() got = %v, want %v", got, tt.want0GetNodeNoK8s)
			}
			if !reflect.DeepEqual(got1, tt.want1GetNodeNoK8s) {
				t.Errorf("getNode() got1 = %v, want %v", got1, tt.want1GetNodeNoK8s)
			}
		})
	}
}

func TestGetNodeNoEnv(t *testing.T) {
	patch0 := gomonkey.ApplyFunc(kubernetes2.GetKubernetesClient,
		func() (kubernetes.Interface, error) { return &fake.Clientset{}, nil })
	defer patch0.Reset()
	patch1 := gomonkey.ApplyFunc(GetEnvVar,
		func(_ string) (string, error) { return "", errors.New("env var not found") })
	defer patch1.Reset()
	type args struct {
		c *cli.Context
	}
	tests := []struct {
		name                string
		argsGetNodeNoEnv    args
		want0GetNodeNoEnv   kubernetes.Interface
		want1GetNodeNoEnv   *corev1.Node
		wantErrGetNodeNoEnv bool
	}{
		{
			name:                "no env error",
			argsGetNodeNoEnv:    args{},
			want0GetNodeNoEnv:   nil,
			want1GetNodeNoEnv:   nil,
			wantErrGetNodeNoEnv: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := GetNode(tt.argsGetNodeNoEnv.c)
			if (err != nil) != tt.wantErrGetNodeNoEnv {
				t.Errorf("getNode() error = %v, wantErr %v", err, tt.wantErrGetNodeNoEnv)
				return
			}
			if !reflect.DeepEqual(got, tt.want0GetNodeNoEnv) {
				t.Errorf("getNode() got = %v, want %v", got, tt.want0GetNodeNoEnv)
			}
			if !reflect.DeepEqual(got1, tt.want1GetNodeNoEnv) {
				t.Errorf("getNode() got1 = %v, want %v", got1, tt.want1GetNodeNoEnv)
			}
		})
	}
}

func TestGetNodeFail(t *testing.T) {
	nodeName := `test-node`
	fakeClient := fake.NewSimpleClientset(&corev1.Node{})
	patch0 := gomonkey.ApplyFunc(kubernetes2.GetKubernetesClient,
		func() (kubernetes.Interface, error) { return fakeClient, nil })
	defer patch0.Reset()
	patch1 := gomonkey.ApplyFunc(GetEnvVar,
		func(_ string) (string, error) { return nodeName, nil })
	defer patch1.Reset()
	type args struct {
		ctxGetNodFail *cli.Context
	}
	tests := []struct {
		name              string
		argsGetNodFail    args
		want0GetNodFail   kubernetes.Interface
		want1GetNodFail   *corev1.Node
		wantErrGetNodFail bool
	}{
		{
			name:              "get node fail",
			argsGetNodFail:    args{&cli.Context{}},
			want0GetNodFail:   nil,
			want1GetNodFail:   nil,
			wantErrGetNodFail: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := GetNode(tt.argsGetNodFail.ctxGetNodFail)
			if (err != nil) != tt.wantErrGetNodFail {
				t.Errorf("getNode() error = %v, wantErr %v", err, tt.wantErrGetNodFail)
				return
			}
			if !reflect.DeepEqual(got, tt.want0GetNodFail) {
				t.Errorf("getNode() got = %v, want %v", got, tt.want0GetNodFail)
			}
			if !reflect.DeepEqual(got1, tt.want1GetNodFail) {
				t.Errorf("getNode() got1 = %v, want %v", got1, tt.want1GetNodFail)
			}
		})
	}
}

func TestGetNodeSuccess(t *testing.T) {
	nodeName := `test-node`
	fakeClient := fake.NewSimpleClientset(&corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
	})
	patch0 := gomonkey.ApplyFunc(GetEnvVar,
		func(_ string) (string, error) { return nodeName, nil })
	defer patch0.Reset()
	patch1 := gomonkey.ApplyFunc(kubernetes2.GetKubernetesClient,
		func() (kubernetes.Interface, error) { return fakeClient, nil })
	defer patch1.Reset()
	type args struct {
		ctxGetNodeSuccess *cli.Context
	}
	tests := []struct {
		name                  string
		argsGetNodeSuccess    args
		want0GetNodeSuccess   kubernetes.Interface
		want1GetNodeSuccess   *corev1.Node
		wantErrGetNodeSuccess bool
	}{
		{
			name:                "success",
			argsGetNodeSuccess:  args{&cli.Context{}},
			want0GetNodeSuccess: fakeClient,
			want1GetNodeSuccess: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
				},
			},
			wantErrGetNodeSuccess: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := GetNode(tt.argsGetNodeSuccess.ctxGetNodeSuccess)
			if (err != nil) != tt.wantErrGetNodeSuccess {
				t.Errorf("getNode() error = %v, wantErr %v", err, tt.wantErrGetNodeSuccess)
				return
			}
			if !reflect.DeepEqual(got, tt.want0GetNodeSuccess) {
				t.Errorf("getNode() got = %v, want %v", got, tt.want0GetNodeSuccess)
			}
			if !reflect.DeepEqual(got1, tt.want1GetNodeSuccess) {
				t.Errorf("getNode() got1 = %v, want %v", got1, tt.want1GetNodeSuccess)
			}
		})
	}
}

func TestHasNPUDeviceLabelsNoLabels(t *testing.T) {
	if HasNPUDeviceLabels(map[string]string{}) {
		t.Error("HasNPUDeviceLabels(empty) should be false")
	}
}

func TestHasNPUDeviceLabelsNoMatch(t *testing.T) {
	labels := map[string]string{
		"other-label":        "value",
		"kubernetes.io/arch": "amd64",
		"some.npu/key":       "true",
	}
	if HasNPUDeviceLabels(labels) {
		t.Error("HasNPUDeviceLabels() should be false when no known NPU device label is present")
	}
}

func TestHasNPUDeviceLabelsWrongValue(t *testing.T) {
	key := common.NpuDeviceLabelPrefix + "d100.present"
	labels := map[string]string{key: "false"}
	if HasNPUDeviceLabels(labels) {
		t.Errorf("HasNPUDeviceLabels() should be false when value is %q", "false")
	}
	labels[key] = "other"
	if HasNPUDeviceLabels(labels) {
		t.Errorf("HasNPUDeviceLabels() should be false when value is %q", "other")
	}
}

func TestHasNPUDeviceLabelsMatch(t *testing.T) {
	key := common.NpuDeviceLabelPrefix + "d100.present"
	labels := map[string]string{
		"other": "x",
		key:     common.NpuDeviceLabelValue,
	}
	if !HasNPUDeviceLabels(labels) {
		t.Errorf("HasNPUDeviceLabels() should be true for key %q with value %q",
			key, common.NpuDeviceLabelValue)
	}
}

func TestGetArchKey(t *testing.T) {
	const unknownVal = "UNKNOWN"
	tests := []struct {
		name string
		node *corev1.Node
		want string
	}{
		{
			name: "arm64 returns aarch64",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"kubernetes.io/arch": "arm64"},
				},
			},
			want: "aarch64",
		},
		{
			name: "amd64 returns x86_64",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"kubernetes.io/arch": "amd64"},
				},
			},
			want: "x86_64",
		},
		{
			name: "missing arch label returns UNKNOWN",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"other": "value"},
				},
			},
			want: unknownVal,
		},
		{
			name: "empty labels returns UNKNOWN",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
			},
			want: unknownVal,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getArchKey(tt.node); got != tt.want {
				t.Errorf("getArchKey() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetCardInfoFromPath(t *testing.T) {
	devicesPath := t.TempDir()
	writePCIDevice(t, devicesPath, "device0", "0x1200", "0x19e5", "0xd803", "0x19e5", "0x3003")

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"kubernetes.io/arch": "amd64"}}}
	got, err := getCardInfo(node, devicesPath)
	if err != nil {
		t.Fatalf("getCardInfo() error = %v", err)
	}
	if got != "Atlas 800I A3" {
		t.Errorf("getCardInfo() = %q, want %q", got, "Atlas 800I A3")
	}
}

func TestGetCardInfoPrefersAcceleratorOverBridge(t *testing.T) {
	devicesPath := t.TempDir()
	writePCIDevice(t, devicesPath, "bridge0", "0x0604", "0x19e5", "0xd801", "0x0200", "0x0100")
	writePCIDevice(t, devicesPath, "card0", "0x1200", "0x19e5", "0xd803", "0x19e5", "0x3003")

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"kubernetes.io/arch": "amd64"}}}
	got, err := getCardInfo(node, devicesPath)
	if err != nil {
		t.Fatalf("getCardInfo() error = %v", err)
	}
	if got != "Atlas 800I A3" {
		t.Errorf("getCardInfo() = %q, want accelerator result %q", got, "Atlas 800I A3")
	}
}

func TestGetCardInfoReturnsBridgeWhenNoAcceleratorExists(t *testing.T) {
	devicesPath := t.TempDir()
	writePCIDevice(t, devicesPath, "bridge0", "0x0604", "0x19e5", "0xd802", "0x19e5", "0x3002")

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"kubernetes.io/arch": "amd64"}}}
	got, err := getCardInfo(node, devicesPath)
	if err != nil {
		t.Fatalf("getCardInfo() error = %v", err)
	}
	if got != "Atlas 800I A2" {
		t.Errorf("getCardInfo() = %q, want bridge result %q", got, "Atlas 800I A2")
	}
}

func TestGetCardInfoReturnsErrorForMissingDevicesPath(t *testing.T) {
	devicesPath := filepath.Join(t.TempDir(), "missing")
	got, err := getCardInfo(&corev1.Node{}, devicesPath)
	if err == nil {
		t.Fatal("getCardInfo() error = nil, want an error")
	}
	if got != unknown {
		t.Errorf("getCardInfo() = %q, want %q", got, unknown)
	}
}

func writePCIDevice(t *testing.T, devicesPath, deviceName, class, vendor, device, subsystemVendor, subsystemDevice string) {
	t.Helper()
	devicePath := filepath.Join(devicesPath, deviceName)
	if err := os.Mkdir(devicePath, 0755); err != nil {
		t.Fatalf("Mkdir(%q) error = %v", devicePath, err)
	}

	files := map[string]string{
		"class":            class + "\n",
		"vendor":           vendor + "\n",
		"device":           device + "\n",
		"subsystem_vendor": subsystemVendor + "\n",
		"subsystem_device": subsystemDevice + "\n",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(devicePath, name), []byte(content), 0644); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", name, err)
		}
	}

}
