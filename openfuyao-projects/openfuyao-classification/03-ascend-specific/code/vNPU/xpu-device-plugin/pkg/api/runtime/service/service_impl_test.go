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

package service

import (
	"context"
	"crypto/md5"
	"fmt"
	"io"
	"io/fs"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	pid1        = 2776170
	pid2        = 2930470
	processMem0 = 40
	processMem1 = 60
)

func isSliceEqual(a, b interface{}) bool {
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false
	}
	sliceA := reflect.ValueOf(a)
	sliceB := reflect.ValueOf(b)
	if sliceA.Len() != sliceB.Len() {
		return false
	}
	for i := 0; i < sliceA.Len(); i++ {
		if !reflect.DeepEqual(sliceA.Index(i).Interface(), sliceB.Index(i).Interface()) {
			return false
		}
	}
	return true
}

func TestReadProcsFile(t *testing.T) {
	realHostPids, err := readProcsFile("./test.cgroup.procs")
	expectedHostPids := []int{2776170, 2930470}
	if err == nil && isSliceEqual(realHostPids, expectedHostPids) {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestReadStatusFile(t *testing.T) {
	realPids, err := readStatusFile("./test.status")
	expectedPids := fmt.Sprintf("%-11s %-11s", "2776170", "1")
	if err == nil && realPids == expectedPids {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestParseDockerCgroupPath(t *testing.T) {
	cgroupPath := "/kubepods/besteffort/pod9b322980-f435-4759-90c7-fa2e7bc36b76" +
		"/53392426fb325c27357920728175c0d18065d633486ee855441d2517ec29a188"
	realPodId, realContainerId, err := parseDockerCgroupPath(cgroupPath)
	expectedPodId := "9b322980-f435-4759-90c7-fa2e7bc36b76"
	expectedContainerId := "53392426fb325c27357920728175c0d18065d633486ee855441d2517ec29a188"
	if err == nil && realPodId == expectedPodId && realContainerId == expectedContainerId {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestParseCgroupPath(t *testing.T) {
	cgroupPath := "/kubepods.slice/kubepods-besteffort.slice" +
		"/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice" +
		"/cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope"
	realPodId, realContainerId, err := parseCgroupPath(cgroupPath)
	expectedPodId := "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
	expectedContainerId := "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5"
	if err == nil && realPodId == expectedPodId && realContainerId == expectedContainerId {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestGetContainerName(t *testing.T) {
	cgroupPath := "/kubepods.slice/kubepods-besteffort.slice" +
		"/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice" +
		"/cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope"

	patch := gomonkey.ApplyFunc(util.ListPods, func(opts metav1.ListOptions) (*v1.PodList, error) {
		pod := v1.Pod{}
		pod.UID = "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
		pod.Status.Phase = v1.PodRunning
		cs := v1.ContainerStatus{
			ContainerID: "containerd://4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			Ready:       true,
			Name:        "container0",
		}
		pod.Status.ContainerStatuses = []v1.ContainerStatus{cs}
		podList := v1.PodList{Items: []v1.Pod{pod}}
		return &podList, nil
	})
	defer patch.Reset()

	realPodId, realContainerName, err := getContainerName(cgroupPath)
	expectedPodId := "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
	expectedContainerName := "container0"
	if err == nil && realPodId == expectedPodId && realContainerName == expectedContainerName {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func fileCheckSum(filename string) string {
	f, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer f.Close()

	h := md5.New()
	if _, err = io.Copy(h, f); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func TestWritePidsConfig(t *testing.T) {
	pidMaps := fmt.Sprintf("%-11s %-11s,%-11s %-11s", "2776170", "1", "2930470", "7")
	err := writePidsConfig("./real.pids.config", pidMaps)
	if err == nil && fileCheckSum("./real.pids.config") == fileCheckSum("./expected.pids.config") {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestReadPidsConfig(t *testing.T) {
	pids, err := readPidsConfig("./expected.pids.config")
	expectedHostPids := []uint32{2776170, 2930470}
	if err == nil && isSliceEqual(pids, expectedHostPids) {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestCleanDestroyedPodDir(t *testing.T) {
	patchGetPodDirNames := gomonkey.ApplyFunc(getPodDirNames, func() ([]string, error) {
		return []string{"d164bc1f-6368-4869-b7bd-7b741647f7c4", "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"}, nil
	})
	defer patchGetPodDirNames.Reset()

	patchListPods := gomonkey.ApplyFunc(util.ListPods, func(opts metav1.ListOptions) (*v1.PodList, error) {
		pod := v1.Pod{}
		pod.UID = "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
		podList := v1.PodList{Items: []v1.Pod{pod}}
		return &podList, nil
	})
	defer patchListPods.Reset()

	err := cleanDestroyedPodDir()
	if err == nil {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func makeXpuDeviceAndVxpuDevices() (map[string]*types.XPUDevice, types.VxpuDevices) {
	xpuDevices := make(map[string]*types.XPUDevice)
	device := types.XPUDevice{
		Index:          0,
		Id:             "id0",
		Type:           "NVIDIA",
		Count:          20,
		MemoryTotal:    200,
		Health:         true,
		VxpuDeviceList: types.VxpuDevices{},
	}
	xpuDevices["id0"] = &device

	vxpuDevices := types.VxpuDevices{}
	dev0 := types.VxpuDevice{
		Id:              "id0-0",
		XpuId:           "id0",
		PodUID:          "pod0",
		ContainerName:   "container0",
		VxpuMemoryLimit: 1 * 1024,
		VxpuCoreLimit:   5,
	}
	dev1 := types.VxpuDevice{
		Id:              "id0-1",
		XpuId:           "id0",
		PodUID:          "pod1",
		ContainerName:   "container1",
		VxpuMemoryLimit: 1 * 1024,
		VxpuCoreLimit:   5,
	}
	vxpuDevices = append(vxpuDevices, dev0)
	vxpuDevices = append(vxpuDevices, dev1)
	return xpuDevices, vxpuDevices
}

func TestSetVxpuDevices(t *testing.T) {
	xpuDevices, vxpuDevices := makeXpuDeviceAndVxpuDevices()
	uidToProcessMap := make(map[string]map[uint32]*types.ProcessUsage)
	processMap := make(map[uint32]*types.ProcessUsage)
	p0 := types.ProcessUsage{ProcessMem: uint64(processMem0), ProcessCoreUtilization: 15}
	p1 := types.ProcessUsage{ProcessMem: uint64(processMem1), ProcessCoreUtilization: 20}
	processMap[pid1] = &p0
	processMap[pid2] = &p1
	uidToProcessMap["id0"] = processMap

	pSet := make(map[string][]uint32)
	var pids1 []uint32
	pids1 = append(pids1, pid1)
	pSet["pod0/container0"] = pids1
	var pids2 []uint32
	pids2 = append(pids2, pid2)
	pSet["pod1/container1"] = pids2

	expectedVxpuDevices := types.VxpuDevices{}
	expectedDev0 := types.VxpuDevice{
		Id:                    "id0-0",
		XpuId:                 "id0",
		PodUID:                "pod0",
		ContainerName:         "container0",
		VxpuMemoryUsed:        40,
		VxpuMemoryUtilization: 20,
		VxpuCoreUtilization:   15,
		VxpuMemoryLimit:       1 * 1024,
		VxpuCoreLimit:         5,
	}
	expectedDev1 := types.VxpuDevice{
		Id:                    "id0-1",
		XpuId:                 "id0",
		PodUID:                "pod1",
		ContainerName:         "container1",
		VxpuMemoryUsed:        60,
		VxpuMemoryUtilization: 30,
		VxpuCoreUtilization:   20,
		VxpuMemoryLimit:       1 * 1024,
		VxpuCoreLimit:         5,
	}
	expectedVxpuDevices = append(expectedVxpuDevices, expectedDev0)
	expectedVxpuDevices = append(expectedVxpuDevices, expectedDev1)
	xpuDevices = setVxpuDevices(vxpuDevices, xpuDevices, uidToProcessMap, pSet)
	if isSliceEqual(xpuDevices["id0"].VxpuDeviceList, expectedVxpuDevices) &&
		xpuDevices["id0"].MemoryUsed == 100 &&
		xpuDevices["id0"].MemoryUtilization == 50 {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestStripContainerIDPrefix(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "containerd prefix",
			input:    "containerd://abc123",
			expected: "abc123",
		},
		{
			name:     "docker prefix",
			input:    "docker://def456",
			expected: "def456",
		},
		{
			name:     "no prefix",
			input:    "ghi789",
			expected: "ghi789",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := stripContainerIDPrefix(tc.input)
			if result == tc.expected {
				t.Log("test succeed")
			} else {
				t.Errorf("test failed: expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestIsCgroupV2(t *testing.T) {
	testCases := []struct {
		name       string
		mockStat   func(name string) (fs.FileInfo, error)
		expectedV2 bool
	}{
		{
			name:       "cgroup v2 - controllers file exists",
			mockStat:   func(name string) (fs.FileInfo, error) { return nil, nil },
			expectedV2: true,
		},
		{
			name:       "cgroup v1 - controllers file not exists",
			mockStat:   func(name string) (fs.FileInfo, error) { return nil, os.ErrNotExist },
			expectedV2: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patchStat := gomonkey.ApplyFunc(os.Stat, tc.mockStat)
			defer patchStat.Reset()

			result := isCgroupV2()
			if result != tc.expectedV2 {
				t.Errorf("test failed: expected %v, got %v", tc.expectedV2, result)
			} else {
				t.Log("test succeed")
			}
		})
	}
}

func TestFindCgroupPathByBaseDir(t *testing.T) {
	testCases := []struct {
		name          string
		baseDir       string
		podUID        string
		containerID   string
		mockWalkDir   func(root string, fn fs.WalkDirFunc) error
		mockReadDir   func(name string) ([]fs.DirEntry, error)
		expectedFound bool
	}{
		{
			name:        "v1 found pod subdir with containerd scope",
			baseDir:     "/sys/fs/cgroup/pids",
			podUID:      "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID: "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/pids/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return []fs.DirEntry{
					&mockDirEntry{name: "cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope", isDir: true},
				}, nil
			},
			expectedFound: true,
		},
		{
			name:        "v1 found podUID:containerID same-level directory",
			baseDir:     "/sys/fs/cgroup/pids",
			podUID:      "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID: "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/pids/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db:4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db:4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return nil, nil
			},
			expectedFound: true,
		},
		{
			name:        "v2 found pod subdir with containerd scope",
			baseDir:     "/sys/fs/cgroup",
			podUID:      "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID: "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return []fs.DirEntry{
					&mockDirEntry{name: "cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope", isDir: true},
				}, nil
			},
			expectedFound: true,
		},
		{
			name:        "v2 found podUID:containerID same-level directory",
			baseDir:     "/sys/fs/cgroup",
			podUID:      "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID: "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db:4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db:4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return nil, nil
			},
			expectedFound: true,
		},
		{
			name:        "not found",
			baseDir:     "/sys/fs/cgroup/pids",
			podUID:      "not_found",
			containerID: "not_found",
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return nil, nil
			},
			expectedFound: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patchWalkDir := gomonkey.ApplyFunc(filepath.WalkDir, tc.mockWalkDir)
			defer patchWalkDir.Reset()

			patchReadDir := gomonkey.ApplyFunc(os.ReadDir, tc.mockReadDir)
			defer patchReadDir.Reset()

			result, err := findCgroupPathByBaseDir(tc.baseDir, tc.podUID, tc.containerID)
			if tc.expectedFound {
				if err != nil {
					t.Errorf("test failed: expected no error, got %v", err)
				} else if result == "" {
					t.Error("test failed: expected non-empty path")
				} else {
					t.Log("test succeed")
				}
			} else {
				if err == nil {
					t.Error("test failed: expected error")
				} else {
					t.Log("test succeed")
				}
			}
		})
	}
}

func TestFindCgroupPath(t *testing.T) {
	testCases := []struct {
		name           string
		podUID         string
		containerID    string
		mockIsCgroupV2 bool
		mockWalkDir    func(root string, fn fs.WalkDirFunc) error
		mockReadDir    func(name string) ([]fs.DirEntry, error)
		expectedFound  bool
	}{
		{
			name:           "cgroup v1 dispatch",
			podUID:         "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID:    "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockIsCgroupV2: false,
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/pids/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return []fs.DirEntry{
					&mockDirEntry{name: "cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope", isDir: true},
				}, nil
			},
			expectedFound: true,
		},
		{
			name:           "cgroup v2 dispatch",
			podUID:         "1aba0c04_4fd5_4691_95ab_dbc6e90fa1db",
			containerID:    "4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockIsCgroupV2: true,
			mockWalkDir: func(root string, fn fs.WalkDirFunc) error {
				podDirPath := "/sys/fs/cgroup/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice"
				fn(podDirPath, &mockDirEntry{name: "kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice", isDir: true}, nil)
				return nil
			},
			mockReadDir: func(name string) ([]fs.DirEntry, error) {
				return []fs.DirEntry{
					&mockDirEntry{name: "cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope", isDir: true},
				}, nil
			},
			expectedFound: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patchIsCgroupV2 := gomonkey.ApplyFunc(isCgroupV2, func() bool { return tc.mockIsCgroupV2 })
			defer patchIsCgroupV2.Reset()

			patchWalkDir := gomonkey.ApplyFunc(filepath.WalkDir, tc.mockWalkDir)
			defer patchWalkDir.Reset()

			patchReadDir := gomonkey.ApplyFunc(os.ReadDir, tc.mockReadDir)
			defer patchReadDir.Reset()

			result, err := findCgroupPath(tc.podUID, tc.containerID)
			if tc.expectedFound {
				if err != nil {
					t.Errorf("test failed: expected no error, got %v", err)
				} else if result == "" {
					t.Error("test failed: expected non-empty path")
				} else {
					t.Log("test succeed")
				}
			} else {
				if err == nil {
					t.Error("test failed: expected error")
				} else {
					t.Log("test succeed")
				}
			}
		})
	}
}

type mockDirEntry struct {
	name  string
	isDir bool
}

func (m *mockDirEntry) Name() string {
	return m.name
}

func (m *mockDirEntry) IsDir() bool {
	return m.isDir
}

func (m *mockDirEntry) Type() fs.FileMode {
	return 0
}

func (m *mockDirEntry) Info() (fs.FileInfo, error) {
	return nil, nil
}

func TestGetPodContainerPids(t *testing.T) {
	testCases := []struct {
		name          string
		podUID        string
		containerID   string
		mockFindPath  func(string, string) (string, error)
		mockReadProcs func(string) ([]int, error)
		expectedPids  int
	}{
		{
			name:        "success get pids",
			podUID:      "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db",
			containerID: "containerd://4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			mockFindPath: func(string, string) (string, error) {
				return "/sys/fs/cgroup/pids/test", nil
			},
			mockReadProcs: func(string) ([]int, error) {
				return []int{2776170, 2930470}, nil
			},
			expectedPids: 2,
		},
		{
			name:        "find cgroup path failed",
			podUID:      "test",
			containerID: "test",
			mockFindPath: func(string, string) (string, error) {
				return "", fmt.Errorf("not found")
			},
			mockReadProcs: func(string) ([]int, error) {
				return []int{}, nil
			},
			expectedPids: 0,
		},
		{
			name:        "read procs file failed",
			podUID:      "test",
			containerID: "test",
			mockFindPath: func(string, string) (string, error) {
				return "/sys/fs/cgroup/pids/test", nil
			},
			mockReadProcs: func(string) ([]int, error) {
				return nil, fmt.Errorf("read error")
			},
			expectedPids: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			patchFind := gomonkey.ApplyFunc(findCgroupPath, tc.mockFindPath)
			defer patchFind.Reset()

			patchRead := gomonkey.ApplyFunc(readProcsFile, tc.mockReadProcs)
			defer patchRead.Reset()

			result := getPodContainerPids(tc.podUID, tc.containerID)
			if len(result) == tc.expectedPids {
				t.Log("test succeed")
			} else {
				t.Errorf("test failed: expected %d pids, got %d", tc.expectedPids, len(result))
			}
		})
	}
}

func TestGetVxpus(t *testing.T) {
	patch := gomonkey.ApplyFunc(util.ListPods, func(opts metav1.ListOptions) (*v1.PodList, error) {
		pod := v1.Pod{}
		pod.UID = "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
		pod.Status.Phase = v1.PodRunning
		pod.Annotations = make(map[string]string)
		pod.Annotations["huawei.com/vnpu-ids-new"] = "0,NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e,ASCEND,1024,10,0,vir01:"

		cs := v1.ContainerStatus{
			ContainerID: "containerd://4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5",
			Ready:       true,
			Name:        "container0",
		}
		pod.Status.ContainerStatuses = []v1.ContainerStatus{cs}

		cont := v1.Container{
			Name: "container0",
		}
		cont.Resources.Limits = make(map[v1.ResourceName]resource.Quantity)
		cont.Resources.Limits["huawei.com/vnpu-number"] = resource.MustParse("1")
		pod.Spec.Containers = []v1.Container{cont}

		podList := v1.PodList{Items: []v1.Pod{pod}}
		return &podList, nil
	})
	defer patch.Reset()

	patchGetVxpuLimit := gomonkey.ApplyFunc(util.GetVxpuLimit, func(device types.ContainerDevice, xpuDevice map[string]*types.XPUDevice) (int64, int64) {
		return 1024, 10
	})
	defer patchGetVxpuLimit.Reset()

	patchGetPids := gomonkey.ApplyFunc(getPodContainerPids, func(string, string) []uint32 {
		return []uint32{2776170, 2930470}
	})
	defer patchGetPids.Reset()

	xpuDevice := make(map[string]*types.XPUDevice)
	device := types.XPUDevice{
		Id:          "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
		Type:        "ASCEND",
		MemoryTotal: 8192,
	}
	xpuDevice["NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e"] = &device

	vxpuDevices, pSet, err := getVxpus(xpuDevice)
	if err != nil {
		t.Errorf("test failed: unexpected error %v", err)
		return
	}

	if len(vxpuDevices) != 1 {
		t.Errorf("test failed: expected 1 vxpu device, got %d", len(vxpuDevices))
		return
	}

	if len(pSet) != 1 {
		t.Errorf("test failed: expected 1 pod in pSet, got %d", len(pSet))
		return
	}

	t.Log("test succeed")
}

func TestGetUidToProcessMap(t *testing.T) {
	testCases := []struct {
		name            string
		isHardMode      bool
		mockGetXPUUsage func(int32, int32) (types.DeviceUsageInfo, map[uint32]*types.ProcessUsage, error)
		expectedEmpty   bool
	}{
		{
			name:       "hard mode",
			isHardMode: true,
			mockGetXPUUsage: func(int32, int32) (types.DeviceUsageInfo, map[uint32]*types.ProcessUsage, error) {
				return types.DeviceUsageInfo{}, nil, nil
			},
			expectedEmpty: true,
		},
		{
			name:       "soft mode success",
			isHardMode: false,
			mockGetXPUUsage: func(int32, int32) (types.DeviceUsageInfo, map[uint32]*types.ProcessUsage, error) {
				processMap := make(map[uint32]*types.ProcessUsage)
				p0 := types.ProcessUsage{ProcessMem: 40, ProcessCoreUtilization: 15}
				processMap[2776170] = &p0
				return types.DeviceUsageInfo{CoreUtil: 50, PowerUsage: 100, Temperature: 60}, processMap, nil
			},
			expectedEmpty: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			originalMode := config.NodeMode
			if tc.isHardMode {
				config.NodeMode = "hard"
			} else {
				config.NodeMode = "soft"
			}
			defer func() { config.NodeMode = originalMode }()

			patchGetXPUUsage := gomonkey.ApplyFunc(xpu.GetXPUUsage, tc.mockGetXPUUsage)
			defer patchGetXPUUsage.Reset()

			xpuDevices := make(map[string]*types.XPUDevice)
			device := types.XPUDevice{
				Id:     "id0",
				Index:  0,
				Type:   "ASCEND",
				Health: true,
			}
			xpuDevices["id0"] = &device

			result, err := getUidToProcessMap(xpuDevices, 10)
			if err != nil {
				t.Errorf("test failed: unexpected error %v", err)
				return
			}

			if tc.expectedEmpty {
				if len(result) != 0 {
					t.Errorf("test failed: expected empty map, got %d entries", len(result))
				} else {
					t.Log("test succeed")
				}
			} else {
				if len(result) != 1 {
					t.Errorf("test failed: expected 1 entry, but got %d", len(result))
				} else {
					t.Log("test succeed")
				}
			}
		})
	}
}

type fakeFileInfo struct {
	name  string
	isDir bool
}

func (f fakeFileInfo) Name() string       { return f.name }
func (f fakeFileInfo) Size() int64        { return 0 }
func (f fakeFileInfo) Mode() os.FileMode  { return 0 }
func (f fakeFileInfo) ModTime() time.Time { return time.Time{} }
func (f fakeFileInfo) IsDir() bool        { return f.isDir }
func (f fakeFileInfo) Sys() interface{}   { return nil }

type mockListener struct{}

func (m *mockListener) Accept() (net.Conn, error) {
	return nil, fmt.Errorf("accept error")
}

func (m *mockListener) Close() error   { return nil }
func (m *mockListener) Addr() net.Addr { return &net.UnixAddr{Name: "test", Net: "unix"} }

func TestGetPidMaps(t *testing.T) {
	patch := gomonkey.ApplyFunc(readStatusFile,
		func(file string) (string, error) {
			return "2776170      1", nil
		})
	defer patch.Reset()

	result := getPidMaps([]int{2776170, 2930470})
	expected := "2776170      1,2776170      1"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestGetPidMaps_ReadError(t *testing.T) {
	patch := gomonkey.ApplyFunc(readStatusFile,
		func(file string) (string, error) {
			return "", fmt.Errorf("read error")
		})
	defer patch.Reset()

	result := getPidMaps([]int{2776170})
	if result != "" {
		t.Errorf("expected empty result on error, got: %s", result)
	}
}

func TestGetPodDirNames(t *testing.T) {
	patch := gomonkey.ApplyFunc(filepath.Walk,
		func(root string, fn filepath.WalkFunc) error {
			_ = fn("/etc/xpu", fakeFileInfo{name: "xpu", isDir: true}, nil)
			_ = fn("/etc/xpu/pod1", fakeFileInfo{name: "pod1", isDir: true}, nil)
			_ = fn("/etc/xpu/file1", fakeFileInfo{name: "file1", isDir: false}, nil)
			return nil
		})
	defer patch.Reset()

	names, err := getPodDirNames()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if len(names) != 1 || names[0] != "pod1" {
		t.Errorf("expected [pod1], got: %v", names)
	}
}

func TestGetPodDirNames_WalkError(t *testing.T) {
	patch := gomonkey.ApplyFunc(filepath.Walk,
		func(root string, fn filepath.WalkFunc) error {
			return fmt.Errorf("walk error")
		})
	defer patch.Reset()

	_, err := getPodDirNames()
	if err == nil {
		t.Error("expected error from walk, got nil")
	}
}

func TestGetPids(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readProcsFile,
		func(file string) ([]int, error) {
			return []int{2776170}, nil
		})
	patches.ApplyFunc(getPidMaps,
		func(hostPids []int) string { return "2776170      1" })
	patches.ApplyFunc(getContainerName,
		func(cgroupPath string) (string, string, error) {
			return "pod-id", "container0", nil
		})
	patches.ApplyFunc(writePidsConfig,
		func(path, pidMaps string) error { return nil })

	svc := PidsServiceServerImpl{}
	resp, err := svc.GetPids(context.Background(),
		&GetPidsRequest{CgroupPath: "/test"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if resp.EncodedPids != "2776170      1" {
		t.Errorf("expected encoded pids, got: %s", resp.EncodedPids)
	}
}

func TestGetPids_ReadProcsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readProcsFile,
		func(file string) ([]int, error) {
			return nil, fmt.Errorf("procs read error")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetPids(context.Background(),
		&GetPidsRequest{CgroupPath: "/test"})
	if err == nil {
		t.Error("expected error from readProcsFile, got nil")
	}
}

func TestGetPids_ContainerNameError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readProcsFile,
		func(file string) ([]int, error) {
			return []int{2776170}, nil
		})
	patches.ApplyFunc(getPidMaps,
		func(hostPids []int) string { return "2776170      1" })
	patches.ApplyFunc(getContainerName,
		func(cgroupPath string) (string, string, error) {
			return "", "", fmt.Errorf("container not found")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetPids(context.Background(),
		&GetPidsRequest{CgroupPath: "/test"})
	if err == nil {
		t.Error("expected error from getContainerName, got nil")
	}
}

func TestGetPids_WriteConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(readProcsFile,
		func(file string) ([]int, error) {
			return []int{2776170}, nil
		})
	patches.ApplyFunc(getPidMaps,
		func(hostPids []int) string { return "2776170      1" })
	patches.ApplyFunc(getContainerName,
		func(cgroupPath string) (string, string, error) {
			return "pod-id", "container0", nil
		})
	patches.ApplyFunc(writePidsConfig,
		func(path, pidMaps string) error {
			return fmt.Errorf("write error")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetPids(context.Background(),
		&GetPidsRequest{CgroupPath: "/test"})
	if err == nil {
		t.Error("expected error from writePidsConfig, got nil")
	}
}

func TestGetAllVxpuInfo(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.GetXPUs,
		func() (map[string]*types.XPUDevice, error) {
			device := &types.XPUDevice{Id: "id0", Index: 0, MemoryTotal: 200}
			return map[string]*types.XPUDevice{"id0": device}, nil
		})
	patches.ApplyFunc(getVxpus,
		func(xpuDevice map[string]*types.XPUDevice) (types.VxpuDevices, map[string][]uint32, error) {
			return types.VxpuDevices{}, map[string][]uint32{}, nil
		})
	patches.ApplyFunc(getUidToProcessMap,
		func(xpuDevices map[string]*types.XPUDevice, period int) (map[string]map[uint32]*types.ProcessUsage, error) {
			return map[string]map[uint32]*types.ProcessUsage{}, nil
		})

	svc := PidsServiceServerImpl{}
	resp, err := svc.GetAllVxpuInfo(context.Background(),
		&GetAllVxpuInfoRequest{Period: "60"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if resp.VxpuInfos == "" {
		t.Error("expected non-empty json, got empty")
	}
}

func TestGetAllVxpuInfo_InvalidPeriod(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.GetXPUs,
		func() (map[string]*types.XPUDevice, error) {
			device := &types.XPUDevice{Id: "id0", Index: 0, MemoryTotal: 200}
			return map[string]*types.XPUDevice{"id0": device}, nil
		})
	patches.ApplyFunc(getVxpus,
		func(xpuDevice map[string]*types.XPUDevice) (types.VxpuDevices, map[string][]uint32, error) {
			return types.VxpuDevices{}, map[string][]uint32{}, nil
		})
	patches.ApplyFunc(getUidToProcessMap,
		func(xpuDevices map[string]*types.XPUDevice, period int) (map[string]map[uint32]*types.ProcessUsage, error) {
			return map[string]map[uint32]*types.ProcessUsage{}, nil
		})

	svc := PidsServiceServerImpl{}
	resp, err := svc.GetAllVxpuInfo(context.Background(),
		&GetAllVxpuInfoRequest{Period: "invalid"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if resp.VxpuInfos == "" {
		t.Error("expected non-empty json, got empty")
	}
}

func TestGetAllVxpuInfo_GetXPUsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.GetXPUs,
		func() (map[string]*types.XPUDevice, error) {
			return nil, fmt.Errorf("get xpus error")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetAllVxpuInfo(context.Background(),
		&GetAllVxpuInfoRequest{Period: "60"})
	if err == nil {
		t.Error("expected error from GetXPUs, got nil")
	}
}

func TestGetAllVxpuInfo_GetVxpusError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.GetXPUs,
		func() (map[string]*types.XPUDevice, error) {
			device := &types.XPUDevice{Id: "id0", Index: 0, MemoryTotal: 200}
			return map[string]*types.XPUDevice{"id0": device}, nil
		})
	patches.ApplyFunc(getVxpus,
		func(xpuDevice map[string]*types.XPUDevice) (types.VxpuDevices, map[string][]uint32, error) {
			return nil, nil, fmt.Errorf("get vxpus error")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetAllVxpuInfo(context.Background(),
		&GetAllVxpuInfoRequest{Period: "60"})
	if err == nil {
		t.Error("expected error from getVxpus, got nil")
	}
}

func TestStart_UnlinkError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(syscall.Unlink,
		func(name string) error { return fmt.Errorf("unlink error") })

	Start()
}

func TestStart_ListenError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(syscall.Unlink,
		func(name string) error { return os.ErrNotExist })
	patches.ApplyFunc(net.Listen,
		func(network, address string) (net.Listener, error) {
			return nil, fmt.Errorf("listen error")
		})

	Start()
}

func TestStart_ChmodError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(syscall.Unlink,
		func(name string) error { return os.ErrNotExist })
	patches.ApplyFunc(net.Listen,
		func(network, address string) (net.Listener, error) {
			return &mockListener{}, nil
		})
	patches.ApplyFunc(os.Chmod,
		func(name string, mode os.FileMode) error {
			return fmt.Errorf("chmod error")
		})

	Start()
}

func TestStart_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(syscall.Unlink,
		func(name string) error { return os.ErrNotExist })
	patches.ApplyFunc(net.Listen,
		func(network, address string) (net.Listener, error) {
			return &mockListener{}, nil
		})
	patches.ApplyFunc(os.Chmod,
		func(name string, mode os.FileMode) error { return nil })
	patches.ApplyMethodFunc(&grpc.Server{}, "Serve",
		func(lis net.Listener) error { return nil })
	patches.ApplyFunc(time.Sleep,
		func(d time.Duration) { _ = 0 })
	patches.ApplyFunc(cleanDestroyedPodDir,
		func() error { return fmt.Errorf("stop cleanup") })

	Start()
}

func TestReadStatusFile_NotFound(t *testing.T) {
	_, err := readStatusFile("/nonexistent/status")
	if err == nil {
		t.Error("expected error for nonexistent file, got nil")
	}
}

func TestReadStatusFile_NSpidNotFound(t *testing.T) {
	f, err := os.CreateTemp("", "status_no_nspid")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Remove(f.Name()); err != nil {
			t.Logf("cleanup remove temp file failed: %v", err)
		}
	}()
	_, _ = fmt.Fprint(f, "Name: test\nUid: 0\n")
	if err := f.Close(); err != nil {
		t.Logf("close temp file failed: %v", err)
	}

	_, err = readStatusFile(f.Name())
	if err == nil {
		t.Error("expected error NSpid not found, got nil")
	}
}

func TestParseCgroupPath_DockerFallback(t *testing.T) {
	path := "/kubepods/besteffort/pod9b322980-f435-4759-90c7-fa2e7bc36b76" +
		"/53392426fb325c27357920728175c0d18065d633486ee855441d2517ec29a188"
	podId, containerId, err := parseCgroupPath(path)
	expectedPod := "9b322980-f435-4759-90c7-fa2e7bc36b76"
	expectedContainer := "53392426fb325c27357920728175c0d18065d633486ee855441d2517ec29a188"
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if podId != expectedPod {
		t.Errorf("expected pod %s, got %s", expectedPod, podId)
	}
	if containerId != expectedContainer {
		t.Errorf("expected container %s, got %s", expectedContainer, containerId)
	}
}

func TestGetContainerName_ListPodsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			return nil, fmt.Errorf("list pods error")
		})

	_, _, err := getContainerName(
		"/kubepods.slice/kubepods-besteffort.slice" +
			"/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice" +
			"/cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope")
	if err == nil {
		t.Error("expected error from ListPods, got nil")
	}
}

func TestGetContainerName_PodNotRunning(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			pod := v1.Pod{}
			pod.UID = "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
			pod.Status.Phase = v1.PodSucceeded
			podList := v1.PodList{Items: []v1.Pod{pod}}
			return &podList, nil
		})

	_, _, err := getContainerName(
		"/kubepods.slice/kubepods-besteffort.slice" +
			"/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice" +
			"/cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope")
	if err == nil {
		t.Error("expected error for pod not running, got nil")
	}
}

func TestGetContainerName_ContainerNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			pod := v1.Pod{}
			pod.UID = "1aba0c04-4fd5-4691-95ab-dbc6e90fa1db"
			pod.Status.Phase = v1.PodRunning
			cs := v1.ContainerStatus{
				ContainerID: "containerd://other-container-id",
				Name:        "other-container",
			}
			pod.Status.ContainerStatuses = []v1.ContainerStatus{cs}
			podList := v1.PodList{Items: []v1.Pod{pod}}
			return &podList, nil
		})

	_, _, err := getContainerName(
		"/kubepods.slice/kubepods-besteffort.slice" +
			"/kubepods-besteffort-pod1aba0c04_4fd5_4691_95ab_dbc6e90fa1db.slice" +
			"/cri-containerd-4e331f94fb1157366826f9d31e056170c7f227fd13d5817243262885ec2544d5.scope")
	if err == nil {
		t.Error("expected error for container not found, got nil")
	}
}

func TestReadPidsConfig_EmptyLines(t *testing.T) {
	f, err := os.CreateTemp("", "pids_config_empty")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Remove(f.Name()); err != nil {
			t.Logf("cleanup remove temp file failed: %v", err)
		}
	}()
	_, _ = fmt.Fprint(f, "\n\n   \n2776170\n")
	if err := f.Close(); err != nil {
		t.Logf("close temp file failed: %v", err)
	}

	pids, err := readPidsConfig(f.Name())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if len(pids) != 1 || pids[0] != 2776170 {
		t.Errorf("expected [2776170], got: %v", pids)
	}
}

func TestReadPidsConfig_InvalidLine(t *testing.T) {
	f, err := os.CreateTemp("", "pids_config_invalid")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Remove(f.Name()); err != nil {
			t.Logf("cleanup remove temp file failed: %v", err)
		}
	}()
	_, _ = fmt.Fprint(f, "not_a_number\n2776170\n")
	if err := f.Close(); err != nil {
		t.Logf("close temp file failed: %v", err)
	}

	pids, err := readPidsConfig(f.Name())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if len(pids) != 1 || pids[0] != 2776170 {
		t.Errorf("expected [2776170], got: %v", pids)
	}
}

func TestCleanDestroyedPodDir_ListPodsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getPodDirNames,
		func() ([]string, error) { return nil, nil })
	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			return nil, fmt.Errorf("list pods error")
		})

	err := cleanDestroyedPodDir()
	if err == nil {
		t.Error("expected error from ListPods, got nil")
	}
}

func TestSetVxpuDevices_VxpuNotExist(t *testing.T) {
	xpuDevices := make(map[string]*types.XPUDevice)
	device := &types.XPUDevice{Id: "id0", Index: 0, MemoryTotal: 200}
	xpuDevices["id0"] = device

	vxpuDevices := types.VxpuDevices{
		{XpuId: "nonexistent", PodUID: "pod0", ContainerName: "c0"},
	}

	result := setVxpuDevices(vxpuDevices, xpuDevices, nil, nil)
	if _, ok := result["id0"]; !ok {
		t.Error("expected device id0 to exist")
	}
}

func TestSetVxpuDevices_SoftModePolicyConflict(t *testing.T) {
	originalMode := config.NodeMode
	config.NodeMode = "soft"
	defer func() { config.NodeMode = originalMode }()

	xpuDevices := make(map[string]*types.XPUDevice)
	device := &types.XPUDevice{
		Id: "id0", Index: 0, MemoryTotal: 200, Policy: 0,
		VxpuDeviceList: types.VxpuDevices{},
	}
	xpuDevices["id0"] = device

	vxpuDevices := types.VxpuDevices{
		{XpuId: "id0", PodUID: "pod0", ContainerName: "c0", Policy: 1},
	}
	uidToProcessMap := make(map[string]map[uint32]*types.ProcessUsage)
	processMap := make(map[uint32]*types.ProcessUsage)
	processMap[2776170] = &types.ProcessUsage{ProcessMem: 40, ProcessCoreUtilization: 15}
	uidToProcessMap["id0"] = processMap
	pSet := map[string][]uint32{"pod0/c0": {2776170}}

	result := setVxpuDevices(vxpuDevices, xpuDevices, uidToProcessMap, pSet)
	if result["id0"].Policy == -1 {
		t.Log("policy conflict detected correctly")
	}
	if result["id0"].MemoryUtilization > 0 {
		t.Log("memory utilization computed correctly")
	}
}

func TestGetVxpus_ListPodsError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			return nil, fmt.Errorf("list pods error")
		})

	xpuDevice := make(map[string]*types.XPUDevice)
	_, _, err := getVxpus(xpuDevice)
	if err == nil {
		t.Error("expected error from ListPods, got nil")
	}
}

func TestGetVxpus_PodNotRunning(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.ListPods,
		func(opts metav1.ListOptions) (*v1.PodList, error) {
			pod := v1.Pod{}
			pod.UID = "test-uid"
			pod.Status.Phase = v1.PodFailed
			podList := v1.PodList{Items: []v1.Pod{pod}}
			return &podList, nil
		})

	xpuDevice := make(map[string]*types.XPUDevice)
	result, pSet, err := getVxpus(xpuDevice)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if len(result) != 0 {
		t.Errorf("expected 0 vxpu devices, got %d", len(result))
	}
	if len(pSet) != 0 {
		t.Errorf("expected 0 pSet entries, got %d", len(pSet))
	}
}

func TestGetAllVxpuInfo_GetUidToProcessMapError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(util.GetXPUs,
		func() (map[string]*types.XPUDevice, error) {
			device := &types.XPUDevice{Id: "id0", Index: 0, MemoryTotal: 200}
			return map[string]*types.XPUDevice{"id0": device}, nil
		})
	patches.ApplyFunc(getVxpus,
		func(xpuDevice map[string]*types.XPUDevice) (types.VxpuDevices, map[string][]uint32, error) {
			return types.VxpuDevices{}, map[string][]uint32{}, nil
		})
	patches.ApplyFunc(getUidToProcessMap,
		func(xpuDevices map[string]*types.XPUDevice, period int) (map[string]map[uint32]*types.ProcessUsage, error) {
			return nil, fmt.Errorf("get uid to process map error")
		})

	svc := PidsServiceServerImpl{}
	_, err := svc.GetAllVxpuInfo(context.Background(),
		&GetAllVxpuInfoRequest{Period: "60"})
	if err == nil {
		t.Error("expected error from getUidToProcessMap, got nil")
	}
}
