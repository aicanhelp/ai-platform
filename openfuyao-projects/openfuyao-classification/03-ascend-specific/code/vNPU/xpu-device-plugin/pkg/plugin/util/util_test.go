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

package util

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"huawei.com/vxpu-device-plugin/pkg/lock"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	vnpuNumber = 40
	vnpuCore   = 5
	vnpuMemory = 2

	testNodeIP = "192.168.1.1"
)

func TestGetPendingPod(t *testing.T) {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
			Annotations: map[string]string{
				types.DeviceBindPhase: types.DeviceBindAllocating,
			},
		},
	}
	fakeNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "host2",
			Annotations: map[string]string{
				types.VXPULockName: "2026-07-01T12:00:00Z,default,pod-test",
			},
		},
	}
	fakeClient := fake.NewSimpleClientset(fakeNode, fakePod)
	lock.SetClient(fakeClient)

	pod, err := GetPendingPod("host2")
	if err == nil && pod != nil &&
		pod.Annotations[types.DeviceBindPhase] == types.DeviceBindAllocating &&
		pod.Name == "pod-test" {
		t.Log("test succeed")
	} else {
		t.Errorf("test failed: pod=%v, err=%v", pod, err)
	}
}

func TestGetPendingPod_NoLock(t *testing.T) {
	fakeNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "host3",
			Annotations: map[string]string{},
		},
	}
	fakeClient := fake.NewSimpleClientset(fakeNode)
	lock.SetClient(fakeClient)

	_, err := GetPendingPod("host3")
	if err == nil {
		t.Error("expected error when node lock is absent")
	}
}

func TestGetPendingPod_WrongPhase(t *testing.T) {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-wrong",
			Annotations: map[string]string{
				types.DeviceBindPhase: "success-done",
			},
		},
	}
	fakeNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "host4",
			Annotations: map[string]string{
				types.VXPULockName: "2026-07-01T12:00:00Z,default,pod-wrong",
			},
		},
	}
	fakeClient := fake.NewSimpleClientset(fakeNode, fakePod)
	lock.SetClient(fakeClient)

	_, err := GetPendingPod("host4")
	if err == nil {
		t.Error("expected error when pod bind phase is not allocating/success")
	}
}

func TestEncodeNodeDevices(t *testing.T) {
	di0 := types.DeviceInfo{
		Index:   0,
		Id:      "id0",
		Count:   10,
		Devmem:  8192,
		Type:    "ASCEND",
		Health:  true,
		Numa:    0,
		Mode:    "soft",
		Corenum: 100,
	}
	di1 := types.DeviceInfo{
		Index:   1,
		Id:      "id1",
		Count:   10,
		Devmem:  8192,
		Type:    "ASCEND",
		Health:  true,
		Numa:    1,
		Mode:    "hard",
		Corenum: 8,
	}
	dlist := []*types.DeviceInfo{&di0, &di1}
	realEncodedNodeDevices := EncodeNodeDevices(dlist)
	expectedEncodedNodeDevices := "0,id0,10,8192,ASCEND,true,0,soft,100:1,id1,10,8192,ASCEND,true,1,hard,8:"
	if realEncodedNodeDevices == expectedEncodedNodeDevices {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestEncodeContainerDevices(t *testing.T) {
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "12345678-abcd-bcde-cdef-123456789000",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
		Template:  "vir01",
	}
	cd1 := types.ContainerDevice{
		Index:     1,
		UUID:      "12345678-abcd-bcde-cdef-123456789001",
		Type:      "ASCEND",
		Usedmem:   2048,
		Usedcores: 20,
		Vid:       11,
		Template:  "vir02",
	}
	realEncodedContainerDevices := EncodeContainerDevices([]types.ContainerDevice{cd0, cd1})
	expectedEncodedContainerDevices := "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:" +
		"1,12345678-abcd-bcde-cdef-123456789001,ASCEND,2048,20,11,vir02:"
	if realEncodedContainerDevices == expectedEncodedContainerDevices {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestEncodePodDevices(t *testing.T) {
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "12345678-abcd-bcde-cdef-123456789000",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
		Template:  "vir01",
	}
	cd1 := types.ContainerDevice{
		Index:     1,
		UUID:      "12345678-abcd-bcde-cdef-123456789001",
		Type:      "ASCEND",
		Usedmem:   2048,
		Usedcores: 20,
		Vid:       11,
		Template:  "vir02",
	}
	cd2 := types.ContainerDevice{
		Index:     2,
		UUID:      "12345678-abcd-bcde-cdef-123456789002",
		Type:      "ASCEND",
		Usedmem:   4096,
		Usedcores: 40,
		Vid:       12,
		Template:  "vir04",
	}
	podDevices := types.PodDevices{types.ContainerDevices{cd0, cd1}, types.ContainerDevices{cd2}}
	realEncodedPodDevices := EncodePodDevices(podDevices)
	expectedEncodedPodDevices := "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:" +
		"1,12345678-abcd-bcde-cdef-123456789001,ASCEND,2048,20,11,vir02:;" +
		"2,12345678-abcd-bcde-cdef-123456789002,ASCEND,4096,40,12,vir04:"
	if realEncodedPodDevices == expectedEncodedPodDevices {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestDecodeNodeDevices(t *testing.T) {
	str := "0,NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e,20,8192,XPU-ASCEND-Tesla P4,true,0,soft,24:"
	result := types.XPUDevice{
		Index:          0,
		Id:             "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
		Type:           "XPU-ASCEND-Tesla P4",
		Health:         true,
		Count:          20,
		VxpuDeviceList: types.VxpuDevices{},
	}
	npuDeviceMap := DecodeNodeDevices(str)
	npuDevice := *npuDeviceMap[result.Id]
	if npuDevice.Index == result.Index &&
		npuDevice.Id == result.Id &&
		npuDevice.Type == result.Type &&
		npuDevice.Health == result.Health &&
		npuDevice.Count == result.Count {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

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

func TestDecodeContainerDevices(t *testing.T) {
	encodedContainerDevices := "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:" +
		"1,12345678-abcd-bcde-cdef-123456789001,ASCEND,2048,20,11,vir01:"
	realDecodedContainerDevices := DecodeContainerDevices(encodedContainerDevices)
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "12345678-abcd-bcde-cdef-123456789000",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
		Template:  "vir01",
	}
	cd1 := types.ContainerDevice{
		Index:     1,
		UUID:      "12345678-abcd-bcde-cdef-123456789001",
		Type:      "ASCEND",
		Usedmem:   2048,
		Usedcores: 20,
		Vid:       11,
		Template:  "vir01",
	}
	expectedDecodedContainerDevices := types.ContainerDevices{cd0, cd1}
	if isSliceEqual(realDecodedContainerDevices, expectedDecodedContainerDevices) {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestDecodePodDevices(t *testing.T) {
	encodedPodDevices := "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:" +
		"1,12345678-abcd-bcde-cdef-123456789001,ASCEND,2048,20,11,vir01:;" +
		"2,12345678-abcd-bcde-cdef-123456789002,ASCEND,4096,40,12,vir01:"
	realDecodedPodDevices := DecodePodDevices(encodedPodDevices)
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "12345678-abcd-bcde-cdef-123456789000",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
		Template:  "vir01",
	}
	cd1 := types.ContainerDevice{
		Index:     1,
		UUID:      "12345678-abcd-bcde-cdef-123456789001",
		Type:      "ASCEND",
		Usedmem:   2048,
		Usedcores: 20,
		Vid:       11,
		Template:  "vir01",
	}
	cd2 := types.ContainerDevice{
		Index:     2,
		UUID:      "12345678-abcd-bcde-cdef-123456789002",
		Type:      "ASCEND",
		Usedmem:   4096,
		Usedcores: 40,
		Vid:       12,
		Template:  "vir01",
	}
	expectedDecodedPodDevices := types.PodDevices{types.ContainerDevices{cd0, cd1}, types.ContainerDevices{cd2}}
	if isSliceEqual(realDecodedPodDevices, expectedDecodedPodDevices) {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestGetNextDeviceRequest(t *testing.T) {
	pod := v1.Pod{}
	pod.Annotations = make(map[string]string)
	pod.Annotations[xpu.AssignedIDsToAllocate] = "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:"
	var cont v1.Container
	cont.Resources.Limits = make(map[v1.ResourceName]resource.Quantity)
	cont.Resources.Limits[xpu.VxpuNumber] = resource.MustParse("1")
	cont.Resources.Limits[xpu.VxpuMemory] = resource.MustParse("2")
	pod.Spec.Containers = []v1.Container{cont}

	_, realContainerDevices, err := GetNextDeviceRequest("ASCEND", pod)
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "12345678-abcd-bcde-cdef-123456789000",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
		Template:  "vir01",
		Policy:    xpu.PolicyFixedShareInt,
	}
	expectedContainerDevices := types.ContainerDevices{cd0}
	if err == nil && isSliceEqual(realContainerDevices, expectedContainerDevices) {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestEraseNextDeviceTypeFromAnnotation(t *testing.T) {
	pod := v1.Pod{}
	pod.Annotations = make(map[string]string)
	pod.Annotations[xpu.AssignedIDsToAllocate] = "0,12345678-abcd-bcde-cdef-123456789000,ASCEND,1024,10,10,vir01:"

	patch := gomonkey.ApplyFunc(PatchPodAnnotations, func(pod *v1.Pod, annotations map[string]string) error {
		if annotations[xpu.AssignedIDsToAllocate] == "" {
			return nil
		} else {
			return errors.New("error occurred")
		}
	})
	defer patch.Reset()

	err := EraseNextDeviceTypeFromAnnotation("ASCEND", pod)
	if err == nil {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestPodAllocationSuccess(t *testing.T) {
	patchPodAnnotations := gomonkey.ApplyFunc(PatchPodAnnotations, func(pod *v1.Pod, annotations map[string]string) error {
		return nil
	})
	defer patchPodAnnotations.Reset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
		},
	}
	PodAllocationSuccess("host2", pod)
	t.Log("test succeed")
}

func TestPodAllocationFailed(t *testing.T) {
	patchPodAnnotations := gomonkey.ApplyFunc(PatchPodAnnotations, func(pod *v1.Pod, annotations map[string]string) error {
		return nil
	})
	defer patchPodAnnotations.Reset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
		},
	}
	PodAllocationFailed("host2", pod)
	t.Log("test succeed")
}

func TestGetXPUs(t *testing.T) {
	result := types.XPUDevice{
		Index:          0,
		Id:             "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
		Type:           "XPU-ASCEND-Tesla P4",
		Health:         true,
		Count:          20,
		VxpuDeviceList: types.VxpuDevices{},
	}
	patch := gomonkey.ApplyFunc(GetNode, func(nodename string) (*v1.Node, error) {
		node := v1.Node{}
		node.Annotations = make(map[string]string)
		node.Annotations[xpu.NodeVXPURegister] =
			"0,NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e,20,8192,XPU-ASCEND-Tesla P4,true,0,0,0:"
		nodeAddress := v1.NodeAddress{Type: v1.NodeInternalIP, Address: "localhost"}
		node.Status.Addresses = []v1.NodeAddress{nodeAddress}
		return &node, nil
	})
	defer patch.Reset()
	deviceMap, err := GetXPUs()
	npuDevice := deviceMap[result.Id]
	if err == nil &&
		len(deviceMap) == 1 &&
		npuDevice.Index == result.Index &&
		npuDevice.Id == result.Id &&
		npuDevice.Type == result.Type &&
		npuDevice.Health == result.Health &&
		npuDevice.Count == result.Count {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestGetVxpuNumber(t *testing.T) {
	resourceList := v1.ResourceList{
		xpu.VxpuNumber: resource.MustParse(strconv.Itoa(vnpuNumber)),
		xpu.VxpuCore:   resource.MustParse(strconv.Itoa(vnpuCore)),
		xpu.VxpuMemory: resource.MustParse(strconv.Itoa(vnpuMemory)),
	}
	number := GetVxpuNumber(resourceList)
	if number == vnpuNumber {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
}

func TestGetVxpuLimit(t *testing.T) {
	xpuDevice := make(map[string]*types.XPUDevice)
	device := types.XPUDevice{
		Id:          "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
		Type:        "NPU-ASCEND-910",
		MemoryTotal: 8192,
	}
	xpuDevice["NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e"] = &device

	testCases := []struct {
		name         string
		device       types.ContainerDevice
		nodeMode     string
		expectedMem  int64
		expectedCore int64
	}{
		{
			name: "soft mode",
			device: types.ContainerDevice{
				UUID:      "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
				Usedmem:   1024,
				Usedcores: 10,
			},
			nodeMode:     "soft",
			expectedMem:  1024,
			expectedCore: 10,
		},
		{
			name: "soft mode",
			device: types.ContainerDevice{
				UUID:      "NPU-882cb9bc-cafd-08e8-4ea7-c59a8974e00e",
				Usedmem:   1024,
				Usedcores: 10,
			},
			nodeMode:     "soft",
			expectedMem:  1024,
			expectedCore: 10,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config.NodeMode = tc.nodeMode
			mem, core := GetVxpuLimit(tc.device, xpuDevice)
			if mem == tc.expectedMem && core == tc.expectedCore {
				t.Log("test succeed")
			} else {
				t.Errorf("test failed: expected mem=%d, core=%d, got mem=%d, core=%d",
					tc.expectedMem, tc.expectedCore, mem, core)
			}
		})
	}
}

func TestIsHardMode(t *testing.T) {
	testCases := []struct {
		name     string
		nodeMode string
		expected bool
	}{
		{
			name:     "hard mode",
			nodeMode: "hard",
			expected: true,
		},
		{
			name:     "soft mode",
			nodeMode: "soft",
			expected: false,
		},
		{
			name:     "empty mode",
			nodeMode: "",
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config.NodeMode = tc.nodeMode
			result := IsHardMode()
			if result == tc.expected {
				t.Log("test succeed")
			} else {
				t.Errorf("test failed: expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestGetNode(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	fakeNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
	}
	fakeClient.CoreV1().Nodes().Create(context.Background(), fakeNode, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	node, err := GetNode("test-node")
	if err != nil {
		t.Errorf("GetNode failed: %v", err)
	}
	if node.Name != "test-node" {
		t.Errorf("expected node name test-node, got %s", node.Name)
	}
}

func TestGetNode_NotFound(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	lock.SetClient(fakeClient)

	_, err := GetNode("not-exist")
	if err == nil {
		t.Errorf("expected error for non-existent node")
	}
}

func TestListPods(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test-pod",
		},
	}
	fakeClient.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	pods, err := ListPods(metav1.ListOptions{})
	if err != nil {
		t.Errorf("ListPods failed: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Errorf("expected 1 pod, got %d", len(pods.Items))
	}
}

func TestPodAllocationTrySuccess(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
			Annotations: map[string]string{
				xpu.AssignedIDsToAllocate: "0,NPU-test,ASCEND,1024,10,0,vir01:",
			},
		},
	}
	fakeClient.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	PodAllocationTrySuccess("host2", pod)
	t.Log("test succeed")
}

func TestPodAllocationTrySuccess_AllComplete(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
			Annotations: map[string]string{
				xpu.AssignedIDsToAllocate: "0,NPU-test,ASCEND,1024,10,0,:",
			},
		},
	}
	fakeClient.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	patchAnno := gomonkey.ApplyFunc(PatchPodAnnotations, func(pod *v1.Pod, annotations map[string]string) error {
		return nil
	})
	defer patchAnno.Reset()

	patchRelease := gomonkey.ApplyFunc(lock.ReleaseNodeLockWithPod, func(nodeName, lockName, namespace, podName string) error {
		return nil
	})
	defer patchRelease.Reset()

	PodAllocationTrySuccess("host2", pod)
	t.Log("test succeed")
}

func TestCreateAnnotationsPatch(t *testing.T) {
	annos := map[string]string{"key1": "val1", "key2": "val2"}
	data, err := createAnnotationsPatch(annos)
	if err != nil {
		t.Errorf("createAnnotationsPatch failed: %v", err)
	}
	if len(data) == 0 {
		t.Errorf("expected non-empty patch data")
	}
}

func TestPatchPodAnnotations(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod-test",
		},
	}
	fakeClient.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	err := PatchPodAnnotations(pod, map[string]string{"new-anno": "value"})
	if err != nil {
		t.Errorf("PatchPodAnnotations failed: %v", err)
	}
}

func TestPatchNodeAnnotations(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	}
	fakeClient.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{})
	lock.SetClient(fakeClient)

	err := PatchNodeAnnotations("test-node", map[string]string{"new-anno": "value"})
	if err != nil {
		t.Errorf("PatchNodeAnnotations failed: %v", err)
	}
}

func TestPatchNodeAnnotations_Error(t *testing.T) {
	lock.SetClient(fake.NewSimpleClientset())
	err := PatchNodeAnnotations("non-existent", map[string]string{"k": "v"})
	if err == nil {
		t.Errorf("expected error patching non-existent node")
	}
}

func TestPatchPodAnnotations_Error(t *testing.T) {
	lock.SetClient(fake.NewSimpleClientset())
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "non-existent",
		},
	}
	err := PatchPodAnnotations(pod, map[string]string{"k": "v"})
	if err == nil {
		t.Errorf("expected error patching non-existent pod")
	}
}

func TestGetContainerIdxByVxpuIdx(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c0",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							xpu.VxpuNumber: resource.MustParse("1"),
						},
					},
				},
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							xpu.VxpuMemory: resource.MustParse("1024"),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{},
					},
				},
			},
		},
	}
	idx := getContainerIdxByVxpuIdx(pod, 1)
	if idx != 1 {
		t.Errorf("expected index 1, got %d", idx)
	}
}

func TestGetContainerIdxByVxpuIdx_NotFound(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c0",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{},
					},
				},
			},
		},
	}
	idx := getContainerIdxByVxpuIdx(pod, 99)
	if idx != -1 {
		t.Errorf("expected -1, got %d", idx)
	}
}

func TestGetXPUs_GetNodeError(t *testing.T) {
	patch := gomonkey.ApplyFunc(GetNode, func(nodename string) (*v1.Node, error) {
		return nil, fmt.Errorf("node not found")
	})
	defer patch.Reset()

	_, err := GetXPUs()
	if err == nil {
		t.Errorf("expected error from GetXPUs when GetNode fails")
	}
}

func TestGetSchedulingPolicy(t *testing.T) {
	tests := []struct {
		name     string
		policy   string
		expected int32
	}{
		{"empty policy", "", xpu.PolicyFixedShareInt},
		{"fixed share", xpu.PolicyFixedShareStr, xpu.PolicyFixedShareInt},
		{"elastic", xpu.PolicyElasticStr, xpu.PolicyElasticInt},
		{"best effort", xpu.PolicyBestEffortStr, xpu.PolicyBestEffortInt},
		{"unknown", "unknown", xpu.PolicyFixedShareInt},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod := v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{xpu.SchedulingPolicy: tc.policy},
				},
			}
			result := GetSchedulingPolicy(pod)
			if result != tc.expected {
				t.Errorf("GetSchedulingPolicy(%q) = %d, want %d", tc.policy, result, tc.expected)
			}
		})
	}
}

func TestGetNodeIp(t *testing.T) {
	node := &v1.Node{
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeInternalIP, Address: testNodeIP},
			},
		},
	}
	ip := getNodeIp(node)
	if ip != testNodeIP {
		t.Errorf("expected %s, got %s", testNodeIP, ip)
	}
}

func TestGetNodeIp_Empty(t *testing.T) {
	node := &v1.Node{}
	ip := getNodeIp(node)
	if ip != "" {
		t.Errorf("expected empty, got %s", ip)
	}
}

func TestGetVxpuLimit_HardMode(t *testing.T) {
	origMode := config.NodeMode
	config.NodeMode = "hard"
	defer func() { config.NodeMode = origMode }()

	device := types.ContainerDevice{
		UUID:     "test-uuid",
		Template: "default",
	}
	xpuDevice := map[string]*types.XPUDevice{
		"test-uuid": {
			MemoryTotal: 8192,
			Type:        "NPU-ASCEND-910",
		},
	}
	mem, core := GetVxpuLimit(device, xpuDevice)
	if mem == 0 && core == 0 {
		t.Logf("hard mode result: mem=%d, core=%d", mem, core)
	}
}
