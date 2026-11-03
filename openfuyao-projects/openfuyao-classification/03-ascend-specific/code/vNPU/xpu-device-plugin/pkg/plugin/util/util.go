/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package util

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"huawei.com/vxpu-device-plugin/pkg/lock"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	deviceLength = 9

	containerLength = 7

	percentage = 100
)

// init initializes the Kubernetes client lock
func init() {
	lock.NewClient()
}

// GetNode retrieves a Kubernetes node by its name
func GetNode(nodename string) (*v1.Node, error) {
	return lock.GetClient().CoreV1().Nodes().Get(context.Background(), nodename, metav1.GetOptions{})
}

// ListPods lists all pods matching the provided list options
func ListPods(opts metav1.ListOptions) (*v1.PodList, error) {
	return lock.GetClient().CoreV1().Pods("").List(context.Background(), opts)
}

// GetPendingPod finds the pending pod on a specific node via node lock annotation.
// Only the node lock is used to identify the pod; there is no fallback path.
func GetPendingPod(nodename string) (*v1.Pod, error) {
	ns, name, err := lock.GetLockPodIdentity(nodename, types.VXPULockName)
	if err != nil {
		return nil, fmt.Errorf("get lock pod identity failed: %v", err)
	}
	if ns == "" || name == "" {
		return nil, fmt.Errorf("node lock has no pod identity on node %s", nodename)
	}
	pod, err := lock.GetClient().CoreV1().Pods(ns).Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get pod %s/%s failed: %v", ns, name, err)
	}
	phase := pod.Annotations[types.DeviceBindPhase]
	if phase != types.DeviceBindAllocating && phase != types.DeviceBindSuccess {
		return nil, fmt.Errorf("pod %s/%s bind phase is %s, expected allocating/success", ns, name, phase)
	}
	log.Infof("GetPendingPod: found via node lock: %s/%s", ns, name)
	return pod, nil
}

// EncodeNodeDevices converts a list of device information to a string representation
func EncodeNodeDevices(dlist []*types.DeviceInfo) string {
	var encodedNodeDevices strings.Builder
	for _, val := range dlist {
		encodedNodeDevices.Write([]byte(strconv.Itoa(int(val.Index))))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(val.Id))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(strconv.Itoa(int(val.Count))))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(strconv.Itoa(int(val.Devmem))))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(val.Type))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(strconv.FormatBool(val.Health)))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(strconv.Itoa(int(val.Numa))))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(val.Mode))
		encodedNodeDevices.Write([]byte(","))
		encodedNodeDevices.Write([]byte(strconv.Itoa(int(val.Corenum))))
		encodedNodeDevices.Write([]byte(":"))
	}
	log.Debugln("Encoded node Devices:", encodedNodeDevices.String())
	return encodedNodeDevices.String()
}

// EncodeContainerDevices converts container device assignments to a string format
func EncodeContainerDevices(cd types.ContainerDevices) string {
	var encodedContainerDevices strings.Builder
	for _, val := range cd {
		encodedContainerDevices.Write([]byte(strconv.Itoa(int(val.Index))))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(val.UUID))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(val.Type))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(strconv.Itoa(int(val.Usedmem))))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(strconv.Itoa(int(val.Usedcores))))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(strconv.Itoa(int(val.Vid))))
		encodedContainerDevices.Write([]byte(","))
		encodedContainerDevices.Write([]byte(val.Template))
		encodedContainerDevices.Write([]byte(":"))
	}
	log.Infoln("Encoded container Devices:", encodedContainerDevices.String())
	return encodedContainerDevices.String()
}

// EncodePodDevices converts pod device assignments to a string format
func EncodePodDevices(pd types.PodDevices) string {
	var ss []string
	for _, cd := range pd {
		ss = append(ss, EncodeContainerDevices(cd))
	}
	return strings.Join(ss, ";")
}

// GetXPUDevice converts device annotation string to XPU device map with additional info
func GetXPUDevice(str string, ip string) map[string]*types.XPUDevice {
	deviceMap := DecodeNodeDevices(str)
	driverVersion, FrameworkVersion, err := xpu.GetVersionInfo()
	if err != nil {
		log.Infof("GetVersionInfo error %v", err)
	}
	for _, device := range deviceMap {
		device.NodeIp = ip
		device.NodeName = config.NodeName
		device.DriverVersion = driverVersion
		device.FrameworkVersion = FrameworkVersion
	}
	return deviceMap
}

// DecodeNodeDevices parses device annotation string into XPU device map
func DecodeNodeDevices(str string) map[string]*types.XPUDevice {
	deviceMap := make(map[string]*types.XPUDevice)
	if !strings.Contains(str, ":") {
		log.Errorf("decode node device failed, wrong annos: %s", str)
		return deviceMap
	}
	tmp := strings.Split(str, ":")
	for _, val := range tmp {
		if !strings.Contains(val, ",") {
			continue
		}
		items := strings.Split(val, ",")
		if len(items) != deviceLength {
			log.Warningf("device string is wrong, device: %s", items)
			continue
		}
		index, err := strconv.Atoi(items[0])
		if err != nil {
			continue
		}
		count, err := strconv.Atoi(items[2])
		if err != nil {
			continue
		}
		devmem, err := strconv.Atoi(items[3])
		if err != nil {
			continue
		}
		health, err := strconv.ParseBool(items[5])
		if err != nil {
			continue
		}
		i := types.XPUDevice{
			Index:          int32(index),
			Id:             items[1],
			Type:           items[4],
			Count:          uint32(count),
			MemoryTotal:    uint64(devmem),
			Health:         health,
			VxpuDeviceList: types.VxpuDevices{},
		}
		deviceMap[items[1]] = &i
	}
	return deviceMap
}

// DecodeContainerDevices parses container device annotation string into container device list
func DecodeContainerDevices(str string) types.ContainerDevices {
	if len(str) == 0 {
		return types.ContainerDevices{}
	}
	cd := strings.Split(str, ":")
	contdev := types.ContainerDevices{}
	for _, val := range cd {
		if strings.Contains(val, ",") == false {
			continue
		}
		fields := strings.Split(val, ",")
		tmpdev := types.ContainerDevice{}
		if len(fields) != containerLength {
			log.Errorln("DecodeContainerDevices invalid parameter:", str)
			return types.ContainerDevices{}
		}
		index, err := strconv.Atoi(fields[0])
		if err != nil {
			log.Errorln("DecodeContainerDevices invalid parameter:", str)
			return types.ContainerDevices{}
		}
		tmpdev.Index = int32(index)
		tmpdev.UUID = fields[1]
		tmpdev.Type = fields[2]
		devmem, err := strconv.Atoi(fields[3])
		if err != nil {
			log.Errorln("DecodeContainerDevices invalid parameter:", str)
			return types.ContainerDevices{}
		}
		tmpdev.Usedmem = int32(devmem)
		devcores, err := strconv.Atoi(fields[4])
		if err != nil {
			log.Errorln("DecodeContainerDevices invalid parameter:", str)
			return types.ContainerDevices{}
		}
		tmpdev.Usedcores = int32(devcores)
		vid, err := strconv.Atoi(fields[5])
		if err != nil {
			log.Errorln("DecodeContainerDevices invalid parameter:", str)
			return types.ContainerDevices{}
		}
		tmpdev.Vid = int32(vid)
		tmpdev.Template = fields[6]
		contdev = append(contdev, tmpdev)
	}
	return contdev
}

// DecodePodDevices parses pod device annotation string into pod device assignments
func DecodePodDevices(str string) types.PodDevices {
	if len(str) == 0 {
		return types.PodDevices{}
	}
	var pd types.PodDevices
	for _, s := range strings.Split(str, ";") {
		cd := DecodeContainerDevices(s)
		pd = append(pd, cd)
	}
	return pd
}

// getContainerIdxByVxpuIdx maps virtual XPU index to container index in pod spec
func getContainerIdxByVxpuIdx(p *v1.Pod, vxpuIdx int) int {
	foundVxpuIdx := -1
	for i, container := range p.Spec.Containers {
		_, ok := container.Resources.Limits[xpu.VxpuNumber]
		if ok {
			foundVxpuIdx++
			if foundVxpuIdx == vxpuIdx {
				return i
			}
			continue
		}
		_, ok = container.Resources.Limits[xpu.VxpuMemory]
		if ok {
			foundVxpuIdx++
			if foundVxpuIdx == vxpuIdx {
				return i
			}
			continue
		}
	}
	return -1
}

// GetVxpuNumber extracts virtual XPU resource limits from container resource list
func GetVxpuNumber(resourceList v1.ResourceList) int64 {
	var number int64 = 0

	if vxpuNumber, ok := resourceList[xpu.VxpuNumber]; ok {
		number = vxpuNumber.Value()
	}

	return number
}

// GetNextDeviceRequest retrieves the next pending device request for allocation
func GetNextDeviceRequest(dtype string, p v1.Pod) (v1.Container, types.ContainerDevices, error) {
	pdevices := DecodePodDevices(p.Annotations[xpu.AssignedIDsToAllocate])
	log.Infoln("pdevices=", pdevices)
	policy := GetSchedulingPolicy(p)
	log.Infoln("policy=", policy)
	res := types.ContainerDevices{}
	for vxpuIdx, val := range pdevices {
		found := false
		for _, dev := range val {
			if strings.Compare(dtype, dev.Type) == 0 {
				dev.Policy = policy
				res = append(res, dev)
				found = true
			}
		}
		if found {
			idx := getContainerIdxByVxpuIdx(&p, vxpuIdx)
			if idx != -1 {
				return p.Spec.Containers[idx], res, nil
			} else {
				log.Errorf("get container idx by vxpuIdx failed, vxpuIdx: %d", vxpuIdx)
			}
			break
		}
	}
	return v1.Container{}, res, errors.New("device request not found")
}

func GetSchedulingPolicy(p v1.Pod) int32 {
	policy := p.Annotations[xpu.SchedulingPolicy]
	if len(policy) == 0 {
		return xpu.PolicyFixedShareInt
	}
	if strings.Compare(policy, xpu.PolicyFixedShareStr) == 0 {
		return xpu.PolicyFixedShareInt
	}
	if strings.Compare(policy, xpu.PolicyElasticStr) == 0 {
		return xpu.PolicyElasticInt
	}
	if strings.Compare(policy, xpu.PolicyBestEffortStr) == 0 {
		return xpu.PolicyBestEffortInt
	}
	return xpu.PolicyFixedShareInt
}

// EraseNextDeviceTypeFromAnnotation removes allocated device types from pod annotations
func EraseNextDeviceTypeFromAnnotation(dtype string, p v1.Pod) error {
	pdevices := DecodePodDevices(p.Annotations[xpu.AssignedIDsToAllocate])
	res := types.PodDevices{}
	found := false
	for _, val := range pdevices {
		if found {
			res = append(res, val)
			continue
		}
		tmp := types.ContainerDevices{}
		for _, dev := range val {
			if strings.Compare(dtype, dev.Type) == 0 {
				found = true
			} else {
				tmp = append(tmp, dev)
			}
		}
		if !found {
			res = append(res, val)
		} else {
			res = append(res, tmp)
		}
	}
	log.Infoln("After erase res=", res)
	newannos := make(map[string]string)
	newannos[xpu.AssignedIDsToAllocate] = EncodePodDevices(res)
	return PatchPodAnnotations(&p, newannos)
}

// PodAllocationTrySuccess checks if all device allocations are complete
func PodAllocationTrySuccess(nodeName string, pod *v1.Pod) {
	refreshed, err := lock.GetClient().CoreV1().Pods(pod.Namespace).Get(context.Background(), pod.Name, metav1.GetOptions{})
	if err != nil {
		log.Errorf("PodAllocationTrySuccess: get pod error: %v, pod: %s/%s", err, pod.Namespace, pod.Name)
		return
	}
	annos := refreshed.Annotations[xpu.AssignedIDsToAllocate]
	log.Infoln("TrySuccess:", annos)
	if strings.Contains(annos, xpu.DeviceType) {
		return
	}
	log.Infoln("AllDevicesAllocateSuccess")
	PodAllocationSuccess(nodeName, pod)
}

// PodAllocationSuccess marks pod allocation as successful and releases node lock
func PodAllocationSuccess(nodeName string, pod *v1.Pod) {
	newannos := make(map[string]string)
	newannos[types.DeviceBindPhase] = types.DeviceBindSuccess
	err := PatchPodAnnotations(pod, newannos)
	if err != nil {
		log.Errorf("patchPodAnnotations failed:%v", err.Error())
	}
	if releaseErr := lock.ReleaseNodeLockWithPod(nodeName, types.VXPULockName, pod.Namespace, pod.Name); releaseErr != nil {
		log.Errorf("ReleaseNodeLockWithPod failed: %v, node: %s, pod: %s/%s", releaseErr, nodeName, pod.Namespace, pod.Name)
	}
}

// PodAllocationFailed marks pod allocation as failed and releases node lock
func PodAllocationFailed(nodeName string, pod *v1.Pod) {
	newannos := make(map[string]string)
	newannos[types.DeviceBindPhase] = types.DeviceBindFailed
	err := PatchPodAnnotations(pod, newannos)
	if err != nil {
		log.Errorf("patchPodAnnotations failed:%v", err.Error())
	}
	if releaseErr := lock.ReleaseNodeLockWithPod(nodeName, types.VXPULockName, pod.Namespace, pod.Name); releaseErr != nil {
		log.Errorf("ReleaseNodeLockWithPod failed: %v, node: %s, pod: %s/%s", releaseErr, nodeName, pod.Namespace, pod.Name)
	}
}

// PatchNodeAnnotations updates annotations on a Kubernetes node
func PatchNodeAnnotations(nodeName string, annotations map[string]string) error {
	nodeBytes, err := createAnnotationsPatch(annotations)
	if err != nil {
		return err
	}
	_, err = lock.GetClient().CoreV1().Nodes().
		Patch(context.Background(), nodeName, k8stypes.StrategicMergePatchType, nodeBytes, metav1.PatchOptions{})
	if err != nil {
		log.Errorf("patch node %s failed, %v", nodeName, err)
	}
	return err
}

// PatchPodAnnotations updates annotations on a Kubernetes pod
func PatchPodAnnotations(pod *v1.Pod, annotations map[string]string) error {
	podBytes, err := createAnnotationsPatch(annotations)
	if err != nil {
		return err
	}
	_, err = lock.GetClient().CoreV1().Pods(pod.Namespace).
		Patch(context.Background(), pod.Name, k8stypes.StrategicMergePatchType, podBytes, metav1.PatchOptions{})
	if err != nil {
		log.Infof("patch pod %v failed, %v", pod.Name, err)
	}
	return err
}

// createAnnotationsPatch creates JSON patch for updating metadata annotations
func createAnnotationsPatch(annotations map[string]string) ([]byte, error) {
	type patchMetadata struct {
		Annotations map[string]string `json:"annotations,omitempty"`
	}
	type patchPod struct {
		Metadata patchMetadata `json:"metadata"`
	}

	p := patchPod{}
	p.Metadata.Annotations = annotations
	return json.Marshal(p)
}

// GetXPUs retrieves XPU device information from node annotations
func GetXPUs() (map[string]*types.XPUDevice, error) {
	node, err := GetNode(config.NodeName)
	if err != nil {
		log.Errorf("k8s get node error: %v, node name: %s", err, config.NodeName)
		return nil, err
	}
	annos, ok := node.ObjectMeta.Annotations[xpu.NodeVXPURegister]
	if !ok {
		errMsg := fmt.Sprintf("node %s annotation %s is not exists",
			config.NodeName, xpu.NodeVXPURegister)
		log.Errorf("%s", errMsg)
		return nil, errors.New(errMsg)
	}
	ip := getNodeIp(node)
	return GetXPUDevice(annos, ip), nil
}

// getNodeIp extracts the internal IP address from node status
func getNodeIp(node *v1.Node) string {
	for _, addr := range node.Status.Addresses {
		if addr.Type == v1.NodeInternalIP {
			return addr.Address
		}
	}
	log.Infoln("no expected node ip")
	return ""
}

func GetVxpuLimit(device types.ContainerDevice, xpuDevice map[string]*types.XPUDevice) (int64, int64) {
	if IsSoftMode() {
		return int64(device.Usedmem), int64(device.Usedcores)
	}

	xpuDev := xpuDevice[device.UUID]
	percent := config.NodeTemplateInfos.GetTemplateCorePercent(xpuDev.Type, device.Template)
	return int64(float64(xpuDev.MemoryTotal) * percent), int64(percent * percentage)
}
