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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	cgroupBaseDir         = "/sys/fs/cgroup/memory"
	cgroupProcs           = "cgroup.procs"
	pidsSockPath          = "/var/lib/xpu/pids.sock"
	hostProcDir           = "/hostproc"
	procStatus            = "status"
	nsPid                 = "NSpid:"
	nsPidFieldCount       = 3
	containerdPodIdPrefix = "-pod"
	dockerPodIdPrefix     = "/pod"
	containerdPodIdSuffix = ".slice"
	dockerPodIdSuffix     = "/"
	containerIdPrefix     = "cri-containerd-"
	containerIdSuffix     = ".scope"

	containerIdPrefixInContainerd = "containerd://"

	containerIdPrefixInDocker = "docker://"
	vxpuConfigBaseDir         = "/etc/xpu"
	pidsConfigFileName        = "pids.config"
	configFilePerm            = 0644
	pidsSockPerm              = 0666
	podDirCleanInterval       = 60
	minPeriod                 = 1
	maxPeriod                 = 86400
	defaultPeriod             = 60
	percentage                = 100
	float64BitSize            = 64

	cgroupV1PidsBaseDir   = "/sys/fs/cgroup/pids"
	cgroupV2BaseDir       = "/sys/fs/cgroup"
	cgroupControllersFile = "/sys/fs/cgroup/cgroup.controllers"
	cgroupMaxWalkDepth    = 6
)

type PidsServiceServerImpl struct {
	*UnimplementedPidsServiceServer
}

// readProcsFile reads process IDs from a cgroup procs file
func readProcsFile(file string) ([]int, error) {
	f, err := os.Open(file)
	if err != nil {
		log.Errorf("can't read %s: %s", file, err.Error())
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	pids := make([]int, 0)
	for scanner.Scan() {
		line := scanner.Text()
		if pid, err := strconv.Atoi(line); err == nil {
			pids = append(pids, pid)
		}
	}

	log.Infof("read from %s, pids: %v", file, pids)
	return pids, nil
}

// readStatusFile reads the NSpid field from a process status file
func readStatusFile(file string) (string, error) {
	f, err := os.Open(file)
	if err != nil {
		log.Errorf("can't read %s: %s", file, err.Error())
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, nsPid) {
			eles := strings.Fields(line)
			if len(eles) != nsPidFieldCount {
				return "", errors.New("NSpid field count error")
			}
			pids := fmt.Sprintf("%-11s %-11s", eles[1], eles[2])
			log.Infof("read from %s, pids: %s", file, pids)
			return pids, nil
		}
	}
	return "", errors.New("NSpid not found")
}

// getPidMaps converts host PIDs to namespace PIDs from status files
func getPidMaps(hostPids []int) string {
	pidMaps := make([]string, 0)
	for _, hp := range hostPids {
		procStatusPath := filepath.Clean(filepath.Join(hostProcDir, strconv.Itoa(hp), procStatus))
		pids, err := readStatusFile(procStatusPath)
		if err != nil {
			log.Errorf("read proc status error: %v, path: %s", err, procStatusPath)
		} else {
			pidMaps = append(pidMaps, pids)
		}
	}
	return strings.Join(pidMaps, ",")
}

// parseDockerCgroupPath extracts pod and container IDs from Docker cgroup paths
func parseDockerCgroupPath(cgroupPath string) (string, string, error) {
	idx := strings.Index(cgroupPath, dockerPodIdPrefix)
	if idx == -1 {
		return "", "", errors.New("pod id prefix not found")
	}
	podIdTmp := cgroupPath[idx+len(dockerPodIdPrefix):]
	idx = strings.Index(podIdTmp, dockerPodIdSuffix)
	if idx == -1 {
		return "", "", errors.New("pod id suffix not found")
	}
	podId := podIdTmp[:idx]
	podId = strings.Replace(podId, "_", "-", -1)

	containerId := podIdTmp[idx+len(dockerPodIdSuffix):]
	return podId, containerId, nil
}

// parseCgroupPath extracts pod and container IDs from container runtime cgroup paths
func parseCgroupPath(cgroupPath string) (string, string, error) {
	idx := strings.Index(cgroupPath, containerdPodIdPrefix)
	if idx == -1 {
		return parseDockerCgroupPath(cgroupPath)
	}
	podIdTmp := cgroupPath[idx+len(containerdPodIdPrefix):]
	idx = strings.Index(podIdTmp, containerdPodIdSuffix)
	if idx == -1 {
		return "", "", errors.New("pod id suffix not found")
	}
	podId := podIdTmp[:idx]
	podId = strings.Replace(podId, "_", "-", -1)

	idx = strings.Index(cgroupPath, containerIdPrefix)
	if idx == -1 {
		return podId, "", errors.New("container id prefix not found")
	}
	containerIdTmp := cgroupPath[idx+len(containerIdPrefix):]
	idx = strings.Index(containerIdTmp, containerIdSuffix)
	if idx == -1 {
		return podId, "", errors.New("container id suffix not found")
	}
	containerId := containerIdTmp[:idx]

	return podId, containerId, nil
}

// getContainerName maps cgroup path to Kubernetes pod and container names
func getContainerName(cgroupPath string) (string, string, error) {
	podId, containerId, err := parseCgroupPath(cgroupPath)
	if err != nil {
		log.Errorf("parse cgroup path error: %v", err)
		return "", "", err
	}

	selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": config.NodeName, "status.phase": string(v1.PodRunning)})
	podList, err := util.ListPods(metav1.ListOptions{
		FieldSelector: selector.String(),
	})
	if err != nil {
		log.Errorf("get pods in current node error: %v", err)
		return podId, "", err
	}

	for _, pod := range podList.Items {
		if string(pod.UID) != podId {
			continue
		}
		if pod.Status.Phase != v1.PodRunning || len(pod.Status.ContainerStatuses) == 0 {
			errMsg := fmt.Sprintf("pod status error: %v, container statuses len: %d",
				pod.Status.Phase, len(pod.Status.ContainerStatuses))
			log.Errorf("%s", errMsg)
			return podId, "", errors.New(errMsg)
		}
		for _, cs := range pod.Status.ContainerStatuses {
			if stripContainerIDPrefix(cs.ContainerID) != containerId {
				continue
			}
			return podId, cs.Name, nil
		}
	}

	return podId, "", errors.New("container not found")
}

// readPidsConfig reads process IDs from the configuration file
func readPidsConfig(pidsConfigPath string) ([]uint32, error) {
	f, err := os.OpenFile(pidsConfigPath, os.O_RDONLY, configFilePerm)
	if err != nil {
		log.Errorf("open pids config file error: %v", err)
		return nil, err
	}
	defer f.Close()

	var pids []uint32
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		tmp := strings.Fields(line)
		if len(tmp) == 0 {
			continue
		}
		pid, err := strconv.Atoi(tmp[0])
		if err != nil {
			log.Warningf("pid is wrong: %v", line)
			continue
		}
		pids = append(pids, uint32(pid))
	}
	return pids, nil
}

// writePidsConfig writes process ID mappings to the configuration file
func writePidsConfig(pidsConfigPath, pidMaps string) error {
	f, err := os.OpenFile(pidsConfigPath, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, configFilePerm)
	if err != nil {
		log.Errorf("open pids config file error: %v", err)
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	pids := strings.Split(pidMaps, ",")
	for _, pid := range pids {
		_, err = w.WriteString(pid + "\n")
		if err != nil {
			log.Errorf("bufio Writer WriteString error: %v", err)
			return err
		}
	}
	return w.Flush()
}

// getPodDirNames lists all pod-specific directories under the base configuration directory
func getPodDirNames() ([]string, error) {
	dirNames := make([]string, 0)
	err := filepath.Walk(vxpuConfigBaseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Errorf("filepath walk func error: %v", err)
			return err
		}
		if info == nil || !info.IsDir() || path == vxpuConfigBaseDir {
			return nil
		}
		dirNames = append(dirNames, info.Name())
		return filepath.SkipDir
	})
	if err != nil {
		log.Errorf("filepath walk error: %v, dir name: %s", err, vxpuConfigBaseDir)
		return []string{}, err
	}
	return dirNames, nil
}

type void struct{}

var val void

// cleanDestroyedPodDir removes directories for pods that no longer exist on the node
func cleanDestroyedPodDir() error {
	podDirNames, err := getPodDirNames()
	if err != nil {
		return err
	}

	selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": config.NodeName})
	podList, err := util.ListPods(metav1.ListOptions{
		FieldSelector: selector.String(),
	})
	if err != nil {
		log.Errorf("get pods in current node error: %v", err)
		return err
	}

	podIdSet := make(map[string]void)
	for _, pod := range podList.Items {
		podIdSet[string(pod.UID)] = val
	}

	for _, podDirName := range podDirNames {
		if _, ok := podIdSet[podDirName]; ok {
			continue
		}
		podAbsoluteDir := filepath.Clean(filepath.Join(vxpuConfigBaseDir, podDirName))
		err = os.RemoveAll(podAbsoluteDir)
		if err != nil {
			log.Infof("remove pod dir error: %v, dir name: %s", err, podAbsoluteDir)
		}
	}

	return nil
}

// GetPids implements the gRPC service to retrieve process IDs for a container
func (PidsServiceServerImpl) GetPids(ctx context.Context, req *GetPidsRequest) (*GetPidsResponse, error) {
	cgroupAbsolutePath := filepath.Clean(filepath.Join(cgroupBaseDir, req.CgroupPath, cgroupProcs))
	hostPids, err := readProcsFile(cgroupAbsolutePath)
	if err != nil {
		log.Errorf("read cgroup procs error: %v, path: %s", err, cgroupAbsolutePath)
		return nil, err
	}
	pidMaps := getPidMaps(hostPids)
	podId, containerName, err := getContainerName(req.CgroupPath)
	if err != nil {
		log.Errorf("get container name error: %v, cgroup path: %s", err, req.CgroupPath)
		return nil, err
	}
	pidsConfigPath := filepath.Clean(filepath.Join(vxpuConfigBaseDir, podId, containerName, pidsConfigFileName))
	err = writePidsConfig(pidsConfigPath, pidMaps)
	if err != nil {
		log.Errorf("write pids config error: %v, podId: %s, containerName: %s", err, podId, containerName)
		return nil, err
	}
	return &GetPidsResponse{EncodedPids: pidMaps}, nil
}

// stripContainerIDPrefix 去除containerID的前缀
func stripContainerIDPrefix(containerID string) string {
	if strings.HasPrefix(containerID, containerIdPrefixInContainerd) {
		return containerID[len(containerIdPrefixInContainerd):]
	}
	if strings.HasPrefix(containerID, containerIdPrefixInDocker) {
		return containerID[len(containerIdPrefixInDocker):]
	}
	return containerID
}

func isCgroupV2() bool {
	_, err := os.Stat(cgroupControllersFile)
	return err == nil
}

func findCgroupPathByBaseDir(baseDir, podUID, containerID string) (string, error) {
	var foundPath string
	var found bool

	err := filepath.WalkDir(baseDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		relPath, relErr := filepath.Rel(baseDir, path)
		if relErr != nil {
			return nil
		}
		depth := len(strings.Split(filepath.ToSlash(relPath), "/"))
		if depth > cgroupMaxWalkDepth {
			return fs.SkipDir
		}

		if !d.IsDir() {
			return nil
		}

		dirName := d.Name()

		if strings.Contains(dirName, "pod") &&
			strings.Contains(dirName, podUID) {

			if strings.Contains(dirName, containerID) {
				foundPath = path
				found = true
				return fs.SkipAll
			}

			subEntries, readErr := os.ReadDir(path)
			if readErr != nil {
				return nil
			}

			for _, subEntry := range subEntries {
				if !subEntry.IsDir() {
					continue
				}

				subDirName := subEntry.Name()

				if strings.Contains(subDirName, containerIdPrefix) &&
					strings.Contains(subDirName, containerID) &&
					strings.HasSuffix(subDirName, containerIdSuffix) {

					foundPath = filepath.Join(path, subDirName)
					found = true
					return fs.SkipAll
				}
			}
		}

		return nil
	})

	if err != nil {
		return "", err
	}

	if !found {
		return "", fmt.Errorf("cgroup path not found in %s for podUID: %s, containerID: %s", baseDir, podUID, containerID)
	}

	return foundPath, nil
}

func findCgroupPath(podUID, containerID string) (string, error) {
	if isCgroupV2() {
		return findCgroupPathByBaseDir(cgroupV2BaseDir, podUID, containerID)
	}
	path, err := findCgroupPathByBaseDir(cgroupV1PidsBaseDir, podUID, containerID)
	if err == nil {
		return path, nil
	}
	return findCgroupPathByBaseDir(cgroupV2BaseDir, podUID, containerID)
}

// setVxpuDevices aggregates virtual XPU device usage statistics from process metrics
func setVxpuDevices(vxpuDevices types.VxpuDevices,
	xpuDevices map[string]*types.XPUDevice,
	uidToProcessMap map[string]map[uint32]*types.ProcessUsage,
	pSet map[string][]uint32) map[string]*types.XPUDevice {
	for _, v := range vxpuDevices {
		if _, ok := xpuDevices[v.XpuId]; !ok {
			log.Warningf("vxpu in pod %s container %s is not exist", v.PodUID, v.ContainerName)
			continue
		}

		processUsage, ok := uidToProcessMap[v.XpuId]
		if !ok {
			xpuDevices[v.XpuId].VxpuDeviceList = append(xpuDevices[v.XpuId].VxpuDeviceList, v)
			continue
		}

		pidkey := fmt.Sprintf("%s/%s", v.PodUID, v.ContainerName)
		for _, pid := range pSet[pidkey] {
			if pUsage, ok := processUsage[pid]; ok {
				v.VxpuCoreUtilization += float64(pUsage.ProcessCoreUtilization)
				v.VxpuMemoryUsed += pUsage.ProcessMem
			}
		}
		if xpuDevices[v.XpuId].MemoryTotal != 0 {
			vxpuMemoryUtil := float64(v.VxpuMemoryUsed*percentage) / float64(xpuDevices[v.XpuId].MemoryTotal)
			vxpuMemoryUtil, err := strconv.ParseFloat(fmt.Sprintf("%.2f", vxpuMemoryUtil), float64BitSize)
			if err != nil {
				log.Warningf("vxpuMemoryUtil Parse failed %f", vxpuMemoryUtil)
			}
			v.VxpuMemoryUtilization = vxpuMemoryUtil
		}

		if util.IsSoftMode() {
			if xpuDevices[v.XpuId].Policy == 0 {
				xpuDevices[v.XpuId].Policy = v.Policy
			} else if xpuDevices[v.XpuId].Policy != v.Policy {
				log.Errorf("device %s policy conflict: expected %d, got %d", v.XpuId, xpuDevices[v.XpuId].Policy, v.Policy)
				xpuDevices[v.XpuId].Policy = -1
			} else {
				// policy matches; no action needed
			}
		}

		xpuDevices[v.XpuId].VxpuDeviceList = append(xpuDevices[v.XpuId].VxpuDeviceList, v)
	}
	for _, device := range xpuDevices {
		device.Mode = config.NodeMode
		for _, vxpuDevice := range device.VxpuDeviceList {
			device.MemoryUsed += vxpuDevice.VxpuMemoryUsed
		}
		memoryUtil := float64(device.MemoryUsed*percentage) / float64(device.MemoryTotal)
		memoryUtil, err := strconv.ParseFloat(fmt.Sprintf("%.2f", memoryUtil), float64BitSize)
		if err != nil {
			log.Warningf("vxpuMemoryUtil Parse failed %f", memoryUtil)
		}
		device.MemoryUtilization = memoryUtil
	}
	return xpuDevices
}

// GetAllVxpuInfo implements the gRPC service to retrieve comprehensive virtual XPU device information
func (PidsServiceServerImpl) GetAllVxpuInfo(ctx context.Context, req *GetAllVxpuInfoRequest) (
	*GetAllVxpuInfoResponse, error) {
	period, err := strconv.Atoi(req.Period)
	if err != nil || period < minPeriod || period > maxPeriod {
		log.Warningf("period %s is uniqualified, err msg: %v", req.Period, err)
		period = defaultPeriod
	}

	xpuDevices, err := util.GetXPUs()
	if err != nil {
		log.Errorf("get xpu device info failed, err msg: %v", err)
		return nil, err
	}

	vxpuDevices, pSet, err := getVxpus(xpuDevices)
	if err != nil {
		log.Errorf("get vxpu device info failed, err msg: %v", err)
		return nil, err
	}

	uidToProcessMap, err := getUidToProcessMap(xpuDevices, period)
	if err != nil {
		return nil, err
	}

	xpuDevices = setVxpuDevices(vxpuDevices, xpuDevices, uidToProcessMap, pSet)
	jsonVxpuInfos, err := json.Marshal(xpuDevices)
	if err != nil {
		log.Errorf("Generate JSON string error: %v", err)
		return nil, err
	}
	return &GetAllVxpuInfoResponse{VxpuInfos: string(jsonVxpuInfos)}, nil
}

// getVxpus retrieves virtual XPU device assignments and process sets for running pods
func getVxpus(xpuDevice map[string]*types.XPUDevice) (types.VxpuDevices, map[string][]uint32, error) {
	selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": config.NodeName, "status.phase": string(v1.PodRunning)})
	podList, err := util.ListPods(metav1.ListOptions{
		FieldSelector: selector.String(),
	})
	if err != nil {
		log.Errorf("get pods in current node error: %v", err)
		return nil, nil, err
	}

	res := types.VxpuDevices{}
	pSet := make(map[string][]uint32)
	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning || len(pod.Status.ContainerStatuses) == 0 {
			errMsg := fmt.Sprintf("pod status error: %v, container statuses len: %d",
				pod.Status.Phase, len(pod.Status.ContainerStatuses))
			log.Errorf("%s", errMsg)
			continue
		}

		pdevices := util.DecodePodDevices(pod.Annotations[xpu.AssignedIDs])
		policy := util.GetSchedulingPolicy(pod)
		pi := 0
		for _, cs := range pod.Spec.Containers {
			number := util.GetVxpuNumber(cs.Resources.Limits)
			if number == 0 {
				continue
			}
			if pi >= len(pdevices) {
				log.Warningf("pod %v does not have enough xpu devices for %v", pod.UID, pi)
				break
			}

			if len(pdevices[pi]) != int(number) {
				log.Warningf("vxpu assgined info error, pod uid: %v, container name: %s", pod.UID, cs.Name)
				continue
			}
			for i := 0; i < int(number); i++ {
				memLimit, coreLimit := util.GetVxpuLimit(pdevices[pi][i], xpuDevice)
				res = append(res, types.VxpuDevice{
					Id:              fmt.Sprintf("%s-%d", pdevices[pi][i].UUID, pdevices[pi][i].Vid),
					XpuId:           pdevices[pi][i].UUID,
					PodUID:          string(pod.UID),
					ContainerName:   cs.Name,
					VxpuMemoryLimit: memLimit,
					VxpuCoreLimit:   coreLimit,
					Policy:          policy,
				})
			}
			pi += 1
			containerID := getContainerID(pod, cs.Name)
			key := fmt.Sprintf("%s/%s", pod.UID, cs.Name)
			pSet[key] = getPodContainerPids(string(pod.UID), containerID)
		}
	}
	return res, pSet, nil
}

func getContainerID(pod v1.Pod, containerName string) string {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == containerName {
			return status.ContainerID
		}
	}
	return ""
}

// getPodContainerPids reads process IDs for pod from cgroup filesystem
func getPodContainerPids(podUID, containerID string) []uint32 {
	podUIDWithUnderscore := strings.ReplaceAll(podUID, "-", "_")
	containerID = stripContainerIDPrefix(containerID)

	cgroupPath, err := findCgroupPath(podUIDWithUnderscore, containerID)
	if err != nil {
		log.Errorf("find cgroup path failed: %v, podUID: %s, containerID: %s",
			err, podUIDWithUnderscore, containerID)
	}

	cgroupProcsPath := filepath.Join(cgroupPath, cgroupProcs)
	hostPids, err := readProcsFile(cgroupProcsPath)
	if err != nil {
		log.Errorf("read cgroup.procs failed: %v, path: %s", err, cgroupProcsPath)
	}

	pids := make([]uint32, 0, len(hostPids))
	for _, pid := range hostPids {
		pids = append(pids, uint32(pid))
	}
	return pids
}

func getUidToProcessMap(xpuDevices map[string]*types.XPUDevice, period int) (map[string]map[uint32]*types.ProcessUsage, error) {
	uidToProcessMap := make(map[string]map[uint32]*types.ProcessUsage)
	if util.IsHardMode() {
		// todo: support hard mode in future, when HDK support DcGetDevProcessInfo
		log.Infoln("hard mode dcmi not support get xpu usage")
		return uidToProcessMap, nil
	}
	for _, v := range xpuDevices {
		deviceUsageInfo, processMap, err := xpu.GetXPUUsage(v.Index, int32(period))
		if err != nil {
			log.Errorf("get xpu usage failed")
			return nil, err
		}
		v.XpuUtilization = float64(deviceUsageInfo.CoreUtil)
		v.PowerUsage = deviceUsageInfo.PowerUsage
		v.Temperature = deviceUsageInfo.Temperature
		uidToProcessMap[v.Id] = processMap
	}
	return uidToProcessMap, nil
}

// Start initializes and runs the gRPC server for process ID management
func Start() {
	srv := grpc.NewServer()
	RegisterPidsServiceServer(srv, PidsServiceServerImpl{})

	err := syscall.Unlink(pidsSockPath)
	if err != nil && !os.IsNotExist(err) {
		log.Errorf("remove pids server sock error: %v", err)
		return
	}

	listener, err := net.Listen("unix", pidsSockPath)
	if err != nil {
		log.Errorf("net listen error: %v", err)
		return
	}
	err = os.Chmod(pidsSockPath, pidsSockPerm)
	if err != nil {
		log.Errorf("modify pids.sock file permissions error: %v", err)
		return
	}

	go func() {
		err := srv.Serve(listener)
		if err != nil {
			log.Errorf("grpc server error: %v", err)
		}
	}()

	go func() {
		for {
			time.Sleep(time.Second * podDirCleanInterval)
			err := cleanDestroyedPodDir()
			if err != nil {
				log.Errorf("clean destroyed pod dir error: %v, stop cleanup...", err)
				break
			}
		}
	}()
}
