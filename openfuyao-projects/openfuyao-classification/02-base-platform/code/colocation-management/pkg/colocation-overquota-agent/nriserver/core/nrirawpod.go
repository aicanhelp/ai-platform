/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/containerd/nri/pkg/api"
	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

// KeyValue is a key-value pair
type KeyValue struct {
	Key   string
	Value string
}

// ContainerEngineType indicates the type of container engine
type ContainerEngineType int8

const (
	// UNDEFINED means undefined container engine
	UNDEFINED ContainerEngineType = iota
	// DOCKER means docker container engine
	DOCKER
	// CONTAINERD means containerd container engine
	CONTAINERD
	// ISULAD means isulad container engine
	ISULAD
	// CRIO means crio container engine
	CRIO
)

const (
	// envVarKeyValueParts Key=value format should have 2 parts
	envVarKeyValueParts = 2
	// envVarSplitLimit Split into max 2 parts: key and value
	envVarSplitLimit = 2
	// envVarSeparator is the separator between key and value
	envVarSeparator = "="
	// podPathIndexFromEnd is the index of pod path from end
	podPathIndexFromEnd = 2
)

type (
	// NRIRawContainer is server container structure
	NRIRawContainer api.Container
	// NRIRawPod is server pod structure
	NRIRawPod api.PodSandbox
)

var (
	supportEnginesPrefixMap = map[ContainerEngineType]string{
		DOCKER:     "docker://",
		CONTAINERD: "containerd://",
		ISULAD:     "iSulad://",
		CRIO:       "cri-o://",
	}
	// defaultContainerEngines is the default container engine
	defaultContainerEngines = CONTAINERD
	// currentContainerEngines is the current container engine
	currentContainerEngines ContainerEngineType = 2
	// containerEngineScopes is the container engine scope
	containerEngineScopes = map[ContainerEngineType]string{
		DOCKER:     "docker",
		CONTAINERD: "cri-containerd",
		ISULAD:     "isulad",
		CRIO:       "crio",
	}
	cgroupDriver = common.CgroupDriverCgroupfs
)

const (
	// server Container Annotations "kubectl.kubernetes.io/last-applied-configuration" means container applied config
	containerAppliedConfiguration string = "kubectl.kubernetes.io/last-applied-configuration"
	// /proc/self/cgroup file
	procSelfCgroupFile             = "/proc/self/cgroup"
	containerSpec      string      = "containers"
	resourcesSpec      string      = "resources"
	fileMode           os.FileMode = 0666
)

// GetExtendedResourceConfiguration returns the extended resource configuration
func GetExtendedResourceConfiguration(annotations map[string]string) (*ExtendedResourceConfiguration, error) {
	info := &ExtendedResourceConfiguration{}
	if len(annotations) == 0 {
		return info, nil
	}
	cfg, ok := annotations[common.ExtenderResourceConfigurationAnnotation]
	if ok {
		err := json.Unmarshal([]byte(cfg), &info)
		if err != nil {
			return nil, err
		}
		return info, nil
	} else {
		return info, nil
	}

}

// GetContainerEngine returns the container engine
func GetContainerEngine(annotations map[string]string) {
	engine, ok := annotations[common.ContainerEngineAnnotation]
	if ok {
		FixCurrentContainerEngine(engine)
	} else {
		currentContainerEngines = defaultContainerEngines
	}
}

// FixCurrentContainerEngine fix current container engine
func FixCurrentContainerEngine(annotationEngine string) string {
	for engine, prefix := range containerEngineScopes {
		if strings.HasPrefix(annotationEngine, prefix) {
			fmt.Printf("The container engine is %v\n", prefix)
			currentContainerEngines = engine
			return prefix
		}
	}
	currentContainerEngines = defaultContainerEngines
	return supportEnginesPrefixMap[currentContainerEngines]
}

// Prefix returns the ID prefix of the container engine
func (engine *ContainerEngineType) Prefix() string {
	prefix, ok := supportEnginesPrefixMap[*engine]
	if !ok {
		return ""
	}
	return prefix
}

// FixContainerCgroupParent fix container cgroup parent
func FixContainerCgroupParent(container *api.Container) (string, error) {

	/*
		Cgroup path conversion examples:

		Systemd driver (burstable):
		Original: kubepods-burstable-pod<uid>.slice:crio:<container_id>
		Converted: kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid>.slice/crio-<container_id>.scope

		Cgroupfs driver with isula/docker/containerd (burstable):
		Original: /kubepods/burstable/pod<uid>/<container_id>
		Converted: kubepods/burstable/pod<uid>/<container_id>

		Cgroupfs driver with crio (burstable):
		Original: /kubepods/burstable/pod<uid>/crio-<container_id>
		Converted: kubepods/burstable/pod<uid>/crio-<container_id>

		Path patterns by QoS:
		- Burstable:    kubepods/burstable/pod<uid>/...
		- BestEffort:   kubepods/besteffort/pod<uid>/...
		- Guaranteed:   kubepods/pod<uid>/...

		Systemd paths use .slice/.scope notation with hierarchical structure.
	*/

	qosClass := getQos(container)
	qosPath, err := getQosPath(qosClass)
	if err != nil {
		return "", err
	}

	if GetCgroupDriver() == common.CgroupDriverSystemd {
		return buildSystemdCgroupPath(container, qosPath)
	}
	return buildCgroupfsPath(container, qosPath)
}

func getQosPath(qosClass string) (string, error) {
	switch v1.PodQOSClass(qosClass) {
	case v1.PodQOSGuaranteed:
		return "", nil
	case v1.PodQOSBurstable:
		return strings.ToLower(string(v1.PodQOSBurstable)), nil
	case v1.PodQOSBestEffort:
		return strings.ToLower(string(v1.PodQOSBestEffort)), nil
	default:
		return "", fmt.Errorf("unknown qos class %v", qosClass)
	}
}
func buildSystemdCgroupPath(container *api.Container, qosPath string) (string, error) {
	if !isSupportedContainerEngine() {
		return "", nil
	}

	baseParts := []string{common.KubepodsCgroup + ".slice"}
	if qosPath != "" {
		baseParts = append(baseParts, common.KubepodsCgroup+"-"+qosPath+".slice")
	}
	baseParts = append(baseParts, getPodCgroupDir(container))

	containerScope := containerEngineScopes[currentContainerEngines] + "-" + container.GetId() + ".scope"
	return filepath.Join(append(baseParts, containerScope)...), nil
}
func buildCgroupfsPath(container *api.Container, qosPath string) (string, error) {
	if !isSupportedContainerEngine() {
		return "", nil
	}

	baseParts := []string{common.KubepodsCgroup}
	if qosPath != "" {
		baseParts = append(baseParts, qosPath)
	}
	baseParts = append(baseParts, getPodCgroupDir(container))

	containerID := getContainerID(container)
	return filepath.Join(append(baseParts, containerID)...), nil
}
func getContainerID(container *api.Container) string {
	if containerEngineScopes[currentContainerEngines] == common.ContainerEngineCrio {
		return containerEngineScopes[currentContainerEngines] + "-" + container.GetId()
	}
	return container.Id
}

func isSupportedContainerEngine() bool {
	switch containerEngineScopes[currentContainerEngines] {
	case common.ContainerEngineContainerd, common.ContainerEngineCrio,
		common.ContainerEngineDocker, common.ContainerEngineIsula:
		return true
	default:
		return false
	}
}

// GetContainerEnv returns the container environment variables
func GetContainerEnv(env []string) map[string]string {

	envs := make(map[string]string, len(env))
	for v := range env {
		e := env[v]
		split := strings.SplitN(e, envVarSeparator, envVarSplitLimit)
		if len(split) == envVarKeyValueParts {
			envs[split[0]] = split[1]
		}
	}
	return envs
}

// GetPodQosLevel returns the pod qos level
func GetPodQosLevel(annotations map[string]string) common.QosLevel {

	if qos, exist := annotations[common.QosClassAnnotationKey]; exist {
		switch common.QosLevel(qos) {
		case common.QosHLS, common.QosLs, common.QosBe:
			klog.Infof("Found valid QoS level from annotation: %s", qos)
			return common.QosLevel(qos)
		default:
			klog.Infof("Found invalid QoS level from annotation: %s, falling back to QosNone", qos)
			return common.QosNone
		}
	}
	return common.QosNone
}

// VerifyPodPriorityAnnotation returns true if the pod has volcano.sh/preemptable annotation
func VerifyPodPriorityAnnotation(annotations map[string]string) bool {
	if _, ok := annotations[common.PriorityAnnotationKey]; ok {
		return true
	} else {
		return false
	}
}

// CheckAnnotationAdmit returns true if the pod has volcano.sh/preemptable annotation
func CheckAnnotationAdmit(annotations map[string]string) AdmitResult {
	// 如果pod使用了离线资源 但是没有离线注解和qos注解 应当拒绝相关请求
	if annotations != nil {
		// 使用了离线资源
		_, ok := annotations[common.ExtenderResourceConfigurationAnnotation]
		if ok {
			// 必须是BE类型的QosLevel
			if GetPodQosLevel(annotations) != common.QosBe {
				return AdmitResult{Admit: false, Reason: "offline resource pod  must have BE QosLevel annotation"}
			}
			// 必须使用volcano.sh/preemptable标签
			if !VerifyPodPriorityAnnotation(annotations) {
				return AdmitResult{Admit: false, Reason: "offline resource pod  must have volcano.sh/preemptable annotation"}
			}
			return AdmitResult{Admit: true}
		}
	}
	return AdmitResult{Admit: true}
}

// GetPodCPUset returns the pod cpu set
func GetPodCPUset(annotations map[string]string) (string, error) {

	nr := NumaResourceSpec{}
	if len(annotations) == 0 {
		return "", nil
	}
	cfg, ok := annotations[common.NumaResourceAnnotation]
	if ok {
		err := json.Unmarshal([]byte(cfg), &nr)
		if err != nil {
			return nr.CPUset, err
		}
		return nr.CPUset, nil
	} else {
		return "", nil
	}
}

// GetCgroupDriver is the getter of global cgroup driver
func GetCgroupDriver() string {
	return cgroupDriver
}

// SetCgroupDriver is the setter of global cgroup driver
func SetCgroupDriver(driver string) {
	cgroupDriver = driver
}

func getQos(container *api.Container) string {
	var podQosClass string
	podQosClass = "Guaranteed"
	if strings.Contains(container.Linux.CgroupsPath, "burstable") {
		podQosClass = "Burstable"
	}
	if strings.Contains(container.Linux.CgroupsPath, "besteffort") {
		podQosClass = "BestEffort"
	}

	return podQosClass
}
func getPodCgroupDir(container *api.Container) string {
	var podPath string

	if GetCgroupDriver() == common.CgroupDriverSystemd {
		podPath = strings.Split(container.Linux.CgroupsPath, ":")[0]
	} else {
		pathInfo := strings.Split(container.Linux.CgroupsPath, "/")
		if len(pathInfo) > 0 {
			podPath = pathInfo[len(pathInfo)-podPathIndexFromEnd]
		}
	}
	return podPath
}

// GetKubeQoSByCgroupParent returns the kube qos class by cgroup parent
func GetKubeQoSByCgroupParent(parent string) v1.PodQOSClass {
	if strings.Contains(parent, "besteffort") {
		return v1.PodQOSBestEffort
	} else if strings.Contains(parent, "burstable") {
		return v1.PodQOSBurstable
	}
	return v1.PodQOSGuaranteed
}
