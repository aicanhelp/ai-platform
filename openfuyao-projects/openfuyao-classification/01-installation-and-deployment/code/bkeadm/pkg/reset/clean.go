/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package reset

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/host"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var needDeleteFile = []string{}

const (
	dockerCleanKubelet = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker rm -f --volumes"
	// todo docker container clean
	// rm all containers in k8s.io namespace
	nerdctlCleanContainer = "nerdctl -n k8s.io ps -a | grep -v CONTAINER | awk '{print $1}' | xargs nerdctl -n k8s.io rm -f -v"
	// umount all mount points in k8s.io namespace
	kubeletDirUnmont = "for m in $(sudo tac /proc/mounts | sudo awk '{print $2}'|sudo grep /var/lib/kubelet);do   sudo umount $m||true;   done"

	// nerdctl only for kubelet container
	nerdctlStopKubelet        = "nerdctl -n k8s.io ps -a | grep kubelet | awk '{print $1}' | xargs nerdctl -n k8s.io stop"
	nerdctlRemoveKubelet      = "nerdctl -n k8s.io ps -a | grep kubelet | awk '{print $1}' | xargs nerdctl -n k8s.io rm --volumes"
	nerdctlForceRemoveKubelet = "nerdctl -n k8s.io ps -a | grep kubelet | awk '{print $1}' | xargs nerdctl -n k8s.io rm -f --volumes"

	// docker only for kubelet container
	dockerStopKubelet        = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker stop"
	dockerRemoveKubelet      = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker rm --volumes"
	dockerForceRemoveKubelet = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker rm -f --volumes"

	// docker for all k8s containers.
	// "|| true" avoids grep exit 1 when there is no matching container.
	dockerListContainers  = "docker ps -a --filter name=k8s_ -q | grep -v kubelet || true"
	dockerStopContainer   = "docker stop"
	dockerRemoveContainer = "docker rm --volumes"
	dockerForceRemovePod  = "docker rm -f --volumes"
	// docker 清理所有数据
	dockerCleanAll = "docker system prune -a -f --volumes"
	// docker 列出所有容器
	dockerListAllContainers = "docker ps -a -q"

	// crictl for all containers
	crictlListContainers  = "crictl pods -q"
	crictlStopContainer   = "crictl stopp"
	crictlRemoveContainer = "crictl rmp"
	crictlForceRemovePod  = "crictl rmp -f"
	// crictl 删除所有容器
	crictlCleanAllContainer = "crictl rmp -a -f"

	// nerdctl for all containers
	nerdctlListContainers       = "nerdctl ps -a -q"
	nerdctlForceRemoveContainer = "nerdctl rm -f --volumes"
	// nerdctl 清理所有数据
	nerdctlCleanAll = "nerdctl --namespace k8s.io system prune -a -f --volumes && nerdctl system image prune -a -f"

	// StopKubeletBin停止二进制安装的kubelet
	StopKubeletBin = "systemctl stop kubelet  && systemctl disable kubelet && rm -f /etc/systemd/system/kubelet.service"

	// RemoveKubeletBin卸载二进制安装的kubelet
	RemoveKubeletBin = "sudo rm -f $(which kubelet)"

	// ForceRemoveKubeletBin强制删除
	ForceRemoveKubeletBin = "sudo rm -rf /var/lib/kubelet " +
		"&& sudo rm -rf /usr/bin/kubelet " +
		"&& sudo rm -rf /etc/systemd/system/kubelet.service"
)

func cleanIpRoute() {
	// clean ip route about boc0 and cali
	cmd := `ip route show all | awk '($1 != "default" && $3 ~ /^(boc0|cali.*)$/) {printf "%s %s %s\n", $1, $2, $3}'`
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("get route failed: %v, output: %s", err, output)
		return
	}
	routes := strings.Split(output, "\n")
	for _, route := range routes {
		// delete route
		if route != "" {
			output, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
				"ip route del "+route)
			if err != nil {
				log.Warnf("delete route failed: %v, output: %s", err, output)
			}
		}
	}
}

func cleanIpNeighbor() {
	// clean ip neighbor about boc0 and cali
	cmd := `ip neigh show all | awk '($3~/^(boc0|cali.*)$/) {print}' | awk '$4 == "lladdr" || 
$4 == "FAILED" {printf "%s %s %s\n", $1, $2, $3}'`
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("delete neighbor failed: %v, output: %s", err, output)
		return
	}
	neighbors := strings.Split(output, "\n")
	for _, neighbor := range neighbors {
		// delete neighbor
		if neighbor != "" {
			output, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
				"ip neigh del "+neighbor)
			if err != nil {
				log.Warnf("delete neighbor failed: %v, output: %s", err, output)
			}
		}
	}
}

func cleanIpLink() {
	needRemoveInters := []string{"vxlan_sys_4789", "gre_sys", "genev_sys_6081", "erspan_sys", "vxlan.calico", "tunl0"}
	cmd := `ip link | awk '/state/ {gsub(/:/, ""); print $2}'`
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("get interface failed: %v, output: %s", err, output)
		return
	}
	inters := strings.Split(output, "\n")
	downCount := 0
	delCount := 0
	for _, inter := range inters {
		inter = strings.Split(inter, "@")[0]
		if inter == "" || (!utils.ContainsString(needRemoveInters, inter) && !strings.HasPrefix(inter, "cali")) {
			continue
		}
		log.Debugf("down interface: %s", inter)
		output, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
			"ip link set "+inter+" down")
		if err != nil {
			log.Warnf("down interface %s failed: %v, output: %s", inter, err, output)
		}
		downCount++
		log.Debugf("delete interface: %s", inter)
		output, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
			"ip link del "+inter)
		if err != nil {
			log.Warnf("delete interface %s failed: %v, output: %s", inter, err, output)
		}
		delCount++
	}
	if downCount > 0 || delCount > 0 {
		log.Infof("Interface cleanup completed: down=%d, deleted=%d", downCount, delCount)
	}
}

func cleanNetwork() {
	log.Info("clean network...")
	cleanIptablesRules()

	// clean ip route
	cleanIpRoute()

	// clean ip neighbor about boc0 and cali
	cleanIpNeighbor()

	// clean virtual interface about calico and fabric
	// fabric interface name vxlan_sys_4789 gre_sys genev_sys_6081 erspan_sys
	// calico interface name
	cleanIpLink()

	deleteInterfaceIfExists("nerdctl0")
	deleteInterfaceIfExists("docker0")

	needDeleteFile = append(needDeleteFile, []string{
		"/etc/kubernetes",
	}...)
}

// cleanIptablesRules flushes iptables rules when the iptables binary is available.
func cleanIptablesRules() {
	if _, err := exec.LookPath("iptables"); err != nil {
		log.Debugf("iptables not found, skip cleaning iptables rules: %v", err)
		return
	}
	cmd := "iptables -F -t raw && iptables -F -t filter && iptables -t nat -F && iptables -t mangle -F && iptables -X -t nat && iptables -X -t raw && iptables -X -t mangle && iptables -X -t filter"
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("clean iptables rule failed: %v, output: %s", err, output)
	}
}

// deleteInterfaceIfExists deletes a network interface only when it currently exists.
func deleteInterfaceIfExists(name string) {
	if _, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
		"ip link show "+name); err != nil {
		log.Debugf("interface %s not found, skip deleting: %v", name, err)
		return
	}
	output, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", "ip link del "+name)
	if err != nil {
		log.Warnf("delete interface %s failed: %v, output: %s", name, err, output)
	}
}

// isSystemdUnitNotLoaded reports whether systemctl failed because the unit is absent.
func isSystemdUnitNotLoaded(output string) bool {
	text := strings.ToLower(output)
	return strings.Contains(text, "not loaded") ||
		strings.Contains(text, "could not be found") ||
		strings.Contains(text, "not found")
}

func cleanKubeletBin() {
	log.Info("clean kubelet...")
	var (
		err error
		out string
	)

	out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", StopKubeletBin)
	if err != nil {
		if isSystemdUnitNotLoaded(out) {
			log.Debugf("stop kubelet skipped, unit not loaded: %v, output: %s", err, out)
		} else {
			log.Warnf("stop kubelet container failed: %v, output: %s", err, out)
		}
	} else {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", RemoveKubeletBin)
		if err != nil {
			log.Warnf("remove kubelet container failed: %v, output: %s", err, out)
		}
	}
	if err != nil {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", ForceRemoveKubeletBin)
		if err != nil {
			log.Warnf("force remove kubelet container failed: %v, output: %s", err, out)
		}
	}

	needDeleteFile = append(needDeleteFile, []string{
		"/etc/cni/net.d/10-calico.conflist",
		"/etc/kubernetes",
	}...)
}

func cleanKubelet() {
	log.Info("clean kubelet...")
	rootDir := bkeinit.DefaultKubeletRootDir

	unmountKubeletDirectories(rootDir)
	cleanKubeletContainer()
	appendKubeletCleanupFiles(rootDir)
}

// unmountKubeletDirectories 卸载 kubelet 相关目录
func unmountKubeletDirectories(rootDir string) {
	if err := unmountKubeletDirectory(rootDir); err != nil {
		log.Warnf("umount kubelet directory failed: %v", err)
	}

	cmd := fmt.Sprintf("for m in $(sudo tac /proc/mounts | "+
		"sudo awk '{print $2}'|sudo grep %s | grep -v %s);do   sudo umount $m||true;   done", rootDir, rootDir)
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("umount kubelet directory failed: %v, output: %s", err, out)
	}
}

// cleanKubeletContainer 清理 kubelet 容器
func cleanKubeletContainer() {
	if infrastructure.IsDocker() {
		cleanDockerKubelet()
	}
	if infrastructure.IsContainerd() {
		cleanContainerdKubelet()
	}
}

// cleanDockerKubelet 使用 Docker 清理 kubelet 容器
func cleanDockerKubelet() {
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerStopKubelet)
	if err != nil {
		log.Warnf("stop kubelet container failed: %v, output: %s", err, out)
	} else {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerRemoveKubelet)
		if err != nil {
			log.Warnf("remove kubelet container failed: %v, output: %s", err, out)
		}
	}
	if err != nil {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerForceRemoveKubelet)
		if err != nil {
			log.Warnf("force remove kubelet container failed: %v, output: %s", err, out)
		}
	}
}

// cleanContainerdKubelet 使用 Containerd 清理 kubelet 容器
func cleanContainerdKubelet() {
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlStopKubelet)
	if err != nil {
		log.Warnf("stop kubelet container failed: %v, output: %s", err, out)
	} else {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlRemoveKubelet)
		if err != nil {
			log.Warnf("remove kubelet container failed: %v, output: %s", err, out)
		}
	}
	if err != nil {
		out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlForceRemoveKubelet)
		if err != nil {
			log.Warnf("force remove kubelet container failed: %v, output: %s", err, out)
		}
	}
}

// appendKubeletCleanupFiles 添加 kubelet 需要删除的文件
func appendKubeletCleanupFiles(rootDir string) {
	needDeleteFile = append(needDeleteFile, rootDir+"/pki")
	needDeleteFile = append(needDeleteFile, rootDir)
	needDeleteFile = append(needDeleteFile, []string{
		rootDir + "/pki",
		rootDir,
		"/etc/cni/net.d/10-calico.conflist",
		"/etc/cni/net.d/boc.conflist",
		"/etc/kubernetes",
	}...)
}

func cleanKubernetesContainer() {
	log.Info("clean kubernetes container...")

	cleanDockerK8sContainers()
	cleanContainerdK8sContainers()

	rootDir := bkeinit.DefaultKubeletRootDir
	unmountKubeletDirectories(rootDir)
	needDeleteFile = append(needDeleteFile, rootDir)
}

// cleanDockerK8sContainers 清理 Docker 中的 K8s 容器
func cleanDockerK8sContainers() bool {
	if !infrastructure.IsDocker() {
		return false
	}
	pods, ok := listContainerIDs(dockerListContainers)
	if !ok {
		return false
	}
	for _, pod := range pods {
		removeDockerContainer(pod)
	}
	return true
}

func isEmptyListOutput(out string) bool {
	out = strings.TrimSpace(out)
	return out == "" || out == "."
}

// listContainerIDs runs a container list command; empty result is treated as nothing to clean.
func listContainerIDs(listCmd string) ([]string, bool) {
	out, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", listCmd)
	if err != nil {
		if isEmptyListOutput(out) {
			return nil, false
		}
		log.Warnf("list containers failed: %v, output: %s", err, out)
		return nil, false
	}
	if isEmptyListOutput(out) {
		return nil, false
	}
	return strings.Fields(out), true
}

// removeDockerContainer 移除单个 Docker 容器
func removeDockerContainer(pod string) {
	out, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", dockerStopContainer+" "+pod)
	if err != nil {
		log.Warnf("stop container %s failed: %v, output: %s", pod, err, out)
	} else {
		out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", dockerRemoveContainer+" "+pod)
		if err != nil {
			log.Warnf("remove container %s failed: %v, output: %s", pod, err, out)
		}
	}
	if err != nil {
		out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", dockerForceRemovePod+" "+pod)
		if err != nil {
			log.Warnf("force remove container %s failed: %v, output: %s; cleaning residual processes as fallback",
				pod, err, out)
			killContainerPid(pod)
			time.Sleep(utils.ContainerRemoveWaitSeconds * time.Second)
			out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", dockerForceRemovePod+" "+pod)
			if err != nil {
				log.Warnf("force remove container %s still failed after process cleanup: %v, output: %s", pod, err, out)
			}
		}
	}
}

// cleanContainerdK8sContainers 清理 Containerd 中的 K8s 容器
func cleanContainerdK8sContainers() {
	if !infrastructure.IsContainerd() {
		return
	}
	pods, ok := listContainerIDs(crictlListContainers)
	if !ok {
		return
	}
	for _, pod := range pods {
		removeContainerdContainer(pod)
	}
}

// removeContainerdContainer 移除单个 Containerd 容器
func removeContainerdContainer(pod string) {
	out, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", crictlStopContainer+" "+pod)
	if err != nil {
		log.Warnf("stop container %s failed: %v, output: %s", pod, err, out)
	} else {
		out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", crictlRemoveContainer+" "+pod)
		if err != nil {
			log.Warnf("remove container %s failed: %v, output: %s", pod, err, out)
		}
	}
	if err != nil {
		out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod)
		if err != nil {
			log.Warnf("force remove container %s failed: %v, output: %s; cleaning residual processes as fallback",
				pod, err, out)
			killCrictlPodPid(pod)
			time.Sleep(utils.ContainerRemoveWaitSeconds * time.Second)
			out, err = global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod)
			if err != nil {
				log.Warnf("force remove container %s still failed after process cleanup: %v, output: %s", pod, err, out)
			}
		}
	}
}

func cleanOtherContainer() {
	log.Info("clean other container...")
	cleanDockerAllContainers()
	cleanContainerdAllContainers()
}

// cleanDockerAllContainers 清理所有 Docker 容器
func cleanDockerAllContainers() {
	if !infrastructure.IsDocker() {
		return
	}
	log.Infof("Skip cleaning all docker containers to preserve docker environment.")
}

// cleanContainerdAllContainers 清理所有 Containerd 容器
func cleanContainerdAllContainers() {
	if !infrastructure.IsContainerd() {
		return
	}
	cleanCrictlContainers()
	cleanNerdctlContainers()
}

// cleanCrictlContainers 使用 crictl 清理容器
func cleanCrictlContainers() {
	pods, ok := listContainerIDs(crictlListContainers)
	if !ok {
		return
	}
	for _, pod := range pods {
		out, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod)
		if err != nil {
			log.Warnf("force remove container %s failed: %v, output: %s", pod, err, out)
		}
	}
}

// cleanNerdctlContainers 使用 nerdctl 清理容器
func cleanNerdctlContainers() {
	pods, ok := listContainerIDs(nerdctlListContainers)
	if !ok {
		return
	}
	for _, pod := range pods {
		out, err := global.Command.ExecuteCommandWithOutput("/bin/sh", "-c", nerdctlForceRemoveContainer+" "+pod)
		if err != nil {
			log.Warnf("force remove container %s failed: %v, output: %s", pod, err, out)
		}
	}
}

func disableDocker() {
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop docker")
	if err != nil {
		log.Debugf("stop docker failed: %v, output: %s", err, output)
	}
	output, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable docker")
	if err != nil {
		log.Debugf("disable docker failed: %v, output: %s", err, output)
	}
}

func disableContainerd() {
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop containerd")
	if err != nil {
		log.Debugf("stop containerd failed: %v, output: %s", err, out)
	}
	out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable containerd")
	if err != nil {
		log.Debugf("disable containerd failed: %v, output: %s", err, out)
	}
}

func disableBkeAgent() {
	out, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl cat bkeagent.service")
	if err != nil {
		log.Debugf("bkeagent unit not found, skip stop/disable: %v, output: %s", err, out)
		return
	}
	out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop bkeagent")
	if err != nil {
		log.Debugf("stop bkeagent failed: %v, output: %s", err, out)
	}
	out, err = global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable bkeagent")
	if err != nil {
		log.Debugf("disable bkeagent failed: %v, output: %s", err, out)
	}
}

func addBkeAgentDeleteFile() {
	matches, err := filepath.Glob("/usr/local/bin/bkeagent*")
	if err != nil {
		log.Warnf("glob /usr/local/bin/bkeagent* failed: %v", err)
	} else {
		needDeleteFile = append(needDeleteFile, matches...)
	}
	needDeleteFile = append(needDeleteFile, []string{
		"/usr/local/bin/bkeagent",
		"/etc/systemd/system/bkeagent.service",
		"/usr/lib/systemd/system/bkeagent.service",
		"/etc/systemd/system/bkeagent.service.d",
		"/usr/lib/systemd/system/bkeagent.service.d",
		"/etc/openFuyao/bkeagent",
		"/etc/bkeagent",
	}...)
}

func cleanBkeAgent() {
	log.Info("clean bkeagent...")
	disableBkeAgent()
	addBkeAgentDeleteFile()
}

func addNeedDeleteFile() {
	skipDockerDelete := infrastructure.IsDocker()

	if !skipDockerDelete {
		needDeleteFile = append(needDeleteFile, []string{
			bkeinit.DefaultCRIDockerDataRootDir,
			"/etc/docker",
		}...)
	}

	needDeleteFile = append(needDeleteFile, []string{
		"/var/lib/cni",
		"/etc/cni",
		"/opt/cni",
	}...)

	needDeleteFile = append(needDeleteFile, []string{
		"/usr/bin/containerd",
		"/usr/bin/containerd-shim",
		"/usr/bin/containerd-shim-runc-v1",
		"/usr/bin/containerd-shim-runc-v2",
		"/usr/bin/crictl",
		"/etc/crictl.yaml",
		"/usr/bin/ctr",
		"/usr/bin/nerdctl",
		"/usr/bin/containerd-stress",
		"/usr/lib/systemd/system/containerd.service",
		"/usr/local/sbin/runc",
		"/etc/containerd",
		"/usr/local/beyondvm",
		bkeinit.DefaultCRIContainerdDataRootDir,
		"/var/lib/cni",
		"/etc/cni",
		"/opt/cni",
		"/var/lib/nerdctl",
	}...)

	if !skipDockerDelete {
		needDeleteFile = append(needDeleteFile, []string{
			"/etc/docker/certs.d",
			"/usr/bin/docker",
			"/usr/bin/dockerd",
			"/usr/bin/docker-init",
			"/usr/bin/docker-proxy",
			"/usr/bin/runc",
			"/usr/lib/systemd/system/docker.service",
			"/etc/systemd/system/containerd.service.d",
		}...)
	} else {
		log.Infof("Skip adding docker related files to delete list because docker is installed.")
	}
}

func cleanContainerRuntime() {
	log.Info("clean container runtime...")

	if infrastructure.IsDocker() {
		log.Infof("Skip uninstalling docker and disabling docker service to preserve docker environment.")
	}

	if infrastructure.IsContainerd() {
		disableContainerd()
	}

	addNeedDeleteFile()
}

func cleanNeedDeleteFile() {
	log.Info("clean file...")
	needDeleteFile = append(needDeleteFile, []string{
		"/etc/openFuyao/haproxy",
		"/etc/openFuyao/keepalived",
		"/usr/bin/calicoctl",
		"/etc/calico",
		"/usr/bin/kubectl",
		"/etc/openFuyao/addons",
		fmt.Sprintf("%s/.kube/config", os.Getenv("HOME")),
	}...)

	for _, f := range needDeleteFile {
		if err := removePath(f); err != nil {
			log.Warnf("remove path %s failed: %v", f, err)
		}
	}
}

// removePath removes a file or directory based on its type.
func removePath(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if info.IsDir() {
		return removeDir(path)
	}
	return removeFile(path)
}

// removeDir removes the directory and all of its contents in one shot.
// Using os.RemoveAll on the root avoids deleting child entries one-by-one, which
// can leave a partially cleaned tree (e.g. containerd content gone but metadata left).
func removeDir(filePath string) error {
	if s, err := os.Stat(filePath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	} else if !s.IsDir() {
		return errors.New(fmt.Sprintf("path %s is not a directory", filePath))
	}
	return os.RemoveAll(filePath)
}

func removeFile(filePath string) error {
	if s, err := os.Stat(filePath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	} else if s.IsDir() {
		return errors.New(fmt.Sprintf("path %s is a directory", filePath))
	}
	return os.Remove(filePath)
}

// unmountKubeletDirectory unmounts all directories that are mounted under the kubelet run directory.
func unmountKubeletDirectory(absoluteKubeletRunDirectory string) error {
	raw, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return err
	}

	if !strings.HasSuffix(absoluteKubeletRunDirectory, "/") {
		// "/" 用于确保路径以斜杠结尾，防止匹配例如 "/var/lib/kubelet"
		absoluteKubeletRunDirectory += "/"
	}

	mounts := strings.Split(string(raw), "\n")
	for _, mount := range mounts {
		//转义空格
		mount = strings.ReplaceAll(mount, `\040`, " ")
		m := strings.Split(mount, " ")

		if len(m) < utils.MatchFields || !strings.HasPrefix(m[1], absoluteKubeletRunDirectory) {
			continue
		}

		// 排除absoluteKubeletRunDirectory本身
		if m[1] == absoluteKubeletRunDirectory || m[1] == absoluteKubeletRunDirectory[:len(absoluteKubeletRunDirectory)-1] {
			continue
		}

		if err = syscall.Unmount(m[1], 0); err != nil {
			log.Warnf("Failed to unmount mounted directory in %s: %s, err: %v", absoluteKubeletRunDirectory, m[1], err)
		}
	}
	return nil
}

func repoRemove(packages ...string) error {
	packageManager := "unknown"
	h, _, _, err := host.PlatformInformation()
	if err != nil {
		log.Warnf("get host platform information failed, err: %v", err)
	}
	switch h {
	case "ubuntu", "debian":
		packageManager = "apt"
	case "centos", "kylin", "redhat", "fedora", "openeuler", "uos":
		packageManager = "yum"
	default:
		packageManager = "unknown"
	}
	output, err := global.Command.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s remove -y %s", packageManager, strings.Join(packages, " ")))
	if err != nil {
		return errors.New(fmt.Sprintf("remove packages %q failed, err: %v, out: %s", strings.Join(packages, " "), err, output))
	}
	return nil
}
