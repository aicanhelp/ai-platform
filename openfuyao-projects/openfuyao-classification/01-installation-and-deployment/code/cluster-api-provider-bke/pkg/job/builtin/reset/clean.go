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

package reset

import (
	"fmt"
	"strings"

	"github.com/blang/semver/v4"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/resetutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/runtime"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	dockerCleanKubelet = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker rm -f --volumes"
	// todo docker container clean
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

	// docker for all k8s containers
	dockerListContainers  = "docker ps -a --filter name=k8s_ -q | grep -v kubelet"
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
)

func CertClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean cert")
	if cfg == nil {
		return extra.CleanAll()
	}
	if cfg.Cluster.CertificatesDir != "" {
		extra.AddDirToClean(cfg.Cluster.CertificatesDir)
	}

	return extra.CleanAll()
}

// GlobalCertClean 删除全局证书目录 /etc/openFuyao/certs
func GlobalCertClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean global cert")
	extra.AddDirToClean("/etc/openFuyao/certs")
	return extra.CleanAll()
}

// KubeletCleanBin 删除systemd部署的kubelet
func KubeletCleanBin(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean kubelet install by systemd")
	cmd := "systemctl stop kubelet && systemctl disable kubelet"
	out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("umount kubelet directory failed: %s, %v", out, err)
	}

	rootDir := bkeinit.DefaultKubeletRootDir
	// unmount all mount points
	if err := resetutil.UnmountKubeletDirectory(rootDir); err != nil {
		log.Warnf("umount kubelet directory failed: %v", err)
	}
	// remove kubelet config dir
	extra.AddDirToClean(rootDir)
	extra.AddFileToClean(utils.KubeletSavePath)
	extra.AddFileToClean(utils.GetKubeletServicePath())

	// remove cni config file
	extra.AddFileToClean("/etc/cni/net.d/10-calico.conflist")
	// remove kubelet volume dir
	for _, dir := range utils.GetRunKubeletPreCreateDirs() {
		if shouldCleanKubeletDir(dir, false, "") {
			extra.AddDirToClean(dir)
		}
	}

	return extra.CleanAll()
}

// KubeletClean 卸载kubelet  待删除
func KubeletClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean kubelet")

	var (
		err error
		out string
	)

	//get kubelet root dir
	rootDir := bkeinit.DefaultKubeletRootDir
	if cfg.Cluster.Kubelet != nil && cfg.Cluster.Kubelet.ExtraVolumes != nil {
		for _, v := range cfg.Cluster.Kubelet.ExtraVolumes {
			if v.Name == "kubelet-root-dir" {
				rootDir = v.HostPath
				break
			}
		}
	}

	// unmount all mount points
	if err := resetutil.UnmountKubeletDirectory(rootDir); err != nil {
		log.Warnf("umount kubelet directory failed: %v", err)
	}

	cmd := fmt.Sprintf("for m in $(sudo tac /proc/mounts | sudo awk '{print $2}'|sudo grep %s | grep -v %s);do   sudo umount $m||true;   done", rootDir, rootDir)
	// 使用命令再来一次
	out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("umount kubelet directory failed: %s, %v", out, err)
	}

	containerRuntime := runtime.DetectRuntime()
	cleanKubeletContainer(extra, containerRuntime)
	cleanKubeletDirsAndFiles(cfg, extra, rootDir, containerRuntime)

	return extra.CleanAll()
}

// cleanKubeletContainer removes kubelet container based on container runtime
func cleanKubeletContainer(extra ExtraClean, containerRuntime string) {
	var (
		err error
		out string
	)
	switch containerRuntime {
	case runtime.ContainerRuntimeDocker:
		out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerStopKubelet)
		if err != nil {
			log.Warnf("stop kubelet container failed: %s, %v", out, err)
		} else {
			out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerRemoveKubelet)
			if err != nil {
				log.Warnf("remove kubelet container failed: %s, %v", out, err)
			}
		}
		if err != nil {
			out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerForceRemoveKubelet)
			if err != nil {
				log.Warnf("force remove kubelet container failed: %s, %v", out, err)
			}
		}
	case runtime.ContainerRuntimeContainerd:
		out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlStopKubelet)
		if err != nil {
			log.Warnf("stop kubelet container failed: %s, %v", out, err)
		} else {
			out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlRemoveKubelet)
			if err != nil {
				log.Warnf("remove kubelet container failed: %s, %v", out, err)
			}
		}
		if err != nil {
			out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlForceRemoveKubelet)
			if err != nil {
				log.Warnf("force remove kubelet container failed: %s, %v", out, err)
			}
		}
	default:
		log.Warnf("unsupported container runtime, skip clean kubelet container")
	}
	if err != nil {
		log.Warnf("clean kubelet container failed: %s , err:%s", out, err)
	}
}

// cleanKubeletDirsAndFiles adds kubelet directories and files to cleanup list
func cleanKubeletDirsAndFiles(cfg *bkev1beta1.BKEConfig, extra ExtraClean, rootDir string, containerRuntime string) {
	// remove kubelet pki dir
	extra.AddDirToClean(rootDir + "/pki")
	// remove kubelet config dir
	extra.AddDirToClean(rootDir)
	// remove cni config file
	extra.AddFileToClean("/etc/cni/net.d/10-calico.conflist")
	extra.AddFileToClean("/etc/cni/net.d/boc.conflist")
	v, ok := cfg.CustomExtra["allInOne"]
	allInOneFlag := ok && v == "true"
	// remove kubelet volume dir
	for _, dir := range utils.GetRunKubeletPreCreateDirs() {
		if shouldCleanKubeletDir(dir, allInOneFlag, containerRuntime) {
			extra.AddDirToClean(dir)
		}
	}
}

// shouldCleanKubeletDir returns whether a kubelet pre-create directory should be
// added to the cleanup list, preserving dirs that must not be removed.
func shouldCleanKubeletDir(dir string, allInOneFlag bool, containerRuntime string) bool {
	// 跳过网络配置目录，这个目录在ubuntu-server上面是不存在的
	if dir == "/etc/kubernetes" || dir == "/etc/sysconfig/network-scripts" {
		return false
	}
	// don't clean all lxc dir
	if dir == "/var/lib/lxc" {
		return false
	}
	// don't remove cni dir for containerd aready installed
	if dir == "/var/lib/cni" || dir == "/etc/cni" || dir == "/opt/cni" {
		if allInOneFlag {
			return false
		}
		if containerRuntime == runtime.ContainerRuntimeContainerd {
			return false
		}
		return false
	}
	return true
}

// ContainerCleanDocker 卸载docker容器的pod
func ContainerCleanDocker(extra ExtraClean) error {
	var (
		err error
		out string
	)
	out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerListContainers)
	if err != nil {
		log.Warnf("list containers failed: %s , err:%s", out, err)
		return nil
	}
	pods := strings.Fields(out)
	for _, pod := range pods {
		out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerStopContainer+" "+pod)
		if err != nil {
			log.Warnf("stop container failed: %s , err:%s", out, err)
		} else {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerRemoveContainer+" "+pod)
			if err != nil {
				log.Warnf("remove container failed: %s , err:%s", out, err)
			}
		}
		if err != nil {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerForceRemovePod+" "+pod)
			if err != nil {
				log.Warnf("force remove container failed: %s , err:%s", out, err)
			}
		}
	}
	return nil
}

// ContainerdCfgClean 删除containerd配置文件
func ContainerdCfgClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean containerd cfg")

	out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop containerd")
	if err != nil {
		return fmt.Errorf("stop containerd failed: %s, %w", out, err)
	}
	out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable containerd")
	if err != nil {
		return fmt.Errorf("disable containerd failed: %s, %w", out, err)
	}

	extra.AddFileToClean("/usr/bin/containerd")
	extra.AddFileToClean("/usr/bin/containerd-stress")
	extra.AddFileToClean("/usr/bin/containerd-shim-shimless-v2")
	extra.AddFileToClean("/usr/bin/containerd-shim-runc-v2")
	extra.AddFileToClean("/usr/bin/crictl")
	extra.AddFileToClean("/etc/crictl.yaml")
	extra.AddFileToClean("/usr/bin/ctr")
	extra.AddFileToClean("/usr/bin/nerdctl")
	extra.AddFileToClean("/usr/lib/systemd/system/containerd.service")
	extra.AddFileToClean("/usr/local/sbin/runc")
	extra.AddDirToClean("/usr/local/beyondvm")
	extra.AddDirToClean("/etc/containerd/")

	return extra.CleanAll()
}

// ContainerClean 卸载容器
func ContainerClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean containers")
	containerRuntime := runtime.DetectRuntime()
	switch containerRuntime {
	case runtime.ContainerRuntimeDocker:
		return ContainerCleanDocker(extra)
	case runtime.ContainerRuntimeContainerd:
		cleanContainerdContainers(extra)
	case "":
		log.Warnf("detect container runtime failed, skip remove containers")
		return nil
	default:
		log.Warnf("unsupported container runtime %q, skip remove containers", containerRuntime)
		return nil
	}
	return extra.CleanAll()
}

// cleanContainerdContainers removes all containers in containerd runtime
func cleanContainerdContainers(extra ExtraClean) {
	out, err := extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlListContainers)
	if err != nil {
		log.Warnf("list containers failed: %s , err:%s", out, err)
		return
	}
	pods := strings.Fields(out)
	for _, pod := range pods {
		cleanContainerdContainer(extra, pod)
	}
}

// cleanContainerdContainer removes a single container in containerd
func cleanContainerdContainer(extra ExtraClean, pod string) {
	out, err := extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlStopContainer+" "+pod)
	if err != nil {
		log.Warnf("stop container %s failed: %s , err:%s", pod, out, err)
		// Try force remove if stop failed
		if out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod); err != nil {
			log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
		}
		return
	}
	// Try normal remove if stop succeeded
	out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlRemoveContainer+" "+pod)
	if err != nil {
		log.Warnf("remove container %s failed: %s , err:%s", pod, out, err)
		// Try force remove if normal remove failed
		if out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod); err != nil {
			log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
		}
	}
}

func ContainerRuntimeClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean container runtime")
	if v, ok := cfg.CustomExtra["allInOne"]; ok && v == "true" {
		log.Infof("this node is all in one mode, skip clean container runtime")
		return nil
	}
	containerRuntime := runtime.DetectRuntime()
	switch containerRuntime {
	case runtime.ContainerRuntimeDocker:
		if err := cleanDockerRuntime(cfg, extra); err != nil {
			return err
		}
	case runtime.ContainerRuntimeContainerd:
		if err := cleanContainerdRuntime(cfg, extra); err != nil {
			return err
		}
	case "":
		log.Info("no available container runtime found")
		extra.AddDirToClean("/etc/cni")
		extra.AddDirToClean("/opt/cni")
	default:
		log.Warnf("unsupported container runtime %q, skip clean", containerRuntime)
	}
	return extra.CleanAll()
}

// cleanDockerRuntime cleans Docker container runtime
func cleanDockerRuntime(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	dataRoot := bkeinit.DefaultCRIDockerDataRootDir
	if cfg.Cluster.ContainerRuntime.Param != nil {
		if v, ok := cfg.Cluster.ContainerRuntime.Param["data-root"]; ok {
			dataRoot = v
		}
	}
	out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerListAllContainers)
	if err != nil {
		return fmt.Errorf("list all containers failed: %s, %w", out, err)
	}
	for _, pod := range strings.Fields(out) {
		if out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerForceRemovePod+" "+pod); err != nil {
			log.Warnf("force remove container failed: %s , err:%s", out, err)
		}
	}
	if out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerCleanAll); err != nil {
		return fmt.Errorf("clean docker failed: %s, %w", out, err)
	}
	output, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop docker")
	if err != nil {
		return fmt.Errorf("stop docker failed: %s, %w", output, err)
	}
	if output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable docker"); err != nil {
		return fmt.Errorf("disable docker failed: %s, %w", output, err)
	}
	if err := httprepo.RepoRemove("docker*", "containerd.io"); err != nil {
		log.Warnf("remove docker packages failed: %v", err)
	}
	extra.AddDirToClean(dataRoot)
	extra.AddDirToClean("/etc/docker")
	extra.AddDirToClean("/var/lib/cni")
	extra.AddDirToClean("/etc/cni")
	extra.AddDirToClean("/opt/cni")
	v, _ := semver.ParseTolerant(cfg.Cluster.KubernetesVersion)
	if v.GTE(semver.MustParse("1.24.0")) {
		output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop cri-dockerd && systemctl stop cri-dockerd.socket")
		if err != nil {
			return fmt.Errorf("stop cri-dockerd failed: %s, %w", output, err)
		}
		if output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable cri-dockerd && systemctl disable cri-dockerd.socket"); err != nil {
			return fmt.Errorf("disable cri-dockerd failed: %s, %w", output, err)
		}
		extra.AddFileToClean("/usr/bin/cri-dockerd")
		extra.AddFileToClean("/etc/systemd/system/cri-dockerd.service")
		extra.AddFileToClean("/etc/systemd/system/cri-dockerd.socket")
	}
	return extra.CleanAll()
}

// cleanContainerdRuntime cleans Containerd container runtime
func cleanContainerdRuntime(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	dataRoot := bkeinit.DefaultCRIContainerdDataRootDir
	if cfg.Cluster.ContainerRuntime.Param != nil {
		if v, ok := cfg.Cluster.ContainerRuntime.Param["data-root"]; ok {
			dataRoot = v
		}
	}
	if out, err := extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlListContainers); err == nil {
		for _, pod := range strings.Fields(out) {
			if out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod); err != nil {
				log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
			}
		}
	}
	if out, err := extra.ExecuteCommandWithOutput("/bin/sh", "-c", nerdctlListContainers); err == nil {
		for _, pod := range strings.Fields(out) {
			if out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", nerdctlForceRemoveContainer+" "+pod); err != nil {
				log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
			}
		}
	}
	out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop containerd")
	if err != nil {
		return fmt.Errorf("stop containerd failed: %s, %w", out, err)
	}
	if out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable containerd"); err != nil {
		return fmt.Errorf("disable containerd failed: %s, %w", out, err)
	}
	extra.AddFileToClean("/usr/bin/containerd")
	extra.AddFileToClean("/usr/bin/containerd-shim")
	extra.AddFileToClean("/usr/bin/containerd-shim-runc-v2")
	extra.AddFileToClean("/usr/bin/crictl")
	extra.AddFileToClean("/etc/crictl.yaml")
	extra.AddFileToClean("/usr/bin/ctr")
	extra.AddFileToClean("/usr/bin/nerdctl")
	extra.AddFileToClean("/usr/bin/containerd-stress")
	extra.AddFileToClean("/usr/lib/systemd/system/containerd.service")
	extra.AddFileToClean("/usr/local/sbin/runc")
	extra.AddDirToClean("/usr/local/beyondvm")
	extra.AddDirToClean("/etc/containerd/")
	extra.AddDirToClean(dataRoot)
	extra.AddDirToClean("/etc/systemd/system/containerd.service.d")
	extra.AddDirToClean("/var/lib/cni")
	extra.AddDirToClean("/etc/cni")
	extra.AddDirToClean("/opt/cni")
	extra.AddDirToClean("/var/lib/nerdctl")
	extra.AddDirToClean("/etc/docker/certs.d")
	return extra.CleanAll()
}

func SourceClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	if err := source.ResetSource(); err != nil {
		log.Infof("reset http repo failed, err: %v", err)
		return err
	}
	if err := httprepo.RepoUpdate(); err != nil {
		log.Infof("update http repo failed, err: %v", err)
		return err
	}
	log.Infof("reset http repo source success")
	return nil
}

func ManifestsClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean manifests")
	if cfg == nil {
		return extra.CleanAll()
	}
	extra.AddDirToClean(mfutil.HAProxyConfPath)
	extra.AddDirToClean(mfutil.KeepAlivedConfPath)
	// keep in sync with the kubeadm/nginx plugin default confPath
	extra.AddDirToClean("/etc/kubernetes/nginx-proxy")

	if cfg.Cluster.Kubelet.ManifestsDir != "" {
		extra.AddDirToClean(cfg.Cluster.Kubelet.ManifestsDir)
	}

	return extra.CleanAll()
}

func ExtraToClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean extra")
	extra.AddFileToClean("/usr/bin/calicoctl")
	extra.AddDirToClean("/etc/calico")

	cleanIptables := func(cmd, name string) {
		if out, err := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd); err != nil {
			log.Warnf("clean %s rule failed: %s, %v", name, out, err)
		}
	}

	if v, ok := cfg.CustomExtra["allInOne"]; !ok || v == "false" {
		extra.AddDirToClean("/etc/openFuyao/addons")
		extra.AddFileToClean("/usr/bin/kubectl")
		cleanIptables("iptables -F -t raw && iptables -F -t filter && iptables -t nat -F && iptables -t mangle -F && iptables -X -t nat && iptables -X -t raw && iptables -X -t mangle && iptables -X -t filter", "iptables")
		cleanIptables("iptables-legacy -F -t raw && iptables-legacy -F -t filter && iptables-legacy -t nat -F && iptables-legacy -t mangle -F && iptables-legacy -X -t nat && iptables-legacy -X -t raw && iptables-legacy -X -t mangle && iptables-legacy -X -t filter", "iptables-legacy")
		cleanIptables("ip6tables -F -t raw && ip6tables -F -t filter && ip6tables -t nat -F && ip6tables -t mangle -F && ip6tables -X -t nat && ip6tables -X -t raw && ip6tables -X -t mangle && ip6tables -X -t filter", "iptables6")
	}

	cleanIPRules := func(cmdGet, cmdDelPrefix string, desc string) {
		out, err := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmdGet)
		if err != nil {
			log.Warnf("get %s failed: %s, %v", desc, out, err)
			return
		}
		for _, item := range strings.Split(out, "\n") {
			if item != "" {
				log.Infof("delete %s: %s", desc, item)
				if out2, err2 := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmdDelPrefix+" "+item); err2 != nil {
					log.Warnf("delete %s failed: %s, %v", desc, out2, err2)
				}
			}
		}
	}

	cleanIPRules(`ip route show all | awk '($1!="default" && $3~/^(boc0|cali.*)$/) {printf "%s %s %s\n",$1,$2,$3}'`, "ip route del", "route")
	cleanIPRules(`ip neigh show all | awk '($3~/^(boc0|cali.*)$/) {print}' | awk '$4 == "lladdr" || $4 == "FAILED" {printf "%s %s %s\n",$1,$2,$3}'`, "ip neigh del", "neighbor")
	cleanNetworkInterfaces(extra)
	extra.AddDirToClean("/etc/kubernetes")
	extra.AddDirToClean(cfg.Cluster.Etcd.DataDir)
	return extra.CleanAll()
}

// cleanNetworkInterfaces removes network interfaces used by CNI
func cleanNetworkInterfaces(extra ExtraClean) {
	needRemoveInters := []string{"vxlan_sys_4789", "gre_sys", "genev_sys_6081", "erspan_sys", "vxlan.calico", "tunl0"}
	out, err := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", `ip link | awk '/state/ {gsub(/:/, ""); print $2}'`)
	if err != nil {
		return
	}
	for _, inter := range strings.Split(out, "\n") {
		inter = strings.Split(inter, "@")[0]
		if inter == "" || (!utils.ContainsString(needRemoveInters, inter) && !strings.HasPrefix(inter, "cali")) {
			continue
		}
		log.Infof("down interface: %s", inter)
		output, err := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "ip link set "+inter+" down")
		if err != nil {
			log.Warnf("down interface failed: %s, %v", output, err)
		}
		log.Infof("delete interface: %s", inter)
		output, err = extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "ip link del "+inter)
		if err != nil {
			log.Warnf("delete interface failed: %s, %v", output, err)
		}
	}
}

// BKEAgentShutdownClean schedules delayed bkeagent uninstall immediately.
// Self-manage delete often loses the API during Reset, so stop cannot wait for
// status writeback. stopDelaySeconds still leaves a window for writeback when
// the API is reachable. Marking/scheduling failures must not fail Reset.
func BKEAgentShutdownClean(_ *bkev1beta1.BKEConfig, _ ExtraClean) error {
	log.Infof("Clean bkeagent (delayed stop scheduled; status writeback best-effort)")
	MarkAgentCleanupPending()
	if err := scheduleAgentCleanupOnce(); err != nil {
		log.Warnf("schedule bkeagent cleanup failed (best-effort): %v", err)
	}
	return nil
}
