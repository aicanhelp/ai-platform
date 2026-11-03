package reset

import (
	"fmt"
	"strings"

	"github.com/blang/semver/v4"
	"github.com/pkg/errors"
	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	"gopkg.openfuyao.cn/bkecommon/source"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/httprepo"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/mfutil"
	"gopkg.openfuyao.cn/bkeagent/utils/resetutil"
	"gopkg.openfuyao.cn/bkeagent/utils/runtime"
)

const (
	dockerCleanKubelet = "docker ps -a | grep kubelet | awk '{print $1}' | xargs docker rm -f --volumes"
	// todo docker container clean
	// dockerCleanContainer = "docker ps -a --filter name=k8s_ | grep -v kubelet | awk '{print $1}' | xargs docker rm -f"
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

// KubeletCleanBin 删除systemd部署的kubelet
func KubeletCleanBin(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	log.Infof("Clean kubelet install by systemd")
	cmd := "systemctl stop kubelet && systemctl disable kubelet"
	out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("umount kubelet directory failed: %s, %v", out, err)
	}

	rootDir := bkeinit.DefaultKubeletRootDir
	// remove kubelet config dir
	extra.AddDirToClean(rootDir)
	extra.AddFileToClean(utils.KubeletSavePath)
	extra.AddFileToClean(utils.GetKubeletServicePath())

	// remove cni config file
	extra.AddFileToClean("/etc/cni/net.d/10-calico.conflist")

	value, ok := cfg.CustomExtra["allInOne"]
	allInOneFlag := ok && value == "true"
	// remove kubelet volume dir
	for _, dir := range utils.RunKubeletPreCreateDirs {
		// 现在先不要删，放到额外清理里面
		if dir == "/etc/kubernetes" {
			continue
		}
		// 跳过网络配置目录，这个目录在ubuntu-server上面是不存在的
		if dir == "/etc/sysconfig/network-scripts" {
			continue
		}
		// don't remove cni dir for containerd aready installed
		if dir == "/var/lib/cni" || dir == "/etc/cni" || dir == "/opt/cni" {
			if allInOneFlag {
				continue
			}
			if runtime.DetectRuntime() == runtime.ContainerRuntimeContainerd {
				continue
			}
			continue
		}
		extra.AddDirToClean(dir)
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

	// remove kubelet
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
		log.Error("unsupported container runtime, skip clean kubelet container")
	}
	if err != nil {
		log.Warnf("clean kubelet container failed: %s , err:%s", out, err)
	}

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
	for _, dir := range utils.RunKubeletPreCreateDirs {
		// 跳过网络配置目录，这个目录在ubuntu-server上面是不存在的
		if dir == "/etc/sysconfig/network-scripts" {
			continue
		}
		// 现在先不要删，放到额外清理里面
		if dir == "/etc/kubernetes" {
			continue
		}
		// don't remove cni dir for containerd aready installed
		if dir == "/var/lib/cni" || dir == "/etc/cni" || dir == "/opt/cni" {
			if allInOneFlag {
				continue
			}
			if containerRuntime == runtime.ContainerRuntimeContainerd {
				continue
			}
			continue
		}
		extra.AddDirToClean(dir)
	}

	return extra.CleanAll()
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
		return errors.Errorf("stop containerd failed: %s, %v", out, err)
	}
	out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable containerd")
	if err != nil {
		return errors.Errorf("disable containerd failed: %s, %v", out, err)
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

	var (
		err error
		out string
	)
	// remove all containers in k8s.io namespace
	switch runtime.DetectRuntime() {
	case runtime.ContainerRuntimeDocker:
		return ContainerCleanDocker(extra)
	case runtime.ContainerRuntimeContainerd:
		out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlListContainers)
		if err != nil {
			log.Warnf("list containers failed: %s , err:%s", out, err)
			return nil
		}
		pods := strings.Fields(out)
		for _, pod := range pods {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlStopContainer+" "+pod)
			if err != nil {
				log.Warnf("stop container %s failed: %s , err:%s", pod, out, err)
			} else {
				out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlRemoveContainer+" "+pod)
				if err != nil {
					log.Warnf("remove container %s failed: %s , err:%s", pod, out, err)
				}
			}
			if err != nil {
				out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod)
				if err != nil {
					log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
				}
			}
		}
	case "":
		log.Error("detect container runtime failed, skip remove containers")
		return nil
	default:
		log.Error("unsupported container runtime, skip remove containers")
		return nil
	}
	return extra.CleanAll()
}

func ContainerRuntimeClean(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error {
	// get current container runtime
	log.Infof("Clean container runtime")

	// skip clean container runtime if all in one mode
	if v, ok := cfg.CustomExtra["allInOne"]; ok && v == "true" {
		log.Infof("this node is all in one mode, skip clean container runtime")
		return nil
	}

	switch runtime.DetectRuntime() {
	case runtime.ContainerRuntimeDocker:
		dataRoot := bkeinit.DefaultCRIDockerDataRootDir
		if cfg.Cluster.ContainerRuntime.Param != nil {
			if v, ok := cfg.Cluster.ContainerRuntime.Param["data-root"]; ok {
				dataRoot = v
			}
		}

		// docker clean all containers
		out, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerListAllContainers)
		if err != nil {
			return errors.Errorf("list all containers failed: %s, %v", out, err)
		}
		pods := strings.Fields(out)
		for _, pod := range pods {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", dockerForceRemovePod+" "+pod)
			if err != nil {
				log.Warnf("force remove container failed: %s , err:%s", out, err)
			}
		}
		// docker clean all data
		out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", dockerCleanAll)
		if err != nil {
			return errors.Errorf("clean docker failed: %s, %v", out, err)
		}
		output, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop docker")
		if err != nil {
			return errors.Errorf("stop docker failed: %s, %v", output, err)
		}
		output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable docker")
		if err != nil {
			return errors.Errorf("disable docker failed: %s, %v", output, err)
		}
		//  remove docker-ce docker-ce-cli containerd.io

		if err := httprepo.RepoRemove("docker*", "containerd.io"); err != nil {
			log.Errorf("remove docker failed: %v", err)
		}

		//output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "yum remove -y docker* containerd.io")
		//if err != nil {
		//	log.Errorf("remove docker failed: %s, %v", output, err)
		//	//return errors.Errorf("remove docker failed: %s, %v", output, err)
		//}
		// rm -rf /var/lib/docker
		extra.AddDirToClean(dataRoot)
		extra.AddDirToClean("/etc/docker")

		// cni dir
		extra.AddDirToClean("/var/lib/cni")
		extra.AddDirToClean("/etc/cni")
		extra.AddDirToClean("/opt/cni")

		//
		v, _ := semver.ParseTolerant(cfg.Cluster.KubernetesVersion)
		if v.GTE(semver.MustParse("1.24.0")) {
			output, err := extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop cri-dockerd && systemctl stop cri-dockerd.socket")
			if err != nil {
				return errors.Errorf("stop cri-dockerd failed: %s, %v", output, err)
			}
			output, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable cri-dockerd && systemctl disable cri-dockerd.socket")
			if err != nil {
				return errors.Errorf("disable cri-dockerd failed: %s, %v", output, err)
			}
			extra.AddFileToClean("/usr/bin/cri-dockerd")
			extra.AddFileToClean("/etc/systemd/system/cri-dockerd.service")
			extra.AddFileToClean("/etc/systemd/system/cri-dockerd.socket")
		}

	case runtime.ContainerRuntimeContainerd:
		dataRoot := bkeinit.DefaultCRIContainerdDataRootDir
		if cfg.Cluster.ContainerRuntime.Param != nil {
			if v, ok := cfg.Cluster.ContainerRuntime.Param["data-root"]; ok {
				dataRoot = v
			}
		}

		// todo 删除容器，并且使用nerdctl 清理 namespace k8s.io 和默认namespace
		// crictl移除所有容器
		out, err := extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlListContainers)
		if err != nil {
			log.Warnf("list containers failed: %s , err:%s", out, err)
			return nil
		}
		pods := strings.Fields(out)
		for _, pod := range pods {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", crictlForceRemovePod+" "+pod)
			if err != nil {
				log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
			}
		}
		// nerdctl移除所有容器
		out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", nerdctlListContainers)
		if err != nil {
			log.Warnf("list containers failed: %s , err:%s", out, err)
			return nil
		}
		pods = strings.Fields(out)
		for _, pod := range pods {
			out, err = extra.ExecuteCommandWithOutput("/bin/sh", "-c", nerdctlForceRemoveContainer+" "+pod)
			if err != nil {
				log.Warnf("force remove container %s failed: %s , err:%s", pod, out, err)
			}
		}

		// 清理containerd空间
		//out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", nerdctlCleanAll)
		//if err != nil {
		//	return errors.Errorf("clean containerd failed: %s, %v", out, err)
		//}

		out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl stop containerd")
		if err != nil {
			return errors.Errorf("stop containerd failed: %s, %v", out, err)
		}
		out, err = extra.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "systemctl disable containerd")
		if err != nil {
			return errors.Errorf("disable containerd failed: %s, %v", out, err)
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

		// remove containerd data
		extra.AddDirToClean(dataRoot)
		extra.AddDirToClean("/etc/systemd/system/containerd.service.d")
		// cni dir
		extra.AddDirToClean("/var/lib/cni")
		extra.AddDirToClean("/etc/cni")
		extra.AddDirToClean("/opt/cni")
		// nerdctl dir
		extra.AddDirToClean("/var/lib/nerdctl")

		// registry cert
		extra.AddDirToClean("/etc/docker/certs.d")
	case "":
		log.Info("no available container runtime found")
		extra.AddDirToClean("/etc/cni")
		extra.AddDirToClean("/opt/cni")
	}
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

	if cfg.Cluster.Kubelet.ManifestsDir != "" {
		extra.AddDirToClean(cfg.Cluster.Kubelet.ManifestsDir)
	}

	nodes := bkenode.Nodes(cfg.Nodes)
	cnode, err := nodes.CurrentNode()
	if err != nil {
		log.Warnf("get current node failed: %v", err)
	} else {
		if cnode.Kubelet != nil && cnode.Kubelet.ManifestsDir != "" {
			extra.AddDirToClean(cnode.Kubelet.ManifestsDir)
		}
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

	needRemoveInters := []string{"vxlan_sys_4789", "gre_sys", "genev_sys_6081", "erspan_sys", "vxlan.calico"}
	out, err := extra.Executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", `ip link | awk '/state/ {gsub(/:/, ""); print $2}'`)
	if err == nil {
		for _, inter := range strings.Split(out, "\n") {
			if utils.ContainsString(needRemoveInters, inter) {
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
	}

	extra.AddDirToClean("/etc/kubernetes")
	extra.AddDirToClean(cfg.Cluster.Etcd.DataDir)
	return extra.CleanAll()
}
