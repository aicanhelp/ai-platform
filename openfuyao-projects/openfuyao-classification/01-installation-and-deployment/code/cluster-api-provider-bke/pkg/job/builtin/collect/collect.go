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

package collect

import (
	"context"
	"fmt"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/client"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	econd "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	edocker "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/runtime"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "Collect"
	two  = 2
)

type CollectPlugin struct {
	k8sClient client.Client
	exec      exec.Executor

	clusterName string
	clusterType string
	nameSpace   string

	controllerRuntime string
}

func New(k8sClient client.Client, exec exec.Executor) *CollectPlugin {
	return &CollectPlugin{
		k8sClient: k8sClient,
		exec:      exec,
	}
}

func (c *CollectPlugin) Name() string {
	return Name
}

func (c *CollectPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"clusterName": {
			Key:         "clusterName",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "cluster name, always is k8s namespace in management cluster",
		},
		"namespace": {
			Key:         "namespace",
			Value:       "",
			Required:    false,
			Default:     "default",
			Description: "namespace to upload certs",
		},
		"clusterType": {
			Key:         "clusterType",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "cluster type",
		},
		"k8sCertDir": {
			Key:         "k8sCertDir",
			Value:       "",
			Required:    false,
			Default:     pkiutil.GetDefaultPkiPath(),
			Description: "k8s cert dir",
		},
		"etcdCertDir": {
			Key:         "etcdCertDir",
			Value:       "",
			Required:    false,
			Default:     pkiutil.GetDefaultEtcdPkiPath(),
			Description: "etcd cert dir",
		},
	}
}

// Execute is the entry point of the plugin
func (c *CollectPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	cc := *c
	cc.exec = exec.Bound(c.exec, ctx)
	c = &cc

	commandMap, err := plugin.ParseCommands(c, commands)
	if err != nil {
		return nil, err
	}
	c.clusterName = commandMap["clusterName"]
	c.nameSpace = commandMap["namespace"]
	c.clusterType = commandMap["clusterType"]

	// 去除最后的 /
	k8sCertDir := strings.TrimSuffix(commandMap["k8sCertDir"], "/")
	etcdCertDir := strings.TrimSuffix(commandMap["etcdCertDir"], "/")

	c.controllerRuntime = runtime.DetectRuntime()

	if err = c.collectCerts(k8sCertDir, etcdCertDir); err != nil {
		return nil, err
	}
	// result 内容 lowLevelRuntime, cgroupDriver, dataRoot, 集群推断类型（bke or bocloud）
	result := c.collectMachineInfo()
	if etcdCertDir == pkiutil.GetDefaultPkiPath() {
		result = append(result, "bke")
	} else {
		result = append(result, "bocloud")
	}

	kubeletRootDir := c.collectKubeletDataRootDir()
	result = append(result, kubeletRootDir)

	return result, nil
}

func (c *CollectPlugin) collectCerts(k8sCertDir, etcdCertDir string) error {
	if etcdCertDir != pkiutil.GetDefaultPkiPath() {
		k8sCerts := pkiutil.GetBocloudCertListWithoutEtcd()
		k8sCerts.SetPkiPath(k8sCertDir)
		etcdCerts := pkiutil.GetBocloudCertListForEtcd()
		etcdCerts.SetPkiPath(etcdCertDir)
		allCerts := append(k8sCerts, etcdCerts...)
		for _, cert := range allCerts {
			err := pkiutil.UploadBocloudCertToClusterAPI(c.k8sClient, cert, c.nameSpace, c.clusterName)
			if err != nil {
				log.Warnf("(ignore)upload cert %s to cluster api failed: %v", cert.Name, err)
			}
		}
	} else {
		k8sCerts := pkiutil.GetCertsWithoutEtcd()
		k8sCerts.SetPkiPath(k8sCertDir)
		etcdCerts := pkiutil.GetEtcdCerts()
		etcdCerts.SetPkiPath(etcdCertDir)
		allCerts := append(k8sCerts, etcdCerts...)
		for _, cert := range allCerts {
			err := pkiutil.UploadBKECertToClusterAPI(c.k8sClient, cert, c.nameSpace, c.clusterName)
			if err != nil {
				log.Warnf("(ignore)upload cert %s to cluster api failed: %v", cert.Name, err)
			}
		}
	}
	return nil
}

// collectDockerMachineInfo 从 Docker 配置中收集机器信息
func (c *CollectPlugin) collectDockerMachineInfo() (string, string, string) {
	lowLevelRuntime := bkeinit.DefaultRuntime
	cgroupDriver := bkeinit.DefaultCgroupDriver
	dataRoot := bkeinit.DefaultCRIDockerDataRootDir

	cfg, err := edocker.GetDockerDaemonConfig(edocker.DockerDaemonConfigFilePath)
	if err != nil {
		log.Warnf("get docker daemon config failed: %v", err)
		return lowLevelRuntime, cgroupDriver, dataRoot
	}

	if cfg == nil {
		log.Warnf("get docker daemon config failed, config is nil")
		return lowLevelRuntime, cgroupDriver, dataRoot
	}

	if cfg.DefaultRuntime != "" {
		lowLevelRuntime = cfg.DefaultRuntime
	}
	if cfg.ExecOptions != nil || len(cfg.ExecOptions) != 0 {
		for _, opt := range cfg.ExecOptions {
			if !strings.Contains(opt, "native.cgroupdriver") {
				continue
			}
			op := strings.Split(opt, "=")
			if len(op) > 1 {
				cgroupDriver = op[1]
			}
		}
	}
	if cfg.Root != "" {
		dataRoot = cfg.Root
	}

	return lowLevelRuntime, cgroupDriver, dataRoot
}

// collectContainerdMachineInfo 从 Containerd 配置中收集机器信息
func (c *CollectPlugin) collectContainerdMachineInfo() (string, string, string) {
	lowLevelRuntime := bkeinit.DefaultRuntime
	cgroupDriver := bkeinit.DefaultCgroupDriver
	dataRoot := bkeinit.DefaultCRIContainerdDataRootDir

	cfg, err := econd.GetContainerdConfig(econd.ContainerdConfigFilePath)
	if err != nil {
		log.Warnf("get containerd config failed: %v", err)
		return lowLevelRuntime, cgroupDriver, dataRoot
	}
	if cfg == nil {
		log.Warnf("get containerd config failed")
		return lowLevelRuntime, cgroupDriver, dataRoot
	}

	// get data root
	if cfg.Root != "" {
		dataRoot = cfg.Root
	}

	plugins := cfg.Plugins
	if v, ok := plugins["io.containerd.grpc.v1.cri"]; ok {
		// get low level runtime
		r := v.Get("containerd.default_runtime_name")
		if r != nil {
			if runtimeName, ok := r.(string); ok {
				lowLevelRuntime = runtimeName
			} else {
				log.Warnf("containerd.default_runtime_name is not a string, got type: %T", r)
			}
		}
		// get cgroup driver
		systemd := v.Get("containerd.runtimes." + lowLevelRuntime + ".options.SystemdCgroup")
		if systemd != nil {
			systemdCgroup, ok := systemd.(bool)
			if !ok {
				log.Warnf("SystemdCgroup is not a bool, got type: %T", systemd)
			}
			if ok && !systemdCgroup {
				cgroupDriver = "cgroupfs"
			}
		}
	}

	return lowLevelRuntime, cgroupDriver, dataRoot
}

func (c *CollectPlugin) collectMachineInfo() []string {
	lowLevelRuntime := bkeinit.DefaultRuntime
	cgroupDriver := bkeinit.DefaultCgroupDriver
	dataRoot := bkeinit.DefaultCRIDockerDataRootDir

	// get docker or containerd config info
	switch c.controllerRuntime {
	case runtime.ContainerRuntimeDocker:
		lowLevelRuntime, cgroupDriver, dataRoot = c.collectDockerMachineInfo()
	case runtime.ContainerRuntimeContainerd:
		lowLevelRuntime, cgroupDriver, dataRoot = c.collectContainerdMachineInfo()
	default:
		log.Warnf("unknown container runtime: %s", c.controllerRuntime)
	}

	return []string{lowLevelRuntime, cgroupDriver, dataRoot}
}

// extractRootDirFromArgs 从容器 inspect 输出中提取 --root-dir 参数值
func extractRootDirFromArgs(output string) string {
	if output == "" {
		log.Warnf("docker inspect kubelet with empty output")
		return bkeinit.DefaultKubeletRootDir
	}
	// 去掉前后的[]
	if len(output) < two {
		return bkeinit.DefaultKubeletRootDir
	}
	output = output[1 : len(output)-1]
	args := strings.Split(output, " ")
	for _, arg := range args {
		if strings.Contains(arg, "--root-dir") {
			tmp := strings.Split(arg, "=")
			if len(tmp) != two {
				log.Warnf("get kubelet root-dir arg in container run args failed, target arg: %s", arg)
				return bkeinit.DefaultKubeletRootDir
			}
			return strings.TrimSuffix(tmp[1], "/")
		}
	}
	return bkeinit.DefaultKubeletRootDir
}

func (c *CollectPlugin) collectKubeletDataRootDir() string {
	if c.exec == nil {
		log.Warnf("executor is nil, use default kubelet root dir")
		return bkeinit.DefaultKubeletRootDir
	}
	switch c.controllerRuntime {
	case runtime.ContainerRuntimeDocker:
		cmd := fmt.Sprintf("docker inspect kubelet --format '{{.Args}}'")
		output, err := c.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
		if err != nil {
			log.Warnf("docker inspect kubelet failed: %v", err)
			return bkeinit.DefaultKubeletRootDir
		}
		return extractRootDirFromArgs(output)
	case runtime.ContainerRuntimeContainerd:
		cmd := fmt.Sprintf("nerdctl -n k8s.io inspect kubelet --format '{{.Args}}'")
		output, err := c.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
		if err != nil {
			log.Warnf("nerdctl inspect kubelet failed: %v", err)
			return bkeinit.DefaultKubeletRootDir
		}
		return extractRootDirFromArgs(output)
	default:
		log.Warnf("unknown container runtime: %s", c.controllerRuntime)
	}
	return bkeinit.DefaultKubeletRootDir
}
