package collect

import (
	"fmt"
	"strings"

	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	"sigs.k8s.io/controller-runtime/pkg/client"

	econd "gopkg.openfuyao.cn/bkeagent/pkg/executor/containerd"
	edocker "gopkg.openfuyao.cn/bkeagent/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
	"gopkg.openfuyao.cn/bkeagent/utils/runtime"
)

const (
	Name = "Collect"
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
func (c *CollectPlugin) Execute(commands []string) ([]string, error) {
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

func (c *CollectPlugin) collectMachineInfo() []string {

	lowLevelRuntime := bkeinit.DefaultRuntime
	cgroupDriver := bkeinit.DefaultCgroupDriver
	dataRoot := bkeinit.DefaultCRIDockerDataRootDir

	result := []string{lowLevelRuntime, cgroupDriver, dataRoot}

	// get docker or containerd config info
	switch c.controllerRuntime {
	case runtime.ContainerRuntimeDocker:
		cfg, err := edocker.GetDockerDaemonConfig(edocker.DockerDaemonConfigFilePath)
		if err != nil {
			log.Errorf("get docker daemon config failed: %v", err)
			return result
		}

		if cfg == nil {
			log.Warnf("get docker daemon config failed, config is nil")
			return result
		}

		if cfg.DefaultRuntime != "" {
			lowLevelRuntime = cfg.DefaultRuntime
		}
		if cfg.ExecOptions != nil || len(cfg.ExecOptions) != 0 {
			for _, opt := range cfg.ExecOptions {
				if strings.Contains(opt, "native.cgroupdriver") {
					op := strings.Split(opt, "=")
					cgroupDriver = op[1]
				}
			}
		}
		if cfg.Root != "" {
			dataRoot = cfg.Root
		}

	case runtime.ContainerRuntimeContainerd:
		dataRoot = bkeinit.DefaultCRIContainerdDataRootDir
		cfg, err := econd.GetContainerdConfig(econd.ContainerdConfigFilePath)
		if err != nil {
			log.Errorf("get containerd config failed: %v", err)
			return result
		}
		if cfg == nil {
			log.Warnf("get containerd config failed")
			return result
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
				lowLevelRuntime = r.(string)
			}
			// get cgroup driver
			systemd := v.Get("containerd.runtimes." + lowLevelRuntime + ".options.SystemdCgroup")
			if systemd != nil && !systemd.(bool) {
				cgroupDriver = "cgroupfs"
			}
		}
	default:
		log.Warnf("unknown container runtime: %s", c.controllerRuntime)
	}

	return []string{lowLevelRuntime, cgroupDriver, dataRoot}
}

func (c *CollectPlugin) collectKubeletDataRootDir() string {
	switch c.controllerRuntime {
	case runtime.ContainerRuntimeDocker:
		cmd := fmt.Sprintf("docker inspect kubelet --format '{{.Args}}'")
		output, err := c.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
		if err != nil {
			log.Errorf("docker inspect kubelet failed: %v", err)
			return bkeinit.DefaultKubeletRootDir
		}
		if output == "" {
			log.Warnf("docker inspect kubelet with empty output")
			return bkeinit.DefaultKubeletRootDir
		}
		// 去掉前后的[]
		output = output[1 : len(output)-1]
		args := strings.Split(output, " ")
		for _, arg := range args {
			if strings.Contains(arg, "--root-dir") {
				tmp := strings.Split(arg, "=")
				if len(tmp) != 2 {
					log.Warnf("get kubelet root-dir arg in  container run args failed, target arg: %s", arg)
					return bkeinit.DefaultKubeletRootDir
				}
				return strings.TrimSuffix(tmp[1], "/")
			}
		}
	case runtime.ContainerRuntimeContainerd:
		cmd := fmt.Sprintf("nerdctl -n k8s.io inspect kubelet --format '{{.Args}}'")
		output, err := c.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", cmd)
		if err != nil {
			log.Errorf("nerdctl inspect kubelet failed: %v", err)
			return bkeinit.DefaultKubeletRootDir
		}
		if output == "" {
			log.Warnf("nerdctl inspect kubelet with empty output")
			return bkeinit.DefaultKubeletRootDir
		}
		// 去掉前后的[]
		output = output[1 : len(output)-1]
		args := strings.Split(output, " ")
		for _, arg := range args {
			if strings.Contains(arg, "--root-dir") {
				tmp := strings.Split(arg, "=")
				if len(tmp) != 2 {
					log.Warnf("get kubelet root-dir arg in container run args failed, target arg: %s", arg)
					return bkeinit.DefaultKubeletRootDir
				}
				return strings.TrimSuffix(tmp[1], "/")
			}
		}
	default:
		log.Warnf("unknown container runtime: %s", c.controllerRuntime)
	}
	return bkeinit.DefaultKubeletRootDir
}
