package docker

import (
	_ "embed"
	"fmt"
	"strings"

	"github.com/pkg/errors"

	edocker "gopkg.openfuyao.cn/bkeagent/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils/httprepo"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const Name = "InstallDocker"

var (
	defaultCgroupDriver = "systemd"
	defaultRuntime      = "runc"
	defaultDataRoot     = "/var/lib/docker"
)

type DockerPlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) plugin.Plugin {
	return &DockerPlugin{
		exec: exec,
	}
}

func (dp DockerPlugin) Name() string {
	return Name
}

func (dp DockerPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"insecureRegistries": {
			Key:         "insecureRegistries",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "insecure insecureRegistries, eg:a.com,b.com split by ',' ",
		},
		"registryMirror": {
			Key:         "registryMirror",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "registry mirror, eg:a.com,b.com split by ',' ",
		},
		"cgroupDriver": {
			Key:         "cgroupDriver",
			Value:       "",
			Required:    false,
			Default:     defaultCgroupDriver,
			Description: "cgroup driver, eg:systemd,cgroupfs",
		},
		"runtime": {
			Key:         "runtime",
			Value:       "",
			Required:    false,
			Default:     defaultRuntime,
			Description: "runtime, eg:runc,richrunc,kata",
		},
		"dataRoot": {
			Key:         "dataRoot",
			Value:       "",
			Required:    false,
			Default:     defaultDataRoot,
			Description: "Specify the data directory",
		},
		"enableDockerTls": {
			Key:         "enableDockerTls",
			Value:       "",
			Required:    false,
			Default:     "false",
			Description: "Enable docker tls, eg:true,false",
		},
		"tlsHost": {
			Key:         "tlsHost",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "tls host ip",
		},
		"runtimeUrl": {
			Key:         "runtimeUrl",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "runtime url for download",
		},
	}
}

func (dp DockerPlugin) Execute(commands []string) ([]string, error) {
	// Parse command
	runtimeParam, err := plugin.ParseCommands(dp, commands)
	if err != nil {
		return nil, err
	}
	// stop docker before install
	_ = dp.exec.ExecuteCommand("sh", "-c", "systemctl stop docker")
	_ = dp.exec.ExecuteCommand("sh", "-c", "systemctl stop docker.socket")

	if runtimeParam["runtime"] == "richrunc" {
		url := fmt.Sprintf("url=%s", runtimeParam["runtimeUrl"])
		cs := []string{
			downloader.Name,
			url,
			"chmod=755",
			"rename=runc",
			"saveto=/usr/local/beyondvm",
		}

		downloaderPlugin := downloader.New()
		if _, err = downloaderPlugin.Execute(cs); err != nil {
			return nil, errors.Errorf("download richrunc %s failed, err: %v", url, err)
		}
	}

	docker := "docker-ce"

	// 对麒麟兼容
	//h, _, _, err := host.PlatformInformation()
	//if err != nil {
	//	log.Errorf("get host platform information failed, err: %v", err)
	//	return nil, errors.Errorf("get host platform information failed, err: %v", err)
	//}
	//
	//if h == "kylin" {
	//	if err := httprepo.RepoSearch("docker-ce"); err != nil {
	//		log.Warnf("[kylin] search docker-ce from repo failed, use docker-engine instead, err: %v", err)
	//		docker = "docker"
	//		// kylin v10 docker-engine强制使用cgroupfs
	//		runtimeParam["cgroupDriver"] = "cgroupfs"
	//	}
	//}
	//优先使用docker-ce
	if err := httprepo.RepoSearch("docker-ce"); err != nil {
		log.Warnf("search docker-ce from repo failed, use docker-engine instead, err: %v", err)
		docker = "docker"
		// docker-engine强制使用cgroupfs
		runtimeParam["cgroupDriver"] = "cgroupfs"
	}

	// install docker
	if err := httprepo.RepoInstall(docker); err != nil {
		log.Errorf("install docker failed, err: %v", err)
		return nil, err
	}

	enableTls := runtimeParam["enableDockerTls"] == "true"

	registries := strings.Split(runtimeParam["insecureRegistries"], ",")
	err = edocker.ConfigDockerDaemon(runtimeParam["cgroupDriver"], runtimeParam["runtime"], runtimeParam["dataRoot"], enableTls, runtimeParam["tlsHost"], registries...)
	if err != nil {
		return nil, err
	}

	out, err := dp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl enable docker")
	if err != nil {
		log.Warnf("enable docker failed, err: %v, out: %s", err, out)
	}

	// start docker
	out, err = dp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart docker")
	if err != nil {
		errorMsg := fmt.Sprintf("start docker failed, err: %v, out: %s", err, out)
		log.Errorf(errorMsg)
		return []string{errorMsg}, errors.New(errorMsg)
	}

	if err = edocker.WaitDockerReady(); err != nil {
		return nil, err
	}
	return nil, nil
}
