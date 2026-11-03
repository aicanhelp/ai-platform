package cri_docker

import (
	_ "embed"
	"fmt"
	"os"
	"text/template"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/httprepo"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const Name = "InstallCRIDocker"

var (
	//go:embed cri-dockerd.service
	criDockerdService string
	//go:embed cri-dockerd.socket
	criDockerdSocket string

	defaultRepo    = fmt.Sprintf("%s:%s", utils.DefaultImageRepo, utils.DefaultImageRepoPort)
	defaultSandbox = fmt.Sprintf("%s/kubernetes/%s", defaultRepo, "pause:3.8")
)

type CRIDockerPlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) plugin.Plugin {
	return &CRIDockerPlugin{
		exec: exec,
	}
}

func (cdp CRIDockerPlugin) Name() string {
	return Name
}

func (cdp CRIDockerPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"criDockerdUrl": {
			Key:         "criDockerdUrl",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "cri-dockerd url for download",
		},
		"sandbox": {
			Key:         "sandbox",
			Value:       "",
			Required:    true,
			Default:     defaultSandbox,
			Description: "Pod sandbox",
		},
	}
}

func (cdp CRIDockerPlugin) Execute(commands []string) ([]string, error) {
	runtimeParam, err := plugin.ParseCommands(cdp, commands)
	if err != nil {
		return nil, err
	}
	// 检查cri-dockerd是否存在
	if utils.Exists("/usr/bin/cri-dockerd") {
		if err = writeCriDockerdConfigToDisk(runtimeParam); err != nil {
			return nil, err
		}
		if err = cdp.startCriDockerd(); err != nil {
			return nil, err
		}
	}

	_ = cdp.exec.ExecuteCommand("sh", "-c", "systemctl stop cri-dockerd")
	_ = cdp.exec.ExecuteCommand("sh", "-c", "systemctl stop cri-dockerd.socket")

	url := fmt.Sprintf("url=%s", runtimeParam["criDockerdUrl"])
	cs := []string{
		downloader.Name,
		url,
		"chmod=755",
		"rename=cri-dockerd",
		"saveto=/usr/bin",
	}

	downloaderPlugin := downloader.New()
	if _, err = downloaderPlugin.Execute(cs); err != nil {
		return nil, errors.Errorf("download cri-dockerd %s failed, err: %v", url, err)
	}

	if err = writeCriDockerdConfigToDisk(runtimeParam); err != nil {
		return nil, err
	}

	if err = httprepo.RepoInstall("socat"); err != nil {
		return nil, err
	}

	if err = cdp.startCriDockerd(); err != nil {
		return nil, err
	}
	return nil, nil
}

func (cdp CRIDockerPlugin) startCriDockerd() error {
	out, err := cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl daemon-reload")
	if err != nil {
		log.Warnf("daemon-reload failed, err: %v, out: %s", err, out)
	}
	out, err = cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl enable cri-dockerd")
	if err != nil {
		log.Warnf("enable cri-dockerd failed, err: %v, out: %s", err, out)
	}
	// start cri-dockerd
	out, err = cdp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart cri-dockerd")
	if err != nil {
		return errors.Errorf("start cri-dockerd failed, err: %v, out: %s", err, out)
	}
	return nil
}

func writeCriDockerdConfigToDisk(runtimeParam map[string]string) error {
	// Render configuration file
	f, err := os.OpenFile("/etc/systemd/system/cri-dockerd.service", os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	tpl, err := template.New("cri-dockerd.service").Parse(criDockerdService)
	if err := tpl.Execute(f, runtimeParam); err != nil {
		return err
	}

	err = os.WriteFile("/etc/systemd/system/cri-dockerd.socket", []byte(criDockerdSocket), 0644)
	if err != nil {
		return err
	}
	return nil
}
