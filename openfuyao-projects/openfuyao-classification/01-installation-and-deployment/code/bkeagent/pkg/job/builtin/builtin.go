package builtin

import (
	"runtime/debug"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/backup"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/collect"
	bcond "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/containerruntime/containerd"
	cridocker "gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/containerruntime/cri_docker"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/containerruntime/docker"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/ha"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/certs"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/env"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/kubelet"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/kubeadm/manifests"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/ping"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/reset"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/selfupdate"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/shutdown"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/switchcluster"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

type BuiltIn interface {
	Execute(execCommands []string) ([]string, error)
}

type Task struct {
}

var pluginRegistry = map[string]plugin.Plugin{}

func New(exec exec.Executor, k8sClient client.Client) BuiltIn {
	t := Task{}
	c := bcond.New(exec)
	pluginRegistry[strings.ToLower(c.Name())] = c
	e := env.New(exec, nil)
	pluginRegistry[strings.ToLower(e.Name())] = e
	s := switchcluster.New(k8sClient)
	pluginRegistry[strings.ToLower(s.Name())] = s
	cert := certs.New(k8sClient, exec, nil)
	pluginRegistry[strings.ToLower(cert.Name())] = cert
	k := kubelet.New(nil, exec)
	pluginRegistry[strings.ToLower(k.Name())] = k
	ka := kubeadm.New(exec, k8sClient)
	pluginRegistry[strings.ToLower(ka.Name())] = ka
	h := ha.New(exec)
	pluginRegistry[strings.ToLower(h.Name())] = h
	d := downloader.New()
	pluginRegistry[strings.ToLower(d.Name())] = d
	r := reset.New()
	pluginRegistry[strings.ToLower(r.Name())] = r
	p := ping.New()
	pluginRegistry[strings.ToLower(p.Name())] = p
	b := backup.New(exec)
	pluginRegistry[strings.ToLower(b.Name())] = b
	dc := docker.New(exec)
	pluginRegistry[strings.ToLower(dc.Name())] = dc
	cc := collect.New(k8sClient, exec)
	pluginRegistry[strings.ToLower(cc.Name())] = cc
	mf := manifests.New(nil, exec)
	pluginRegistry[strings.ToLower(mf.Name())] = mf
	sd := shutdown.New()
	pluginRegistry[strings.ToLower(sd.Name())] = sd
	updatePlugin := selfupdate.New(exec)
	pluginRegistry[strings.ToLower(updatePlugin.Name())] = updatePlugin
	cdp := cridocker.New(exec)
	pluginRegistry[strings.ToLower(cdp.Name())] = cdp
	return &t
}

// Execute Execute built-in instructions
func (t *Task) Execute(execCommands []string) (res []string, err error) {
	defer func() {
		if e := recover(); e != nil {
			log.Error(string(debug.Stack()))
			if recoverErr, ok := e.(error); ok {
				log.Errorf("panic: %v", recoverErr)
				err = errors.Errorf("panic: %v", recoverErr)
			}
		}
	}()

	// Distribute execution instruction
	if len(execCommands) == 0 {
		return []string{}, errors.Errorf("Instructions cannot be null")
	}
	if v, ok := pluginRegistry[strings.ToLower(execCommands[0])]; ok {
		return v.Execute(execCommands)
	}
	return nil, errors.Errorf("Instruction not found")
}
