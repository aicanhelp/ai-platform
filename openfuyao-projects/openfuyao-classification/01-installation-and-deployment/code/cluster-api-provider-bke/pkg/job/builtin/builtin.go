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

package builtin

import (
	"context"
	"fmt"
	"runtime/debug"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/backup"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/collect"
	bcond "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/containerruntime/containerd"
	cridocker "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/containerruntime/cridocker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/containerruntime/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/downloader"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/ha"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/env"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/kubelet"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/kubeadm/manifests"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/ping"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/postprocess"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/preprocess"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/reset"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/selfupdate"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/switchcluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type BuiltIn interface {
	Execute(ctx context.Context, execCommands []string) ([]string, error)
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
	// Wire HA IPVS init without making ha import kubeadm/env (keeps ha.test small under coverpkg=./...).
	// Capture the Executor param: inside New, name "exec" shadows the exec package.
	runnerExec := exec
	ha.KernelEnvRunner = func(ctx context.Context, commands []string) error {
		_, err := env.New(runnerExec, nil).Execute(ctx, commands)
		if err != nil {
			return errors.WithMessage(err, "HA kernel env init")
		}
		return nil
	}
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
	updatePlugin := selfupdate.New(exec)
	pluginRegistry[strings.ToLower(updatePlugin.Name())] = updatePlugin
	cdp := cridocker.New(exec)
	pluginRegistry[strings.ToLower(cdp.Name())] = cdp
	prepro := preprocess.New(exec, k8sClient)
	pluginRegistry[strings.ToLower(prepro.Name())] = prepro // "preprocess"
	postpro := postprocess.New(exec, k8sClient)
	pluginRegistry[strings.ToLower(postpro.Name())] = postpro // "postprocess"
	return &t
}

// Execute Execute built-in instructions
func (t *Task) Execute(ctx context.Context, execCommands []string) ([]string, error) {
	var panicErr error
	defer func() {
		if e := recover(); e != nil {
			log.Error(string(debug.Stack()))
			if recoverErr, ok := e.(error); ok {
				log.Errorf("panic: %v", recoverErr)
				panicErr = fmt.Errorf("panic: %w", recoverErr)
			} else {
				panicErr = errors.Errorf("panic: %v", e)
			}
		}
	}()

	// Distribute execution instruction
	if len(execCommands) == 0 {
		return []string{}, errors.Errorf("Instructions cannot be null")
	}
	if v, ok := pluginRegistry[strings.ToLower(execCommands[0])]; ok {
		res, err := v.Execute(ctx, execCommands)
		if panicErr != nil {
			return nil, panicErr
		}
		return res, err
	}
	if panicErr != nil {
		return nil, panicErr
	}
	return nil, errors.Errorf("Instruction not found")
}
