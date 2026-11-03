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

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/resetutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// CleanInterface is the interface for clean phase
type CleanInterface interface {
	Clean(*bkev1beta1.BKEConfig) error
	AddDirToClean(string)
	AddFileToClean(string)
	AddIPToClean(string)
}

// CleanPhase is the struct for clean phase
type CleanPhase struct {
	Name      string
	CleanFunc func(cfg *bkev1beta1.BKEConfig, extra ExtraClean) error
	extra     ExtraClean
}

// Clean is the function to clean phase
func (c *CleanPhase) Clean(cfg *bkev1beta1.BKEConfig) error {
	c.extra.Executor = &exec.CommandExecutor{}
	return c.CleanFunc(cfg, c.extra)
}

// AddDirToClean is the function to add dir to clean list
func (c *CleanPhase) AddDirToClean(dir string) {
	c.extra.AddDirToClean(dir)
}

// AddFileToClean is the function to add file to clean list
func (c *CleanPhase) AddFileToClean(file string) {
	c.extra.AddFileToClean(file)
}

// AddIPToClean is the function to add ip to clean list
func (c *CleanPhase) AddIPToClean(ip string) {
	c.extra.AddIPToClean(ip)
}

// ExtraInterface is the interface for extra clean
type ExtraInterface interface {
	CleanAll() error
	CleanFile() error
	CleanDir() error
	CleanIP() error
	AddDirToClean(string)
	AddFileToClean(string)
	AddIPToClean(string)
}

// ExtraClean is the struct for extra clean
type ExtraClean struct {
	File []string
	Dir  []string
	Ips  []string
	exec.Executor
}

// CleanAll is the function to clean all extra files and dirs
func (e *ExtraClean) CleanAll() error {
	if err := e.CleanFile(); err != nil {
		return err
	}

	if err := e.CleanDir(); err != nil {
		return err
	}

	if err := e.CleanIP(); err != nil {
		return err
	}
	return nil
}

// CleanFile is the function to clean extra files
func (e *ExtraClean) CleanFile() error {
	for _, file := range e.File {
		if err := resetutil.CleanFile(file); err != nil {
			log.Warnf("clean file %s failed: %v", file, err)
		}
		log.Infof("clean file %s success", file)
	}
	return nil
}

// CleanDir is the function to clean extra dirs
func (e *ExtraClean) CleanDir() error {
	for _, dir := range e.Dir {
		if err := resetutil.CleanDir(dir); err != nil {
			log.Warnf("clean dir %s failed: %v", dir, err)
		}
		log.Infof("clean dir %s success", dir)
	}
	return nil
}

// CleanIP is the function to clean extra ips
func (e *ExtraClean) CleanIP() error {
	if len(e.Ips) == 0 {
		return nil
	}
	for _, ip := range e.Ips {
		intfName, err := bkenet.GetInterfaceFromIp(ip)
		if err != nil {
			log.Warnf("get interface from ip %s failed: %v", ip, err)
			continue
		}
		if addr, err := net.InterfaceIpExit(intfName, ip); err != nil {
			log.Warnf("check ip %s exist failed: %v", ip, err)
			return nil
		} else if addr != "" {
			deleteCommand := fmt.Sprintf("ip addr del %s dev %s", addr, intfName)
			output, err := e.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", deleteCommand)
			if err != nil {
				log.Warnf("clean ip addr %q from interface %q failed %s, err:%v", ip, intfName, output, err)
				continue
			}
			log.Infof("clean ip addr %q from interface %q success", ip, intfName)
		} else {
			log.Infof("ip addr %q not exit in interface %q, skip clean", ip, intfName)
		}
	}
	return nil
}

// AddDirToClean is the function to add dir to clean list
func (e *ExtraClean) AddDirToClean(dir string) {
	// unique e.Dir
	for _, d := range e.Dir {
		if d == dir {
			return
		}
	}
	e.Dir = append(e.Dir, dir)
}

// AddFileToClean is the function to add file to clean list
func (e *ExtraClean) AddFileToClean(file string) {
	// unique e.File
	for _, f := range e.File {
		if f == file {
			return
		}
	}
	e.File = append(e.File, file)
}

// AddIPToClean is the function to add ip to clean list
func (e *ExtraClean) AddIPToClean(ip string) {
	// unique e.Ips
	for _, i := range e.Ips {
		if i == ip {
			return
		}
	}
	e.Ips = append(e.Ips, ip)
}

// CleanPhases is the function to clean extra files and dirs
type CleanPhases []CleanPhase

// DefaultCleanPhases is the default clean phases
func DefaultCleanPhases() CleanPhases {
	return CleanPhases{
		CleanKubeletPhase(),
		CleanContainerdCfgPhase(),
		CleanContainerPhase(),
		CleanContainerRuntimePhase(),
		CleanCertPhase(),
		CleanManifestsPhase(),
		CleanSourcePhase(),
		CleanExtraPhase(),
		CleanGlobalCertPhase(),
		CleanShutdownPhase(),
	}
}

// CleanCertPhase is the function to clean all phases
func CleanCertPhase() CleanPhase {
	return CleanPhase{
		Name:      "cert",
		CleanFunc: CertClean,
		extra: ExtraClean{
			Dir: []string{
				pkiutil.GetDefaultPkiPath(),
			},
		},
	}
}

// CleanManifestsPhase is the function to clean manifests
func CleanManifestsPhase() CleanPhase {
	return CleanPhase{
		Name:      "manifests",
		CleanFunc: ManifestsClean,
		extra: ExtraClean{
			Dir: []string{
				mfutil.GetDefaultManifestsPath(),
			},
			File: []string{
				mfutil.GetAuditPolicyFilePath(),
			},
		},
	}
}

// CleanContainerdCfgPhase is the function to clean old containerd cfg
func CleanContainerdCfgPhase() CleanPhase {
	return CleanPhase{
		Name:      "containerd-cfg",
		CleanFunc: ContainerdCfgClean,
	}
}

// CleanContainerPhase is the function to clean container
func CleanContainerPhase() CleanPhase {
	return CleanPhase{
		Name:      "container",
		CleanFunc: ContainerClean,
	}
}

// CleanKubeletPhase is the function to clean kubelet
func CleanKubeletPhase() CleanPhase {
	return CleanPhase{
		Name:      "kubelet",
		CleanFunc: KubeletCleanBin,
		extra: ExtraClean{
			Dir: []string{utils.KubeletConfigPath},
		},
	}
}

// CleanContainerRuntimePhase is the function to clean container runtime
func CleanContainerRuntimePhase() CleanPhase {
	return CleanPhase{
		Name:      "containerRuntime",
		CleanFunc: ContainerRuntimeClean,
	}
}

// CleanSourcePhase is the function to reset repo source
func CleanSourcePhase() CleanPhase {
	return CleanPhase{
		Name:      "source",
		CleanFunc: SourceClean,
	}
}

// CleanExtraPhase is the function to clean extra files and dirs
func CleanExtraPhase() CleanPhase {
	return CleanPhase{
		Name:      "extra",
		CleanFunc: ExtraToClean,
	}
}

// CleanGlobalCertPhase is the function to clean global cert directory
func CleanGlobalCertPhase() CleanPhase {
	return CleanPhase{
		Name:      "global-cert",
		CleanFunc: GlobalCertClean,
	}
}

// CleanShutdownPhase schedules delayed bkeagent uninstall. Last so kubelet/
// runtime are gone first; stop uses delayed nohup (see agent_cleanup.go).
func CleanShutdownPhase() CleanPhase {
	return CleanPhase{
		Name:      ScopeShutdown,
		CleanFunc: BKEAgentShutdownClean,
	}
}
