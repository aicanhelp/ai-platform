/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hub

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// HarborCfg is harbor config
type HarborCfg struct {
	Nodes        fuyaov1beta1.Nodes
	CertPath     string
	OriginDomain string
	RegistryHost string

	routineMaximum *utils.GoRoutineLimit
}

// NewHarborCfg is new HarborCfg
func NewHarborCfg(nodes fuyaov1beta1.Nodes, certPath string, originDomain string, host string) *HarborCfg {
	return &HarborCfg{
		Nodes:          nodes,
		routineMaximum: utils.NewGoRoutineLimits(),
		CertPath:       certPath,
		OriginDomain:   originDomain,
		RegistryHost:   host,
	}
}

func (c *HarborCfg) copyCert(node *fuyaov1beta1.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer c.routineMaximum.Done()

	if node.IP == c.RegistryHost {
		return
	}

	if _, err := utils.ExecLocalSudoCmd(fmt.Sprintf("chmod %o %s", constants.RwxRxRx, c.CertPath)); err != nil {
		zlog.Fatalf("failed to chmod local hub cert before copy: %v", err)
	}

	client := node.Client()
	err := client.CopyFileToRemote(c.CertPath, constants.HubCaPath)
	if err != nil {
		zlog.Fatalf("failed to copy hub cert: %v", err)
	}
	if _, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.Rw, constants.HubCaPath)); err != nil {
		zlog.Fatalf("failed to chmod remote hub cert after copy: %v", err)
	}
	if _, err := utils.ExecLocalSudoCmd(fmt.Sprintf("chmod %o %s", constants.Rw, constants.HubCaBkPath)); err != nil {
		zlog.Fatalf("failed to chmod local harbor cert after copy: %v", err)
	}
}

func (c *HarborCfg) setProxy(node *fuyaov1beta1.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer c.routineMaximum.Done()

	client := node.Client()
	cmd := fmt.Sprintf(`echo "%s %s" | sudo tee -a /etc/hosts`, c.RegistryHost, c.OriginDomain)
	zlog.Info(fmt.Sprintf("Exec cmd : %s", cmd))
	_, err := client.ExecCmd(cmd)
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("Failed to set hub proxy: %v", err))
	}
}

// Executor 拷贝harbor证书
func (c *HarborCfg) Executor(function func(*fuyaov1beta1.Node, *sync.WaitGroup)) {
	var wg sync.WaitGroup

	for _, worker := range c.Nodes.Workers {
		wg.Add(1)
		c.routineMaximum.Add()
		go function(worker, &wg)
	}

	for _, master := range c.Nodes.Masters {
		wg.Add(1)
		c.routineMaximum.Add()
		go function(master, &wg)
	}

	for _, registry := range c.Nodes.Registry {
		wg.Add(1)
		c.routineMaximum.Add()
		go function(registry, &wg)
	}

	wg.Wait()
}

// InstallHarborCfg 拷贝证书与配置host
func InstallHarborCfg(runtime *common.KubeRuntime) error {
	// 非离线安装不处理
	if !runtime.Arg.EnableOffline {
		return nil
	}
	fmt.Println("[prepare] Processing hub config")

	// 获取镜像仓配置
	registryCfg, err := common.GetRegistryConfigYaml()
	if err != nil {
		return err
	} else if err == nil && registryCfg == nil {
		// 离线使用用户自己镜像仓场景
		return nil
	}

	harborCfg := NewHarborCfg(runtime.Cluster.Spec.Nodes, registryCfg.Spec.CaPath, registryCfg.Spec.OriginDomain,
		registryCfg.Spec.IP)
	harborCfg.Executor(harborCfg.copyCert)
	if registryCfg.Spec.OriginDomain != registryCfg.Spec.IP {
		harborCfg.Executor(harborCfg.setProxy)
	}
	return nil
}

// RecordHarborConfig 记录harbor的配置
func RecordHarborConfig(domain, ip string, node *fuyaov1beta1.Node) error {
	client := node.Client()
	if _, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.RwxRxRx, constants.HubCaPath)); err != nil {
		return fmt.Errorf("failed to chmod remote harbor cert before backup: %v", err)
	}
	if err := client.DownloadFile(constants.HubCaBkPath, constants.HubCaPath); err != nil {
		zlog.Error("failed to download harbor cert")
		return fmt.Errorf("failed to download harbor cert: %v", err)
	}
	if _, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.Rw, constants.HubCaPath)); err != nil {
		return fmt.Errorf("failed to chmod remote harbor cert adter backup: %v", err)
	}
	if _, err := utils.ExecLocalSudoCmd(fmt.Sprintf("chmod %o %s", constants.Rw, constants.HubCaBkPath)); err != nil {
		return fmt.Errorf("failed to chmod harbor cert backup: %v", err)
	}

	cfgYaml := metadata.RegistryConfig{
		APIVersion: constants.ApiVersion,
		Kind:       constants.HubCfg,
		Metadata:   base.Metadata{Name: "hub-config"},
		Spec: metadata.Spec{
			IP:           ip,
			Domain:       domain + constants.Slash + constants.DefaultProject,
			CaPath:       constants.HubCaBkPath,
			OriginDomain: domain,
		},
	}
	assetPath, err := common.GetAssetAbsPathReal()
	if err != nil {
		return err
	}

	return common.MarshalYaml[metadata.RegistryConfig](filepath.Join(assetPath, constants.RegistryConfig), cfgYaml)
}

// UploadImagePrepare 准备上传镜像
func UploadImagePrepare(passwd, domain string, nodes fuyaov1beta1.Nodes) error {
	registry := nodes.Registry[0]
	ip, err := utils.GetLocalIP()
	if err != nil {
		return err
	}
	// 下载证书
	if ip != registry.IP {
		client := registry.Client()
		if err := client.DownloadFile(constants.HubCaPath, constants.HubCaPath); err != nil {
			zlog.Error("failed to download harbor cert")
			return err
		}
	}

	// 配置hosts
	if domain != registry.IP && !strings.Contains(domain, registry.IP) {
		cmd := fmt.Sprintf(`echo "%s %s" | sudo tee -a /etc/hosts`, registry.IP, domain)
		if _, err := utils.ExecLocalCmd(cmd); err != nil {
			zlog.Error("failed to set harbor proxy")
			return fmt.Errorf(fmt.Sprintf("Failed to exec cmd %s : %v", cmd, err))
		}
	}

	// 登录harbor
	cmd := fmt.Sprintf("nerdctl login %s -u %s -p %s", domain, constants.DefaultHarborUsername, passwd)
	if _, err := utils.ExecLocalCmdWithoutLog(cmd); err != nil {
		zlog.Error("failed to login harbor")
		return err
	}

	return nil
}

// GetHubInfo is get harbor domain passwd and ip
func GetHubInfo(components []fuyaov1beta1.Component) (string, string, string) {
	domain := ""
	passwd := ""
	ip := ""
	for _, component := range components {
		if component.Name != constants.Harbor {
			continue
		}

		for _, param := range component.Param {
			switch param.Name {
			case common.RegistryDomain:
				domain = param.Value
			case common.RegistryPassword:
				passwd = param.Value
			case common.IP:
				ip = param.Value
			default:
				continue
			}
		}
		break
	}

	domain = domain + ":" + constants.DefaultRegistryHttpsPort
	return domain, passwd, ip
}
