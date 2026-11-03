/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pipelines

import (
	"fmt"
	"strings"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/prepare/hub"
)

// NewCreateRegistryPipeline 执行创建镜像仓并上传制品
func NewCreateRegistryPipeline(runtime *common.KubeRuntime) error {
	if err := common.UnCompressPkgToOptPath(runtime.Cluster.Spec.Nodes.Registry[0]); err != nil {
		return err
	}

	tools := []string{constants.Tar, constants.OpenSSL, constants.Iptables}
	installCheck := precheck.NewCreateRegistryFlightCheck(runtime.Cluster.Spec.Nodes)
	installCheck.FlightCheck.NodesOperate.Tools = tools
	if err := installCheck.StartCheck(); err != nil {
		return err
	}

	// 渲染shell脚本，并拷贝安装脚本
	pretreatment := asset.NewPretreatment(runtime.Cluster)
	if err := pretreatment.PretreatmentPackage(); err != nil {
		return err
	}

	// 生成组件依赖图
	builder := executor.NewGraphBuilder()
	// 此依赖图也用于安装组件
	graph, err := builder.InstallGraphBuilder(runtime.Cluster.Spec.Components, runtime.Cluster.Spec.Nodes.Registry[0])
	if err != nil {
		return err
	}

	// 执行前置检查
	exec := executor.NewExecutor(constants.PreCheck, runtime.Cluster.Spec.Nodes, graph)
	exec.FlightCheck()

	// 拷贝二进制文件
	prepareArtifact := binary.NewPrepareArtifact(graph, runtime.Cluster.Spec.Nodes, runtime.Arg.FilePath)
	prepareArtifact.EnableCopyImage = true
	prepareArtifact.EnableCopyBinary = true
	if err := prepareArtifact.CreateRegistryCopyBinary(); err != nil {
		return err
	}

	// 根据依赖图进行安装
	exec.Operate = constants.Install
	exec.RegistryInstall()

	// 上传镜像准备
	domain, passwd, ip := hub.GetHubInfo(runtime.Cluster.Spec.Components)
	if err := hub.UploadImagePrepare(passwd, domain, runtime.Cluster.Spec.Nodes); err != nil {
		return err
	}

	// 上传镜像
	if err := prepareArtifact.UploadImages(domain); err != nil {
		return err
	}
	// 记录域名与证书文件
	if err := hub.RecordHarborConfig(domain, ip, runtime.Cluster.Spec.Nodes.Registry[0]); err != nil {
		return err
	}
	common.RemoveTmpFiles(runtime.Cluster.Spec.Nodes, runtime.Arg.SkipRmTmpFile)
	fmt.Println("The image warehouse is created successfully.")
	return nil
}

func clearSensitiveData(args common.Argument, runtime *common.KubeRuntime) {
	args.ClearSensitiveData()
	runtime.ClearSensitiveData()
}

// CreateRegistry 创建镜像仓并上传制品入口
func CreateRegistry(args common.Argument) error {
	ok, err := params.VerifyPasswd(args.RegistryPasswd)
	if !ok {
		return err
	}
	ok, err = params.VerifyPasswd(args.RegistryDBPasswd)
	if !ok {
		return err
	}
	if err := params.VerifyRegistryCertExpiryDays(args.RegistryCertExpiryDays); err != nil {
		return err
	}

	if args.RegistryDomain != "" {
		// 生成证书时域名中带有http://字样会有出错，同时域名不带有字样也不影响域名设置，所以删除https://与http://
		args.RegistryDomain = strings.TrimPrefix(args.RegistryDomain, constants.Https)
		args.RegistryDomain = strings.TrimPrefix(args.RegistryDomain, constants.Http)

		ok, err = params.VerifyDomain(constants.RegistryDomain, args.RegistryDomain)
		if !ok {
			return err
		}
	}

	runtime, err := common.NewKubeRuntime(args)
	if err != nil {
		return err
	}

	defer clearSensitiveData(args, runtime)

	if err = NewCreateRegistryPipeline(runtime); err != nil {
		return err
	}
	return nil
}
