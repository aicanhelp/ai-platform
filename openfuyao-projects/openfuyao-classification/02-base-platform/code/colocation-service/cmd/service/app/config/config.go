/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package config for colocation-service
package config

import (
	apiserver "k8s.io/apiserver/pkg/server"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"

	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned"
)

// DefaultServerPort Server的默认端口号
const DefaultServerPort = 8080

// OverallConfig 拥有运行colocation-service服务器的配置
type OverallConfig struct {
	SecureServing    *apiserver.SecureServingInfo
	Client           clientset.Interface
	ColocationClient colocationclientset.Interface
	KubeConfig       *restclient.Config
	// LoopbackClientConfig 是特权环回连接的配置
	LoopbackClientConfig *restclient.Config
}

type completedConfig struct {
	*OverallConfig
}

// CompletedConfig 同Config，仅仅为了交换私有对象。
type CompletedConfig struct {
	// 嵌入不能在此包之外实例化的私有指针。
	*completedConfig
}

// Complete 填充任何未设置的字段，这些字段必须具有有效数据。它让接收器发生变异。
func (c *OverallConfig) Complete() CompletedConfig {
	cc := completedConfig{OverallConfig: c}

	return CompletedConfig{completedConfig: &cc}
}
