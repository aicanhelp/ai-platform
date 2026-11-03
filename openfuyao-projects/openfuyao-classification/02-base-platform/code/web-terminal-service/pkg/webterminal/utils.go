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

package webterminal

import "time"

const (
	endOfWindow = "\u0004"
	// WaitWirte 写入等待时间
	WaitWirte = 10 * time.Second
	pongWait  = 30 * time.Second
	pingSend  = (pongWait * 9) / 10
)

const (
	// UserPodNamespace user pod 名字
	UserPodNamespace = "openfuyao-system"
	// UserContainerName 容器名
	UserContainerName = "user-container"
	// RootPath 路径
	RootPath = "/root/.kube"
	// Finalizer 标识
	Finalizer = "openfuyao.com.finalizer.webterminal"
	// KubectlApi 请求标识
	KubectlApi = "/rest/webterminal/v1/user/"
	ImagePath  = "/mnt/data/imagePath.txt"
)
