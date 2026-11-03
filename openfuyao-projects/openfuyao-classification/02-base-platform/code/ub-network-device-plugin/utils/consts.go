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

package utils

import "time"

const (
	ResourceNamePrefix    = "unifiedbus.com"
	URMAResource          = "ub_net_device"
	MaxResourceNameLength = 63
	ResourceName          = ResourceNamePrefix + "/" + URMAResource

	KubeletCheckpoint = "/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint"
	RuntimeSock       = "/run/containerd/containerd.sock"

	TidDevicePath = "/dev/ummu/tid"
	UBCorePath    = "/dev/ubcore/ubcore"
	Permission    = 0666

	GrpcTimeOut = 30 * time.Second
)
