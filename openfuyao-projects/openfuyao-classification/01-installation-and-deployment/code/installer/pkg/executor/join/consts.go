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

package join

const (
	// etcd 证书存放路径
	etcd = "etcd"
	// K8sCerts k8s证书临时存放目录
	k8sCerts = "certs"
	// 集群配置文件证书存放路径参数名
	certDir = "certDir"

	// k8s证书名
	caCrt           = "ca.crt"
	caKey           = "ca.key"
	saKey           = "sa.key"
	saPub           = "sa.pub"
	frontProxyCaCrt = "front-proxy-ca.crt"
	frontProxyCaKey = "front-proxy-ca.key"
	etcdCaCrt       = "ca.crt"
	etcdCaKey       = "ca.key"

	masterCountMin = 1
)
