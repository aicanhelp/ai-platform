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

package factory

import (
	"crypto/sha512"
	"crypto/tls"
	"time"

	"go.etcd.io/etcd/client/v3/kubernetes"
	"go.etcd.io/etcd/server/v3/etcdserver"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/fy-ext/dynamicloader"
)

const (
	size                       = sha512.Size384
	fyFastPerceiveNodeNotReady = "FY_FAST_PERCEIVE_NODENOTREADY_CONFIG"
	fastKeepaliveTimeout       = 5 * time.Second
	fastKeepaliveTime          = 10 * time.Second
)

func fyExtendTLSConfig(tlsConfig *tls.Config, c storagebackend.TransportConfig) error {
	if len(c.TrustedCAFile) > 0 {
		if err := dynamicloader.UpdateCADynamicallyForClient(tlsConfig, c.TrustedCAFile); err != nil {
			return err
		}
	}
	return nil
}

// SetEtcdServer for kubeone
func SetEtcdServer(server *etcdserver.EtcdServer) {
	if server == nil {
		return
	}

	newETCD3Client = func(c storagebackend.TransportConfig) (*kubernetes.Client, error) {
		kc := &kubernetes.Client{Client: v3client.New(server)}
		kc.Kubernetes = kc
		return kc, nil
	}
}
