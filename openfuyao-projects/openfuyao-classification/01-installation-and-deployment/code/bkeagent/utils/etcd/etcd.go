/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package etcd

import (
	"context"
	"time"

	"github.com/pkg/errors"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/snapshot"
	"google.golang.org/grpc"
	k8setcd "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"

	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const etcdTimeout = 2 * time.Second

// Save generates an etcd snapshot and writes it to the specified file path.
//
// It connects to the first endpoint defined in the provided etcd client,
// establishes a secure connection using the client's TLS configuration,
// and then saves a consistent etcd snapshot to the given destination path.
//
// The operation runs with a 4-minute timeout context. If the snapshot process
// fails, an error describing the failure will be returned.
func Save(c *k8setcd.Client, dbPath string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	cfg := clientv3.Config{
		Endpoints:   c.Endpoints[:1],
		DialTimeout: etcdTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	}

	if err := snapshot.Save(ctx, log.Desugar(), cfg, dbPath); err != nil {
		return errors.Errorf("failed to save etcd snapshot: %v", err)
	}
	return nil
}
