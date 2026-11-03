/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package etcd

import (
	"context"
	"fmt"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/snapshot"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	k8setcd "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
)

const etcdTimeout = 2 * time.Second
const etcdSaveTimeout = 4 * time.Minute

// Save generates an etcd snapshot and writes it to the specified file path.
//
// It connects to the first endpoint defined in the provided etcd client,
// establishes a secure connection using the client's TLS configuration,
// and then saves a consistent etcd snapshot to the given destination path.
//
// The operation runs with a 4-minute timeout context. If the snapshot process
// fails, an error describing the failure will be returned.
func Save(c *k8setcd.Client, dbPath string) error {
	ctx, cancel := context.WithTimeout(context.Background(), etcdSaveTimeout)
	defer cancel()
	cfg := clientv3.Config{
		Endpoints:   c.Endpoints[:1],
		DialTimeout: etcdTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	}

	snapshotLogger, err := zap.NewProduction()
	if err != nil {
		return fmt.Errorf("failed to create etcd snapshot logger: %w", err)
	}
	defer func() { _ = snapshotLogger.Sync() }()

	if err := snapshot.Save(ctx, snapshotLogger, cfg, dbPath); err != nil {
		return fmt.Errorf("failed to save etcd snapshot: %w", err)
	}
	return nil
}
