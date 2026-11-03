/*
Copyright 2017 The Kubernetes Authors.
Copyright (c) 2025 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// package grpc
package grpc

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/api"
	decryptapi "go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/grpc/v1"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/log"
)

const (
	// unixProtocol is the only supported protocol for remote Decryption provider.
	unixProtocol = "unix"
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	decryptClient decryptapi.DecryptionServiceClient
	connection    *grpc.ClientConn
	callTimeout   time.Duration
}

// CreateGRPCService returns a cert.Service which use gRPC to communicate the remote Decryption provider.
func CreateGRPCService(endpoint string, callTimeout time.Duration) (api.Service, error) {
	log.GetLogger().Info("Configure Decryption provider with endpoint", zap.String("endpoint", endpoint))

	addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	s := &gRPCService{callTimeout: callTimeout}
	s.connection, err = grpc.Dial(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.WaitForReady(true)),
		grpc.WithContextDialer(
			func(context.Context, string) (net.Conn, error) {
				// Ignoring addr and timeout arguments:
				// addr - comes from the closure
				c, err := net.DialUnix(unixProtocol, nil, &net.UnixAddr{Name: addr})
				if err != nil {
					log.GetLogger().Error("Failed to create unix socket connection, address",
						zap.String("address", addr), zap.Error(err))
				} else {
					log.GetLogger().Info("Successfully dialed Unix socket address:", zap.String("address", addr))
				}
				return c, err
			}))

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, error: %v", endpoint, err)
	}

	s.decryptClient = decryptapi.NewDecryptionServiceClient(s.connection)
	return s, nil
}

// DecryptCert a given cert data byte to obtain the original byte data.
func (g *gRPCService) DecryptCert(cipher []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()

	request := &decryptapi.DecryptRequest{
		Ciphertext: cipher,
	}
	response, err := g.decryptClient.DecryptCert(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Plaintext, nil
}

// DecryptSecret a given secret data byte to obtain the original byte data.
func (g *gRPCService) DecryptSecret(cipher []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()

	request := &decryptapi.DecryptRequest{
		Ciphertext: cipher,
	}
	response, err := g.decryptClient.DecryptSecret(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Plaintext, nil
}

func (g *gRPCService) Release() {
	if g.connection == nil {
		return
	}
	err := g.connection.Close()
	if err != nil {
		log.GetLogger().Error("Ignore for closing uds connection failed", zap.Error(err))
	}
}

// Parse the endpoint to extract schema, host or path.
func parseEndpoint(endpoint string) (string, error) {
	if len(endpoint) == 0 {
		return "", fmt.Errorf("remote Decryption provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q for remote Decryption provider, error: %v", endpoint, err)
	}

	if u.Scheme != unixProtocol {
		return "", fmt.Errorf("unsupported scheme %q for remote Decryption provider", u.Scheme)
	}

	// Linux abstract namespace socket - no physical file required
	// Warning: Linux Abstract sockets have not concept of ACL (unlike traditional file based sockets).
	// However, Linux Abstract sockets are subject to Linux networking namespace, so will only be accessible to
	// containers within the same pod (unless host networking is used).
	if strings.HasPrefix(u.Path, "/@") {
		return strings.TrimPrefix(u.Path, "/"), nil
	}

	return u.Path, nil
}
