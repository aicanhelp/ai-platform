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

package containerd

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/pkg/errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pb "k8s.io/cri-api/pkg/apis/runtime/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// unixProtocol is the network protocol of unix socket.
	unixProtocol = "unix"
	// DefaultConnectionTimeoutSeconds specifies the default connection timeout in seconds
	DefaultConnectionTimeoutSeconds = 10
	// DefaultConnectionTimeout specifies the default connection timeout duration
	DefaultConnectionTimeout = DefaultConnectionTimeoutSeconds * time.Second
)

var (
	defaultRuntimeEndpoints = []string{"unix:///var/run/dockershim.sock", "unix:///run/containerd/containerd.sock", "unix:///run/crio/crio.sock"}
)

func getRuntimeClient(runtimeEndpoint string) (pb.RuntimeServiceClient, *grpc.ClientConn, error) {
	trimmed := strings.TrimSpace(runtimeEndpoint)
	if trimmed == "" {
		return nil, nil, errors.New("runtime endpoint must be provided")
	}
	conn, err := getRuntimeClientConnection(trimmed)
	if err != nil {
		return nil, nil, errors.Wrap(err, "connect runtime")
	}
	runtimeClient := pb.NewRuntimeServiceClient(conn)
	return runtimeClient, conn, nil
}

func getRuntimeClientConnection(runtimeEndpoint string) (*grpc.ClientConn, error) {
	log.Debug("get runtime connection")
	// If no EP set then use the default endpoint types
	if runtimeEndpoint == "" {
		log.Debugf("runtime connect using default endpoints: %v. "+
			"As the default settings are now deprecated, you should set the "+
			"endpoint instead.", defaultRuntimeEndpoints)
		log.Debug("Note that performance maybe affected as each default " +
			"connection attempt takes n-seconds to complete before timing out " +
			"and going to the next in sequence.")
		return getConnection(defaultRuntimeEndpoints)
	}
	return getConnection([]string{runtimeEndpoint})
}

func getImageClient(imageEndpoint string) (pb.ImageServiceClient, *grpc.ClientConn, error) {
	// Create gRPC connection to image service
	grpcConn, err := getImageClientConnection(imageEndpoint)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to establish image service connection")
	}
	client := pb.NewImageServiceClient(grpcConn)
	return client, grpcConn, nil
}

func getImageClientConnection(imageEndpoint string) (*grpc.ClientConn, error) {

	log.Debugf("get image connection")
	// If no EP set then use the default endpoint types
	if imageEndpoint == "" {
		log.Debugf("image connect using default endpoints: %v. "+
			"As the default settings are now deprecated, you should set the "+
			"endpoint instead.", defaultRuntimeEndpoints)
		log.Debug("Note that performance maybe affected as each default " +
			"connection attempt takes n-seconds to complete before timing out " +
			"and going to the next in sequence.")
		return getConnection(defaultRuntimeEndpoints)
	}
	return getConnection([]string{imageEndpoint})
}

func getConnection(endPoints []string) (*grpc.ClientConn, error) {
	endPointsLen := len(endPoints)
	if endPointsLen == 0 {
		return nil, fmt.Errorf("endpoint len is invalid")
	}

	var conn *grpc.ClientConn
	for idx, endPoint := range endPoints {
		log.Debugf("connect using endpoint '%s' with '%s' timeout", endPoint, DefaultConnectionTimeout)
		addr, dialer, err := GetAddressAndDialer(endPoint)
		if err != nil {
			if idx == endPointsLen-1 {
				return nil, err
			}
			log.Debugf("resolve endpoint %s failed, try next: %v", endPoint, err)
			continue
		}
		conn, err = grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock(), grpc.WithTimeout(DefaultConnectionTimeout), grpc.WithContextDialer(dialer))
		if err != nil {
			errMsg := errors.Wrapf(err, "connect endpoint '%s', make sure you are running as root and the endpoint has been started", endPoint)
			if idx == endPointsLen-1 {
				return nil, errMsg
			}
			log.Debugf("connect endpoint %s failed, try next: %v", endPoint, errMsg)
		} else {
			log.Debugf("connected successfully using endpoint: %s", endPoint)
			break
		}
	}
	return conn, nil
}

// GetAddressAndDialer returns the address parsed from the given endpoint and a context dialer.
func GetAddressAndDialer(endpoint string) (string, func(ctx context.Context, addr string) (net.Conn, error), error) {
	// Parse with Unix protocol fallback
	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)
	if err != nil {
		return "", nil, fmt.Errorf("endpoint validation failed: %w", err)
	}

	// Protocol whitelist check
	if protocol != unixProtocol {
		return "", nil, fmt.Errorf("unsupported protocol %q: only unix sockets are allowed", protocol)
	}

	return addr, dial, nil
}

// dial establishes a network connection using the Unix domain socket protocol.
func dial(ctx context.Context, addr string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, unixProtocol, addr)
}

// parseEndpointWithFallbackProtocol attempts to parse an endpoint URL with a fallback protocol.
func parseEndpointWithFallbackProtocol(endpoint string, fallbackProtocol string) (string, string, error) {
	const deprecatedFormatMsg = "endpoint format deprecated - please use full URL (e.g. %s://%s)"

	// First try direct parsing
	protocol, addr, err := parseEndpoint(endpoint)
	if err == nil {
		return protocol, addr, nil
	}

	// Only attempt fallback for protocol-less endpoints
	if protocol != "" {
		return "", "", fmt.Errorf("invalid endpoint format: %w", err)
	}

	// Construct and try fallback URL
	fallbackEndpoint := fmt.Sprintf("%s://%s", fallbackProtocol, endpoint)
	protocol, addr, err = parseEndpoint(fallbackEndpoint)
	if err != nil {
		return "", "", fmt.Errorf("invalid endpoint and fallback parse failed: %w", err)
	}

	// Log deprecation warning (rate limited in production)
	log.Info(fmt.Sprintf(deprecatedFormatMsg, fallbackProtocol, endpoint))
	return protocol, addr, nil
}

// parseEndpoint parses a network endpoint URL into protocol and address components.
func parseEndpoint(endpoint string) (string, string, error) {
	// Strict validation of URL format
	if strings.TrimSpace(endpoint) == "" {
		return "", "", fmt.Errorf("empty endpoint URL provided")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", fmt.Errorf("invalid endpoint URL format: %w", err)
	}

	// Normalize scheme to lowercase for case-insensitive comparison
	scheme := strings.ToLower(u.Scheme)

	switch scheme {
	case "tcp":
		if u.Host == "" {
			return "", "", fmt.Errorf("missing host address in TCP endpoint")
		}
		return "tcp", u.Host, nil

	case "unix":
		if u.Path == "" {
			return "", "", fmt.Errorf("missing socket path in Unix endpoint")
		}
		return "unix", u.Path, nil

	case "":
		// Empty scheme indicates legacy format - reject for security
		return "", "", fmt.Errorf("legacy endpoint format not allowed - use full URL (tcp:// or unix://)")

	default:
		// Reject unsupported protocols
		return "", "", fmt.Errorf("unsupported network protocol %q (allowed: tcp, unix)", scheme)
	}
}
