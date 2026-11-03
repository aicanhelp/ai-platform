/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package containerd

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
)

func TestGetRuntimeClientWithValidEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getRuntimeClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	runtimeClient, conn, err := getRuntimeClient("unix:///var/run/containerd.sock")

	assert.NoError(t, err)
	assert.NotNil(t, runtimeClient)
	assert.Equal(t, testConn, conn)
}

func TestGetRuntimeClientWithConnectionError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getRuntimeClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return nil, assert.AnError
	})

	runtimeClient, conn, err := getRuntimeClient("unix:///var/run/containerd.sock")

	assert.Error(t, err)
	assert.Nil(t, runtimeClient)
	assert.Nil(t, conn)
	assert.Contains(t, err.Error(), "connect")
}

func TestGetRuntimeClientConnectionWithDefaultEndpoints(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	conn, err := getRuntimeClientConnection("")

	assert.NoError(t, err)
	assert.Equal(t, testConn, conn)
}

func TestGetRuntimeClientConnectionWithCustomEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	conn, err := getRuntimeClientConnection("unix:///custom/path.sock")

	assert.NoError(t, err)
	assert.Equal(t, testConn, conn)
}

func TestGetRuntimeClientConnectionWithConnectionError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return nil, assert.AnError
	})

	conn, err := getRuntimeClientConnection("unix:///custom/path.sock")

	assert.Error(t, err)
	assert.Nil(t, conn)
}

func TestGetImageClientWithValidEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getImageClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	imageClient, conn, err := getImageClient("unix:///var/run/containerd.sock")

	assert.NoError(t, err)
	assert.NotNil(t, imageClient)
	assert.Equal(t, testConn, conn)
}

func TestGetImageClientWithConnectionError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return nil, assert.AnError
	})

	imageClient, conn, err := getImageClient("unix:///var/run/containerd.sock")

	assert.Error(t, err)
	assert.Nil(t, imageClient)
	assert.Nil(t, conn)
	assert.Contains(t, err.Error(), "connect")
}

func TestGetImageClientConnectionWithDefaultEndpoints(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	conn, err := getImageClientConnection("")

	assert.NoError(t, err)
	assert.Equal(t, testConn, conn)
}

func TestGetImageClientConnectionWithCustomEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testConn := &grpc.ClientConn{}
	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return testConn, nil
	})

	conn, err := getImageClientConnection("unix:///custom/image.sock")

	assert.NoError(t, err)
	assert.Equal(t, testConn, conn)
}

func TestGetImageClientConnectionWithConnectionError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getConnection, func(endPoints []string) (*grpc.ClientConn, error) {
		return nil, assert.AnError
	})

	conn, err := getImageClientConnection("unix:///custom/image.sock")

	assert.Error(t, err)
	assert.Nil(t, conn)
}

func TestGetAddressAndDialerWithHTTPFallback(t *testing.T) {
	endpoint := "http://localhost:9000"

	addr, dialer, err := GetAddressAndDialer(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "//localhost:9000", addr)
	assert.NotNil(t, dialer)
}

func TestParseEndpointWithValidHTTPS(t *testing.T) {
	endpoint := "https://registry.example.com"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported network protocol")
}

func TestParseEndpointWithHTTP(t *testing.T) {
	endpoint := "http://registry.example.com"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported network protocol")
}

func TestParseEndpointWithEmptyPathUnix(t *testing.T) {
	endpoint := "unix:///path/to/socket"

	protocol, addr, err := parseEndpoint(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "unix", protocol)
	assert.Equal(t, "/path/to/socket", addr)
}

func TestParseEndpointWithWhitespaceOnly(t *testing.T) {
	endpoint := "   "

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty endpoint")
}

func TestParseEndpointWithFallbackProtocolEmptyEndpoint(t *testing.T) {
	_, _, err := parseEndpointWithFallbackProtocol("", unixProtocol)

	assert.Error(t, err)
}

func TestParseEndpointWithFallbackProtocolInvalidFallback(t *testing.T) {
	_, _, err := parseEndpointWithFallbackProtocol("::invalid", "tcp")

	assert.Error(t, err)
}

func TestGetRuntimeClientWithNilConnection(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getRuntimeClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return nil, nil
	})

	runtimeClient, conn, err := getRuntimeClient("unix:///var/run/containerd.sock")

	assert.NoError(t, err)
	assert.NotNil(t, runtimeClient)
	assert.Nil(t, conn)
}

func TestGetImageClientWithNilConnection(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(getImageClientConnection, func(endpoint string) (*grpc.ClientConn, error) {
		return nil, nil
	})

	imageClient, conn, err := getImageClient("unix:///var/run/containerd.sock")

	assert.NoError(t, err)
	assert.NotNil(t, imageClient)
	assert.Nil(t, conn)
}

func TestGetAddressAndDialerWithHttpsFallback(t *testing.T) {
	endpoint := "https://secure.example.com:443"

	addr, dialer, err := GetAddressAndDialer(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "//secure.example.com:443", addr)
	assert.NotNil(t, dialer)
}

func TestGetAddressAndDialerWithValidUnixSocket(t *testing.T) {
	endpoint := "unix:///var/run/containerd/containerd.sock"

	addr, dialer, err := GetAddressAndDialer(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "/var/run/containerd/containerd.sock", addr)
	assert.NotNil(t, dialer)
}

func TestGetAddressAndDialerWithTCPEndpoint(t *testing.T) {
	endpoint := "tcp://localhost:9000"

	_, _, err := GetAddressAndDialer(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported protocol")
}

func TestGetAddressAndDialerWithEmptyEndpoint(t *testing.T) {
	endpoint := ""

	_, _, err := GetAddressAndDialer(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "endpoint validation failed")
}

func TestGetAddressAndDialerWithUnsupportedProtocol(t *testing.T) {
	endpoint := "http://localhost:9000"

	addr, dialer, err := GetAddressAndDialer(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "//localhost:9000", addr)
	assert.NotNil(t, dialer)
}

func TestParseEndpointWithUnixSocket(t *testing.T) {
	endpoint := "unix:///var/run/containerd/containerd.sock"

	protocol, addr, err := parseEndpoint(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "unix", protocol)
	assert.Equal(t, "/var/run/containerd/containerd.sock", addr)
}

func TestParseEndpointWithTCP(t *testing.T) {
	endpoint := "tcp://localhost:9000"

	protocol, addr, err := parseEndpoint(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "tcp", protocol)
	assert.Equal(t, "localhost:9000", addr)
}

func TestParseEndpointWithEmptyEndpoint(t *testing.T) {
	endpoint := ""

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty endpoint")
}

func TestParseEndpointWithMissingHostInTCP(t *testing.T) {
	endpoint := "tcp://"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "missing host")
}

func TestParseEndpointWithMissingSocketPath(t *testing.T) {
	endpoint := "unix://"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "missing socket path")
}

func TestParseEndpointWithLegacyFormat(t *testing.T) {
	endpoint := "/var/run/containerd/containerd.sock"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "legacy endpoint format")
}

func TestParseEndpointWithUnsupportedScheme(t *testing.T) {
	endpoint := "ftp://localhost:9000"

	_, _, err := parseEndpoint(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported network protocol")
}

func TestParseEndpointWithCaseInsensitiveScheme(t *testing.T) {
	endpoint := "UNIX:///var/run/containerd/containerd.sock"

	protocol, addr, err := parseEndpoint(endpoint)

	assert.NoError(t, err)
	assert.Equal(t, "unix", protocol)
	assert.Equal(t, "/var/run/containerd/containerd.sock", addr)
}

func TestGetConnectionWithEmptyEndpoints(t *testing.T) {
	endpoints := []string{}

	conn, err := getConnection(endpoints)

	assert.Error(t, err)
	assert.Nil(t, conn)
	assert.Contains(t, err.Error(), "endpoint len is invalid")
}

func TestParseEndpointWithFallbackProtocol(t *testing.T) {
	endpoint := "/var/run/containerd/containerd.sock"

	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)

	assert.NoError(t, err)
	assert.Equal(t, unixProtocol, protocol)
	assert.Equal(t, "/var/run/containerd/containerd.sock", addr)
}

func TestParseEndpointWithFallbackProtocolValidURL(t *testing.T) {
	endpoint := "unix:///var/run/containerd/containerd.sock"

	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)

	assert.NoError(t, err)
	assert.Equal(t, "unix", protocol)
	assert.Equal(t, "/var/run/containerd/containerd.sock", addr)
}

func TestParseEndpointWithFallbackProtocolInvalidURL(t *testing.T) {
	endpoint := "::invalid"

	_, _, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)

	assert.Error(t, err)
}

func TestParseEndpointWithFallbackProtocolProtocolSet(t *testing.T) {
	endpoint := "http://localhost:9000"

	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)

	assert.NoError(t, err)
	assert.Equal(t, unixProtocol, protocol)
	assert.Equal(t, "//localhost:9000", addr)
}

func TestGetAddressAndDialerWithIPv4Loopback(t *testing.T) {
	endpoint := "tcp://127.0.0.1:9000"

	_, _, err := GetAddressAndDialer(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported protocol")
}

func TestGetAddressAndDialerWithIPv4PrivateClassB(t *testing.T) {
	endpoint := "tcp://192.168.1.1:9000"

	_, _, err := GetAddressAndDialer(endpoint)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported protocol")
}
