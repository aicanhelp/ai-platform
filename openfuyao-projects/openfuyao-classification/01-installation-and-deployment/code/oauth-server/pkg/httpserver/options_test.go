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

package httpserver

import (
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
)

const (
	fileMode         = 0644
	httpsPort        = 8443
	defaultHttpsPort = 9096
)

var perm = fs.FileMode(fileMode)

func TestNewDefaultHttpServerOptions(t *testing.T) {
	options := NewDefaultHttpServerOptions()
	assert.NotNil(t, options)
	assert.Equal(t, defaultHttpsPort, options.HttpPort)
	assert.Equal(t, 0, options.HttpsPort)
	assert.Equal(t, "", options.TlsCertFile)
	assert.Equal(t, "", options.TlsPrivateKeyFile)
	assert.Equal(t, "", options.RootCAFile)
}

func TestServerOptionsValidateValidConfig(t *testing.T) {
	options := &ServerOptions{
		HttpPort:          defaultHttpsPort,
		HttpsPort:         0,
		TlsCertFile:       "",
		TlsPrivateKeyFile: "",
		RootCAFile:        "",
	}

	errs := options.Validate()
	assert.Empty(t, errs)
}

func TestServerOptionsValidateInvalidPorts(t *testing.T) {
	options := &ServerOptions{
		HttpPort:  constants.MinHttpPort,
		HttpsPort: constants.MinHttpPort,
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.Contains(t, errs, fuyaoerrors.ErrInvalidHttpAndHttpsPort)
}

func TestServerOptionsValidateValidHttpsConfig(t *testing.T) {
	// Create temporary files for testing
	tempDir := t.TempDir()
	certFile := filepath.Join(tempDir, "cert.crt")
	keyFile := filepath.Join(tempDir, "key.key")
	caFile := filepath.Join(tempDir, "ca.crt")

	// Create dummy files
	err := os.WriteFile(certFile, []byte("cert"), perm)
	if err != nil {
		return
	}
	err = os.WriteFile(keyFile, []byte("key"), perm)
	if err != nil {
		return
	}
	err = os.WriteFile(caFile, []byte("ca"), perm)
	if err != nil {
		return
	}

	options := &ServerOptions{
		HttpPort:          0,
		HttpsPort:         httpsPort,
		TlsCertFile:       certFile,
		TlsPrivateKeyFile: keyFile,
		RootCAFile:        caFile,
	}

	errs := options.Validate()
	assert.Empty(t, errs)
}

func TestServerOptionsValidateMissingCertFile(t *testing.T) {
	options := &ServerOptions{
		HttpPort:          0,
		HttpsPort:         httpsPort,
		TlsCertFile:       "",
		TlsPrivateKeyFile: "key.key",
		RootCAFile:        "ca.crt",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.Contains(t, errs, fuyaoerrors.ErrEmptyCertFile)
}

func TestServerOptionsValidateMissingKeyFile(t *testing.T) {
	tempDir := t.TempDir()
	certFile := filepath.Join(tempDir, "cert.crt")
	err := os.WriteFile(certFile, []byte("cert"), perm)
	if err != nil {
		return
	}
	options := &ServerOptions{
		HttpsPort:         httpsPort,
		HttpPort:          0,
		TlsCertFile:       certFile,
		TlsPrivateKeyFile: "",
		RootCAFile:        "ca.crt",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.Contains(t, errs, fuyaoerrors.ErrEmptyPrivateKeyFile)
}

func TestServerOptionsValidateMissingCAFile(t *testing.T) {
	tempDir := t.TempDir()
	certFile := filepath.Join(tempDir, "cert.crt")
	keyFile := filepath.Join(tempDir, "key.key")
	err := os.WriteFile(certFile, []byte("cert"), perm)
	if err != nil {
		return
	}
	err = os.WriteFile(keyFile, []byte("key"), perm)
	if err != nil {
		return
	}

	options := &ServerOptions{
		HttpPort:          0,
		HttpsPort:         httpsPort,
		TlsCertFile:       certFile,
		TlsPrivateKeyFile: keyFile,
		RootCAFile:        "",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.Contains(t, errs, fuyaoerrors.ErrEmptyMasterCAFile)
}

func TestServerOptionsValidateCertFileNotFound(t *testing.T) {
	options := &ServerOptions{
		HttpPort:          0,
		HttpsPort:         httpsPort,
		TlsCertFile:       "/nonexistent/cert.crt",
		TlsPrivateKeyFile: "key.key",
		RootCAFile:        "ca.crt",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.True(t, os.IsNotExist(errs[1]))
}

func TestServerOptionsValidateKeyFileNotFound(t *testing.T) {
	tempDir := t.TempDir()
	certFile := filepath.Join(tempDir, "cert.crt")
	err := os.WriteFile(certFile, []byte("cert"), perm)
	if err != nil {
		return
	}

	options := &ServerOptions{
		HttpPort:          0,
		HttpsPort:         httpsPort,
		TlsCertFile:       certFile,
		TlsPrivateKeyFile: "/nonexistent/key.key",
		RootCAFile:        "ca.crt",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.True(t, os.IsNotExist(errs[1]))
}

func TestServerOptionsValidateCAFileNotFound(t *testing.T) {
	tempDir := t.TempDir()
	keyFile := filepath.Join(tempDir, "key.key")
	certFile := filepath.Join(tempDir, "cert.crt")
	err := os.WriteFile(certFile, []byte("cert"), perm)
	if err != nil {
		return
	}
	err = os.WriteFile(keyFile, []byte("key"), perm)
	if err != nil {
		return
	}

	options := &ServerOptions{
		HttpsPort:         httpsPort,
		HttpPort:          0,
		TlsCertFile:       certFile,
		TlsPrivateKeyFile: keyFile,
		RootCAFile:        "/nonexistent/ca.crt",
	}

	errs := options.Validate()
	assert.NotEmpty(t, errs)
	assert.True(t, os.IsNotExist(errs[0]))
}

func TestNewHttpServerHttpOnly(t *testing.T) {
	options := &ServerOptions{
		HttpPort:  defaultHttpsPort,
		HttpsPort: 0,
	}

	server, err := NewHttpServer(options)
	assert.NoError(t, err)
	assert.NotNil(t, server)
	assert.Equal(t, ":9096", server.Addr)
	assert.Nil(t, server.TLSConfig)
}

func TestNewHttpServerNilOptions(t *testing.T) {
	server, err := NewHttpServer(nil)
	assert.Error(t, err)
	assert.Equal(t, fuyaoerrors.ErrHttpServerOptionsNil, err)
	assert.Nil(t, server)
}
