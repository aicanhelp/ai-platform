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

package httputil

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"console-service/pkg/constant"
)

const (
	hours     = 24
	days      = 365
	serialNum = 2
	keyLen    = 2048
)

func TestGetResponseJson(t *testing.T) {
	got := GetResponseJson(constant.Success, "ok", "test")
	want := &ResponseJson{Code: constant.Success, Msg: "ok", Data: "test"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetResponseJson() = %v, want %v", got, want)
	}
}

func TestGetDefaultSuccessResponseJson(t *testing.T) {
	got := GetDefaultSuccessResponseJson()
	want := &ResponseJson{Code: constant.Success, Msg: "success", Data: nil}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetDefaultSuccessResponseJson() = %v, want %v", got, want)
	}
}

func TestGetDefaultClientFailureResponseJson(t *testing.T) {
	got := GetDefaultClientFailureResponseJson()
	want := &ResponseJson{Code: constant.ClientError, Msg: "bad request", Data: nil}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetDefaultClientFailureResponseJson() = %v, want %v", got, want)
	}
}

func TestGetDefaultServerFailureResponseJson(t *testing.T) {
	got := GetDefaultServerFailureResponseJson()
	want := &ResponseJson{Code: constant.ServerError, Msg: "remote server busy", Data: nil}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetDefaultServerFailureResponseJson() = %v, want %v", got, want)
	}
}

func TestGetParamsEmptyErrorResponseJson(t *testing.T) {
	got := GetParamsEmptyErrorResponseJson()
	want := &ResponseJson{Code: constant.ClientError, Msg: "parameters not found", Data: nil}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetParamsEmptyErrorResponseJson() = %v, want %v", got, want)
	}
}

func TestGetHttpConfigTLSFailed(t *testing.T) {
	dummyConfig := &tls.Config{
		MinVersion:         tls.VersionTLS12,
		MaxVersion:         tls.VersionTLS13,
		InsecureSkipVerify: true,
	}

	tests := []struct {
		name      string
		enableTLS bool
		want      *tls.Config
		wantErr   bool
	}{
		{
			name:      "TestDisableTLS",
			enableTLS: false,
			want: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12,
				MaxVersion: tls.VersionTLS13},
			wantErr: false,
		},
		{
			name:      "TestLoadCertFailed",
			enableTLS: true,
			want:      dummyConfig,
			wantErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetHttpConfig(tt.enableTLS)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetHttpConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetHttpConfig() got = %v, want %v", got, tt.want)
			}
		})
	}

	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(certFile, keyFile string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	defer patch1.Reset()

	t.Run("TestLoadCAFailed", func(t *testing.T) {
		got, err := GetHttpConfig(true)
		if err == nil {
			t.Errorf("GetHttpConfig() error = %v, wantErr true", err)
		}
		if !reflect.DeepEqual(got, dummyConfig) {
			t.Errorf("GetHttpConfig() got = %v, want nil", got)
		}
	})
}

func TestGetHttpConfigSuccess(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(certFile, keyFile string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	defer patch1.Reset()

	fakeCABytes := []byte("-----BEGIN CERTIFICATE-----\nxxxxxxxx\n-----END CERTIFICATE-----")
	fakeCAPool := x509.NewCertPool()
	fakeCAPool.AppendCertsFromPEM(fakeCABytes)

	patch2 := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return fakeCABytes, nil
	})
	defer patch2.Reset()

	got, err := GetHttpConfig(true)
	want := &tls.Config{
		Certificates: []tls.Certificate{{}},
		MinVersion:   tls.VersionTLS12,
		MaxVersion:   tls.VersionTLS13,
		RootCAs:      fakeCAPool,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TLS Config is incorrect: %v, want %v", got, want)
		return
	}
	if err != nil {
		t.Errorf("error = %v, wantErr false", err)
		return
	}
}

func TestGetHttpTransport(t *testing.T) {
	type args struct {
		enableTLS bool
	}
	tests := []struct {
		name    string
		args    args
		want    *http.Transport
		wantErr bool
	}{
		{
			name: "GetHTTPConfigFailed",
			args: args{
				enableTLS: true,
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "GetHTTPConfigSucceeded",
			args: args{
				enableTLS: false,
			},
			want: &http.Transport{
				TLSClientConfig: &tls.Config{
					MinVersion:         tls.VersionTLS12,
					MaxVersion:         tls.VersionTLS13,
					InsecureSkipVerify: true,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetHttpTransport(tt.args.enableTLS)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetHttpTransport() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetHttpTransport() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsHttpsEnabled(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer patch.Reset()
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err != nil {
			t.Errorf("IsHttpsEnabled error = %v, want nil", err)
		}
		if got {
			t.Error("IsHttpsEnabled got = true, want false")
		}
	})

	patch = gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, os.ErrPermission
	})
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err == nil {
			t.Error("IsHttpsEnabled error = nil, want not nil")
		}
		if got {
			t.Error("IsHttpsEnabled got = true, want false")
		}
	})

	patch = gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, nil
	})
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err != nil {
			t.Errorf("IsHttpsEnabled error = %v, want nil", err)
		}
		if !got {
			t.Errorf("IsHttpsEnabled got = %v, want true", got)
		}
	})
}

// Helper function to generate a self-signed certificate and CA for testing
type testCertificate struct {
	certPEM []byte
	keyPEM  []byte
	caPEM   []byte
}

func generateTestCertificate() (*testCertificate, error) {
	caPriv, caCert, err := generateCACertificate()
	if err != nil {
		return nil, err
	}

	serverCertPEM, serverKeyPEM, err := generateServerCertificate(caCert, caPriv)
	if err != nil {
		return nil, err
	}

	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCert.Raw})

	return &testCertificate{
		certPEM: serverCertPEM,
		keyPEM:  serverKeyPEM,
		caPEM:   caPEM,
	}, nil
}

func generateCACertificate() (*rsa.PrivateKey, *x509.Certificate, error) {
	caPriv, err := rsa.GenerateKey(rand.Reader, keyLen)
	if err != nil {
		return nil, nil, err
	}

	caTemplate := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{"Test CA"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour * hours * days),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	caDerBytes, err := x509.CreateCertificate(rand.Reader, &caTemplate, &caTemplate, &caPriv.PublicKey, caPriv)
	if err != nil {
		return nil, nil, err
	}

	caCert, err := x509.ParseCertificate(caDerBytes)
	if err != nil {
		return nil, nil, err
	}

	return caPriv, caCert, nil
}

func generateServerCertificate(caCert *x509.Certificate, caPriv *rsa.PrivateKey) (certPEM, keyPEM []byte, err error) {
	serverPriv, err := rsa.GenerateKey(rand.Reader, keyLen)
	if err != nil {
		return nil, nil, err
	}

	serverTemplate := x509.Certificate{
		SerialNumber: big.NewInt(serialNum),
		Subject: pkix.Name{
			Organization: []string{"Test Server"},
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(time.Hour * hours * days),
		KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
	}

	serverDerBytes, err := x509.CreateCertificate(rand.Reader, &serverTemplate, caCert, &serverPriv.PublicKey, caPriv)
	if err != nil {
		return nil, nil, err
	}

	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverDerBytes})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(serverPriv)})

	return certPEM, keyPEM, nil
}

// Helper function to create temporary files
func createTempFile(content []byte) (string, error) {
	file, err := os.CreateTemp("", "test")
	if err != nil {
		return "", err
	}
	defer file.Close()

	_, err = file.Write(content)
	if err != nil {
		return "", err
	}

	return file.Name(), nil
}

func TestGetCustomizedHttpConfigByPath(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	certPath, err := createTempFile(certs.certPEM)
	assert.NoError(t, err)
	defer os.Remove(certPath)

	keyPath, err := createTempFile(certs.keyPEM)
	assert.NoError(t, err)
	defer os.Remove(keyPath)

	t.Run("Empty cert and key paths", func(t *testing.T) {
		tlsCfg, err := GetCustomizedHttpConfigByPath("", "", "")
		assert.NoError(t, err)
		assert.NotNil(t, tlsCfg)
		assert.True(t, tlsCfg.InsecureSkipVerify)
	})

	t.Run("Non-existent cert path", func(t *testing.T) {
		tlsCfg, err := GetCustomizedHttpConfigByPath("/nonexistent/cert", "/nonexistent/key", "")
		assert.NoError(t, err)
		assert.NotNil(t, tlsCfg)
	})
}

func TestGetCustomizedHttpConfigByRawSkipTLS(t *testing.T) {
	tlsCfg, err := GetCustomizedHttpConfigByRaw(true, nil, nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, tlsCfg)
	assert.True(t, tlsCfg.InsecureSkipVerify)
}

func TestGetCustomizedHttpConfigByRawValidCAOnly(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	tlsCfg, err := GetCustomizedHttpConfigByRaw(false, nil, nil, certs.caPEM)
	assert.NoError(t, err)
	assert.NotNil(t, tlsCfg)
	assert.NotNil(t, tlsCfg.RootCAs)
	assert.Empty(t, tlsCfg.Certificates)
}

func TestGetCustomizedHttpConfigByRawValidClientCert(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	tlsCfg, err := GetCustomizedHttpConfigByRaw(false, certs.certPEM, certs.keyPEM, certs.caPEM)
	assert.NoError(t, err)
	assert.NotNil(t, tlsCfg)
	assert.NotNil(t, tlsCfg.RootCAs)
	assert.Len(t, tlsCfg.Certificates, 1)
}

func TestGetCustomizedHttpConfigByRawMissingCA(t *testing.T) {
	_, err := GetCustomizedHttpConfigByRaw(false, nil, nil, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "required tls verification but no ca is provided")

	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	_, err = GetCustomizedHttpConfigByRaw(false, certs.certPEM, certs.keyPEM, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "required tls verification but no ca is provided")
}

func TestGetCustomizedHttpConfigByRawMissingCertOrKey(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	_, err = GetCustomizedHttpConfigByRaw(false, certs.certPEM, nil, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "both cert and key must be provided")

	_, err = GetCustomizedHttpConfigByRaw(false, nil, certs.keyPEM, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "both cert and key must be provided")
}

func TestGetCustomizedHttpConfigByRawInvalidCA(t *testing.T) {
	invalidCA := []byte("invalid ca certificate")
	_, err := GetCustomizedHttpConfigByRaw(false, nil, nil, invalidCA)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ca is illegal, failed to load ca")
}

func TestGetCustomizedHttpConfigByRawInvalidClientCert(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	invalidCert := []byte("invalid certificate")
	invalidKey := []byte("invalid key")

	_, err = GetCustomizedHttpConfigByRaw(false, invalidCert, invalidKey, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load client certificate/key")
}

func TestLoadClientCert(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	certPath, err := createTempFile(certs.certPEM)
	assert.NoError(t, err)
	defer os.Remove(certPath)

	keyPath, err := createTempFile(certs.keyPEM)
	assert.NoError(t, err)
	defer os.Remove(keyPath)

	t.Run("Valid cert and key paths", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadClientCert(tlsCfg, certPath, keyPath)
		assert.NoError(t, err)
		assert.Len(t, tlsCfg.Certificates, 1)
	})

	t.Run("Empty paths", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadClientCert(tlsCfg, "", "")
		assert.NoError(t, err)
		assert.Len(t, tlsCfg.Certificates, 0)
	})

	t.Run("Non-existent paths", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadClientCert(tlsCfg, "/nonexistent/cert", "/nonexistent/key")
		assert.NoError(t, err)
		assert.Len(t, tlsCfg.Certificates, 0)
	})
}

func TestLoadCA(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	caPath, err := createTempFile(certs.caPEM)
	assert.NoError(t, err)
	defer os.Remove(caPath)

	t.Run("Valid CA path", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadCA(tlsCfg, caPath)
		assert.NoError(t, err)
		assert.NotNil(t, tlsCfg.RootCAs)
	})

	t.Run("Empty CA path", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadCA(tlsCfg, "")
		assert.NoError(t, err)
		assert.Nil(t, tlsCfg.RootCAs)
	})

	t.Run("Non-existent CA path", func(t *testing.T) {
		tlsCfg := &tls.Config{}
		err := loadCA(tlsCfg, "/nonexistent/ca")
		assert.NoError(t, err)
		assert.Nil(t, tlsCfg.RootCAs)
	})

	t.Run("Invalid CA content", func(t *testing.T) {
		invalidCAPath, err := createTempFile([]byte("invalid cert"))
		assert.NoError(t, err)
		defer os.Remove(invalidCAPath)

		tlsCfg := &tls.Config{}
		err = loadCA(tlsCfg, invalidCAPath)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "append CA")
	})
}

func TestGetCustomizedHttpTransportByPath(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	certPath, err := createTempFile(certs.certPEM)
	assert.NoError(t, err)
	defer os.Remove(certPath)

	keyPath, err := createTempFile(certs.keyPEM)
	assert.NoError(t, err)
	defer os.Remove(keyPath)

	t.Run("Valid cert and key paths", func(t *testing.T) {
		transport, err := GetCustomizedHttpTransportByPath(certPath, keyPath, "")
		assert.NoError(t, err)
		assert.NotNil(t, transport)
		assert.NotNil(t, transport.TLSClientConfig)
		assert.Len(t, transport.TLSClientConfig.Certificates, 1)
	})

	t.Run("Empty paths", func(t *testing.T) {
		transport, err := GetCustomizedHttpTransportByPath("", "", "")
		assert.NoError(t, err)
		assert.NotNil(t, transport)
		assert.NotNil(t, transport.TLSClientConfig)
		assert.True(t, transport.TLSClientConfig.InsecureSkipVerify)
	})
}

func TestGetCustomizedHttpTransportByRawSkipTLS(t *testing.T) {
	transport, err := GetCustomizedHttpTransportByRaw(true, nil, nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, transport)
	assert.NotNil(t, transport.TLSClientConfig)
	assert.True(t, transport.TLSClientConfig.InsecureSkipVerify)
}

func TestGetCustomizedHttpTransportByRawValidCAOnly(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	transport, err := GetCustomizedHttpTransportByRaw(false, nil, nil, certs.caPEM)
	assert.NoError(t, err)
	assert.NotNil(t, transport)
	assert.NotNil(t, transport.TLSClientConfig)
	assert.NotNil(t, transport.TLSClientConfig.RootCAs)
	assert.Empty(t, transport.TLSClientConfig.Certificates)
}

func TestGetCustomizedHttpTransportByRawValidClientCert(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	transport, err := GetCustomizedHttpTransportByRaw(false, certs.certPEM, certs.keyPEM, certs.caPEM)
	assert.NoError(t, err)
	assert.NotNil(t, transport)
	assert.NotNil(t, transport.TLSClientConfig)
	assert.NotNil(t, transport.TLSClientConfig.RootCAs)
	assert.Len(t, transport.TLSClientConfig.Certificates, 1)
}

func TestGetCustomizedHttpTransportByRawMissingCA(t *testing.T) {
	_, err := GetCustomizedHttpTransportByRaw(false, nil, nil, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "required tls verification but no ca is provided")

	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	_, err = GetCustomizedHttpTransportByRaw(false, certs.certPEM, certs.keyPEM, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "required tls verification but no ca is provided")
}

func TestGetCustomizedHttpTransportByRawMissingCertOrKey(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	_, err = GetCustomizedHttpTransportByRaw(false, certs.certPEM, nil, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "both cert and key must be provided")

	_, err = GetCustomizedHttpTransportByRaw(false, nil, certs.keyPEM, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "both cert and key must be provided")
}

func TestGetCustomizedHttpTransportByRawInvalidCA(t *testing.T) {
	invalidCA := []byte("invalid ca certificate")
	_, err := GetCustomizedHttpTransportByRaw(false, nil, nil, invalidCA)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ca is illegal, failed to load ca")
}

func TestGetCustomizedHttpTransportByRawInvalidClientCert(t *testing.T) {
	certs, err := generateTestCertificate()
	assert.NoError(t, err)

	invalidCert := []byte("invalid certificate")
	invalidKey := []byte("invalid key")

	_, err = GetCustomizedHttpTransportByRaw(false, invalidCert, invalidKey, certs.caPEM)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load client certificate/key")
}
