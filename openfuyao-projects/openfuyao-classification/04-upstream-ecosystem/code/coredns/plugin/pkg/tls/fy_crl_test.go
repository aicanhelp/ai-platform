// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package tls

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"
)

var (
	caPEMBytes = []byte(`-----BEGIN CERTIFICATE-----
MIIDwDCCAqigAwIBAgIUWIPzxB4UcBEbXPDAFk7hw45Az3wwDQYJKoZIhvcNAQEL
BQAwdzELMAkGA1UEBhMCQ04xEDAOBgNVBAgTB0JlaWppbmcxEDAOBgNVBAcTB0Jl
aWppbmcxEzARBgNVBAoTCkt1YmVybmV0ZXMxGjAYBgNVBAsTEUt1YmVybmV0ZXMt
bWFudWFsMRMwEQYDVQQDEwprdWJlcm5ldGVzMCAXDTIzMDMyODA4MDAwMFoYDzIx
MjMwMzA0MDgwMDAwWjB3MQswCQYDVQQGEwJDTjEQMA4GA1UECBMHQmVpamluZzEQ
MA4GA1UEBxMHQmVpamluZzETMBEGA1UEChMKS3ViZXJuZXRlczEaMBgGA1UECxMR
S3ViZXJuZXRlcy1tYW51YWwxEzARBgNVBAMTCmt1YmVybmV0ZXMwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQDAIExRcZ3eJaxfL4lvVdgJbc6halNeEF2m
7TXkBTm53LqSgEE2P7RIfBD4mTlp43HUl6+Pahn/4hm3gqCRKnu+SHJex4FmJ2xE
Levc5xXsf8NuNq+3Y1RcmXYo4UhjdlbhC351K7acu2zw+eQxHxJ1HDlCyP3ZeoQ4
ADvx1MAY/FAWLWOgyVg8sxAdpILEWPGxUdSbEKM25ofLFcFjp8Pb8ugOw5PuoylI
++//VlTPpt0/z1QlDNie7RQ9c9OCY2zV7o+KrWVC/BwNtWZPV9CHAIsjuN/c0d1F
UlxdG06Xl/kchEs3h1/rtLM/piIHgHqG5mpxnNGsrOitkkqDLZPdAgMBAAGjQjBA
MA4GA1UdDwEB/wQEAwIBBjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBTSlFif
q+LVB8d/Gt4dPEDhfEvMOzANBgkqhkiG9w0BAQsFAAOCAQEAUp0DXKY3DvwKsRpr
ExzCb4StdhcIbXJ8iPwmA7VMRqxSDP0p4PMUWeHkD74OVavf+XawyyOFTlMrFl7/
ui8dgFb6YqSiRw1XeueLoUvWJsixFLKJm2kf1qOegadpyMAc+mTuyp+ZFAxiI8DX
efozLEUj+mldAs6NZdNWA1oG3GQLw0E9Of1kWatsYbfeph57gCqmfacCgZ/wEV/m
gL4a01274UITOeWICV/n7VsOvyuVwcs65bbO4gCl/0C1CUMQ0e1uAg4LhvQvDdSQ
m0ycc1t0LW1ef9uU+xoUI//Y1dWmx2cwQdNyOGlu/d4H3A1tZXzUm8uPZHT0h1Li
jK4c6g==
-----END CERTIFICATE-----`)

	revokedCertPEMBytes = []byte(`-----BEGIN CERTIFICATE-----
MIID/DCCAuSgAwIBAgIUAZHOLWE4ymaPF4HY09KEtxs/8mgwDQYJKoZIhvcNAQEL
BQAwdzELMAkGA1UEBhMCQ04xEDAOBgNVBAgTB0JlaWppbmcxEDAOBgNVBAcTB0Jl
aWppbmcxEzARBgNVBAoTCkt1YmVybmV0ZXMxGjAYBgNVBAsTEUt1YmVybmV0ZXMt
bWFudWFsMRMwEQYDVQQDEwprdWJlcm5ldGVzMCAXDTIzMDMyODA4MDAwMFoYDzIx
MjMwMzA0MDgwMDAwWjB2MQswCQYDVQQGEwJDTjEQMA4GA1UECBMHQmVpamluZzEQ
MA4GA1UEBxMHQmVpamluZzEXMBUGA1UEChMOc3lzdGVtOm1hc3RlcnMxGjAYBgNV
BAsTEUt1YmVybmV0ZXMtbWFudWFsMQ4wDAYDVQQDEwVhZG1pbjCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAMPZ1a9WyVxglvVWW4VVkcV9kzBBtIUiYZBy
wwmufnrpKeHAiiKYJ0oRawNiXEX1IzddjBUoeXq/e2ObV8JsLcdq64FmVkzY1rOg
FQJGS2C1cF4OxYC3MpmidVlKkwUAgZLja7aeKmDzI5x9/xPP+x7XPL+cbZPlBdab
JomfeiiKH3ny0inXDwxNhyAuBslm0OBY/S+fe6HXl1tDSHpVhZ6eHbgqU4kzFFPw
rLGJlDfGZKv2BZdC46Q8wDpf1qcA0cofXjdMutEr2AdIjEVfBELyc8CQDQPZc3lS
BIMGrnu0QtMpZYFVVXxkKqNxJa+oQ+N+/+prWF7UhgbMGt5TpjECAwEAAaN/MH0w
DgYDVR0PAQH/BAQDAgWgMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAM
BgNVHRMBAf8EAjAAMB0GA1UdDgQWBBSW4T/834vEDyxYbeFfNjny5q3dzjAfBgNV
HSMEGDAWgBTSlFifq+LVB8d/Gt4dPEDhfEvMOzANBgkqhkiG9w0BAQsFAAOCAQEA
dyKTuO2H3Y9mU78XuBrHTofYmtdbbeP+v5bgIydVXAbc6pKXGIdAQjMy4B7b70c+
dLwm0z7kPqvzAigO/qQVdWPglQX+X35kog/8naceKVW+xP8Deteh1jQKuY5sH80+
zzvmoyqLu2sgskr7OJBCx1uTS6B+M+xHtyeNdfCYxkI6cvJG/fnv0XADN/sRwG6P
UZfWV0NAYY9T6GN6mQSJRWUtMZ0jZjm7w6a+pep+eGxTlj2SK7AZkbPE5Kvf3e4p
XJmeApekrt2Yy7WnYjoov9zyUnHwa+jPDgvluyib1ETJBGuaqSSmntHn7E/WeuUy
zNscFyVDcHTtKqTl2DMvMg==
-----END CERTIFICATE-----`)

	crlPEMBytes = []byte(`-----BEGIN X509 CRL-----
MIICJDCCAQwCAQEwDQYJKoZIhvcNAQELBQAwdzELMAkGA1UEBhMCQ04xEDAOBgNV
BAgTB0JlaWppbmcxEDAOBgNVBAcTB0JlaWppbmcxEzARBgNVBAoTCkt1YmVybmV0
ZXMxGjAYBgNVBAsTEUt1YmVybmV0ZXMtbWFudWFsMRMwEQYDVQQDEwprdWJlcm5l
dGVzFw0yMzAzMzAxNDQxNTNaGA8yMTIzMDEwMjE1MDQwNVowJzAlAhQBkc4tYTjK
Zo8XgdjT0oS3Gz/yaBcNMjMwMzMwMTQ0MTUzWqA2MDQwHwYDVR0jBBgwFoAU0pRY
n6vi1QfHfxreHTxA4XxLzDswEQYDVR0UBAoCCBCyC3eWO4+rMA0GCSqGSIb3DQEB
CwUAA4IBAQALm/VsV1iAsje3QJ91pg5+JntU/fVgeiUJsTkEZxZaArweZxf9FSLL
crQ6upQ7BnAsWzsUTXrnymxVaRKdZ78Z7d8zKX28LnF201lo49+VT8ou7aJZzxgm
RfrdaFM60wqcJCAeQv5Y1Oe/mVRmPEu94eF8/E3QgugQKb8Hu/KoVZx+z4nNjmA6
zvncwy4MCDwQMuv3xdsrtXPySJajqoyll82ucKoGCapNuAQWBITD7Pr6RgpHoz0A
QW/kcqb5EBNhs8NYhVPnwlZO+dAvu9oDxEq1GOMZNt2YOI2ewZYD/UV5+XgVnZlT
wUETiNYeJp+lavt7Ahbv/3D02DU/FkIS
-----END X509 CRL-----`)

	emptyCRLPEMBytes = []byte(`-----BEGIN X509 CRL-----
MIIB+jCB4wIBATANBgkqhkiG9w0BAQsFADB3MQswCQYDVQQGEwJDTjEQMA4GA1UE
CBMHQmVpamluZzEQMA4GA1UEBxMHQmVpamluZzETMBEGA1UEChMKS3ViZXJuZXRl
czEaMBgGA1UECxMRS3ViZXJuZXRlcy1tYW51YWwxEzARBgNVBAMTCmt1YmVybmV0
ZXMXDTIzMDMzMDE0NDMxNVoYDzIxMjMwMTAyMTUwNDA1WqA2MDQwHwYDVR0jBBgw
FoAU0pRYn6vi1QfHfxreHTxA4XxLzDswEQYDVR0UBAoCCD5hrAlSrx68MA0GCSqG
SIb3DQEBCwUAA4IBAQC+hKSnOylPDGs7b8iodv286zXHM9hAzZnDWZmgvdalU72R
bzEU9NyOYXViVJMJw6SBeSLGPOHfa8g159zrGRmZfO2sjfel4WfrA0mv5ucIWETX
Ozv8pAr4Y56Q2IO4T++NMsjelc8409q020gLr8n3i3vD/Hy/WugCE3kTNabHQBGG
N/Z4Vl2tLxpmLHcitFeImbtyXMgFTukwjYQh8FYg+ONekgmfGvSkTWo8SMd7gfAE
oQwTMOBlI/8re0XLcNshzEJwdFDVALKG4w4VWoEQlora3u17BhBlfSs71hWjIo/J
YRUXwV4I+BrltmGnQCJVyXmsv1MS1YIDPIkCEI/j
-----END X509 CRL-----`)

	expiredCRLPEMBytes = []byte(`-----BEGIN X509 CRL-----
MIICIjCCAQoCAQEwDQYJKoZIhvcNAQELBQAwdzELMAkGA1UEBhMCQ04xEDAOBgNV
BAgTB0JlaWppbmcxEDAOBgNVBAcTB0JlaWppbmcxEzARBgNVBAoTCkt1YmVybmV0
ZXMxGjAYBgNVBAsTEUt1YmVybmV0ZXMtbWFudWFsMRMwEQYDVQQDEwprdWJlcm5l
dGVzFw0yMzAxMDIxNTA0MDVaFw0yMzAyMDIxNTA0MDVaMCcwJQIUAZHOLWE4ymaP
F4HY09KEtxs/8mgXDTIzMDMzMDE0NTQ0OFqgNjA0MB8GA1UdIwQYMBaAFNKUWJ+r
4tUHx38a3h08QOF8S8w7MBEGA1UdFAQKAggaJ+W/ctYdbDANBgkqhkiG9w0BAQsF
AAOCAQEAdfWmZzQZyLUX1cqw37Z/i2GlpMdfW8YWaIHobLT1/RZNNwNb7VTCoCiK
4sZ2xC8I3L+CCG9CyHLpJ6hJFZOsQpSVyQDhiD/6glxutsXevG6CC3IOlVTLEWlY
wBiV7aNBloXR8GMNtlNWEYGjIJxiwM7ku6sA3YZE3+SCc/2IJYEh+rRTOfYywkJ8
xCKicDEt0Vhicxa+FoE1mEI7mWmJWQYnhg2hteVLbuaAN5xcpGMCbjlEynT23Of+
zgVx6t13ybw/ldTGgEu0TUir067sH9O0HU5CVVhlQTYoJ/ROtFXIJml+sQ8eW8Ri
7cYQvleGmTHSXTxmfJ5JhG5Icy01uw==
-----END X509 CRL-----`)

	foreverCertPEMBytes = []byte(`-----BEGIN CERTIFICATE-----
MIIDTTCCAjWgAwIBAgIIWdWhbcQeoTIwDQYJKoZIhvcNAQELBQAwdzELMAkGA1UE
BhMCQ04xEDAOBgNVBAgTB0JlaWppbmcxEDAOBgNVBAcTB0JlaWppbmcxEzARBgNV
BAoTCkt1YmVybmV0ZXMxGjAYBgNVBAsTEUt1YmVybmV0ZXMtbWFudWFsMRMwEQYD
VQQDEwprdWJlcm5ldGVzMCAXDTIzMDMzMDE0NTAxN1oYDzIxMjMwMTAyMTUwNDA1
WjAAMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsN0PN0pgWmOvlrDc
pp4kZ9MUpW66Ef8ZMZBd5pcjdfSV1l9wYPZwyqf+VgMe5uRod3SLj0WMAUB+LErA
vmvuClsc5rvPStQxmcLpOC/bo69UKUr6rcibjnbxxNVcaW3bqJxJ+jn7K4DPxMiE
zJn+PIDYz7ead3aB6Z36h65uQBYZa108GPgRl3R4otg7Ej2alx7YNi1sFQuF2kFX
Yl9M6mDnAAHmNBmh6cdR8HrFUJcUu1rxDhoWFIM3VZxaCXw1CWhOERFcU5V8DyPj
yIOEGun6e49zzlKxj0H328cmMOThmDiOJ4lHZV+qvqEUf/HNuLCTepxh2xjJtiGp
EAw6YQIDAQABo1IwUDAOBgNVHQ8BAf8EBAMCAqQwHQYDVR0lBBYwFAYIKwYBBQUH
AwIGCCsGAQUFBwMBMB8GA1UdIwQYMBaAFNKUWJ+r4tUHx38a3h08QOF8S8w7MA0G
CSqGSIb3DQEBCwUAA4IBAQCc+hs+63sefq/bmviBqL3L8mT0PJbOdlvd7cmyfIfw
nm7pZ/XRoVDx+c0nk2Yzej/ALjkoeJxPKRwGfuE51f0tGNSoVZZ+BrG8gynqsoEU
DydAcvCHqTWiKIP2ZSy2a5Az8AAdpKR52rVk5dyUfv3ZvA5CFlLX2VeBJOcZKKpT
GTtsk7e1hNkowILGozKJ3Ed0UlzP6iHfuGvCenF/Fq+dSh4e6XAaJ89ld753IEcF
5uQZ++G3gpN90/flPrN8XVmPsNZy4241mVQkKCRCXRVbOvq0Afzed6wzOw+osYqm
til1NyCU60iG77XrV7CalYqEKYd8HDZZNhq6wqr9lFLB
-----END CERTIFICATE-----`)
)

const (
	safeMode fs.FileMode = 0600
)

func writeFile(filename string, data []byte) (string, error) {
	tempName := filepath.Join(os.TempDir(), filename)
	if err := os.WriteFile(tempName, data, safeMode); err != nil {
		return "", err
	}
	return tempName, nil
}

func der(pemBytes []byte) []byte {
	block, _ := pem.Decode(pemBytes)
	return block.Bytes
}

func TestGetVerifyCallBackFromFile(t *testing.T) {
	type args struct {
		fileData []byte
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		preset  func()
		reset   func()
	}{
		{
			name:    "return error when crl file is empty",
			args:    args{},
			wantErr: true,
		},
		{
			name:    "return error when crl file is expired",
			args:    args{expiredCRLPEMBytes},
			wantErr: false,
		},
		{
			name:    "return error when crl file is invalid",
			args:    args{[]byte("invalid data")},
			wantErr: true,
		},
		{
			name:    "return nil when crl file is existed and valid PEM data",
			args:    args{crlPEMBytes},
			wantErr: false,
		},
		{
			name:    "return nil when crl file is existed and valid DER data",
			args:    args{der(crlPEMBytes)},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename, err := writeFile("kubernetes-test.crl", tt.args.fileData)
			if err != nil {
				t.Fatalf("Unable to create the test file %q: %v", filename, err)
			}
			defer func() {
				if err := os.Remove(filename); err != nil {
					t.Errorf("Unable to clean up test file %q: %v", filename, err)
				}
			}()
			if _, err := getVerifyCallBackFromFile(filename); (err != nil) != tt.wantErr {
				t.Errorf("getVerifyCallBackFromFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestGetVerifyCallBackFromFile_Update(t *testing.T) {
	filename, err := writeFile("kubernetes-test-updating.crl", emptyCRLPEMBytes)
	if err != nil {
		t.Fatalf("Unable to create the test file %q: %v", filename, err)
	}
	defer func() {
		if err := os.Remove(filename); err != nil {
			t.Errorf("Unable to clean up test file %q: %v", filename, err)
		}
	}()
	verify, err := getVerifyCallBackFromFile(filename)
	if err != nil {
		t.Error(err)
		return
	}
	time.Sleep(time.Second)
	type args struct {
		newCRLBytes []byte
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		preset  func()
		reset   func()
	}{
		{
			name:    "verify succeed when updated crl file is expired",
			args:    args{expiredCRLPEMBytes},
			wantErr: true,
		},
		{
			name:    "verify succeed when updated crl file is not revoke cert",
			args:    args{emptyCRLPEMBytes},
			wantErr: false,
		},
		{
			name:    "verify failed when updated crl file has revoked cert",
			args:    args{crlPEMBytes},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(filename, tt.args.newCRLBytes, safeMode); err != nil {
				t.Errorf("Unable to write the test file %q: %v", filename, err)
				return
			}
			ca, _ := x509.ParseCertificate(der(caPEMBytes))
			err := verify([][]byte{der(revokedCertPEMBytes)}, [][]*x509.Certificate{{ca}})
			if (err != nil) != tt.wantErr {
				t.Errorf("verifyPeerCertRevoke() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			time.Sleep(time.Second)
		})
	}
}

func TestVerifyPeerCertRevoke(t *testing.T) {
	crl, _ := x509.ParseRevocationList(der(crlPEMBytes))
	ca, _ := x509.ParseCertificate(der(caPEMBytes))
	type args struct {
		rawCerts       [][]byte
		verifiedChains [][]*x509.Certificate
		crl            *x509.RevocationList
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "return error when cert is revoked",
			args: args{
				rawCerts:       [][]byte{der(revokedCertPEMBytes)},
				verifiedChains: [][]*x509.Certificate{{ca}},
				crl:            crl,
			},
			wantErr: true,
		},
		{
			name: "return nil when cert is not revoked",
			args: args{
				rawCerts:       [][]byte{der(foreverCertPEMBytes)},
				verifiedChains: [][]*x509.Certificate{{ca}},
				crl:            crl,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := verifyPeerCertRevoke(tt.args.rawCerts, tt.args.verifiedChains, tt.args.crl); (err != nil) != tt.wantErr {
				t.Errorf("verifyPeerCertRevoke() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSetVerifyCRLCallback(t *testing.T) {
	filename, err := writeFile("kubernetes-test-updating.crl", emptyCRLPEMBytes)
	if err != nil {
		t.Fatalf("Unable to create the test file %q: %v", filename, err)
	}
	defer func() {
		if err := os.Remove(filename); err != nil {
			t.Errorf("Unable to clean up test file %q: %v", filename, err)
		}
	}()
	type args struct {
		filename string
		config   *tls.Config
		wantErr  bool
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "don't update tls.Config when file name is empty",
			args: args{
				filename: "",
				config:   new(tls.Config),
				wantErr:  true,
			},
		},
		{
			name: "update VerifyPeerCertificate of tls.Config success when crl file valid",
			args: args{
				filename: filename,
				config:   new(tls.Config),
				wantErr:  false,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetVerifyCRLCallback(tt.args.filename, tt.args.config)
			if (tt.args.config != nil && tt.args.config.VerifyPeerCertificate != nil) == tt.args.wantErr {
				t.Errorf("SetVerifyCRLCallback() test failed, wantErr %v", tt.args.wantErr)
			}
		})
	}
}
