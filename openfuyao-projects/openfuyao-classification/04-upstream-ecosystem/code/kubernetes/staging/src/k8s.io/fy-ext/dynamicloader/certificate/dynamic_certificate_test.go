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

package certificate

import (
	"context"
	"testing"
	"time"

	"k8s.io/fy-ext/dynamicloader/contentprovider"
)

var (
	caData1 = []byte(`-----BEGIN CERTIFICATE-----
MIIDfTCCAmWgAwIBAgIBZDANBgkqhkiG9w0BAQsFADBYMQswCQYDVQQGEwJVUzEJ
MAcGA1UECBMAMRYwFAYDVQQHEw1TYW4gRnJhbmNpc2NvMQkwBwYDVQQJEwAxCTAH
BgNVBBETADEQMA4GA1UEChMHVGVzdCBDQTAeFw0yNTA5MjYwODUzNDVaFw0yNjA5
MjYwODUzNDVaMFgxCzAJBgNVBAYTAlVTMQkwBwYDVQQIEwAxFjAUBgNVBAcTDVNh
biBGcmFuY2lzY28xCTAHBgNVBAkTADEJMAcGA1UEERMAMRAwDgYDVQQKEwdUZXN0
IENBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtziWLxLVEi0v+j5h
Vt1J9vTvr2l8JSYve/RrE7BwMWYhCmS87snfZoSLPv9NdYaVU1pYMuES5SylXuqI
CBjdkyNs/npa5/RNxlDLsSUV49ZzzXaw68qQVjrn1eceuiGmz7+0j40ei01ORUHi
mbHDi9OQw2pZeYeywapKDgT0o9qFK8jL2/vF2WkOkGcEHPrUUIpOgKiErSbW1Qjc
wmFeiEAoWCYD2qn2eiJD7MTiVO7dEi/+BTKRH/u0FBwxyMsKZ7Bj6+7j+Ce/38Az
W7weET5/MBfE1nkhXFhZTHP35oTQvtf/B63f+G0x0NWgcntRmNAlVpwPO9XnALPQ
qBY0UQIDAQABo1IwUDAOBgNVHQ8BAf8EBAMCAoQwHQYDVR0lBBYwFAYIKwYBBQUH
AwIGCCsGAQUFBwMBMA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0OBAcEBQECAwQHMA0G
CSqGSIb3DQEBCwUAA4IBAQAugHy2wfguolTL/Upb6XONbbwHeAMUMFUdVV92TqjC
XiXNZOJi1a74ZSeyQ8kx5xEuFLTkqVLcOXthAT9DUw1PChV80r7mjKWFXnyRkEN1
9cSns6QNLGaPpm+93JTODwT/UXXjKeN1SB3EMmM19R4KSuUeMk6GaJTXft/ATdr6
cLz+j3t94TlchCJnshSjXsdWmkkEluYbwzHWeHCdaC6mBCmodssyvIklKNg05oHZ
X7gplhGMaotTdErVy5x5/owUDjBlVDLEfuz1kK6DvUaXV5T7uhgSWcQyxia/mv+b
czoDbds0sHTVgFfglzcqFfDxAYS78qdpPCjtWXoUeOzU
-----END CERTIFICATE-----
`)
	caData2 = []byte(`-----BEGIN CERTIFICATE-----
MIIDfTCCAmWgAwIBAgIBZDANBgkqhkiG9w0BAQsFADBYMQswCQYDVQQGEwJVUzEJ
MAcGA1UECBMAMRYwFAYDVQQHEw1TYW4gRnJhbmNpc2NvMQkwBwYDVQQJEwAxCTAH
BgNVBBETADEQMA4GA1UEChMHVGVzdCBDQTAeFw0yNTA5MjYwODUzNDVaFw0yNjA5
MjYwODUzNDVaMFgxCzAJBgNVBAYTAlVTMQkwBwYDVQQIEwAxFjAUBgNVBAcTDVNh
biBGcmFuY2lzY28xCTAHBgNVBAkTADEJMAcGA1UEERMAMRAwDgYDVQQKEwdUZXN0
IENBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtziWLxLVEi0v+j5h
Vt1J9vTvr2l8JSYve/RrE7BwMWYhCmS87snfZoSLPv9NdYaVU1pYMuES5SylXuqI
CBjdkyNs/npa5/RNxlDLsSUV49ZzzXaw68qQVjrn1eceuiGmz7+0j40ei01ORUHi
mbHDi9OQw2pZeYeywapKDgT0o9qFK8jL2/vF2WkOkGcEHPrUUIpOgKiErSbW1Qjc
wmFeiEAoWCYD2qn2eiJD7MTiVO7dEi/+BTKRH/u0FBwxyMsKZ7Bj6+7j+Ce/38Az
W7weET5/MBfE1nkhXFhZTHP35oTQvtf/B63f+G0x0NWgcntRmNAlVpwPO9XnALPQ
qBY0UQIDAQABo1IwUDAOBgNVHQ8BAf8EBAMCAoQwHQYDVR0lBBYwFAYIKwYBBQUH
AwIGCCsGAQUFBwMBMA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0OBAcEBQECAwQHMA0G
CSqGSIb3DQEBCwUAA4IBAQAugHy2wfguolTL/Upb6XONbbwHeAMUMFUdVV92TqjC
XiXNZOJi1a74ZSeyQ8kx5xEuFLTkqVLcOXthAT9DUw1PChV80r7mjKWFXnyRkEN1
9cSns6QNLGaPpm+93JTODwT/UXXjKeN1SB3EMmM19R4KSuUeMk6GaJTXft/ATdr6
cLz+j3t94TlchCJnshSjXsdWmkkEluYbwzHWeHCdaC6mBCmodssyvIklKNg05oHZ
X7gplhGMaotTdErVy5x5/owUDjBlVDLEfuz1kK6DvUaXV5T7uhgSWcQyxia/mv+b
czoDbds0sHTVgFfglzcqFfDxAYS78qdpPCjtWXoUeOzU
-----END CERTIFICATE-----
`)
	certData1 = []byte(`-----BEGIN CERTIFICATE-----
MIIDbjCCAlagAwIBAgIBATANBgkqhkiG9w0BAQsFADBZMQswCQYDVQQGEwJVUzEJ
MAcGA1UECBMAMRYwFAYDVQQHEw1TYW4gRnJhbmNpc2NvMQkwBwYDVQQJEwAxCTAH
BgNVBBETADERMA8GA1UEChMIVGVzdCBPcmcwHhcNMjUwOTI2MDg1MzQ0WhcNMjYw
OTI2MDg1MzQ0WjBZMQswCQYDVQQGEwJVUzEJMAcGA1UECBMAMRYwFAYDVQQHEw1T
YW4gRnJhbmNpc2NvMQkwBwYDVQQJEwAxCTAHBgNVBBETADERMA8GA1UEChMIVGVz
dCBPcmcwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCk+zVHCsrAlXqg
RYuksR+2wq3M7eL8IZPnATrw6PNBSRAEMj9PO9Fofery3jXFUgsVdO/3Anvbv6rR
f6c3REu0R0XESgqEEjL1wpsStUWxfstmJjti7quiGYeKXUyG7nkqpxRfy64mO4lH
oWSrfp329lf8/Ka1jaeLGpunZiCgInhsVYVgZ63JvkmDAggDwngXswJl91sW6isR
PzXb0vD/wp+7XvENYeoRdjzcsI79mhSjXDfkFRiUTOsy9t9pxH0iRukzKTNcRbdv
l4CkBnUAYLZ9wrYhKDjVjSDZrN0pg5la6ph0YVqP9tXIw3TPBTSDgVC6Mnj/Tf+Z
umOL8og/AgMBAAGjQTA/MA4GA1UdDwEB/wQEAwIHgDAdBgNVHSUEFjAUBggrBgEF
BQcDAgYIKwYBBQUHAwEwDgYDVR0OBAcEBQECAwQFMA0GCSqGSIb3DQEBCwUAA4IB
AQAdZFS74gYu+hfPkHMe3QcKKuTrhqmXiAG0UFzSLGMDHwoRNM3+gqg+4PfXAHV4
t7BrKEi58Gwj4tx8lZVa5zR5LNF2it14Lm9ac0mn/drWdo/S5Zkj56lIhoGGEc1p
bUINvKQCwkRJDqm4KSo0H/SL1K/iOGS/knrlcB4q9M49CThjW3Klwdu7KFvJhwMT
RD7wju7RjSkXmWQ8StJGvsOy/f0WXmCacY8dul0gGgfJaNVYrIMHQTAS3ZUrdr1+
WMOym7PNe3uISNl9ijtVql9NxyWr8Rjfk1aOxS1NKIs9lUTQUMdBQSVgYoRu0kSS
1U9OMMLRkYSdS5OkGzl7KIqc
-----END CERTIFICATE-----
`)
	keyData1 = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEApPs1RwrKwJV6oEWLpLEftsKtzO3i/CGT5wE68OjzQUkQBDI/
TzvRaH3q8t41xVILFXTv9wJ727+q0X+nN0RLtEdFxEoKhBIy9cKbErVFsX7LZiY7
Yu6rohmHil1Mhu55KqcUX8uuJjuJR6Fkq36d9vZX/PymtY2nixqbp2YgoCJ4bFWF
YGetyb5JgwIIA8J4F7MCZfdbFuorET8129Lw/8Kfu17xDWHqEXY83LCO/ZoUo1w3
5BUYlEzrMvbfacR9IkbpMykzXEW3b5eApAZ1AGC2fcK2ISg41Y0g2azdKYOZWuqY
dGFaj/bVyMN0zwU0g4FQujJ4/03/mbpji/KIPwIDAQABAoIBACJEjW3qp291/OkC
WP2FztUKvcfll3KQVOS0BDaa5D/ouZtthmniPl5KXT1B+4A3SNcVmWxAh1J/Dq+C
KMzZcjIJZcveJy4Ei7IxUD2ECmxSLlRS1532fg7m55jNRUJ3kRJTbu9w4oTNc/e9
ZagbXhs4UtFY9xWZzTWe+rq8A0XPXZCx1NgK5+9pFAM9SXshodK736/kdXmEC3mT
pNKsf22ltmJScpHFJS49ICabgQgn0qR3ac1MVfwa5wJFqdQy8HruagBOowJvSJW3
eyfAi34y1OJahvQn+1IXCwQBe6a21qTpLLq1vT4mu+5OuanHG7JEnWuzxcWyV/kL
mxt+IDkCgYEA2xA6s/6+zjM1GP38n69pmJYLxa/wrnd+Ch4fAA2EZsYc9BMv3zTJ
G2elsdtaEvJEd14CezIrtF7Jgr5dAs+QVpYSRXA3MiTeFvu4Gsv6ikcE3G7WOk12
nolPoVJXlxE/cIZfRGseKKznOmwBLnkaoTku3siQl2as01sjIz0WQCkCgYEAwMyM
mrbPREod0RTTq5dvsf90NTidJ6wnYfXqk5/PzYB1nE939JHwFB3yYwwGuYJtfh9Q
WQSU0udY5126uUcExjcSXdhaiIaof6BWudGe1ghZK5ur3nslUWdJQm0eAjqPQ9AB
hdso2SRw+YPYQBntvqQbSs+/z9WUFJttJoqo8icCgYAIKMWW+LEQqxWaNbOYGwd6
2HPdzBzR0r+ODusmrFo3Hb/sU16gGK446BZi1oG3XPCx5X2AKzE3iaRFy2NZfHOq
1fIeRfy4UGk5P+zuipRtXBHRSAuQplv5sbGWl6ONi+XqAgdAXCEGcNI3+dfdbENS
p/eX8q54BSoAwU9Au8C9aQKBgCV2GUPejXtZY13CzDYlIKDXX/8PVwnkW2qfW8r/
XGMzuKZ8knRE/v8JqpUwOwZEn33IG9aQG1K3fQDHp4ZklDrnbHwrk7u/XnId93v3
jQNNb/YXM1fxWZjtlsB/bztDoZ6thf46TT+T4xe4VIZwA0/tlxf8MeryNumiIZ1B
h/41AoGAR3d7egruAC+ZJyRm45dJpIzENN3HD+K+kdvnbJJ65v4mwponB+vSqhzf
F26N84JXhssHp/Eef0WKehLLhsiLUBJtczsUPzvjxWRBx4uH2BaW0geDj14TmygT
cLzPlwPht5IHoarpHWFZQAQdzZSbGTkyUSjEvMAoC1XT/OijLow=
-----END RSA PRIVATE KEY-----
`)
	certData2 = []byte(`-----BEGIN CERTIFICATE-----
MIIDcjCCAlqgAwIBAgIBAjANBgkqhkiG9w0BAQsFADBbMQswCQYDVQQGEwJVUzEJ
MAcGA1UECBMAMRYwFAYDVQQHEw1TYW4gRnJhbmNpc2NvMQkwBwYDVQQJEwAxCTAH
BgNVBBETADETMBEGA1UEChMKVGVzdCBPcmcgMjAeFw0yNTA5MjYwODUzNDVaFw0y
NjA5MjYwODUzNDVaMFsxCzAJBgNVBAYTAlVTMQkwBwYDVQQIEwAxFjAUBgNVBAcT
DVNhbiBGcmFuY2lzY28xCTAHBgNVBAkTADEJMAcGA1UEERMAMRMwEQYDVQQKEwpU
ZXN0IE9yZyAyMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArYX3YzP0
/ScsY+9EYi66kO09bufAnCxQ4iz0LOgWB7pi56C+S045QGQb0DvaHvBdAkQ0miTs
zGP7hN1uTuxO6C6YrWbLAI9g+oemOfKRGF7xaoDc9Z4IP1jrR6qMlSUr9PtbsX7J
+B/HFfCZiuEkcwtuT7jkYd7/iQyGzjfiVqk7ZKAn7K1zUVUbRVBC3GaQ0E3rrXei
ZKqpoCtaa+eAjT/OjeB5dUV0ar+sEoXXDsqr2YVrdXmtg0PyXZCijAcIi3FUpyNJ
Yv750GNsQASKaObj9/bwVMq/CZjfwEpV/t5pKMb1jTqsGgR42+p5uKZdVO/Bbz01
d0glGwI4lmIyIQIDAQABo0EwPzAOBgNVHQ8BAf8EBAMCB4AwHQYDVR0lBBYwFAYI
KwYBBQUHAwIGCCsGAQUFBwMBMA4GA1UdDgQHBAUBAgMEBjANBgkqhkiG9w0BAQsF
AAOCAQEAVG9DWO/OBpk3sx/Ef5wIByyk0650+u+ORJ2Suoq0yELHSX1GgaBc1cnx
k64kauNnV5Huax8rnNTxaJ5Sy8Ur0drXOl4Qrf8NUyXNckM6gQEWQ8bc5zBsS8pU
jEuU1JjeHpESE3xdJtCZVKHButqAlW4ouF1tuuKClL4l1f/W09RI/oeS30SSEZWo
V9EyLRYedRQ2fsBirZH+c3JgzUVJK7B/szMLdcUl72JC1xY/QS3jW753/a37XzUN
s4q+ld1etMttzHT12w7BA9/6ZVzZ3kUZJLx5yd9E2pNBcOBY1hLWgbAaOEDYarCY
9QMZtKeZhwA79No67+DNuu/zuLnivg==
-----END CERTIFICATE-----
`)
	keyData2 = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEArYX3YzP0/ScsY+9EYi66kO09bufAnCxQ4iz0LOgWB7pi56C+
S045QGQb0DvaHvBdAkQ0miTszGP7hN1uTuxO6C6YrWbLAI9g+oemOfKRGF7xaoDc
9Z4IP1jrR6qMlSUr9PtbsX7J+B/HFfCZiuEkcwtuT7jkYd7/iQyGzjfiVqk7ZKAn
7K1zUVUbRVBC3GaQ0E3rrXeiZKqpoCtaa+eAjT/OjeB5dUV0ar+sEoXXDsqr2YVr
dXmtg0PyXZCijAcIi3FUpyNJYv750GNsQASKaObj9/bwVMq/CZjfwEpV/t5pKMb1
jTqsGgR42+p5uKZdVO/Bbz01d0glGwI4lmIyIQIDAQABAoIBAAe9tJCaDnGGCtvr
yPrcUiRfAQuZOSV+cJ6hGhowFwjXOXb928oTk52aGbI2HqqcrHmKaL0BArNc49Pv
O5mNFI6/wr3m77ghrVY1MPw1ebWYSWAn9DOHjkn31zSGQ5PvvutElLhGtg85/lwD
NMZWTur29T/cjrZdsrv+xrtyZh2EMlOkMQuS3kRFMx2ljfijRGZYY5vtfPj4oySw
Wxcv90N9dPHqpHFKEGaP3B/9btuYf/82kc3hLwdKNRqUIgxnOCzrWfbObdtczpVc
0meOCQMVlJrbznvOCvHxLWZggqhmpAOYNGhahBfCI2UDIStUmqGK7ZW9P5ux17h6
bG4CIzECgYEAwGc50ec/NptU1VodVEGhaZ0wShFXDsjW7dMOc350ACtD9Um5GN+0
0uRz9BMZzU3+ywG8SaLE3t6Zyb/kiAEoO7KX4ZYdDnd3HVOHmRdc6QNdNrue6Ici
bgB8d2btsBTH2qLAFcNSVQu9atyBzf3Po+8Yea3C5wu95iVI5WHySh0CgYEA5uEo
3cMCJGLOY0Pl8NXLdEmbuTlDFaBMNThcZz6shdR65CZF7xTkoR6ZtHM3fEffQ9dq
PkyuIaOX4dsem612dsqW01dFvdenkC2FmO0/7/ys0OShOx59JDwXEMwxGRC0p/Q/
UmMAJZa8dgHfgTjq7NAIcwpI8yC5JHWFhqcuKNUCgYEAoGowQfrMsImnnE6K34Hs
wzKkXUzByfjXgegTR2Xp724jNfJmmf+CMsb4ikoWnB7oTXe/9aD5BVyQ2i3cqyxT
0BoiGZpf1XzzWgcXqiNZeoksgYLChdqNcSuEmsS9ba6VxoslHcAA+BlWry70Jrv0
EBk7zc97DcmzZZw2f9E/Al0CgYEAoDpejBiL4giYRxeV+p/FBL0Yz2DIB7gpaz3k
EtrSDEa7yp4tJpEPYinEbhEPzrZqYcCbwQSiTEImY94Xzd6COkq6hY0pn2iMp2iQ
MONl3f0zSf2BT3AHVSBGuaeb41NOrMi7MUf3kB6eSB9gp2OC1jpBwuS+X8yz8HOG
6T4YsMkCgYEAk4JZHxUoF+hCGWZYApRgGCJiHZnAGw8ViyZ3Ct5ELuxojweWFsmS
I1UuCUgk7DidOJd6U8OdwP0BZamxel+mQvusYBCJRpBQmM6edYRXmxp4xUh5WK59
4wQM8ReMxDXbdQhECRv9+IAuhNTpo5gtX5cLnS9nIyzcdvlSvdkUiBc=
-----END RSA PRIVATE KEY-----
`)
)

type mockDynamicContentProvider struct {
	content  []byte
	listener contentprovider.Listener
}

func (p *mockDynamicContentProvider) CurrentContent() []byte { return p.content }

func (p *mockDynamicContentProvider) AddListener(listener contentprovider.Listener) {
	p.listener = listener
}

func (p *mockDynamicContentProvider) RunOnce() error { return nil }

func (p *mockDynamicContentProvider) Run(ctx context.Context) error { return nil }

func (p *mockDynamicContentProvider) SetContent(content []byte) {
	p.content = content
	p.listener(content)
}

// keyPair/capool would be refreshed when certificates changed
func TestDynamicCertificateContent(t *testing.T) {
	certProvider := &mockDynamicContentProvider{
		content: certData1,
	}
	keyProvider := &mockDynamicContentProvider{
		content: keyData1,
	}
	caProvider := &mockDynamicContentProvider{
		content: caData1,
	}
	certContentProvider := NewDynamicCertificateContent(certProvider, keyProvider, caProvider)
	certContentProvider.RunOnce()

	keyPair := certContentProvider.loadKeyPair()
	if keyPair == nil {
		t.Fatal("expected key pair not nil")
	}
	caPool := certContentProvider.loadCAPool()
	if caPool == nil {
		t.Fatal("expected ca pool not nil")
	}

	ctx, _ := context.WithCancel(context.Background())
	certContentProvider.Run(ctx)

	certProvider.SetContent(certData2)
	time.Sleep(100 * time.Millisecond)
	keyPairNew := certContentProvider.loadKeyPair()
	if keyPairNew != keyPair {
		t.Fatal("expected key pair not nil and equal with old one")
	}

	keyProvider.SetContent(keyData2)
	time.Sleep(100 * time.Millisecond)
	keyPairNew = certContentProvider.loadKeyPair()
	if keyPairNew == nil || keyPairNew == keyPair {
		t.Fatal("expected key pair not nil and not equal with old one")
	}

	caProvider.SetContent(caData2)
	time.Sleep(100 * time.Millisecond)
	caPoolNew := certContentProvider.loadCAPool()
	if caPoolNew == nil || caPoolNew == caPool {
		t.Fatal("expected ca pool not nil and not equal with old one")
	}
}

// keypair/capool would be not refreshed when certificates format not correct
func TestDynamicCertificateContentWhenErrorFormat(t *testing.T) {
	certProvider := &mockDynamicContentProvider{
		content: certData1,
	}
	keyProvider := &mockDynamicContentProvider{
		content: keyData1,
	}
	caProvider := &mockDynamicContentProvider{
		content: caData1,
	}
	certContentProvider := NewDynamicCertificateContent(certProvider, keyProvider, caProvider)
	certContentProvider.RunOnce()

	keyPair := certContentProvider.loadKeyPair()
	if keyPair == nil {
		t.Fatal("expected key pair not nil")
	}
	caPool := certContentProvider.loadCAPool()
	if caPool == nil {
		t.Fatal("expected ca pool not nil")
	}

	ctx, _ := context.WithCancel(context.Background())
	certContentProvider.Run(ctx)

	certProvider.SetContent([]byte("wrong"))
	time.Sleep(100 * time.Millisecond)
	keyPairNew := certContentProvider.loadKeyPair()
	if keyPairNew == nil || keyPairNew != keyPair {
		t.Fatal("expected key pair not nil and equal with old one")
	}

	keyProvider.SetContent([]byte("wrong"))
	time.Sleep(100 * time.Millisecond)
	keyPairNew = certContentProvider.loadKeyPair()
	if keyPairNew == nil || keyPairNew != keyPair {
		t.Fatal("expected key pair not nil and equal with old one")
	}

	caProvider.SetContent([]byte("wrong"))
	time.Sleep(100 * time.Millisecond)
	caPoolNew := certContentProvider.loadCAPool()
	if caPoolNew == nil || caPoolNew != caPool {
		t.Fatal("expected ca pool not nil and equal with old one")
	}
}
