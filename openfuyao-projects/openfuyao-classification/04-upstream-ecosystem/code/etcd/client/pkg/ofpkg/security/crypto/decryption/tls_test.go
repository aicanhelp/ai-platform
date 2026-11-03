package decryption

import (
	"crypto/tls"
	"encoding/base64"
	"reflect"
	"sync"
	"testing"
)

var (
	certPEMBlock = []byte(`
-----BEGIN CERTIFICATE-----
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
-----END CERTIFICATE-----`)

	keyPEMBlock = []byte(`-----BEGIN RSA PRIVATE KEY-----
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
	encryptedKeyPEMBlock = []byte(base64.StdEncoding.EncodeToString(keyPEMBlock))
)

func TestShouldBeSupportLoadEncryptedKeyCert(t *testing.T) {
	mockReInit(t, true)
	mockDecryptKeyFunc(t, mockDecryptKey)

	gotCert, gotFailed, gotEnable := OFTryDecryptWithCache(certPEMBlock, encryptedKeyPEMBlock, tls.X509KeyPair)
	if gotCert == nil {
		t.Errorf("OFTryDecryptWithCache should got cert object by encrypt key pem")
	}
	if gotFailed != nil {
		t.Errorf("OFTryDecryptWithCache should be not error while parse cert from encrypted cert")
	}
	if !gotEnable {
		t.Errorf("OFTryDecryptWithCache should be enable while cert parse")
	}
}

func TestShouldBeSupportLoadKeyCert(t *testing.T) {
	mockReInit(t, true)
	decryptedTime := 0
	mockDecryptKeyFunc(t, func(keyBlock []byte) ([]byte, error) {
		decryptedTime++
		return mockDecryptKey(keyBlock)
	})
	gotCert1, _, _ := OFTryDecryptWithCache(certPEMBlock, encryptedKeyPEMBlock, tls.X509KeyPair)
	gotCert2, gotFailed, gotEnable := OFTryDecryptWithCache(certPEMBlock, encryptedKeyPEMBlock, tls.X509KeyPair)
	if !reflect.DeepEqual(gotCert1, gotCert2) {
		t.Errorf("OFTryDecryptWithCache should deep equal while given same pem block")
	}
	if &gotCert1 == &gotCert2 {
		t.Errorf("OFTryDecryptWithCache should return a new one while incoke every time")
	}
	if decryptedTime != 1 {
		t.Errorf("OFTryDecryptWithCache should be decrypt only one time while given same input")
	}
	if gotFailed != nil {
		t.Errorf("OFTryDecryptWithCache should be not error while decrypt cert")
	}
	if !gotEnable {
		t.Errorf("OFTryDecryptWithCache should be enable while key parse")
	}
}

func TestShouldReturnNilWhileDisable(t *testing.T) {
	mockReInit(t, false)
	gotCert, gotFailed, gotEnable := OFTryDecryptWithCache(certPEMBlock, keyPEMBlock, tls.X509KeyPair)
	if gotCert != nil {
		t.Errorf("OFTryDecryptWithCache should return nil when disable")
	}
	if gotFailed != nil {
		t.Errorf("OFTryDecryptWithCache should not be failed when disable")
	}
	if gotEnable {
		t.Errorf("OFTryDecryptWithCache should be false when disable")
	}
}

func TestShouldFailedWhileDecryptError(t *testing.T) {
	mockReInit(t, true)
	mockDecryptKeyFunc(t, mockDecryptKey)

	gotCert, gotFailed, gotEnable := OFTryDecryptWithCache(certPEMBlock, keyPEMBlock, tls.X509KeyPair)
	if gotCert != nil {
		t.Errorf("OFTryDecryptWithCache should not return cert while decrypt failed")
	}
	if gotFailed == nil {
		t.Errorf("OFTryDecryptWithCache should return an error while decrypt failed")
	}
	if !gotEnable {
		t.Errorf("OFTryDecryptWithCache should be enable while just decrypt failed")
	}
}

func TestShouldFailedWhileParseCertFailed(t *testing.T) {
	mockReInit(t, true)
	mockDecryptKeyFunc(t, mockDecryptKey)

	gotCert, gotFailed, gotEnable := OFTryDecryptWithCache(keyPEMBlock, encryptedKeyPEMBlock, tls.X509KeyPair)
	if gotCert != nil {
		t.Errorf("OFTryDecryptWithCache should not return cert while cert parse failed")
	}
	if gotFailed == nil {
		t.Errorf("OFTryDecryptWithCache should return an error while cert parse failed")
	}
	if !gotEnable {
		t.Errorf("OFTryDecryptWithCache should be enable while just cert parse failed")
	}
}

func TestToKey(t *testing.T) {
	certPEMBlock := []byte{1, 2, 3}
	keyPEMBlock := []byte{4, 5, 6}

	key1 := toKey(certPEMBlock, keyPEMBlock)
	key2 := toKey(certPEMBlock, keyPEMBlock)

	if !reflect.DeepEqual(key1, key2) {
		t.Errorf("toKeys should be equal for multi time")
	}
}

func TestDisableDecryptReturnOriginKey(t *testing.T) {
	result, err := disableDecrypt(keyPEMBlock)
	if err != nil {
		t.Errorf("disableDecrypt should never return error while decrypt data")
	}
	if !reflect.DeepEqual(result, keyPEMBlock) {
		t.Errorf("disableDecrypt should return euqal ones")
	}
}

func mockDecryptKeyFunc(t *testing.T, mockDecryptKey func(keyBlock []byte) ([]byte, error)) {
	old := decryptKey
	t.Cleanup(func() {
		decryptKey = old
	})
	decryptKey = mockDecryptKey
}

func mockDecryptKey(keyBlock []byte) ([]byte, error) {
	return base64.StdEncoding.DecodeString(string(keyBlock))
}

func mockReInit(t *testing.T, enable bool) {
	udsKeyValue := ""
	if enable {
		udsKeyValue = "unix:///enable.sock"
	} else {
		cache = &sync.Map{}
	}
	t.Setenv(udsKey, udsKeyValue)
	runInit()
}
