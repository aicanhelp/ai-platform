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

package app

import (
	"context"
	"os"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/fy-ext/dynamicloader/contentprovider"
	"k8s.io/kubernetes/pkg/controller"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

var privateKey1 = []byte(`-----BEGIN RSA PRIVATE KEY-----
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

var privateKey2 = []byte(`-----BEGIN RSA PRIVATE KEY-----
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

var rootCa1 = []byte(`-----BEGIN CERTIFICATE-----
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
var rootCa2 = []byte(`-----BEGIN CERTIFICATE-----
MIIDgjCCAmqgAwIBAgICAMgwDQYJKoZIhvcNAQELBQAwWjELMAkGA1UEBhMCVVMx
CTAHBgNVBAgTADEWMBQGA1UEBxMNU2FuIEZyYW5jaXNjbzEJMAcGA1UECRMAMQkw
BwYDVQQREwAxEjAQBgNVBAoTCVRlc3QgQ0EgMjAeFw0yNTA5MjYwOTI2NDhaFw0y
NjA5MjYwOTI2NDhaMFoxCzAJBgNVBAYTAlVTMQkwBwYDVQQIEwAxFjAUBgNVBAcT
DVNhbiBGcmFuY2lzY28xCTAHBgNVBAkTADEJMAcGA1UEERMAMRIwEAYDVQQKEwlU
ZXN0IENBIDIwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDLGr5yitJw
l/ORBKvrwfyVDu4RbQx41i2koGooadU7DDF9ZYEvQAWZbFvO9k37Rx2vqHZhsW5h
8BkMWmIGyingVelaxIXfDPJSZgRE2VvTLQVQqpuO4Z2TNM+qip+aTlei/DyP6W5Y
uVBETqh9KfxroSduHf5UWOvzfqxc36D3ODDbkj3QFxIWpoEEMx3YhM6NSQTJz7/D
fPjSh7fKw45M+rDK5mwbUoafAL8Q1q8g+MUBYVCuVRdBiIggJTOpoGj1zZBFSf3T
Fj2Dtra+Wm6Cv07K0JOHHeUXwPVef+axBjgVg9nZyQ3AN8DgnUHBm55ccWk9ZURo
eVxHkNQxIH/rAgMBAAGjUjBQMA4GA1UdDwEB/wQEAwIChDAdBgNVHSUEFjAUBggr
BgEFBQcDAgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ4EBwQFAQID
BAgwDQYJKoZIhvcNAQELBQADggEBAKzxsBSm+DvHqZFjAKgknyDaekGTBssXi2g5
Vv/fXSjNPjqCYzu5bKo8iJ8yVJs1XWHki7I79Jldnc7fHQoUDfbkV7JnJAzJZcw6
V6B8fzUYkVUuHCFDMhQsQc3m3K5YyAwZLpxojF7CTa8w89aSJzQ8WSXbFXhKDSUC
ADVGWR0AiQ4GhFjax7284hrAoDmRu1rKTJBkFb57IVIsO+5ej4l9HBQZ9jx4JC9s
Xx4gNPoxYMms/4M2t0V/MIgqIhoSWn7tFodCEjYIZeoTR5Wz6NlWcvN2W6AJzHUu
+lnIWxQuaK0poCoYsVfwjMWjoPb0+/UJ8QOwjMK+j4eD//SghSE=
-----END CERTIFICATE-----
`)

func TestFyUpdateKeysDynamically(t *testing.T) {
	// prepare params
	keyFile, _ := os.CreateTemp("", "testing-*")
	defer os.Remove(keyFile.Name())
	os.WriteFile(keyFile.Name(), privateKey1, 0600)

	sk, _ := keyutil.PrivateKeyFromFile(keyFile.Name())
	tokenGenerator, _ := serviceaccount.JWTTokenGenerator("test", sk)

	filedValue := func(generator serviceaccount.TokenGenerator) interface{} {
		v := reflect.ValueOf(generator).Elem().FieldByName("signer")
		s := reflect.NewAt(v.Type(), v.Addr().UnsafePointer()).Elem().Interface().(atomic.Value)
		return s.Load()
	}
	signer1 := filedValue(tokenGenerator)

	fyUpdateKeysDynamically(tokenGenerator, keyFile.Name())
	os.WriteFile(keyFile.Name(), privateKey2, 0600)

	time.Sleep(contentprovider.DefaultCallBackDuration + 2*time.Second)
	signer2 := filedValue(tokenGenerator)

	if signer1 == signer2 {
		t.Errorf("expect signer1 not equal with signer2")
	}
}

func TestFyUpdateCADynamically(t *testing.T) {
	// prepare params
	keyFile, _ := os.CreateTemp("", "testfile-*")
	defer os.Remove(keyFile.Name())
	os.WriteFile(keyFile.Name(), rootCa1, 0600)

	rootCa, _ := readCA(keyFile.Name())
	client := fake.NewSimpleClientset()
	makeNameSpace := func(labelValue string) *v1.Namespace {
		return &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test",
				Labels: map[string]string{"test": labelValue},
			},
		}
	}
	client.CoreV1().Namespaces().Update(context.TODO(), makeNameSpace("test"), metav1.UpdateOptions{})

	controller := &serviceaccountcontroller.TokensController{}
	controller.SetRootCa(rootCa)
	controller.SetClient(client)

	oldCa := controller.GetRootCa()

	fyUpdateCADynamically(context.TODO(), controller, keyFile.Name())
	os.WriteFile(keyFile.Name(), rootCa2, 0600)

	time.Sleep(contentprovider.DefaultCallBackDuration + 2*time.Second)
	currentCa := controller.GetRootCa()

	if reflect.DeepEqual(oldCa, currentCa) {
		t.Errorf("expect oldCa not equal with currentCA")
	}
}

func TestFyUpdatePublishCADynamically(t *testing.T) {
	// prepare params
	keyFile, _ := os.CreateTemp("", "testfile-*")
	defer os.Remove(keyFile.Name())
	os.WriteFile(keyFile.Name(), rootCa1, 0600)

	rootCa, _ := readCA(keyFile.Name())
	client := fake.NewSimpleClientset()
	makeNamespace := func(labelValue string) *v1.Namespace {
		return &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test",
				Labels: map[string]string{"test": labelValue},
			},
		}
	}
	client.CoreV1().Namespaces().Update(context.TODO(), makeNamespace("test"), metav1.UpdateOptions{})

	publisher := &rootcacertpublisher.Publisher{}
	publisher.SetRootCa(rootCa)
	publisher.SetClient(client)
	saController := serviceaccountconfig.SAControllerConfiguration{
		RootCAFile: keyFile.Name(),
	}
	configration := kubectrlmgrconfig.KubeControllerManagerConfiguration{
		SAController: saController,
	}
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	controllerContext := ControllerContext{
		ComponentConfig: configration,
		InformerFactory: informerFactory,
	}

	oldCa := publisher.GetRootCa()

	fyUpdatePublishCADynamically(publisher, controllerContext)
	os.WriteFile(keyFile.Name(), rootCa2, 0600)

	time.Sleep(contentprovider.DefaultCallBackDuration + 2*time.Second)
	currentCa := publisher.GetRootCa()

	if reflect.DeepEqual(oldCa, currentCa) {
		t.Errorf("expect oldCa not equal with currentCA")
	}
}
