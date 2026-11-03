/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1beta1

import (
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	numZero                             = 0
	numOne                              = 1
	numTwo                              = 2
	numThree                            = 3
	numFour                             = 4
	numSix                              = 6
	numTen                              = 10
	numTwentyFour                       = 24
	numOneHundred                       = 100
	numOneNinetyTwo                     = 192
	numFortyThree                       = 43
	numFiveThousand                     = 5000
	numSixThousand                      = 6000
	numSixThousandFourHundredFortyThree = 6443
)

var (
	testLoopbackIP = net.IPv4(numOneHundred, numTwentyFour, numThree, numOne)
	testPrivateIP  = net.IPv4(numOneHundred, numOneNinetyTwo, numOne, numOne)
)

func TestAPIEndpointIsZero(t *testing.T) {
	tests := []struct {
		name     string
		endpoint APIEndpoint
		expected bool
	}{
		{
			name:     "empty endpoint should be zero",
			endpoint: APIEndpoint{},
			expected: true,
		},
		{
			name:     "only host should not be zero",
			endpoint: APIEndpoint{Host: "example.com"},
			expected: false,
		},
		{
			name:     "only port should not be zero",
			endpoint: APIEndpoint{Port: numSixThousandFourHundredFortyThree},
			expected: false,
		},
		{
			name:     "both host and port should not be zero",
			endpoint: APIEndpoint{Host: "example.com", Port: numSixThousandFourHundredFortyThree},
			expected: false,
		},
		{
			name:     "empty host with non-zero port should not be zero",
			endpoint: APIEndpoint{Port: numOne},
			expected: false,
		},
		{
			name:     "zero port with non-empty host should not be zero",
			endpoint: APIEndpoint{Host: "localhost", Port: numZero},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.endpoint.IsZero()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestAPIEndpointIsValid(t *testing.T) {
	tests := []struct {
		name     string
		endpoint APIEndpoint
		expected bool
	}{
		{
			name:     "empty endpoint should be invalid",
			endpoint: APIEndpoint{},
			expected: false,
		},
		{
			name:     "only host should be invalid",
			endpoint: APIEndpoint{Host: "example.com"},
			expected: false,
		},
		{
			name:     "only port should be invalid",
			endpoint: APIEndpoint{Port: numSixThousandFourHundredFortyThree},
			expected: false,
		},
		{
			name:     "both host and port should be valid",
			endpoint: APIEndpoint{Host: "example.com", Port: numSixThousandFourHundredFortyThree},
			expected: true,
		},
		{
			name:     "localhost with port should be valid",
			endpoint: APIEndpoint{Host: "localhost", Port: numSixThousandFourHundredFortyThree},
			expected: true,
		},
		{
			name:     "ip address with port should be valid",
			endpoint: APIEndpoint{Host: "192.168.1.1", Port: numSixThousandFourHundredFortyThree},
			expected: true,
		},
		{
			name:     "zero port should be invalid",
			endpoint: APIEndpoint{Host: "example.com", Port: numZero},
			expected: false,
		},
		{
			name:     "empty host should be invalid",
			endpoint: APIEndpoint{Host: "", Port: numSixThousandFourHundredFortyThree},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.endpoint.IsValid()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestAPIEndpointString(t *testing.T) {
	tests := []struct {
		name     string
		endpoint APIEndpoint
		expected string
	}{
		{
			name:     "standard port",
			endpoint: APIEndpoint{Host: "example.com", Port: numSixThousandFourHundredFortyThree},
			expected: "example.com:6443",
		},
		{
			name:     "custom port",
			endpoint: APIEndpoint{Host: "example.com", Port: numThree},
			expected: "example.com:3",
		},
		{
			name:     "localhost",
			endpoint: APIEndpoint{Host: "localhost", Port: numSixThousandFourHundredFortyThree},
			expected: "localhost:6443",
		},
		{
			name:     "ip address",
			endpoint: APIEndpoint{Host: "192.168.1.1", Port: numSixThousandFourHundredFortyThree},
			expected: "192.168.1.1:6443",
		},
		{
			name:     "low port number",
			endpoint: APIEndpoint{Host: "example.com", Port: numOne},
			expected: "example.com:1",
		},
		{
			name:     "high port number",
			endpoint: APIEndpoint{Host: "example.com", Port: numFiveThousand},
			expected: "example.com:5000",
		},
		{
			name:     "port with zero value",
			endpoint: APIEndpoint{Host: "example.com", Port: numZero},
			expected: "example.com:0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.endpoint.String()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEClusterSpecFields(t *testing.T) {
	t.Run("basic fields", func(t *testing.T) {
		spec := BKEClusterSpec{
			ControlPlaneEndpoint: APIEndpoint{
				Host: "example.com",
				Port: numSixThousandFourHundredFortyThree,
			},
			ClusterConfig: &BKEConfig{
				Cluster: Cluster{
					KubernetesVersion: "v1.25.6",
				},
			},
			KubeletConfigRef: &KubeletConfigRef{
				Name:      "kubelet-config",
				Namespace: "kube-system",
			},
			Pause:  true,
			DryRun: true,
			Reset:  false,
		}

		assert.True(t, spec.ControlPlaneEndpoint.IsValid())
		assert.False(t, spec.ControlPlaneEndpoint.IsZero())
		assert.Equal(t, "example.com:6443", spec.ControlPlaneEndpoint.String())
		assert.NotNil(t, spec.ClusterConfig)
		assert.NotNil(t, spec.KubeletConfigRef)
		assert.True(t, spec.Pause)
		assert.True(t, spec.DryRun)
		assert.False(t, spec.Reset)
	})

	t.Run("optional fields nil", func(t *testing.T) {
		spec := BKEClusterSpec{
			Pause:  false,
			DryRun: false,
			Reset:  false,
		}

		assert.Nil(t, spec.ClusterConfig)
		assert.Nil(t, spec.KubeletConfigRef)
		assert.True(t, spec.ControlPlaneEndpoint.IsZero())
		assert.False(t, spec.ControlPlaneEndpoint.IsValid())
	})
}

func TestKubeletConfigRefFields(t *testing.T) {
	t.Run("with namespace", func(t *testing.T) {
		ref := KubeletConfigRef{
			Name:      "kubelet-config",
			Namespace: "kube-system",
		}

		assert.Equal(t, "kubelet-config", ref.Name)
		assert.Equal(t, "kube-system", ref.Namespace)
	})

	t.Run("without namespace", func(t *testing.T) {
		ref := KubeletConfigRef{
			Name: "kubelet-config",
		}

		assert.Equal(t, "kubelet-config", ref.Name)
		assert.Equal(t, "", ref.Namespace)
	})
}

func TestBKEConfigFields(t *testing.T) {
	t.Run("basic fields", func(t *testing.T) {
		config := BKEConfig{
			Cluster: Cluster{
				KubernetesVersion: "v1.25.6",
				NTPServer:         "pool.ntp.org",
			},
			Addons: []Product{
				{
					Name:    "calico",
					Version: "v3.24.5",
				},
			},
			CustomExtra: map[string]string{
				"key1": "value1",
			},
		}

		assert.NotNil(t, config.Cluster)
		assert.Equal(t, "v1.25.6", config.Cluster.KubernetesVersion)
		assert.Len(t, config.Addons, numOne)
		assert.Len(t, config.CustomExtra, numOne)
	})

	t.Run("empty config", func(t *testing.T) {
		config := BKEConfig{}

		assert.NotNil(t, config)
		assert.Empty(t, config.Addons)
		assert.Nil(t, config.CustomExtra)
	})
}

func TestLabelFields(t *testing.T) {
	t.Run("with key and value", func(t *testing.T) {
		label := Label{
			Key:   "node-role.kubernetes.io/master",
			Value: "",
		}

		assert.Equal(t, "node-role.kubernetes.io/master", label.Key)
		assert.Equal(t, "", label.Value)
	})

	t.Run("empty label", func(t *testing.T) {
		label := Label{}

		assert.Equal(t, "", label.Key)
		assert.Equal(t, "", label.Value)
	})
}

func TestClusterFields(t *testing.T) {
	t.Run("full cluster config", func(t *testing.T) {
		cluster := Cluster{
			KubernetesVersion: "v1.25.6",
			EtcdVersion:       "3.5.9",
			CertificatesDir:   "/etc/kubernetes/pki",
			NTPServer:         "pool.ntp.org",
			AgentHealthPort:   "10256",
			Networking: Networking{
				ServiceSubnet: "10.96.0.0/12",
				PodSubnet:     "10.244.0.0/16",
				DNSDomain:     "cluster.local",
			},
			Labels: []Label{
				{Key: "env", Value: "prod"},
			},
		}

		assert.Equal(t, "v1.25.6", cluster.KubernetesVersion)
		assert.Equal(t, "3.5.9", cluster.EtcdVersion)
		assert.Equal(t, "/etc/kubernetes/pki", cluster.CertificatesDir)
		assert.Equal(t, "pool.ntp.org", cluster.NTPServer)
		assert.Equal(t, "10256", cluster.AgentHealthPort)
		assert.Equal(t, "10.96.0.0/12", cluster.Networking.ServiceSubnet)
		assert.Len(t, cluster.Labels, numOne)
	})
}

func TestContainerRuntimeFields(t *testing.T) {
	t.Run("with cri and runtime", func(t *testing.T) {
		cri := ContainerRuntime{
			CRI:     "containerd",
			Runtime: "runc",
			Param: map[string]string{
				"data-root": "/var/lib/containerd",
			},
		}

		assert.Equal(t, "containerd", cri.CRI)
		assert.Equal(t, "runc", cri.Runtime)
		assert.NotNil(t, cri.Param)
	})

	t.Run("empty container runtime", func(t *testing.T) {
		cri := ContainerRuntime{}

		assert.Equal(t, "", cri.CRI)
		assert.Equal(t, "", cri.Runtime)
		assert.Nil(t, cri.Param)
	})
}

func TestRepoFields(t *testing.T) {
	t.Run("full repo config", func(t *testing.T) {
		repo := Repo{
			Domain:                "registry.example.com",
			Ip:                    "192.168.1.1",
			Port:                  "5000",
			Prefix:                "kubernetes",
			InsecureSkipTLSVerify: true,
			AuthSecretRef: &AuthSecretRef{
				Name:        "docker-secret",
				Namespace:   "default",
				UsernameKey: "username",
				PasswordKey: "password",
			},
			TlsSecretRef: &TlsSecretRef{
				Name:      "tls-secret",
				Namespace: "default",
				CaKey:     "ca.crt",
				CertKey:   "tls.crt",
				KeyKey:    "tls.key",
			},
		}

		assert.Equal(t, "registry.example.com", repo.Domain)
		assert.Equal(t, "192.168.1.1", repo.Ip)
		assert.Equal(t, "5000", repo.Port)
		assert.Equal(t, "kubernetes", repo.Prefix)
		assert.True(t, repo.InsecureSkipTLSVerify)
		assert.NotNil(t, repo.AuthSecretRef)
		assert.NotNil(t, repo.TlsSecretRef)
	})

	t.Run("minimal repo config", func(t *testing.T) {
		repo := Repo{
			Domain: "registry.example.com",
			Prefix: "kubernetes",
		}

		assert.Equal(t, "registry.example.com", repo.Domain)
		assert.Equal(t, "kubernetes", repo.Prefix)
		assert.Empty(t, repo.Port)
		assert.Nil(t, repo.AuthSecretRef)
		assert.Nil(t, repo.TlsSecretRef)
	})
}

func TestAuthSecretRefFields(t *testing.T) {
	t.Run("with all fields", func(t *testing.T) {
		auth := AuthSecretRef{
			Name:        "secret",
			Namespace:   "default",
			UsernameKey: "username",
			PasswordKey: "password",
		}

		assert.Equal(t, "secret", auth.Name)
		assert.Equal(t, "default", auth.Namespace)
		assert.Equal(t, "username", auth.UsernameKey)
		assert.Equal(t, "password", auth.PasswordKey)
	})

	t.Run("default keys", func(t *testing.T) {
		auth := AuthSecretRef{
			Name: "secret",
		}

		assert.Equal(t, "secret", auth.Name)
		assert.Empty(t, auth.Namespace)
		assert.Empty(t, auth.UsernameKey)
		assert.Empty(t, auth.PasswordKey)
	})
}

func TestTlsSecretRefFields(t *testing.T) {
	t.Run("with all fields", func(t *testing.T) {
		tls := TlsSecretRef{
			Name:      "tls-secret",
			Namespace: "default",
			CaKey:     "ca.crt",
			CertKey:   "tls.crt",
			KeyKey:    "tls.key",
		}

		assert.Equal(t, "tls-secret", tls.Name)
		assert.Equal(t, "default", tls.Namespace)
		assert.Equal(t, "ca.crt", tls.CaKey)
		assert.Equal(t, "tls.crt", tls.CertKey)
		assert.Equal(t, "tls.key", tls.KeyKey)
	})

	t.Run("default keys", func(t *testing.T) {
		tls := TlsSecretRef{
			Name: "tls-secret",
		}

		assert.Equal(t, "tls-secret", tls.Name)
		assert.Empty(t, tls.Namespace)
		assert.Empty(t, tls.CaKey)
		assert.Empty(t, tls.CertKey)
		assert.Empty(t, tls.KeyKey)
	})
}

func TestNetworkingFields(t *testing.T) {
	t.Run("full networking config", func(t *testing.T) {
		networking := Networking{
			ServiceSubnet: "10.96.0.0/12",
			PodSubnet:     "10.244.0.0/16",
			DNSDomain:     "cluster.local",
		}

		assert.Equal(t, "10.96.0.0/12", networking.ServiceSubnet)
		assert.Equal(t, "10.244.0.0/16", networking.PodSubnet)
		assert.Equal(t, "cluster.local", networking.DNSDomain)
	})

	t.Run("empty networking", func(t *testing.T) {
		networking := Networking{}

		assert.Empty(t, networking.ServiceSubnet)
		assert.Empty(t, networking.PodSubnet)
		assert.Empty(t, networking.DNSDomain)
	})
}

func TestProductFields(t *testing.T) {
	t.Run("full product config", func(t *testing.T) {
		product := Product{
			Name:        "calico",
			Version:     "v3.24.5",
			Type:        "yaml",
			ReleaseName: "calico",
			Namespace:   "calico-system",
			Timeout:     300,
			Block:       true,
			Param: map[string]string{
				"mode": "bgp",
			},
			ValuesConfigMapRef: &ValuesConfigMapRef{
				Name:      "calico-values",
				Namespace: "default",
				ValuesKey: "values.yaml",
			},
		}

		assert.Equal(t, "calico", product.Name)
		assert.Equal(t, "v3.24.5", product.Version)
		assert.Equal(t, "yaml", product.Type)
		assert.Equal(t, "calico", product.ReleaseName)
		assert.Equal(t, "calico-system", product.Namespace)
		assert.Equal(t, 300, product.Timeout)
		assert.True(t, product.Block)
		assert.NotNil(t, product.Param)
		assert.NotNil(t, product.ValuesConfigMapRef)
	})

	t.Run("minimal product", func(t *testing.T) {
		product := Product{
			Name: "nginx",
		}

		assert.Equal(t, "nginx", product.Name)
		assert.Empty(t, product.Version)
		assert.Empty(t, product.Type)
		assert.Empty(t, product.ReleaseName)
		assert.Empty(t, product.Namespace)
		assert.Equal(t, numZero, product.Timeout)
		assert.False(t, product.Block)
		assert.Nil(t, product.Param)
		assert.Nil(t, product.ValuesConfigMapRef)
	})
}

func TestValuesConfigMapRefFields(t *testing.T) {
	t.Run("with all fields", func(t *testing.T) {
		ref := ValuesConfigMapRef{
			Name:      "values",
			Namespace: "default",
			ValuesKey: "values.yaml",
		}

		assert.Equal(t, "values", ref.Name)
		assert.Equal(t, "default", ref.Namespace)
		assert.Equal(t, "values.yaml", ref.ValuesKey)
	})

	t.Run("default values key", func(t *testing.T) {
		ref := ValuesConfigMapRef{
			Name: "values",
		}

		assert.Equal(t, "values", ref.Name)
		assert.Empty(t, ref.Namespace)
		assert.Empty(t, ref.ValuesKey)
	})
}

func TestControlPlaneFields(t *testing.T) {
	t.Run("full control plane", func(t *testing.T) {
		cp := ControlPlane{
			ControllerManager: &ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"bind-address": "0.0.0.0",
				},
			},
			Scheduler: &ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"bind-address": "0.0.0.0",
				},
			},
			APIServer: &APIServer{
				APIEndpoint: APIEndpoint{
					Port: numSixThousandFourHundredFortyThree,
				},
				CertSANs: []string{"example.com", "localhost"},
			},
			Etcd: &Etcd{
				DataDir:        "/var/lib/etcd",
				ServerCertSANs: []string{"etcd1", "etcd2"},
				PeerCertSANs:   []string{"etcd1", "etcd2"},
			},
		}

		assert.NotNil(t, cp.ControllerManager)
		assert.NotNil(t, cp.Scheduler)
		assert.NotNil(t, cp.APIServer)
		assert.NotNil(t, cp.Etcd)
		assert.Len(t, cp.APIServer.CertSANs, numTwo)
		assert.Len(t, cp.Etcd.ServerCertSANs, numTwo)
	})

	t.Run("nil control plane", func(t *testing.T) {
		cp := ControlPlane{}

		assert.Nil(t, cp.ControllerManager)
		assert.Nil(t, cp.Scheduler)
		assert.Nil(t, cp.APIServer)
		assert.Nil(t, cp.Etcd)
	})
}

func TestControlPlaneComponentFields(t *testing.T) {
	t.Run("with extra args and volumes", func(t *testing.T) {
		component := ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"feature-gates": "DynamicKubeletConfig=true",
			},
			ExtraVolumes: []HostPathMount{
				{
					Name:      "certs",
					HostPath:  "/etc/kubernetes/pki",
					MountPath: "/etc/kubernetes/pki",
					ReadOnly:  true,
					PathType:  "Directory",
				},
			},
		}

		assert.NotNil(t, component.ExtraArgs)
		assert.Len(t, component.ExtraVolumes, numOne)
		assert.Equal(t, "certs", component.ExtraVolumes[numZero].Name)
		assert.True(t, component.ExtraVolumes[numZero].ReadOnly)
	})

	t.Run("empty component", func(t *testing.T) {
		component := ControlPlaneComponent{}

		assert.Nil(t, component.ExtraArgs)
		assert.Empty(t, component.ExtraVolumes)
	})
}

func TestHostPathMountFields(t *testing.T) {
	t.Run("full host path mount", func(t *testing.T) {
		mount := HostPathMount{
			Name:      "certs",
			HostPath:  "/etc/kubernetes/pki",
			MountPath: "/etc/kubernetes/pki",
			ReadOnly:  true,
			PathType:  "Directory",
		}

		assert.Equal(t, "certs", mount.Name)
		assert.Equal(t, "/etc/kubernetes/pki", mount.HostPath)
		assert.Equal(t, "/etc/kubernetes/pki", mount.MountPath)
		assert.True(t, mount.ReadOnly)
		assert.Equal(t, "Directory", mount.PathType)
	})

	t.Run("minimal host path mount", func(t *testing.T) {
		mount := HostPathMount{
			Name:      "data",
			HostPath:  "/data",
			MountPath: "/data",
		}

		assert.Equal(t, "data", mount.Name)
		assert.False(t, mount.ReadOnly)
		assert.Empty(t, mount.PathType)
	})
}

func TestAPIServerFields(t *testing.T) {
	t.Run("full api server", func(t *testing.T) {
		server := APIServer{
			APIEndpoint: APIEndpoint{
				Host: "example.com",
				Port: numSixThousandFourHundredFortyThree,
			},
			ControlPlaneComponent: ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"authorization-mode": "Node,RBAC",
				},
			},
			CertSANs: []string{"example.com", "localhost", "10.0.0.1"},
		}

		assert.Equal(t, "example.com", server.Host)
		assert.Equal(t, int32(numSixThousandFourHundredFortyThree), server.Port)
		assert.NotNil(t, server.ExtraArgs)
		assert.Len(t, server.CertSANs, numThree)
	})

	t.Run("minimal api server", func(t *testing.T) {
		server := APIServer{}

		assert.Empty(t, server.Host)
		assert.Equal(t, int32(numZero), server.Port)
		assert.Nil(t, server.ExtraArgs)
		assert.Empty(t, server.CertSANs)
	})
}

func TestEtcdFields(t *testing.T) {
	t.Run("full etcd", func(t *testing.T) {
		etcd := Etcd{
			ControlPlaneComponent: ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"listen-peer-urls": "https://0.0.0.0:2380",
				},
			},
			DataDir:        "/var/lib/etcd",
			ServerCertSANs: []string{"etcd1", "etcd2"},
			PeerCertSANs:   []string{"etcd1", "etcd2"},
		}

		assert.Equal(t, "/var/lib/etcd", etcd.DataDir)
		assert.NotNil(t, etcd.ExtraArgs)
		assert.Len(t, etcd.ServerCertSANs, numTwo)
		assert.Len(t, etcd.PeerCertSANs, numTwo)
	})

	t.Run("minimal etcd", func(t *testing.T) {
		etcd := Etcd{}

		assert.Empty(t, etcd.DataDir)
		assert.Nil(t, etcd.ExtraArgs)
		assert.Empty(t, etcd.ServerCertSANs)
		assert.Empty(t, etcd.PeerCertSANs)
	})
}

func TestKubeletFields(t *testing.T) {
	t.Run("full kubelet", func(t *testing.T) {
		kubelet := Kubelet{
			ControlPlaneComponent: ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"feature-gates": "RotateKubeletServerCertificate=true",
				},
			},
			ManifestsDir: "/etc/kubernetes/manifests",
		}

		assert.NotNil(t, kubelet.ExtraArgs)
		assert.Equal(t, "/etc/kubernetes/manifests", kubelet.ManifestsDir)
	})

	t.Run("minimal kubelet", func(t *testing.T) {
		kubelet := Kubelet{}

		assert.Empty(t, kubelet.ManifestsDir)
		assert.Nil(t, kubelet.ExtraArgs)
		assert.Empty(t, kubelet.ExtraVolumes)
	})
}

func TestContainerdConfigRefFields(t *testing.T) {
	t.Run("with namespace", func(t *testing.T) {
		ref := ContainerdConfigRef{
			Name:      "containerd-config",
			Namespace: "kube-system",
		}

		assert.Equal(t, "containerd-config", ref.Name)
		assert.Equal(t, "kube-system", ref.Namespace)
	})

	t.Run("without namespace", func(t *testing.T) {
		ref := ContainerdConfigRef{
			Name: "containerd-config",
		}

		assert.Equal(t, "containerd-config", ref.Name)
		assert.Empty(t, ref.Namespace)
	})
}
