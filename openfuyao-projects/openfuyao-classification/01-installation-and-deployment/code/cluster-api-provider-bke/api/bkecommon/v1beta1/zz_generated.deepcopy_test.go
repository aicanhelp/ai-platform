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

package v1beta1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// createTestCluster creates a test Cluster with common configuration to avoid repetition
func createTestCluster() Cluster {
	return Cluster{
		Networking: Networking{
			DNSDomain: "cluster.local",
		},
		KubernetesVersion: "v1.25.6",
		ImageRepo: Repo{
			Domain: "registry.example.com",
			Port:   "5000",
			Prefix: "k8s",
		},
		ChartRepo: Repo{
			Domain: "chart.example.com",
			Port:   "5001",
			Prefix: "charts",
			AuthSecretRef: &AuthSecretRef{
				Name:        "auth",
				Namespace:   "default",
				UsernameKey: "username",
				PasswordKey: "password",
			},
			TlsSecretRef: &TlsSecretRef{
				Name:      "tls",
				Namespace: "default",
				CertKey:   "tls.crt",
				KeyKey:    "tls.key",
				CaKey:     "ca.crt",
			},
		},
	}
}

// createTestBKEConfigCluster creates a simple test Cluster for BKEConfig to avoid repetition
func createTestBKEConfigCluster() Cluster {
	return Cluster{
		KubernetesVersion: "v1.25.6",
	}
}

// TestAPIEndpointDeepCopy tests APIEndpoint deep copy
func TestAPIEndpointDeepCopy(t *testing.T) {
	original := &APIEndpoint{
		Port: 6443,
	}

	if original == nil {
		t.Fatal("Original object is nil")
	}
	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Host != original.Host {
		t.Errorf("Expected Host %s, got %s", original.Host, copied.Host)
	}

	if copied.Port != original.Port {
		t.Errorf("Expected Port %d, got %d", original.Port, copied.Port)
	}

}

// TestAPIServerDeepCopy tests APIServer deep copy
func TestAPIServerDeepCopy(t *testing.T) {
	original := &APIServer{
		APIEndpoint: APIEndpoint{
			Port: 6443,
		},
		ControlPlaneComponent: ControlPlaneComponent{
			ExtraVolumes: []HostPathMount{
				{
					Name:      "certs",
					HostPath:  "/etc/kubernetes/pki",
					MountPath: "/etc/kubernetes/pki",
					ReadOnly:  true,
					PathType:  "Directory",
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

}

// TestBKEConfigDeepCopy tests BKEConfig deep copy
func TestBKEConfigDeepCopy(t *testing.T) {
	original := &BKEConfig{
		Cluster: createTestBKEConfigCluster(),
		CustomExtra: map[string]string{
			"custom-arg": "value",
		},
	}

	if original == nil {
		t.Fatal("Original object is nil")
	}
	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.Cluster.KubernetesVersion != original.Cluster.KubernetesVersion {
		t.Errorf("Expected KubernetesVersion %s, got %s",
			original.Cluster.KubernetesVersion, copied.Cluster.KubernetesVersion)
	}

	// Modify copied map
	copied.CustomExtra["new-key"] = "new-value"
	if _, exists := original.CustomExtra["new-key"]; exists {
		t.Error("Modifying copied CustomExtra affected original")
	}
}

// TestBKENodeDeepCopy tests deep copy of BKENode
func TestBKENodeDeepCopy(t *testing.T) {
	original := &BKENode{
		Spec: BKENodeSpec{
			Role:     []string{"master", "etcd"},
			IP:       "192.168.1.1",
			Port:     "22",
			Username: "root",
			Hostname: "master-1",
		},
		Status: BKENodeStatus{
			State:     NodeReady,
			StateCode: 15,
			Message:   "Node is ready",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Spec.IP != original.Spec.IP {
		t.Errorf("Expected IP %s, got %s", original.Spec.IP, copied.Spec.IP)
	}

	if copied.Status.State != original.Status.State {
		t.Errorf("Expected State %s, got %s", original.Status.State, copied.Status.State)
	}
}

// TestBKEConfigAddonsDeepCopy tests deep copy of addons in BKEConfig
func TestBKEConfigAddonsDeepCopy(t *testing.T) {
	original := &BKEConfig{
		Addons: []Product{
			{
				Name:    "calico",
				Version: "v3.24.5",
				Type:    "yaml",
				Param: map[string]string{
					"mode": "bgp",
				},
			},
			{
				Name:        "victoriametrics",
				Version:     "0.58.2",
				Type:        "chart",
				ReleaseName: "vmks",
				Namespace:   "vm",
				ValuesConfigMapRef: &ValuesConfigMapRef{
					Name:      "vm-values",
					Namespace: "vm",
					ValuesKey: "values.yaml",
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify copied Addons
	copied.Addons[0].Version = "v3.25.0"
	if original.Addons[0].Version == copied.Addons[0].Version {
		t.Error("Modifying copied addon affected original")
	}
	// Verify Product struct
	if original.Addons[1].ReleaseName != copied.Addons[1].ReleaseName {
		t.Error("Copied addon is not a deep copy")
	}
	// Verify ValuesConfigMapRef struct
	if original.Addons[1].ValuesConfigMapRef.Name != copied.Addons[1].ValuesConfigMapRef.Name {
		t.Error("Copied addon is not a deep copy")
	}
}

// TestClusterDeepCopy tests Cluster deep copy
func TestClusterDeepCopy(t *testing.T) {
	original := createTestCluster()
	original.HTTPRepo = Repo{
		Domain: "yum.example.com",
		Port:   "80",
	}
	original.ContainerRuntime = ContainerRuntime{
		CRI:     "containerd",
		Runtime: "runc",
		Param: map[string]string{
			"data-root": "/var/lib/containerd",
		},
	}
	original.Labels = []Label{
		{Key: "env", Value: "prod"},
		{Key: "team", Value: "backend"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.KubernetesVersion != original.KubernetesVersion {
		t.Errorf("Expected KubernetesVersion %s, got %s",
			original.KubernetesVersion, copied.KubernetesVersion)
	}

	// Verify Repo struct
	if copied.ImageRepo.Domain != original.ImageRepo.Domain {
		t.Errorf("Expected ImageRepo Domain %s, got %s",
			original.ImageRepo.Domain, copied.ImageRepo.Domain)
	}

	// Modify copied Param
	copied.ContainerRuntime.Param["new-param"] = "new-value"
	if _, exists := original.ContainerRuntime.Param["new-param"]; exists {
		t.Error("Modifying copied ContainerRuntime Param affected original")
	}

	// Modify copied Labels
	copied.Labels[0].Value = "dev"
	if original.Labels[0].Value == copied.Labels[0].Value {
		t.Error("Modifying copied label affected original")
	}

	// Verify ChartRepo Domain
	if copied.ChartRepo.Domain != original.ChartRepo.Domain {
		t.Errorf("Expected ChartRepo Domain %s, got %s",
			original.ChartRepo.Domain, copied.ChartRepo.Domain)
	}

	// Verify ChartRepo AuthSecretRef name
	if copied.ChartRepo.AuthSecretRef.Name != original.ChartRepo.AuthSecretRef.Name {
		t.Errorf("Expected ChartRepo AuthSecretRef name %s, got %s",
			original.ChartRepo.AuthSecretRef.Name, copied.ChartRepo.AuthSecretRef.Name)
	}

	// Verify ChartRepo TlsSecretRef name
	if copied.ChartRepo.TlsSecretRef.Name != original.ChartRepo.TlsSecretRef.Name {
		t.Errorf("Expected ChartRepo TlsSecretRef name %s, got %s",
			original.ChartRepo.TlsSecretRef.Name, copied.ChartRepo.TlsSecretRef.Name)
	}
}

// TestProductDeepCopy tests Product deep copy
func TestProductDeepCopy(t *testing.T) {
	original := &Product{
		Name:    "nginx-ingress",
		Version: "v1.5.1",
		Param: map[string]string{
			"replicas": "2",
			"type":     "daemonset",
		},
		Block: true,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	// Modify copied Param
	copied.Param["new-param"] = "new-value"
	if _, exists := original.Param["new-param"]; exists {
		t.Error("Modifying copied Param affected original")
	}

	// Modify copied values to verify independence
	copied.Version = "v1.6.0"
	if original.Version == copied.Version {
		t.Error("Modifying copied version affected original")
	}
}

// TestControlPlaneComponentDeepCopy tests ControlPlaneComponent deep copy
func TestControlPlaneComponentDeepCopy(t *testing.T) {
	original := &ControlPlaneComponent{
		ExtraArgs: map[string]string{
			"feature-gates":      "DynamicKubeletConfig=true",
			"authorization-mode": "Node,RBAC",
		},
		ExtraVolumes: []HostPathMount{
			{
				Name:      "kubeconfig",
				HostPath:  "/etc/kubernetes/admin.conf",
				MountPath: "/etc/kubernetes/admin.conf",
				ReadOnly:  true,
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Modify copied ExtraArgs
	copied.ExtraArgs["new-arg"] = "new-value"
	if _, exists := original.ExtraArgs["new-arg"]; exists {
		t.Error("Modifying copied ExtraArgs affected original")
	}

	// Modify copied ExtraVolumes
	copied.ExtraVolumes[0].Name = "new-name"
	if original.ExtraVolumes[0].Name == copied.ExtraVolumes[0].Name {
		t.Error("Modifying copied ExtraVolumes affected original")
	}
}

// TestHostPathMountDeepCopy tests HostPathMount deep copy
func TestHostPathMountDeepCopy(t *testing.T) {
	original := &HostPathMount{
		Name:      "certs",
		HostPath:  "/etc/ssl/certs",
		MountPath: "/etc/ssl/certs",
		ReadOnly:  true,
		PathType:  "Directory",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify all fields
	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}
	if copied.HostPath != original.HostPath {
		t.Errorf("Expected HostPath %s, got %s", original.HostPath, copied.HostPath)
	}
	if copied.MountPath != original.MountPath {
		t.Errorf("Expected MountPath %s, got %s", original.MountPath, copied.MountPath)
	}
	if copied.ReadOnly != original.ReadOnly {
		t.Errorf("Expected ReadOnly %t, got %t", original.ReadOnly, copied.ReadOnly)
	}
	if copied.PathType != original.PathType {
		t.Errorf("Expected PathType %s, got %s", original.PathType, copied.PathType)
	}

	// Modify copied values to verify independence
	copied.Name = "new-certs"
	if original.Name == copied.Name {
		t.Error("Modifying copied name affected original")
	}
}

// TestRepoDeepCopy tests Repo deep copy
func TestRepoDeepCopy(t *testing.T) {
	original := &Repo{
		Domain: "registry.example.com",
		Port:   "5000",
		Prefix: "kubernetes",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify all fields
	if copied.Domain != original.Domain {
		t.Errorf("Expected Domain %s, got %s", original.Domain, copied.Domain)
	}
	if copied.Ip != original.Ip {
		t.Errorf("Expected IP %s, got %s", original.Ip, copied.Ip)
	}
	if copied.Port != original.Port {
		t.Errorf("Expected Port %s, got %s", original.Port, copied.Port)
	}
	if copied.Prefix != original.Prefix {
		t.Errorf("Expected Prefix %s, got %s", original.Prefix, copied.Prefix)
	}

	// Modify copied values to verify independence
	copied.Domain = "new-registry.example.com"
	if original.Domain == copied.Domain {
		t.Error("Modifying copied domain affected original")
	}
}

// TestNetworkingDeepCopy tests Networking deep copy
func TestNetworkingDeepCopy(t *testing.T) {
	original := &Networking{
		DNSDomain: "cluster.local",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify all fields
	if copied.ServiceSubnet != original.ServiceSubnet {
		t.Errorf("Expected ServiceSubnet %s, got %s", original.ServiceSubnet, copied.ServiceSubnet)
	}
	if copied.PodSubnet != original.PodSubnet {
		t.Errorf("Expected PodSubnet %s, got %s", original.PodSubnet, copied.PodSubnet)
	}
	if copied.DNSDomain != original.DNSDomain {
		t.Errorf("Expected DNSDomain %s, got %s", original.DNSDomain, copied.DNSDomain)
	}

}

// TestControlPlaneDeepCopy tests ControlPlane deep copy
func TestControlPlaneDeepCopy(t *testing.T) {
	original := &ControlPlane{
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
				Port: 6443,
			},
		},
		Etcd: &Etcd{
			DataDir:        "/var/lib/openFuyao/etcd",
			ServerCertSANs: []string{"etcd.example.com"},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify ControllerManager pointer deep copy
	if copied.ControllerManager == original.ControllerManager {
		t.Error("ControllerManager should be different instances")
	}
	if copied.ControllerManager.ExtraArgs["bind-address"] != original.ControllerManager.ExtraArgs["bind-address"] {
		t.Error("ControllerManager ExtraArgs values should match original")
	}

	// Verify Scheduler pointer deep copy
	if copied.Scheduler == original.Scheduler {
		t.Error("Scheduler should be different instances")
	}

	// Verify APIServer pointer deep copy
	if copied.APIServer == original.APIServer {
		t.Error("APIServer should be different instances")
	}

	// Verify Etcd pointer deep copy
	if copied.Etcd == original.Etcd {
		t.Error("Etcd should be different instances")
	}
	if copied.Etcd.DataDir != original.Etcd.DataDir {
		t.Errorf("Expected Etcd DataDir %s, got %s", original.Etcd.DataDir, copied.Etcd.DataDir)
	}

}

// TestEtcdDeepCopy tests Etcd deep copy
func TestEtcdDeepCopy(t *testing.T) {
	original := &Etcd{
		DataDir:               "/var/lib/openFuyao/etcd",
		ControlPlaneComponent: ControlPlaneComponent{},
		ServerCertSANs:        []string{"etcd1", "etcd2"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.DataDir != original.DataDir {
		t.Errorf("Expected DataDir %s, got %s", original.DataDir, copied.DataDir)
	}

}

// TestKubeletDeepCopy tests Kubelet deep copy
func TestKubeletDeepCopy(t *testing.T) {
	original := &Kubelet{
		ManifestsDir: "/etc/kubernetes/manifests",
		ControlPlaneComponent: ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"feature-gates": "RotateKubeletServerCertificate=true",
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.ManifestsDir != original.ManifestsDir {
		t.Errorf("Expected ManifestsDir %s, got %s", original.ManifestsDir, copied.ManifestsDir)
	}

	// Modify copied ExtraArgs
	copied.ExtraArgs["new-arg"] = "new-value"
	if _, exists := original.ExtraArgs["new-arg"]; exists {
		t.Error("Modifying copied Kubelet ExtraArgs affected original")
	}
}

// TestLabelDeepCopy tests Label deep copy
func TestLabelDeepCopy(t *testing.T) {
	original := &Label{
		Key:   "environment",
		Value: "production",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify fields
	if copied.Key != original.Key {
		t.Errorf("Expected Key %s, got %s", original.Key, copied.Key)
	}
	if copied.Value != original.Value {
		t.Errorf("Expected Value %s, got %s", original.Value, copied.Value)
	}

	// Modify copied values to verify independence
	copied.Value = "development"
	if original.Value == copied.Value {
		t.Error("Modifying copied label value affected original")
	}
}

// TestContainerRuntimeDeepCopy tests ContainerRuntime deep copy
func TestContainerRuntimeDeepCopy(t *testing.T) {
	original := &ContainerRuntime{
		CRI:     "containerd",
		Runtime: "runc",
		Param: map[string]string{
			"data-root": "/var/lib/containerd",
			"cgroupDir": "/containerd",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.CRI != original.CRI {
		t.Errorf("Expected CRI %s, got %s", original.CRI, copied.CRI)
	}
	if copied.Runtime != original.Runtime {
		t.Errorf("Expected Runtime %s, got %s", original.Runtime, copied.Runtime)
	}
	// Modify copied Param
	copied.Param["new-param"] = "new-value"
	if _, exists := original.Param["new-param"]; exists {
		t.Error("Modifying copied ContainerRuntime Param affected original")
	}
}

// TestBKEClusterSpecDeepCopy tests BKEClusterSpec deep copy
func TestBKEClusterSpecDeepCopy(t *testing.T) {
	original := &BKEClusterSpec{
		ControlPlaneEndpoint: APIEndpoint{
			Port: 6443,
		},
		ClusterConfig: &BKEConfig{
			Cluster: createTestBKEConfigCluster(),
		},
		KubeletConfigRef: &KubeletConfigRef{
			Name:      "default-kubelet-config",
			Namespace: "kube-system",
		},
		Pause:  false,
		DryRun: true,
		Reset:  false,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify basic fields
	if copied.Pause != original.Pause {
		t.Errorf("Expected Pause %t, got %t", original.Pause, copied.Pause)
	}

	// Verify ControlPlaneEndpoint deep copy
	if copied.ControlPlaneEndpoint.Host != original.ControlPlaneEndpoint.Host {
		t.Errorf("Expected ControlPlaneEndpoint Host %s, got %s",
			original.ControlPlaneEndpoint.Host, copied.ControlPlaneEndpoint.Host)
	}

	// Verify ClusterConfig pointer deep copy
	if copied.ClusterConfig == original.ClusterConfig {
		t.Error("ClusterConfig should be different instances")
	}
	if copied.ClusterConfig.Cluster.KubernetesVersion != original.ClusterConfig.Cluster.KubernetesVersion {
		t.Errorf("Expected ClusterConfig KubernetesVersion %s, got %s",
			original.ClusterConfig.Cluster.KubernetesVersion, copied.ClusterConfig.Cluster.KubernetesVersion)
	}

	// Verify KubeletConfigRef pointer deep copy
	if copied.KubeletConfigRef == original.KubeletConfigRef {
		t.Error("KubeletConfigRef should be different instances")
	}
	if copied.KubeletConfigRef.Name != original.KubeletConfigRef.Name {
		t.Errorf("Expected KubeletConfigRef Name %s, got %s",
			original.KubeletConfigRef.Name, copied.KubeletConfigRef.Name)
	}
}

// TestKubeletConfigRefDeepCopy tests KubeletConfigRef deep copy
func TestKubeletConfigRefDeepCopy(t *testing.T) {
	original := &KubeletConfigRef{
		Name:      "kubelet-config",
		Namespace: "kube-system",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify fields
	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}
	if copied.Namespace != original.Namespace {
		t.Errorf("Expected Namespace %s, got %s", original.Namespace, copied.Namespace)
	}

	// Modify copied values to verify independence
	copied.Name = "new-kubelet-config"
	if original.Name == copied.Name {
		t.Error("Modifying copied name affected original")
	}
}

// TestEmptyAndNilCases tests empty values and nil cases
func TestEmptyAndNilCases(t *testing.T) {
	t.Run("EmptyAddonSlice", func(t *testing.T) {
		original := &BKEConfig{
			Addons: []Product{},
		}
		copied := original.DeepCopy()
		if len(copied.Addons) != 0 {
			t.Error("Empty slice should remain empty after copy")
		}
	})

	t.Run("NilMaps", func(t *testing.T) {
		original := &BKEConfig{
			CustomExtra: nil,
		}
		copied := original.DeepCopy()
		if copied.CustomExtra != nil {
			t.Error("Nil map should remain nil after copy")
		}
	})

	t.Run("NilSlices", func(t *testing.T) {
		original := &BKEConfig{
			Addons: nil,
		}
		copied := original.DeepCopy()
		if copied.Addons != nil {
			t.Error("Nil slice should remain nil after copy")
		}
	})
}

// TestRuntimeObjectImplementation tests runtime.Object interface implementation
func TestRuntimeObjectImplementation(t *testing.T) {
	// Test BKECluster DeepCopyObject implementation
	cluster := &BKECluster{
		TypeMeta: metav1.TypeMeta{
			Kind:       "BKECluster",
			APIVersion: "cluster.bke.io/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: BKEClusterSpec{
			Pause: false,
		},
	}

	obj := cluster.DeepCopyObject()
	if obj == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	clusterObj, ok := obj.(*BKECluster)
	if !ok {
		t.Error("DeepCopyObject should return a BKECluster")
	}

	if clusterObj.Name != cluster.Name {
		t.Errorf("Expected Name %s, got %s", cluster.Name, clusterObj.Name)
	}

	// Ensure different instances
	if clusterObj == cluster {
		t.Error("DeepCopyObject should return a different instance")
	}

	// Test BKEClusterList DeepCopyObject implementation
	list := &BKEClusterList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "BKEClusterList",
			APIVersion: "cluster.bke.io/v1beta1",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "12345",
		},
		Items: []BKECluster{*cluster},
	}

	listObj := list.DeepCopyObject()
	if listObj == nil {
		t.Error("DeepCopyObject should not return nil for list")
	}

	listClusterObj, ok := listObj.(*BKEClusterList)
	if !ok {
		t.Error("DeepCopyObject should return a BKEClusterList")
	}

	if listClusterObj.ResourceVersion != list.ResourceVersion {
		t.Errorf("Expected ResourceVersion %s, got %s", list.ResourceVersion, listClusterObj.ResourceVersion)
	}
}

// TestDeepCopyInto tests DeepCopyInto method
func TestDeepCopyInto(t *testing.T) {
	original := &APIEndpoint{
		Host: "original.host",
		Port: 1234,
	}

	copied := &APIEndpoint{}
	original.DeepCopyInto(copied)

	if copied.Host != original.Host {
		t.Errorf("Expected Host %s, got %s", original.Host, copied.Host)
	}
	if copied.Port != original.Port {
		t.Errorf("Expected Port %d, got %d", original.Port, copied.Port)
	}

	// Modify copied values to verify independence
	copied.Host = "modified.host"
	if original.Host == copied.Host {
		t.Error("Modifying copy affected original in DeepCopyInto")
	}
}

// TestTimeRelatedStructs tests structures that may contain time-related fields
func TestTimeRelatedStructs(t *testing.T) {
	// Create a metav1.ObjectMeta with time for testing
	now := metav1.Now()
	original := &BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-cluster",
			CreationTimestamp: now,
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	// Time should also be properly copied
	if !copied.CreationTimestamp.Time.Equal(original.CreationTimestamp.Time) {
		t.Error("CreationTimestamp was not properly copied")
	}
}

// TestBKEConfigControlPlaneDeepCopy tests deep copy of ControlPlane in BKEConfig
func TestBKEConfigControlPlaneDeepCopy(t *testing.T) {
	original := &BKEConfig{
		Cluster: Cluster{
			ControlPlane: ControlPlane{
				ControllerManager: &ControlPlaneComponent{
					ExtraArgs: map[string]string{
						"feature-gates": "DynamicKubeletConfig=true",
						"bind-address":  "0.0.0.0",
					},
					ExtraVolumes: []HostPathMount{
						{
							Name:      "config",
							HostPath:  "/etc/kubernetes/config",
							MountPath: "/etc/kubernetes/config",
							ReadOnly:  true,
						},
					},
				},
				Scheduler: &ControlPlaneComponent{
					ExtraArgs: map[string]string{
						"bind-address": "0.0.0.0",
					},
				},
				APIServer: &APIServer{
					CertSANs: []string{"api.example.com", "localhost"},
				},
				Etcd: &Etcd{
					ServerCertSANs: []string{"etcd1", "etcd2"},
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify deep copy of multiple nested structures
	controllerManager := copied.Cluster.ControlPlane.ControllerManager
	if controllerManager == original.Cluster.ControlPlane.ControllerManager {
		t.Error("ControllerManager should be different instances")
	}

	// Verify slices content is also deep copied
	controllerManager.ExtraArgs["new-arg"] = "new-value"
	if _, exists := original.Cluster.ControlPlane.ControllerManager.ExtraArgs["new-arg"]; exists {
		t.Error("Modifying copied nested structure affected original")
	}
}

// TestBKENodeSpecDeepCopy tests deep copy of BKENodeSpec with nested ControlPlane
func TestBKENodeSpecDeepCopy(t *testing.T) {
	original := &BKENode{
		Spec: BKENodeSpec{
			Role: []string{"master", "etcd"},
			IP:   "192.168.1.1",
			ControlPlane: ControlPlane{
				APIServer: &APIServer{
					APIEndpoint: APIEndpoint{
						Port: 6443,
					},
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify nested structures in BKENode
	if copied.Spec.ControlPlane.APIServer == original.Spec.ControlPlane.APIServer {
		t.Error("BKENode APIServer should be different instances")
	}
}

// TestAuthSecretRefDeepCopy tests AuthSecretRef deep copy
func TestAuthSecretRefDeepCopy(t *testing.T) {
	original := &AuthSecretRef{
		Name:        "secret",
		Namespace:   "default",
		UsernameKey: "username",
		PasswordKey: "password",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	// Modify copied values to verify independence
	copied.Name = "new-secret"
	if original.Name == copied.Name {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilAuth *AuthSecretRef
	if nilAuth.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKEAgentStatusDeepCopy tests BKEAgentStatus deep copy
func TestBKEAgentStatusDeepCopy(t *testing.T) {
	original := &BKEAgentStatus{
		Replies:            10,
		UnavailableReplies: 2,
		Status:             "8/10",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Replies != original.Replies {
		t.Errorf("Expected Replies %d, got %d", original.Replies, copied.Replies)
	}

	// Modify copied values to verify independence
	copied.Replies = 20
	if original.Replies == copied.Replies {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilStatus *BKEAgentStatus
	if nilStatus.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKEClusterPhasesDeepCopy tests BKEClusterPhases deep copy
func TestBKEClusterPhasesDeepCopy(t *testing.T) {
	original := BKEClusterPhases{
		"Provisioning",
		"Provisioned",
		"Running",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied) != len(original) {
		t.Errorf("Expected length %d, got %d", len(original), len(copied))
	}

	// Modify copied values to verify independence
	copied[0] = "Failed"
	if original[0] == copied[0] {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilPhases BKEClusterPhases
	if nilPhases.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeListDeepCopy tests BKENodeList deep copy
func TestBKENodeListDeepCopy(t *testing.T) {
	original := &BKENodeList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "BKENodeList",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "1000",
		},
		Items: []BKENode{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-1",
				},
				Spec: BKENodeSpec{
					IP: "192.168.1.1",
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-2",
				},
				Spec: BKENodeSpec{
					IP: "192.168.1.2",
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied.Items) != len(original.Items) {
		t.Errorf("Expected %d items, got %d", len(original.Items), len(copied.Items))
	}

	// Modify copied items to verify independence
	copied.Items[0].Name = "modified-node"
	if original.Items[0].Name == copied.Items[0].Name {
		t.Error("Modifying copied items should not affect original")
	}

	// Test DeepCopyObject
	obj := original.DeepCopyObject()
	if obj == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	// Test nil case
	var nilList *BKENodeList
	if nilList.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeStatusDeepCopy tests BKENodeStatus deep copy
func TestBKENodeStatusDeepCopy(t *testing.T) {
	original := &BKENodeStatus{
		State:     NodeReady,
		StateCode: 15,
		Message:   "Node is ready",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.State != original.State {
		t.Errorf("Expected State %s, got %s", original.State, copied.State)
	}

	// Modify copied values to verify independence
	copied.Message = "modified message"
	if original.Message == copied.Message {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilStatus *BKENodeStatus
	if nilStatus.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestClusterConditionWithTime tests ClusterCondition deep copy with LastTransitionTime
func TestClusterConditionWithTime(t *testing.T) {
	now := metav1.Now()
	original := &ClusterCondition{
		Type:               ClusterConditionType("Ready"),
		AddonName:          "calico",
		Status:             ConditionTrue,
		LastTransitionTime: &now,
		Reason:             "ClusterReady",
		Message:            "Cluster is ready",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Type != original.Type {
		t.Errorf("Expected Type %s, got %s", original.Type, copied.Type)
	}

	// Verify LastTransitionTime is properly deep copied
	if copied.LastTransitionTime == nil {
		t.Fatal("LastTransitionTime should not be nil")
	}
	if !copied.LastTransitionTime.Time.Equal(original.LastTransitionTime.Time) {
		t.Error("LastTransitionTime was not properly copied")
	}

	// Note: LastTransitionTime uses DeepCopy() which creates a new time value
	// so modification to copied won't affect original

	// Test nil LastTransitionTime
	originalNil := &ClusterCondition{
		Type:   ClusterConditionType("Ready"),
		Status: ConditionTrue,
	}
	copiedNil := originalNil.DeepCopy()
	if copiedNil.LastTransitionTime != nil {
		t.Error("LastTransitionTime should be nil when original is nil")
	}
}

// TestClusterConditionsSliceDeepCopy tests ClusterConditions slice deep copy
func TestClusterConditionsSliceDeepCopy(t *testing.T) {
	original := ClusterConditions{
		{
			Type:   ClusterConditionType("Ready"),
			Status: ConditionTrue,
		},
		{
			Type:   ClusterConditionType("Health"),
			Status: ConditionFalse,
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied) != len(original) {
		t.Errorf("Expected length %d, got %d", len(original), len(copied))
	}

	// Modify copied to verify independence
	copied[0].Status = ConditionFalse
	if original[0].Status == copied[0].Status {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilConditions ClusterConditions
	if nilConditions.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestCommandSpecDeepCopy tests CommandSpec deep copy
func TestCommandSpecDeepCopy(t *testing.T) {
	original := &CommandSpec{
		Command:    "/bin/bash",
		Args:       []string{"-c", "echo hello"},
		WorkingDir: "/home/user",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Command != original.Command {
		t.Errorf("Expected Command %s, got %s", original.Command, copied.Command)
	}

	// Verify Args slice is deep copied
	if len(copied.Args) != len(original.Args) {
		t.Errorf("Expected Args length %d, got %d", len(original.Args), len(copied.Args))
	}

	// Modify copied to verify independence
	copied.Args[0] = "-x"
	if original.Args[0] == copied.Args[0] {
		t.Error("Modifying copied Args should not affect original")
	}

	// Test nil Args
	originalNilArgs := &CommandSpec{
		Command: "/bin/bash",
	}
	copiedNilArgs := originalNilArgs.DeepCopy()
	if copiedNilArgs.Args != nil {
		t.Error("Args should be nil when original is nil")
	}

	// Test nil case
	var nilCmd *CommandSpec
	if nilCmd.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestFileSpecDeepCopy tests FileSpec deep copy
func TestFileSpecDeepCopy(t *testing.T) {
	original := &FileSpec{
		Path:        "/etc/config.conf",
		Content:     "key=value",
		Permissions: "0644",
		Owner:       "root:root",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Path != original.Path {
		t.Errorf("Expected Path %s, got %s", original.Path, copied.Path)
	}

	// Modify copied to verify independence
	copied.Content = "modified content"
	if original.Content == copied.Content {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilFile *FileSpec
	if nilFile.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestContainerdConfigSpecDeepCopy tests ContainerdConfigSpec deep copy
func TestContainerdConfigSpecDeepCopy(t *testing.T) {
	original := &ContainerdConfigSpec{
		ConfigType:  "combined",
		Description: "Test configuration",
		Service: &ServiceConfig{
			ExecStart: "/usr/bin/containerd --config /etc/containerd/config.toml",
			Slice:     "system.slice",
		},
		Main: &MainConfig{
			Root: "/var/lib/containerd",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.ConfigType != original.ConfigType {
		t.Errorf("Expected ConfigType %s, got %s", original.ConfigType, copied.ConfigType)
	}

	// Verify Service pointer is deep copied
	if copied.Service == original.Service {
		t.Error("Service should be different instances")
	}

	// Verify Main pointer is deep copied
	if copied.Main == original.Main {
		t.Error("Main should be different instances")
	}

	// Modify copied to verify independence
	copied.Service.ExecStart = "modified"
	if original.Service.ExecStart == copied.Service.ExecStart {
		t.Error("Modifying copied Service should not affect original")
	}

	// Test nil Service and Main
	originalNil := &ContainerdConfigSpec{
		ConfigType: "service",
	}
	copiedNil := originalNil.DeepCopy()
	if copiedNil.Service != nil {
		t.Error("Service should be nil when original is nil")
	}
	if copiedNil.Main != nil {
		t.Error("Main should be nil when original is nil")
	}

	// Test nil case
	var nilSpec *ContainerdConfigSpec
	if nilSpec.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeSpecNilCases tests BKENodeSpec deep copy with nil fields
func TestBKENodeSpecNilCases(t *testing.T) {
	original := &BKENodeSpec{
		Role: []string{"master"},
		IP:   "192.168.1.1",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Role slice is deep copied (compare with nil)
	if copied.Role == nil || original.Role == nil {
		t.Error("Role should not be nil")
	}
	if &copied.Role[0] == &original.Role[0] {
		t.Error("Role should be different instances")
	}

	// Modify copied to verify independence
	copied.Role[0] = "worker"
	if original.Role[0] == copied.Role[0] {
		t.Error("Modifying copied Role should not affect original")
	}

	// Test nil Role
	originalNilRole := &BKENodeSpec{
		IP: "192.168.1.1",
	}
	copiedNilRole := originalNilRole.DeepCopy()
	if copiedNilRole.Role != nil {
		t.Error("Role should be nil when original is nil")
	}

	// Test nil Kubelet
	originalNilKubelet := &BKENodeSpec{
		Role: []string{"master"},
		IP:   "192.168.1.1",
	}
	copiedNilKubelet := originalNilKubelet.DeepCopy()
	if copiedNilKubelet.Kubelet != nil {
		t.Error("Kubelet should be nil when original is nil")
	}

	// Test nil Labels
	originalNilLabels := &BKENodeSpec{
		Role: []string{"master"},
		IP:   "192.168.1.1",
	}
	copiedNilLabels := originalNilLabels.DeepCopy()
	if copiedNilLabels.Labels != nil {
		t.Error("Labels should be nil when original is nil")
	}
}

// TestClusterNilCases tests Cluster deep copy with nil fields
func TestClusterNilCases(t *testing.T) {
	original := &Cluster{
		KubernetesVersion: "v1.25.6",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil Kubelet
	if copied.Kubelet != nil {
		t.Error("Kubelet should be nil when original is nil")
	}

	// Test nil ContainerdConfigRef
	if copied.ContainerdConfigRef != nil {
		t.Error("ContainerdConfigRef should be nil when original is nil")
	}

	// Test nil Labels
	if copied.Labels != nil {
		t.Error("Labels should be nil when original is nil")
	}

	// Test nil ControlPlane
	originalNilCP := &Cluster{
		KubernetesVersion: "v1.25.6",
	}
	copiedNilCP := originalNilCP.DeepCopy()
	if copiedNilCP.ControlPlane.APIServer != nil {
		t.Error("ControlPlane.APIServer should be nil when original is nil")
	}
}

// TestBKEClusterStatusNilCases tests BKEClusterStatus deep copy with nil fields
func TestBKEClusterStatusNilCases(t *testing.T) {
	original := &BKEClusterStatus{
		Ready: true,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil AddonStatus
	if copied.AddonStatus != nil {
		t.Error("AddonStatus should be nil when original is nil")
	}

	// Test nil PhaseStatus
	if copied.PhaseStatus != nil {
		t.Error("PhaseStatus should be nil when original is nil")
	}

	// Test nil Conditions
	if copied.Conditions != nil {
		t.Error("Conditions should be nil when original is nil")
	}
}

// TestEtcdNilCases tests Etcd deep copy with nil fields
func TestEtcdNilCases(t *testing.T) {
	original := &Etcd{
		DataDir: "/var/lib/etcd",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil ServerCertSANs
	if copied.ServerCertSANs != nil {
		t.Error("ServerCertSANs should be nil when original is nil")
	}

	// Test nil PeerCertSANs
	if copied.PeerCertSANs != nil {
		t.Error("PeerCertSANs should be nil when original is nil")
	}

	// Test nil ExtraArgs
	if copied.ExtraArgs != nil {
		t.Error("ExtraArgs should be nil when original is nil")
	}

	// Test nil ExtraVolumes
	if copied.ExtraVolumes != nil {
		t.Error("ExtraVolumes should be nil when original is nil")
	}
}

// TestControlPlaneNilCases tests ControlPlane deep copy with nil fields
func TestControlPlaneNilCases(t *testing.T) {
	original := &ControlPlane{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil ControllerManager
	if copied.ControllerManager != nil {
		t.Error("ControllerManager should be nil when original is nil")
	}

	// Test nil Scheduler
	if copied.Scheduler != nil {
		t.Error("Scheduler should be nil when original is nil")
	}

	// Test nil APIServer
	if copied.APIServer != nil {
		t.Error("APIServer should be nil when original is nil")
	}

	// Test nil Etcd
	if copied.Etcd != nil {
		t.Error("Etcd should be nil when original is nil")
	}
}

// TestControlPlaneComponentNilCases tests ControlPlaneComponent deep copy with nil fields
func TestControlPlaneComponentNilCases(t *testing.T) {
	original := &ControlPlaneComponent{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil ExtraArgs
	if copied.ExtraArgs != nil {
		t.Error("ExtraArgs should be nil when original is nil")
	}

	// Test nil ExtraVolumes
	if copied.ExtraVolumes != nil {
		t.Error("ExtraVolumes should be nil when original is nil")
	}
}

// TestContainerRuntimeNilCases tests ContainerRuntime deep copy with nil fields
func TestContainerRuntimeNilCases(t *testing.T) {
	original := &ContainerRuntime{
		CRI: "containerd",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil Param
	if copied.Param != nil {
		t.Error("Param should be nil when original is nil")
	}
}

// TestBKENodeDeepCopyNilCases tests BKENode deep copy with nil Status
func TestBKENodeDeepCopyNilCases(t *testing.T) {
	original := &BKENode{
		Spec: BKENodeSpec{
			IP: "192.168.1.1",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test that Spec is properly copied
	if copied.Spec.IP != original.Spec.IP {
		t.Errorf("Expected IP %s, got %s", original.Spec.IP, copied.Spec.IP)
	}
}

// TestBKEClusterListNilItems tests BKEClusterList with empty items
func TestBKEClusterListNilItems(t *testing.T) {
	original := &BKEClusterList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "BKEClusterList",
		},
		Items: nil,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Items is nil
	if copied.Items != nil {
		t.Error("Items should be nil when original is nil")
	}
}

// TestBKENodeListNilItems tests BKENodeList with empty items
func TestBKENodeListNilItems(t *testing.T) {
	original := &BKENodeList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "BKENodeList",
		},
		Items: nil,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Items is nil
	if copied.Items != nil {
		t.Error("Items should be nil when original is nil")
	}
}

// TestBKEConfigNilCases tests BKEConfig deep copy with nil fields
func TestBKEConfigNilCases(t *testing.T) {
	original := &BKEConfig{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil Cluster
	if copied.Cluster.KubernetesVersion != "" {
		t.Error("Cluster should have default values")
	}

	// Test nil Addons
	if copied.Addons != nil {
		t.Error("Addons should be nil when original is nil")
	}

	// Test nil CustomExtra
	if copied.CustomExtra != nil {
		t.Error("CustomExtra should be nil when original is nil")
	}
}

// TestBKEClusterSpecNilCases tests BKEClusterSpec deep copy with nil fields
func TestBKEClusterSpecNilCases(t *testing.T) {
	original := &BKEClusterSpec{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil ClusterConfig
	if copied.ClusterConfig != nil {
		t.Error("ClusterConfig should be nil when original is nil")
	}

	// Test nil KubeletConfigRef
	if copied.KubeletConfigRef != nil {
		t.Error("KubeletConfigRef should be nil when original is nil")
	}
}

// TestProductNilCases tests Product deep copy with nil fields
func TestProductNilCases(t *testing.T) {
	original := &Product{
		Name: "test-product",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil Param
	if copied.Param != nil {
		t.Error("Param should be nil when original is nil")
	}

	// Test nil ValuesConfigMapRef
	if copied.ValuesConfigMapRef != nil {
		t.Error("ValuesConfigMapRef should be nil when original is nil")
	}
}

// TestServiceLoggingDeepCopy tests ServiceLogging deep copy
func TestServiceLoggingDeepCopy(t *testing.T) {
	original := &ServiceLogging{
		StandardOutput: "journal",
		StandardError:  "journal",
		LogLevelMax:    "info",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.StandardOutput != original.StandardOutput {
		t.Errorf("Expected StandardOutput %s, got %s", original.StandardOutput, copied.StandardOutput)
	}

	// Modify copied to verify independence
	copied.StandardOutput = "modified"
	if original.StandardOutput == copied.StandardOutput {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilLogging *ServiceLogging
	if nilLogging.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestTLSConfigDeepCopy tests TLSConfig deep copy
func TestTLSConfigDeepCopy(t *testing.T) {
	original := &TLSConfig{
		InsecureSkipVerify: true,
		CAFile:             "/etc/ssl/certs/ca.crt",
		CertFile:           "/etc/ssl/certs/tls.crt",
		KeyFile:            "/etc/ssl/private/tls.key",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.CAFile != original.CAFile {
		t.Errorf("Expected CAFile %s, got %s", original.CAFile, copied.CAFile)
	}

	// Modify copied to verify independence
	copied.CAFile = "modified"
	if original.CAFile == copied.CAFile {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilTLS *TLSConfig
	if nilTLS.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestRegistryHostConfigDeepCopy tests RegistryHostConfig deep copy
func TestRegistryHostConfigDeepCopy(t *testing.T) {
	original := &RegistryHostConfig{
		Host:         "https://docker.io",
		Capabilities: []string{"pull", "resolve"},
		TLS: &TLSConfig{
			InsecureSkipVerify: false,
		},
		Auth: &RegistryAuthConfig{
			Username: "user",
			Password: "pass",
		},
		Header: map[string][]string{
			"User-Agent": {"containerd/1.0"},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Host != original.Host {
		t.Errorf("Expected Host %s, got %s", original.Host, copied.Host)
	}

	// Verify nested pointers are deep copied
	if copied.TLS == original.TLS {
		t.Error("TLS should be different instances")
	}
	if copied.Auth == original.Auth {
		t.Error("Auth should be different instances")
	}

	// Verify Header map is deep copied
	if copied.Header == nil || original.Header == nil {
		t.Error("Header should not be nil")
	}
	if len(copied.Header) != len(original.Header) {
		t.Error("Header length should match")
	}

	// Modify copied to verify independence
	copied.TLS.InsecureSkipVerify = true
	if original.TLS.InsecureSkipVerify == copied.TLS.InsecureSkipVerify {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilHost *RegistryHostConfig
	if nilHost.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestRegistryAuthConfigDeepCopy tests RegistryAuthConfig deep copy
func TestRegistryAuthConfigDeepCopy(t *testing.T) {
	original := &RegistryAuthConfig{
		Username: "user",
		Password: "password",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Username != original.Username {
		t.Errorf("Expected Username %s, got %s", original.Username, copied.Username)
	}

	// Modify copied to verify independence
	copied.Username = "modified"
	if original.Username == copied.Username {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilAuth *RegistryAuthConfig
	if nilAuth.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestMainConfigDeepCopy tests MainConfig deep copy
func TestMainConfigDeepCopy(t *testing.T) {
	original := &MainConfig{
		MetricsAddress: "0.0.0.0:1338",
		Root:           "/var/lib/containerd",
		State:          "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Root != original.Root {
		t.Errorf("Expected Root %s, got %s", original.Root, copied.Root)
	}

	// Modify copied to verify independence
	copied.Root = "modified"
	if original.Root == copied.Root {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilMain *MainConfig
	if nilMain.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestPhaseStatusDeepCopy tests PhaseStatus deep copy
func TestPhaseStatusDeepCopy(t *testing.T) {
	now := metav1.Now()
	original := PhaseStatus{
		{
			Name:      BKEClusterPhase("Provisioning"),
			StartTime: &now,
			Status:    BKEClusterPhaseStatus("InProgress"),
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied) != len(original) {
		t.Errorf("Expected length %d, got %d", len(original), len(copied))
	}

	// Modify copied to verify independence
	copied[0].Status = BKEClusterPhaseStatus("Success")
	if original[0].Status == copied[0].Status {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilPhaseStatus PhaseStatus
	if nilPhaseStatus.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestPhaseStateDeepCopy tests PhaseState deep copy
func TestPhaseStateDeepCopy(t *testing.T) {
	now := metav1.Now()
	original := &PhaseState{
		Name:      BKEClusterPhase("Provisioned"),
		StartTime: &now,
		Status:    BKEClusterPhaseStatus("Success"),
		Message:   "Phase completed",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	// Verify StartTime is properly copied
	if copied.StartTime == nil {
		t.Fatal("StartTime should not be nil")
	}
	if !copied.StartTime.Time.Equal(original.StartTime.Time) {
		t.Error("StartTime was not properly copied")
	}

	// Test nil StartTime
	originalNil := &PhaseState{
		Name:   BKEClusterPhase("Provisioning"),
		Status: BKEClusterPhaseStatus("InProgress"),
	}
	copiedNil := originalNil.DeepCopy()
	if copiedNil.StartTime != nil {
		t.Error("StartTime should be nil when original is nil")
	}

	// Test nil case
	var nilPhaseState *PhaseState
	if nilPhaseState.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletConfigDeepCopy tests KubeletConfig deep copy
func TestKubeletConfigDeepCopy(t *testing.T) {
	original := &KubeletConfig{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "KubeletConfig",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-kubelet-config",
			Namespace: "default",
		},
		Spec: KubeletConfigSpec{
			KubeletConfig: map[string]runtime.RawExtension{
				"kubelet": {
					Raw: []byte(`{"maxPods":110}`),
				},
			},
			KubeletService: &KubeletServiceSpec{
				ServiceName: "kubelet",
				Enabled:     true,
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	// Verify Spec is deep copied
	if copied.Spec.KubeletService == original.Spec.KubeletService {
		t.Error("KubeletService should be different instances")
	}

	// Test DeepCopyObject
	obj := original.DeepCopyObject()
	if obj == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	kubeletConfigObj, ok := obj.(*KubeletConfig)
	if !ok {
		t.Error("DeepCopyObject should return a KubeletConfig")
	}

	if kubeletConfigObj.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, kubeletConfigObj.Name)
	}

	// Test nil case
	var nilConfig *KubeletConfig
	if nilConfig.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletConfigListDeepCopy tests KubeletConfigList deep copy
func TestKubeletConfigListDeepCopy(t *testing.T) {
	original := &KubeletConfigList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "KubeletConfigList",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "1000",
		},
		Items: []KubeletConfig{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "config-1",
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "config-2",
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied.Items) != len(original.Items) {
		t.Errorf("Expected %d items, got %d", len(original.Items), len(copied.Items))
	}

	// Verify items are deep copied
	copied.Items[0].Name = "modified"
	if original.Items[0].Name == copied.Items[0].Name {
		t.Error("Modifying copied items should not affect original")
	}

	// Test DeepCopyObject
	obj := original.DeepCopyObject()
	if obj == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	listObj, ok := obj.(*KubeletConfigList)
	if !ok {
		t.Error("DeepCopyObject should return a KubeletConfigList")
	}

	if listObj.ResourceVersion != original.ResourceVersion {
		t.Errorf("Expected ResourceVersion %s, got %s", original.ResourceVersion, listObj.ResourceVersion)
	}

	// Test nil case
	var nilList *KubeletConfigList
	if nilList.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletConfigListNilItems tests KubeletConfigList with nil items
func TestKubeletConfigListNilItems(t *testing.T) {
	original := &KubeletConfigList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "KubeletConfigList",
		},
		Items: nil,
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Items != nil {
		t.Error("Items should be nil when original is nil")
	}
}

// TestKubeletConfigSpecDeepCopy tests KubeletConfigSpec deep copy
func TestKubeletConfigSpecDeepCopy(t *testing.T) {
	original := &KubeletConfigSpec{
		KubeletConfig: map[string]runtime.RawExtension{
			"kubelet": {
				Raw: []byte(`{"maxPods":110}`),
			},
		},
		KubeletService: &KubeletServiceSpec{
			ServiceName: "kubelet",
			Enabled:     true,
			Unit: KubeletUnit{
				Description: "kubelet service",
				After:       []string{"network.target"},
				Wants:       []string{"network-online.target"},
			},
			Service: KubeletService{
				ExecStart:   "/usr/bin/kubelet",
				Environment: []string{"KUBELET_EXTRA_ARGS=--container-runtime=remote"},
			},
			Install: KubeletInstall{
				WantedBy:   []string{"multi-user.target"},
				RequiredBy: []string{"kubelet.service"},
			},
			Variables: map[string]string{
				"kubelet_version": "v1.25.6",
			},
		},
		Files: []FileSpec{
			{
				Path:        "/etc/kubernetes/kubelet.conf",
				Content:     "server: https://localhost:6443",
				Permissions: "0600",
			},
		},
		Commands: []CommandSpec{
			{
				Command:    "/bin/bash",
				Args:       []string{"-c", "echo configured"},
				WorkingDir: "/tmp",
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify KubeletService pointer is deep copied
	if copied.KubeletService == original.KubeletService {
		t.Error("KubeletService should be different instances")
	}

	// Verify Files slice is deep copied
	if len(copied.Files) != len(original.Files) {
		t.Errorf("Expected Files length %d, got %d", len(original.Files), len(copied.Files))
	}

	// Verify Commands slice is deep copied
	if len(copied.Commands) != len(original.Commands) {
		t.Errorf("Expected Commands length %d, got %d", len(original.Commands), len(copied.Commands))
	}

	// Modify copied to verify independence
	copied.KubeletService.ServiceName = "modified-kubelet"
	if original.KubeletService.ServiceName == copied.KubeletService.ServiceName {
		t.Error("Modifying copy affected original")
	}

	// Test nil KubeletService
	originalNilService := &KubeletConfigSpec{
		KubeletConfig: map[string]runtime.RawExtension{},
	}
	copiedNilService := originalNilService.DeepCopy()
	if copiedNilService.KubeletService != nil {
		t.Error("KubeletService should be nil when original is nil")
	}

	// Test nil Files
	originalNilFiles := &KubeletConfigSpec{
		KubeletConfig: map[string]runtime.RawExtension{},
	}
	copiedNilFiles := originalNilFiles.DeepCopy()
	if copiedNilFiles.Files != nil {
		t.Error("Files should be nil when original is nil")
	}

	// Test nil Commands
	originalNilCommands := &KubeletConfigSpec{
		KubeletConfig: map[string]runtime.RawExtension{},
	}
	copiedNilCommands := originalNilCommands.DeepCopy()
	if copiedNilCommands.Commands != nil {
		t.Error("Commands should be nil when original is nil")
	}

	// Test nil case
	var nilSpec *KubeletConfigSpec
	if nilSpec.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletInstallDeepCopy tests KubeletInstall deep copy
func TestKubeletInstallDeepCopy(t *testing.T) {
	original := &KubeletInstall{
		WantedBy:   []string{"multi-user.target"},
		RequiredBy: []string{"kubelet.service"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if len(copied.WantedBy) != len(original.WantedBy) {
		t.Errorf("Expected WantedBy length %d, got %d", len(original.WantedBy), len(copied.WantedBy))
	}

	// Modify copied to verify independence
	copied.WantedBy[0] = "modified-target"
	if original.WantedBy[0] == copied.WantedBy[0] {
		t.Error("Modifying copy affected original")
	}

	// Test nil WantedBy
	originalNilWanted := &KubeletInstall{
		RequiredBy: []string{"kubelet.service"},
	}
	copiedNilWanted := originalNilWanted.DeepCopy()
	if copiedNilWanted.WantedBy != nil {
		t.Error("WantedBy should be nil when original is nil")
	}

	// Test nil RequiredBy
	originalNilRequired := &KubeletInstall{
		WantedBy: []string{"multi-user.target"},
	}
	copiedNilRequired := originalNilRequired.DeepCopy()
	if copiedNilRequired.RequiredBy != nil {
		t.Error("RequiredBy should be nil when original is nil")
	}

	// Test nil case
	var nilInstall *KubeletInstall
	if nilInstall.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletServiceDeepCopy tests KubeletService deep copy
func TestKubeletServiceDeepCopy(t *testing.T) {
	original := &KubeletService{
		ExecStart:       "/usr/bin/kubelet",
		Restart:         "always",
		Environment:     []string{"KUBELET_EXTRA_ARGS=--container-runtime=remote"},
		EnvironmentFile: []string{"/etc/kubernetes/kubelet.env"},
		ExecStartPre:    []string{"/bin/mkdir -p /etc/kubernetes/manifests"},
		CustomExtra: map[string]string{
			"custom-arg": "value",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.ExecStart != original.ExecStart {
		t.Errorf("Expected ExecStart %s, got %s", original.ExecStart, copied.ExecStart)
	}

	// Verify Environment slice is deep copied
	if len(copied.Environment) != len(original.Environment) {
		t.Errorf("Expected Environment length %d, got %d", len(original.Environment), len(copied.Environment))
	}

	// Verify CustomExtra map is deep copied
	if original.CustomExtra != nil && len(copied.CustomExtra) != len(original.CustomExtra) {
		t.Error("CustomExtra should have same length")
	}

	// Modify copied to verify independence
	copied.CustomExtra["new-key"] = "new-value"
	if _, exists := original.CustomExtra["new-key"]; exists {
		t.Error("Modifying copy affected original")
	}

	// Test nil Environment
	originalNilEnv := &KubeletService{
		ExecStart: "/usr/bin/kubelet",
	}
	copiedNilEnv := originalNilEnv.DeepCopy()
	if copiedNilEnv.Environment != nil {
		t.Error("Environment should be nil when original is nil")
	}

	// Test nil EnvironmentFile
	originalNilEnvFile := &KubeletService{
		ExecStart:   "/usr/bin/kubelet",
		Environment: []string{},
	}
	copiedNilEnvFile := originalNilEnvFile.DeepCopy()
	if copiedNilEnvFile.EnvironmentFile != nil {
		t.Error("EnvironmentFile should be nil when original is nil")
	}

	// Test nil ExecStartPre
	originalNilPre := &KubeletService{
		ExecStart:   "/usr/bin/kubelet",
		Environment: []string{},
	}
	copiedNilPre := originalNilPre.DeepCopy()
	if copiedNilPre.ExecStartPre != nil {
		t.Error("ExecStartPre should be nil when original is nil")
	}

	// Test nil CustomExtra
	originalNilExtra := &KubeletService{
		ExecStart:   "/usr/bin/kubelet",
		Environment: []string{},
	}
	copiedNilExtra := originalNilExtra.DeepCopy()
	if copiedNilExtra.CustomExtra != nil {
		t.Error("CustomExtra should be nil when original is nil")
	}

	// Test nil case
	var nilService *KubeletService
	if nilService.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletServiceSpecDeepCopy tests KubeletServiceSpec deep copy
func TestKubeletServiceSpecDeepCopy(t *testing.T) {
	original := &KubeletServiceSpec{
		Enabled:     true,
		ServiceName: "kubelet",
		Unit: KubeletUnit{
			Description: "kubelet service",
			After:       []string{"network.target"},
			Wants:       []string{"network-online.target"},
			Requires:    []string{"network-online.target"},
		},
		Service: KubeletService{
			ExecStart: "/usr/bin/kubelet",
		},
		Install: KubeletInstall{
			WantedBy: []string{"multi-user.target"},
		},
		Variables: map[string]string{
			"kubelet_version": "v1.25.6",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.ServiceName != original.ServiceName {
		t.Errorf("Expected ServiceName %s, got %s", original.ServiceName, copied.ServiceName)
	}

	// Verify Unit is deep copied
	if copied.Unit.Description != original.Unit.Description {
		t.Errorf("Expected Unit Description %s, got %s", original.Unit.Description, copied.Unit.Description)
	}

	// Verify Variables map is deep copied
	if original.Variables != nil && len(copied.Variables) != len(original.Variables) {
		t.Error("Variables should have same length")
	}

	// Modify copied to verify independence
	copied.Variables["new-key"] = "new-value"
	if _, exists := original.Variables["new-key"]; exists {
		t.Error("Modifying copy affected original")
	}

	// Test nil Variables
	originalNilVars := &KubeletServiceSpec{
		ServiceName: "kubelet",
	}
	copiedNilVars := originalNilVars.DeepCopy()
	if copiedNilVars.Variables != nil {
		t.Error("Variables should be nil when original is nil")
	}

	// Test nil case
	var nilSpec *KubeletServiceSpec
	if nilSpec.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletUnitDeepCopy tests KubeletUnit deep copy
func TestKubeletUnitDeepCopy(t *testing.T) {
	original := &KubeletUnit{
		Description:   "kubelet service",
		Documentation: "https://kubernetes.io/docs/reference/generated/kubelet",
		After:         []string{"network.target", "local-fs.target"},
		Wants:         []string{"network-online.target"},
		Requires:      []string{"network-online.target"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Description != original.Description {
		t.Errorf("Expected Description %s, got %s", original.Description, copied.Description)
	}

	// Verify After slice is deep copied
	if len(copied.After) != len(original.After) {
		t.Errorf("Expected After length %d, got %d", len(original.After), len(copied.After))
	}

	// Modify copied to verify independence
	copied.After[0] = "modified-target"
	if original.After[0] == copied.After[0] {
		t.Error("Modifying copy affected original")
	}

	// Test nil After
	originalNilAfter := &KubeletUnit{
		Description: "kubelet service",
	}
	copiedNilAfter := originalNilAfter.DeepCopy()
	if copiedNilAfter.After != nil {
		t.Error("After should be nil when original is nil")
	}

	// Test nil Wants
	originalNilWants := &KubeletUnit{
		Description: "kubelet service",
		After:       []string{},
	}
	copiedNilWants := originalNilWants.DeepCopy()
	if copiedNilWants.Wants != nil {
		t.Error("Wants should be nil when original is nil")
	}

	// Test nil Requires
	originalNilRequires := &KubeletUnit{
		Description: "kubelet service",
		After:       []string{},
		Wants:       []string{},
	}
	copiedNilRequires := originalNilRequires.DeepCopy()
	if copiedNilRequires.Requires != nil {
		t.Error("Requires should be nil when original is nil")
	}

	// Test nil case
	var nilUnit *KubeletUnit
	if nilUnit.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestContainerdConfigRefDeepCopy tests ContainerdConfigRef deep copy
func TestContainerdConfigRefDeepCopy(t *testing.T) {
	original := &ContainerdConfigRef{
		Name:      "containerd-config",
		Namespace: "kube-system",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	if copied.Namespace != original.Namespace {
		t.Errorf("Expected Namespace %s, got %s", original.Namespace, copied.Namespace)
	}

	// Modify copied to verify independence
	copied.Name = "modified-config"
	if original.Name == copied.Name {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilRef *ContainerdConfigRef
	if nilRef.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestTlsSecretRefDeepCopy tests TlsSecretRef deep copy
func TestTlsSecretRefDeepCopy(t *testing.T) {
	original := &TlsSecretRef{
		Name:      "tls-secret",
		Namespace: "default",
		CaKey:     "ca.crt",
		CertKey:   "tls.crt",
		KeyKey:    "tls.key",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	if copied.CertKey != original.CertKey {
		t.Errorf("Expected CertKey %s, got %s", original.CertKey, copied.CertKey)
	}

	// Modify copied to verify independence
	copied.Name = "modified-secret"
	if original.Name == copied.Name {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilRef *TlsSecretRef
	if nilRef.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestValuesConfigMapRefDeepCopy tests ValuesConfigMapRef deep copy
func TestValuesConfigMapRefDeepCopy(t *testing.T) {
	original := &ValuesConfigMapRef{
		Name:      "values-config",
		Namespace: "default",
		ValuesKey: "values.yaml",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}

	if copied.ValuesKey != original.ValuesKey {
		t.Errorf("Expected ValuesKey %s, got %s", original.ValuesKey, copied.ValuesKey)
	}

	// Modify copied to verify independence
	copied.Name = "modified-config"
	if original.Name == copied.Name {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilRef *ValuesConfigMapRef
	if nilRef.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeDeepCopyObject tests BKENode DeepCopyObject
func TestBKENodeDeepCopyObject(t *testing.T) {
	original := &BKENode{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "BKENode",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-node",
			Namespace: "default",
		},
		Spec: BKENodeSpec{
			IP: "192.168.1.1",
		},
	}

	obj := original.DeepCopyObject()
	if obj == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	nodeObj, ok := obj.(*BKENode)
	if !ok {
		t.Error("DeepCopyObject should return a BKENode")
	}

	if nodeObj.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, nodeObj.Name)
	}

	// Ensure different instances
	if nodeObj == original {
		t.Error("DeepCopyObject should return a different instance")
	}

	// Test nil case
	var nilNode *BKENode
	if nilNode.DeepCopyObject() != nil {
		t.Error("DeepCopyObject of nil should return nil")
	}
}

// TestAPIServerWithCertSANs tests APIServer deep copy with CertSANs
func TestAPIServerWithCertSANs(t *testing.T) {
	original := &APIServer{
		APIEndpoint: APIEndpoint{
			Host: "api.example.com",
			Port: 6443,
		},
		ControlPlaneComponent: ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"bind-address": "0.0.0.0",
			},
		},
		CertSANs: []string{"api.example.com", "192.168.1.100"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify CertSANs slice is deep copied
	if len(copied.CertSANs) != len(original.CertSANs) {
		t.Errorf("Expected CertSANs length %d, got %d", len(original.CertSANs), len(copied.CertSANs))
	}

	// Modify copied to verify independence
	copied.CertSANs[0] = "modified.example.com"
	if original.CertSANs[0] == copied.CertSANs[0] {
		t.Error("Modifying copied CertSANs affected original")
	}

	// Test nil CertSANs
	originalNilSANs := &APIServer{
		APIEndpoint: APIEndpoint{
			Port: 6443,
		},
	}
	copiedNilSANs := originalNilSANs.DeepCopy()
	if copiedNilSANs.CertSANs != nil {
		t.Error("CertSANs should be nil when original is nil")
	}

	// Test nil case
	var nilServer *APIServer
	if nilServer.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestEtcdWithCertSANs tests Etcd deep copy with ServerCertSANs and PeerCertSANs
func TestEtcdWithCertSANs(t *testing.T) {
	original := &Etcd{
		DataDir:               "/var/lib/etcd",
		ControlPlaneComponent: ControlPlaneComponent{},
		ServerCertSANs:        []string{"etcd-0.example.com", "etcd-1.example.com"},
		PeerCertSANs:          []string{"etcd-0-peer.example.com", "etcd-1-peer.example.com"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify ServerCertSANs is deep copied
	if len(copied.ServerCertSANs) != len(original.ServerCertSANs) {
		t.Errorf("Expected ServerCertSANs length %d, got %d", len(original.ServerCertSANs), len(copied.ServerCertSANs))
	}

	// Verify PeerCertSANs is deep copied
	if len(copied.PeerCertSANs) != len(original.PeerCertSANs) {
		t.Errorf("Expected PeerCertSANs length %d, got %d", len(original.PeerCertSANs), len(copied.PeerCertSANs))
	}

	// Modify copied to verify independence
	copied.ServerCertSANs[0] = "modified.example.com"
	if original.ServerCertSANs[0] == copied.ServerCertSANs[0] {
		t.Error("Modifying copied ServerCertSANs affected original")
	}

	copied.PeerCertSANs[0] = "modified-peer.example.com"
	if original.PeerCertSANs[0] == copied.PeerCertSANs[0] {
		t.Error("Modifying copied PeerCertSANs affected original")
	}

	// Test nil case
	var nilEtcd *Etcd
	if nilEtcd.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestKubeletWithKubelet tests Kubelet deep copy with nested Kubelet pointer
func TestKubeletWithKubelet(t *testing.T) {
	original := &Kubelet{
		ControlPlaneComponent: ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"feature-gates": "RotateKubeletServerCertificate=true",
			},
		},
		ManifestsDir: "/etc/kubernetes/manifests",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil case
	var nilKubelet *Kubelet
	if nilKubelet.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestControlPlaneComponentWithNilFields tests ControlPlaneComponent with nil fields
func TestControlPlaneComponentWithNilFields(t *testing.T) {
	original := &ControlPlaneComponent{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil case
	var nilComponent *ControlPlaneComponent
	if nilComponent.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestRegistryConfigDeepCopy tests RegistryConfig deep copy
func TestRegistryConfigDeepCopy(t *testing.T) {
	original := &RegistryConfig{
		ConfigPath: "/etc/containerd/certs.d",
		Configs: map[string]RegistryHostConfig{
			"docker.io": {
				Host:         "https://docker.io",
				Capabilities: []string{"pull", "resolve"},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Configs map is deep copied
	if len(copied.Configs) != len(original.Configs) {
		t.Errorf("Expected Configs length %d, got %d", len(original.Configs), len(copied.Configs))
	}

	// Test nil Configs
	originalNilConfigs := &RegistryConfig{}
	copiedNilConfigs := originalNilConfigs.DeepCopy()
	if copiedNilConfigs.Configs != nil {
		t.Error("Configs should be nil when original is nil")
	}

	// Test nil case
	var nilConfig *RegistryConfig
	if nilConfig.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestRegistryHostConfigWithNilFields tests RegistryHostConfig with nil fields
func TestRegistryHostConfigWithNilFields(t *testing.T) {
	original := &RegistryHostConfig{
		Host:         "https://docker.io",
		Capabilities: []string{"pull"},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil TLS
	originalNilTLS := &RegistryHostConfig{
		Host:         "https://docker.io",
		Capabilities: []string{"pull"},
	}
	copiedNilTLS := originalNilTLS.DeepCopy()
	if copiedNilTLS.TLS != nil {
		t.Error("TLS should be nil when original is nil")
	}

	// Test nil Auth
	originalNilAuth := &RegistryHostConfig{
		Host:         "https://docker.io",
		Capabilities: []string{"pull"},
	}
	copiedNilAuth := originalNilAuth.DeepCopy()
	if copiedNilAuth.Auth != nil {
		t.Error("Auth should be nil when original is nil")
	}

	// Test nil Header
	originalNilHeader := &RegistryHostConfig{
		Host:         "https://docker.io",
		Capabilities: []string{"pull"},
	}
	copiedNilHeader := originalNilHeader.DeepCopy()
	if copiedNilHeader.Header != nil {
		t.Error("Header should be nil when original is nil")
	}

	// Test nil Capabilities
	originalNilCaps := &RegistryHostConfig{
		Host: "https://docker.io",
	}
	copiedNilCaps := originalNilCaps.DeepCopy()
	if copiedNilCaps.Capabilities != nil {
		t.Error("Capabilities should be nil when original is nil")
	}
}

// TestRegistryHostConfigHeaderWithNilValue tests RegistryHostConfig Header with nil value
func TestRegistryHostConfigHeaderWithNilValue(t *testing.T) {
	original := &RegistryHostConfig{
		Host: "https://docker.io",
		Header: map[string][]string{
			"User-Agent": nil,
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Header with nil value is handled
	if copied.Header == nil {
		t.Error("Header should not be nil")
	}
}

// TestServiceConfigDeepCopy tests ServiceConfig deep copy
func TestServiceConfigDeepCopy(t *testing.T) {
	original := &ServiceConfig{
		ExecStart: "/usr/bin/containerd",
		Slice:     "system.slice",
		Logging: &ServiceLogging{
			StandardOutput: "journal",
		},
		CustomExtra: map[string]string{
			"custom-arg": "value",
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify CustomExtra map is deep copied
	if original.CustomExtra != nil && len(copied.CustomExtra) != len(original.CustomExtra) {
		t.Error("CustomExtra should have same length")
	}

	// Test nil Logging
	originalNilLogging := &ServiceConfig{
		ExecStart: "/usr/bin/containerd",
	}
	copiedNilLogging := originalNilLogging.DeepCopy()
	if copiedNilLogging.Logging != nil {
		t.Error("Logging should be nil when original is nil")
	}

	// Test nil CustomExtra
	originalNilExtra := &ServiceConfig{
		ExecStart: "/usr/bin/containerd",
	}
	copiedNilExtra := originalNilExtra.DeepCopy()
	if copiedNilExtra.CustomExtra != nil {
		t.Error("CustomExtra should be nil when original is nil")
	}

	// Test nil case
	var nilConfig *ServiceConfig
	if nilConfig.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestScriptConfigDeepCopy tests ScriptConfig deep copy
func TestScriptConfigDeepCopy(t *testing.T) {
	original := &ScriptConfig{
		Content:     "#!/bin/bash\necho hello",
		Path:        "/usr/local/bin/init.sh",
		Args:        []string{"--init", "--verbose"},
		Interpreter: "/bin/bash",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Args is deep copied
	if len(copied.Args) != len(original.Args) {
		t.Errorf("Expected Args length %d, got %d", len(original.Args), len(copied.Args))
	}

	// Modify copied to verify independence
	copied.Args[0] = "--modified"
	if original.Args[0] == copied.Args[0] {
		t.Error("Modifying copy affected original")
	}

	// Test nil Args
	originalNilArgs := &ScriptConfig{
		Content: "#!/bin/bash",
	}
	copiedNilArgs := originalNilArgs.DeepCopy()
	if copiedNilArgs.Args != nil {
		t.Error("Args should be nil when original is nil")
	}

	// Test nil case
	var nilScript *ScriptConfig
	if nilScript.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestRepoWithNilFields tests Repo deep copy with nil fields
func TestRepoWithNilFields(t *testing.T) {
	original := &Repo{
		Domain: "registry.example.com",
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil AuthSecretRef
	originalNilAuth := &Repo{
		Domain: "registry.example.com",
	}
	copiedNilAuth := originalNilAuth.DeepCopy()
	if copiedNilAuth.AuthSecretRef != nil {
		t.Error("AuthSecretRef should be nil when original is nil")
	}

	// Test nil TlsSecretRef
	originalNilTLS := &Repo{
		Domain: "registry.example.com",
	}
	copiedNilTLS := originalNilTLS.DeepCopy()
	if copiedNilTLS.TlsSecretRef != nil {
		t.Error("TlsSecretRef should be nil when original is nil")
	}

	// Test nil case
	var nilRepo *Repo
	if nilRepo.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestClusterWithNilFields tests Cluster deep copy with nil fields
func TestClusterWithNilFields(t *testing.T) {
	original := &Cluster{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil Networking
	if copied.Networking.DNSDomain != "" {
		t.Error("Networking should have default values")
	}

	// Test nil ImageRepo
	if copied.ImageRepo.Domain != "" {
		t.Error("ImageRepo should have default values")
	}

	// Test nil ChartRepo
	if copied.ChartRepo.Domain != "" {
		t.Error("ChartRepo should have default values")
	}

	// Test nil ContainerRuntime
	if copied.ContainerRuntime.CRI != "" {
		t.Error("ContainerRuntime should have default values")
	}

	// Test nil HTTPRepo
	if copied.HTTPRepo.Domain != "" {
		t.Error("HTTPRepo should have default values")
	}

	// Test nil case
	var nilCluster *Cluster
	if nilCluster.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeSpecWithLabels tests BKENodeSpec with Labels field
func TestBKENodeSpecWithLabels(t *testing.T) {
	original := &BKENodeSpec{
		IP: "192.168.1.1",
		Labels: []Label{
			{Key: "node-role", Value: "master"},
			{Key: "environment", Value: "prod"},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Labels is deep copied
	if len(copied.Labels) != len(original.Labels) {
		t.Errorf("Expected Labels length %d, got %d", len(original.Labels), len(copied.Labels))
	}

	// Modify copied to verify independence
	copied.Labels[0].Value = "worker"
	if original.Labels[0].Value == copied.Labels[0].Value {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilSpec *BKENodeSpec
	if nilSpec.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestBKENodeSpecWithKubelet tests BKENodeSpec with Kubelet field
func TestBKENodeSpecWithKubelet(t *testing.T) {
	original := &BKENodeSpec{
		IP:   "192.168.1.1",
		Role: []string{"master"},
		Kubelet: &Kubelet{
			ManifestsDir: "/etc/kubernetes/manifests",
			ControlPlaneComponent: ControlPlaneComponent{
				ExtraArgs: map[string]string{
					"feature-gates": "DynamicKubeletConfig=true",
				},
			},
		},
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Verify Kubelet pointer is deep copied
	if copied.Kubelet == original.Kubelet {
		t.Error("Kubelet should be different instances")
	}

	// Modify copied to verify independence
	copied.Kubelet.ManifestsDir = "/modified/path"
	if original.Kubelet.ManifestsDir == copied.Kubelet.ManifestsDir {
		t.Error("Modifying copy affected original")
	}

	// Test nil case
	var nilSpec *BKENodeSpec
	if nilSpec.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}

// TestAPIServerNilFields tests APIServer with nil fields
func TestAPIServerNilFields(t *testing.T) {
	original := &APIServer{}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	// Test nil case
	var nilServer *APIServer
	if nilServer.DeepCopy() != nil {
		t.Error("DeepCopy of nil should return nil")
	}
}
