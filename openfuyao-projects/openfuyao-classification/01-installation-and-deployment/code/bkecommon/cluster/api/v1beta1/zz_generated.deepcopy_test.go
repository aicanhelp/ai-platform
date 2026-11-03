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
			DataDir:        "/var/lib/etcd",
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
		DataDir:               "/var/lib/etcd",
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
