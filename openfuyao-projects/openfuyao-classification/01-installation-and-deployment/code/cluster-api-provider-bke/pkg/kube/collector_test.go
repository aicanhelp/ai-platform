/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

func TestCollectorStruct(t *testing.T) {
	c := &Collector{
		availableCollectNode: nil,
		result:               nil,
		errs:                 nil,
		warns:                nil,
	}

	if c.availableCollectNode != nil {
		t.Error("Expected availableCollectNode to be nil")
	}
	if c.result != nil {
		t.Error("Expected result to be nil")
	}
	if c.errs != nil {
		t.Error("Expected errs to be nil")
	}
	if c.warns != nil {
		t.Error("Expected warns to be nil")
	}
}

func TestCollectResultStruct(t *testing.T) {
	cr := &CollectResult{
		ControlPlaneEndpoint: confv1beta1.APIEndpoint{
			Host: "10.0.0.1",
			Port: 6443,
		},
		Nodes:                  nil,
		KubernetesVersion:      "v1.21.0",
		EtcdCertificatesDir:    "/etc/etcd/ssl",
		K8sCertificatesDir:     "/etc/kubernetes/pki",
		AvailableCollectedNode: nil,
	}

	if cr.ControlPlaneEndpoint.Host != "10.0.0.1" {
		t.Errorf("Expected Host to be 10.0.0.1, got %s", cr.ControlPlaneEndpoint.Host)
	}
	if cr.ControlPlaneEndpoint.Port != 6443 {
		t.Errorf("Expected Port to be 6443, got %d", cr.ControlPlaneEndpoint.Port)
	}
	if cr.KubernetesVersion != "v1.21.0" {
		t.Errorf("Expected KubernetesVersion to be v1.21.0, got %s", cr.KubernetesVersion)
	}
	if cr.EtcdCertificatesDir != "/etc/etcd/ssl" {
		t.Errorf("Expected EtcdCertificatesDir to be /etc/etcd/ssl, got %s", cr.EtcdCertificatesDir)
	}
	if cr.K8sCertificatesDir != "/etc/kubernetes/pki" {
		t.Errorf("Expected K8sCertificatesDir to be /etc/kubernetes/pki, got %s", cr.K8sCertificatesDir)
	}
}

func TestCollectorPackageConstants(t *testing.T) {
	if bocloudEtcdCertVolumeName1 != "ssl" {
		t.Errorf("Expected bocloudEtcdCertVolumeName1 to be ssl, got %s", bocloudEtcdCertVolumeName1)
	}
	if bocloudEtcdCertVolumeName2 != "etcd-certs" {
		t.Errorf("Expected bocloudEtcdCertVolumeName2 to be etcd-certs, got %s", bocloudEtcdCertVolumeName2)
	}
	if bocloudPkiCertVolumeName != "pki" {
		t.Errorf("Expected bocloudPkiCertVolumeName to be pki, got %s", bocloudPkiCertVolumeName)
	}
	if bocloudEtcdCertDefaultPath != "/etc/etcd/ssl" {
		t.Errorf("Expected bocloudEtcdCertDefaultPath to be /etc/etcd/ssl, got %s", bocloudEtcdCertDefaultPath)
	}
	if bocloudPkiCertDefaultPath != "/etc/kubernetes/pki" {
		t.Errorf("Expected bocloudPkiCertDefaultPath to be /etc/kubernetes/pki, got %s", bocloudPkiCertDefaultPath)
	}
	if kubeadmK8sCertsVolumeName != "k8s-certs" {
		t.Errorf("Expected kubeadmK8sCertsVolumeName to be k8s-certs, got %s", kubeadmK8sCertsVolumeName)
	}
	if kubeadmEtcdCertsVolumeName != "etcd-certs" {
		t.Errorf("Expected kubeadmEtcdCertsVolumeName to be etcd-certs, got %s", kubeadmEtcdCertsVolumeName)
	}
	if base != 10 {
		t.Errorf("Expected base to be 10, got %d", base)
	}
	if bitsize != 32 {
		t.Errorf("Expected bitsize to be 32, got %d", bitsize)
	}
}

func TestGetNodeIP(t *testing.T) {
	tests := []struct {
		name string
		node *corev1.Node
		want string
	}{
		{
			name: "node with internal IP",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{Type: corev1.NodeInternalIP, Address: "192.168.1.10"},
					},
				},
			},
			want: "192.168.1.10",
		},
		{
			name: "node with multiple addresses",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{Type: corev1.NodeHostName, Address: "node1"},
						{Type: corev1.NodeInternalIP, Address: "10.0.0.1"},
					},
				},
			},
			want: "10.0.0.1",
		},
		{
			name: "node without internal IP",
			node: &corev1.Node{
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{Type: corev1.NodeHostName, Address: "node1"},
					},
				},
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetNodeIP(tt.node)
			if got != tt.want {
				t.Errorf("GetNodeIP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetNodeK8sVersion(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			NodeInfo: corev1.NodeSystemInfo{
				KubeletVersion: "v1.21.0",
			},
		},
	}

	got := getNodeK8sVersion(node)
	if got != "v1.21.0" {
		t.Errorf("getNodeK8sVersion() = %v, want v1.21.0", got)
	}
}

func TestGetNodeObjHostname(t *testing.T) {
	tests := []struct {
		name string
		node *corev1.Node
		want string
	}{
		{
			name: "nodeWithHostnameLabel",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "node1",
					Labels: map[string]string{corev1.LabelHostname: "test-host"},
				},
			},
			want: "test-host",
		},
		{
			name: "nodeWithoutLabels",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
			},
			want: "node2",
		},
		{
			name: "nodeWithEmptyHostnameLabel",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "node3",
					Labels: map[string]string{corev1.LabelHostname: ""},
				},
			},
			want: "node3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getNodeObjHostname(tt.node)
			if got != tt.want {
				t.Errorf("getNodeObjHostname() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetNodeRole(t *testing.T) {
	const (
		masterLabel = "node-role.kubernetes.io/master"
		nodeLabel   = "node-role.kubernetes.io/node"
	)

	tests := []struct {
		name string
		node *corev1.Node
		want string
	}{
		{
			name: "masterWorkerNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						masterLabel: "",
						nodeLabel:   "",
					},
				},
			},
			want: bkenode.MasterWorkerNodeRole,
		},
		{
			name: "masterOnlyNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{masterLabel: ""},
				},
			},
			want: bkenode.MasterNodeRole,
		},
		{
			name: "workerOnlyNode",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{nodeLabel: ""},
				},
			},
			want: bkenode.WorkerNodeRole,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getNodeRole(tt.node)
			if len(got) == 0 {
				t.Error("getNodeRole() returned empty slice")
				return
			}
			if got[0] != tt.want {
				t.Errorf("getNodeRole() = %v, want %v", got[0], tt.want)
			}
		})
	}
}

func TestNewCollector(t *testing.T) {
	client := &Client{}
	collector := NewCollector(client)

	if collector == nil {
		t.Fatal("NewCollector() returned nil")
	}
	if collector.client != client {
		t.Error("collector.client not set correctly")
	}
	if collector.result == nil {
		t.Error("collector.result should be initialized")
	}
	if collector.keyWord == nil {
		t.Error("collector.keyWord should be initialized")
	}
}

func TestNewResult(t *testing.T) {
	result := newResult()

	if result == nil {
		t.Fatal("newResult() returned nil")
	}
	if result.Nodes == nil {
		t.Error("result.Nodes should be initialized")
	}
}

func TestInitializeContainerRuntimeMap(t *testing.T) {
	runtimeMap := initializeContainerRuntimeMap()

	if runtimeMap == nil {
		t.Fatal("initializeContainerRuntimeMap() returned nil")
	}
	if runtimeMap["docker"] != 0 {
		t.Errorf("docker count = %d, want 0", runtimeMap["docker"])
	}
	if runtimeMap["containerd"] != 0 {
		t.Errorf("containerd count = %d, want 0", runtimeMap["containerd"])
	}
}

func TestUpdateContainerRuntimeMap(t *testing.T) {
	const (
		numZero = 0
		numOne  = 1
	)

	tests := []struct {
		name           string
		runtime        string
		wantDocker     int
		wantContainerd int
	}{
		{
			name:           "dockerRuntime",
			runtime:        "docker://20.10.7",
			wantDocker:     numOne,
			wantContainerd: numZero,
		},
		{
			name:           "containerdRuntime",
			runtime:        "containerd://1.5.5",
			wantDocker:     numZero,
			wantContainerd: numOne,
		},
		{
			name:           "emptyRuntime",
			runtime:        "",
			wantDocker:     numZero,
			wantContainerd: numZero,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runtimeMap := initializeContainerRuntimeMap()
			updateContainerRuntimeMap(runtimeMap, tt.runtime)

			if runtimeMap["docker"] != tt.wantDocker {
				t.Errorf("docker count = %d, want %d", runtimeMap["docker"], tt.wantDocker)
			}
			if runtimeMap["containerd"] != tt.wantContainerd {
				t.Errorf("containerd count = %d, want %d", runtimeMap["containerd"], tt.wantContainerd)
			}
		})
	}
}

func TestDetermineContainerRuntime(t *testing.T) {
	const (
		numOne   = 1
		numTwo   = 2
		numThree = 3
	)

	tests := []struct {
		name            string
		dockerCount     int
		containerdCount int
		wantCRI         string
	}{
		{
			name:            "moreDocker",
			dockerCount:     numThree,
			containerdCount: numOne,
			wantCRI:         bkeinit.CRIDocker,
		},
		{
			name:            "moreContainerd",
			dockerCount:     numOne,
			containerdCount: numThree,
			wantCRI:         bkeinit.CRIContainerd,
		},
		{
			name:            "equalCounts",
			dockerCount:     numTwo,
			containerdCount: numTwo,
			wantCRI:         bkeinit.CRIContainerd,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collector := &Collector{}
			runtimeMap := map[string]int{
				"docker":     tt.dockerCount,
				"containerd": tt.containerdCount,
			}

			result := collector.determineContainerRuntime(runtimeMap)

			if result.CRI != tt.wantCRI {
				t.Errorf("CRI = %v, want %v", result.CRI, tt.wantCRI)
			}
		})
	}
}

func TestProcessNode(t *testing.T) {
	const testIP = "192.168.1.10"
	collector := &Collector{}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-node",
			Labels: map[string]string{corev1.LabelHostname: "test-host"},
		},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{Type: corev1.NodeInternalIP, Address: testIP},
			},
		},
	}

	result := collector.processNode(node)

	if result.Hostname != "test-host" {
		t.Errorf("Hostname = %v, want test-host", result.Hostname)
	}
	if result.IP != testIP {
		t.Errorf("IP = %v, want %s", result.IP, testIP)
	}
}

func TestCommandExit(t *testing.T) {
	tests := []struct {
		name     string
		except   string
		commands []string
		wantVal  string
		wantOk   bool
	}{
		{
			name:     "foundWithEquals",
			except:   "--cluster-cidr",
			commands: []string{"--cluster-cidr=10.244.0.0/16"},
			wantVal:  "10.244.0.0/16",
			wantOk:   true,
		},
		{
			name:     "foundWithSpace",
			except:   "--service-cluster-ip-range",
			commands: []string{"--service-cluster-ip-range", "10.96.0.0/12"},
			wantVal:  "",
			wantOk:   true,
		},
		{
			name:     "notFound",
			except:   "--missing-flag",
			commands: []string{"--other-flag=value"},
			wantVal:  "",
			wantOk:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			val, ok := commandExit(tt.except, tt.commands)
			if ok != tt.wantOk {
				t.Errorf("commandExit() ok = %v, want %v", ok, tt.wantOk)
			}
			if val != tt.wantVal {
				t.Errorf("commandExit() val = %v, want %v", val, tt.wantVal)
			}
		})
	}
}

func TestNewCertCommandKeyWords(t *testing.T) {
	keywords := NewCertCommandKeyWords()

	if keywords == nil {
		t.Fatal("NewCertCommandKeyWords() returned nil")
	}
	if len(keywords[mfutil.KubeAPIServer]) == 0 {
		t.Error("KubeAPIServer keywords should not be empty")
	}
	if len(keywords[mfutil.Etcd]) == 0 {
		t.Error("Etcd keywords should not be empty")
	}
}

func TestCertCommandExit(t *testing.T) {
	keywords := NewCertCommandKeyWords()

	tests := []struct {
		name      string
		component string
		command   string
		wantVal   string
		wantOk    bool
	}{
		{
			name:      "apiServerCert",
			component: mfutil.KubeAPIServer,
			command:   "--tls-cert-file=/etc/kubernetes/pki/apiserver.crt",
			wantVal:   "/etc/kubernetes/pki/apiserver.crt",
			wantOk:    true,
		},
		{
			name:      "etcdCert",
			component: mfutil.Etcd,
			command:   "--cert-file=/etc/etcd/ssl/etcd.crt",
			wantVal:   "/etc/etcd/ssl/etcd.crt",
			wantOk:    true,
		},
		{
			name:      "notFound",
			component: mfutil.KubeAPIServer,
			command:   "--unknown-flag=value",
			wantVal:   "",
			wantOk:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			val, ok := keywords.certCommandExit(tt.component, tt.command)
			if ok != tt.wantOk {
				t.Errorf("certCommandExit() ok = %v, want %v", ok, tt.wantOk)
			}
			if val != tt.wantVal {
				t.Errorf("certCommandExit() val = %v, want %v", val, tt.wantVal)
			}
		})
	}
}

func TestCertVolumeExit(t *testing.T) {
	keywords := NewCertCommandKeyWords()

	tests := []struct {
		name          string
		volumeName    string
		componentType string
		want          bool
	}{
		{
			name:          "etcdCertsVolume",
			volumeName:    "etcd-certs",
			componentType: mfutil.Etcd,
			want:          true,
		},
		{
			name:          "k8sCertsVolume",
			volumeName:    "k8s-certs",
			componentType: mfutil.KubeAPIServer,
			want:          true,
		},
		{
			name:          "pkiVolume",
			volumeName:    "pki",
			componentType: "",
			want:          true,
		},
		{
			name:          "unknownVolume",
			volumeName:    "unknown",
			componentType: mfutil.KubeAPIServer,
			want:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := keywords.certVolumeExit(tt.volumeName, tt.componentType)
			if got != tt.want {
				t.Errorf("certVolumeExit() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetCertificatesDirFromCertPaths(t *testing.T) {
	tests := []struct {
		name      string
		certPaths []string
	}{
		{
			name: "commonPrefix",
			certPaths: []string{
				"/etc/kubernetes/pki/apiserver.crt",
				"/etc/kubernetes/pki/apiserver.key",
				"/etc/kubernetes/pki/ca.crt",
			},
		},
		{
			name: "differentPaths",
			certPaths: []string{
				"/etc/etcd/ssl/etcd.crt",
				"/etc/etcd/ssl/etcd.key",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getCertificatesDirFromCertPaths(tt.certPaths)
			if got == "" {
				t.Error("getCertificatesDirFromCertPaths() returned empty string")
			}
		})
	}
}

func TestCollectorSetDefaults(t *testing.T) {
	tests := []struct {
		name              string
		etcdCertDir       string
		k8sCertDir        string
		wantEtcdCertDir   string
		wantK8sCertDir    string
		wantWarningsCount int
	}{
		{
			name:              "bothEmpty",
			etcdCertDir:       "",
			k8sCertDir:        "",
			wantEtcdCertDir:   bocloudEtcdCertDefaultPath,
			wantK8sCertDir:    bocloudPkiCertDefaultPath,
			wantWarningsCount: 2,
		},
		{
			name:              "etcdEmpty",
			etcdCertDir:       "",
			k8sCertDir:        "/custom/pki",
			wantEtcdCertDir:   bocloudEtcdCertDefaultPath,
			wantK8sCertDir:    "/custom/pki",
			wantWarningsCount: 1,
		},
		{
			name:              "k8sEmpty",
			etcdCertDir:       "/custom/etcd",
			k8sCertDir:        "",
			wantEtcdCertDir:   "/custom/etcd",
			wantK8sCertDir:    bocloudPkiCertDefaultPath,
			wantWarningsCount: 1,
		},
		{
			name:              "noneEmpty",
			etcdCertDir:       "/custom/etcd",
			k8sCertDir:        "/custom/pki",
			wantEtcdCertDir:   "/custom/etcd",
			wantK8sCertDir:    "/custom/pki",
			wantWarningsCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collector := &Collector{
				result: &CollectResult{
					EtcdCertificatesDir: tt.etcdCertDir,
					K8sCertificatesDir:  tt.k8sCertDir,
				},
				warns: []error{},
			}

			collector.setDefaults()

			if collector.result.EtcdCertificatesDir != tt.wantEtcdCertDir {
				t.Errorf("EtcdCertificatesDir = %v, want %v",
					collector.result.EtcdCertificatesDir, tt.wantEtcdCertDir)
			}
			if collector.result.K8sCertificatesDir != tt.wantK8sCertDir {
				t.Errorf("K8sCertificatesDir = %v, want %v",
					collector.result.K8sCertificatesDir, tt.wantK8sCertDir)
			}
			if len(collector.warns) != tt.wantWarningsCount {
				t.Errorf("warnings count = %v, want %v",
					len(collector.warns), tt.wantWarningsCount)
			}
		})
	}
}

func TestCollectorCheckEtcdRole(t *testing.T) {
	const testHostname = "test-node"

	tests := []struct {
		name         string
		getPodErr    error
		wantEtcdRole bool
	}{
		{
			name:         "etcdPodExists",
			getPodErr:    nil,
			wantEtcdRole: true,
		},
		{
			name:         "etcdPodNotExists",
			getPodErr:    errors.New("not found"),
			wantEtcdRole: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}
			collector := &Collector{client: client}

			patches := gomonkey.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					return &corev1.Pod{}, tt.getPodErr
				})
			defer patches.Reset()

			bkeNode := confv1beta1.Node{
				Hostname: testHostname,
				Role:     []string{bkenode.MasterNodeRole},
			}

			result := collector.checkEtcdRole(bkeNode)

			hasEtcdRole := false
			for _, role := range result.Role {
				if role == bkenode.EtcdNodeRole {
					hasEtcdRole = true
					break
				}
			}

			if hasEtcdRole != tt.wantEtcdRole {
				t.Errorf("hasEtcdRole = %v, want %v", hasEtcdRole, tt.wantEtcdRole)
			}
		})
	}
}

func TestCollectorSelectAvailableCollectNode(t *testing.T) {
	const (
		testIP       = "192.168.1.10"
		testHostname = "master-node"
	)

	tests := []struct {
		name             string
		nodeRole         []string
		healthCheckErr   error
		wantNodeSelected bool
	}{
		{
			name:             "healthyMasterNode",
			nodeRole:         []string{bkenode.MasterNodeRole},
			healthCheckErr:   nil,
			wantNodeSelected: true,
		},
		{
			name:             "unhealthyMasterNode",
			nodeRole:         []string{bkenode.MasterNodeRole},
			healthCheckErr:   errors.New("unhealthy"),
			wantNodeSelected: false,
		},
		{
			name:             "workerNode",
			nodeRole:         []string{bkenode.WorkerNodeRole},
			healthCheckErr:   nil,
			wantNodeSelected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				BKELog: &bkev1beta1.BKELogger{},
			}
			collector := &Collector{
				client: client,
				result: newResult(),
			}

			patches := gomonkey.ApplyMethod(client, "CheckComponentHealth",
				func(_ *Client, _ *corev1.Node) error {
					return tt.healthCheckErr
				})
			patches.ApplyMethod(client.BKELog, "Info",
				func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
			patches.ApplyMethod(client.BKELog, "Warn",
				func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
			defer patches.Reset()

			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testHostname},
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{Type: corev1.NodeInternalIP, Address: testIP},
					},
				},
			}

			bkeNode := confv1beta1.Node{
				Hostname: testHostname,
				Role:     tt.nodeRole,
				IP:       testIP,
			}

			collector.selectAvailableCollectNode(node, bkeNode)

			if tt.wantNodeSelected && collector.availableCollectNode == nil {
				t.Error("expected node to be selected but it was not")
			}
			if !tt.wantNodeSelected && collector.availableCollectNode != nil {
				t.Error("expected node not to be selected but it was")
			}
		})
	}
}

func TestCollectorCollectKubernetesVersion(t *testing.T) {
	const testVersion = "v1.28.0"

	tests := []struct {
		name       string
		version    *version.Info
		versionErr error
		wantErr    bool
	}{
		{
			name:       "success",
			version:    &version.Info{GitVersion: testVersion},
			versionErr: nil,
			wantErr:    false,
		},
		{
			name:       "error",
			version:    nil,
			versionErr: errors.New("failed to get version"),
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				ClientSet: &kubernetes.Clientset{},
			}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
			}

			patches := gomonkey.ApplyMethod(&discovery.DiscoveryClient{}, "ServerVersion",
				func(_ *discovery.DiscoveryClient) (*version.Info, error) {
					return tt.version, tt.versionErr
				})
			defer patches.Reset()

			collector.collectKubernetesVersion()

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if !tt.wantErr && collector.result.KubernetesVersion != testVersion {
				t.Errorf("KubernetesVersion = %v, want %v",
					collector.result.KubernetesVersion, testVersion)
			}
		})
	}
}

func TestCollectorCollectControlPlaneEndpoint(t *testing.T) {
	const (
		testHost = "192.168.1.100"
		testPort = 6443
	)

	tests := []struct {
		name    string
		host    string
		wantErr bool
	}{
		{
			name:    "validEndpoint",
			host:    "https://192.168.1.100:6443",
			wantErr: false,
		},
		{
			name:    "invalidURL",
			host:    "://invalid",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				RestConfig: &rest.Config{Host: tt.host},
			}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
			}

			collector.collectControlPlaneEndpoint()

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if !tt.wantErr {
				if collector.result.ControlPlaneEndpoint.Host != testHost {
					t.Errorf("Host = %v, want %v",
						collector.result.ControlPlaneEndpoint.Host, testHost)
				}
				if collector.result.ControlPlaneEndpoint.Port != testPort {
					t.Errorf("Port = %v, want %v",
						collector.result.ControlPlaneEndpoint.Port, testPort)
				}
			}
		})
	}
}

func TestCollectorCollectAPIServerInfo(t *testing.T) {
	const (
		testHostname      = "master-node"
		testPodSubnet     = "10.244.0.0/16"
		testServiceSubnet = "10.96.0.0/12"
		testDNSDomain     = "cluster.local"
		testPkiPath       = "/etc/kubernetes/pki"
	)

	tests := []struct {
		name         string
		pod          *corev1.Pod
		getPodErr    error
		wantErr      bool
		wantWarnings int
	}{
		{
			name: "successWithAllInfo",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Command: []string{
								"kube-apiserver",
								"--service-cluster-ip-range=" + testServiceSubnet,
								"--service-account-issuer=https://kubernetes.default.svc." + testDNSDomain,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "pki",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: testPkiPath,
								},
							},
						},
					},
				},
			},
			getPodErr:    nil,
			wantErr:      false,
			wantWarnings: 0,
		},
		{
			name:         "podNotFound",
			pod:          nil,
			getPodErr:    errors.New("not found"),
			wantErr:      true,
			wantWarnings: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
				warns:  []error{},
				availableCollectNode: &confv1beta1.Node{
					Hostname: testHostname,
				},
			}

			patches := gomonkey.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					return tt.pod, tt.getPodErr
				})
			defer patches.Reset()

			collector.collectAPIServerInfo(testHostname)

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if len(collector.warns) != tt.wantWarnings {
				t.Errorf("warnings count = %v, want %v",
					len(collector.warns), tt.wantWarnings)
			}
		})
	}
}

func TestCollectorCollectEtcdInfo(t *testing.T) {
	const (
		testHostname    = "master-node"
		testEtcdCertDir = "/etc/kubernetes/pki/"
	)

	tests := []struct {
		name      string
		pod       *corev1.Pod
		getPodErr error
		wantErr   bool
	}{
		{
			name: "success",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "etcd-certs",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: testEtcdCertDir + "etcd",
								},
							},
						},
					},
				},
			},
			getPodErr: nil,
			wantErr:   false,
		},
		{
			name:      "podNotFound",
			pod:       nil,
			getPodErr: errors.New("not found"),
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
			}

			patches := gomonkey.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					return tt.pod, tt.getPodErr
				})
			defer patches.Reset()

			collector.collectEtcdInfo(testHostname)

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if !tt.wantErr && collector.result.EtcdCertificatesDir != testEtcdCertDir {
				t.Errorf("EtcdCertificatesDir = %v, want %v",
					collector.result.EtcdCertificatesDir, testEtcdCertDir)
			}
		})
	}
}

func TestCollectorCollectControllerManagerInfo(t *testing.T) {
	const (
		testHostname  = "master-node"
		testPodSubnet = "10.244.0.0/16"
	)

	tests := []struct {
		name         string
		pod          *corev1.Pod
		getPodErr    error
		wantErr      bool
		wantWarnings int
	}{
		{
			name: "successWithClusterCIDR",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Command: []string{
								"kube-controller-manager",
								"--cluster-cidr=" + testPodSubnet,
							},
						},
					},
				},
			},
			getPodErr:    nil,
			wantErr:      false,
			wantWarnings: 0,
		},
		{
			name: "missingClusterCIDR",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Command: []string{"kube-controller-manager"}},
					},
				},
			},
			getPodErr:    nil,
			wantErr:      false,
			wantWarnings: 1,
		},
		{
			name:         "podNotFound",
			pod:          nil,
			getPodErr:    errors.New("not found"),
			wantErr:      true,
			wantWarnings: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
				warns:  []error{},
			}

			patches := gomonkey.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					return tt.pod, tt.getPodErr
				})
			defer patches.Reset()

			collector.collectControllerManagerInfo(testHostname)

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if len(collector.warns) != tt.wantWarnings {
				t.Errorf("warnings count = %v, want %v",
					len(collector.warns), tt.wantWarnings)
			}
		})
	}
}

func TestCollectorCollectClusterInfo(t *testing.T) {
	const testHostname = "master-node"

	client := &Client{
		ClientSet:  &kubernetes.Clientset{},
		RestConfig: &rest.Config{Host: "https://192.168.1.100:6443"},
	}
	collector := &Collector{
		client: client,
		result: newResult(),
		errs:   []error{},
		warns:  []error{},
		availableCollectNode: &confv1beta1.Node{
			Hostname: testHostname,
		},
	}

	patches := gomonkey.ApplyMethod(&discovery.DiscoveryClient{}, "ServerVersion",
		func(_ *discovery.DiscoveryClient) (*version.Info, error) {
			return &version.Info{GitVersion: "v1.28.0"}, nil
		})
	patches.ApplyMethod(client, "GetPod",
		func(_ *Client, _, podName string) (*corev1.Pod, error) {
			return &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Command: []string{}}},
					Volumes:    []corev1.Volume{},
				},
			}, nil
		})
	defer patches.Reset()

	collector.collectClusterInfo()

	if len(collector.errs) > 0 {
		t.Errorf("unexpected errors: %v", collector.errs)
	}
}

func TestCollectorCollectClusterInfoNoAvailableNode(t *testing.T) {
	client := &Client{ClientSet: &kubernetes.Clientset{}}
	collector := &Collector{
		client: client,
		result: newResult(),
		errs:   []error{},
	}

	patches := gomonkey.ApplyMethod(&discovery.DiscoveryClient{}, "ServerVersion",
		func(_ *discovery.DiscoveryClient) (*version.Info, error) {
			return &version.Info{GitVersion: "v1.28.0"}, nil
		})
	defer patches.Reset()

	collector.collectClusterInfo()

	if len(collector.errs) == 0 {
		t.Error("expected error when no available collect node")
	}
}

func TestCollectorCollectNodeInfo(t *testing.T) {
	const (
		testIP       = "192.168.1.10"
		testHostname = "test-node"
		numOne       = 1
	)

	tests := []struct {
		name      string
		nodes     *corev1.NodeList
		listErr   error
		getPodErr error
		wantErr   bool
	}{
		{
			name: "successWithNodes",
			nodes: &corev1.NodeList{
				Items: []corev1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:   testHostname,
							Labels: map[string]string{"node-role.kubernetes.io/master": ""},
						},
						Status: corev1.NodeStatus{
							Addresses: []corev1.NodeAddress{
								{Type: corev1.NodeInternalIP, Address: testIP},
							},
							NodeInfo: corev1.NodeSystemInfo{
								ContainerRuntimeVersion: "containerd://1.5.5",
							},
						},
					},
				},
			},
			listErr:   nil,
			getPodErr: errors.New("not found"),
			wantErr:   false,
		},
		{
			name:      "listNodesError",
			nodes:     nil,
			listErr:   errors.New("failed to list"),
			getPodErr: nil,
			wantErr:   true,
		},
		{
			name:      "noNodesFound",
			nodes:     &corev1.NodeList{Items: []corev1.Node{}},
			listErr:   nil,
			getPodErr: nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{
				BKELog: &bkev1beta1.BKELogger{},
			}
			collector := &Collector{
				client: client,
				result: newResult(),
				errs:   []error{},
			}

			patches := gomonkey.ApplyMethod(client, "ListNodes",
				func(_ *Client, _ *metav1.ListOptions) (*corev1.NodeList, error) {
					return tt.nodes, tt.listErr
				})
			patches.ApplyMethod(client, "GetPod",
				func(_ *Client, _, _ string) (*corev1.Pod, error) {
					return nil, tt.getPodErr
				})
			patches.ApplyMethod(client.BKELog, "Info",
				func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
			patches.ApplyMethod(client.BKELog, "Warn",
				func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
			defer patches.Reset()

			collector.collectNodeInfo()

			if tt.wantErr && len(collector.errs) == 0 {
				t.Error("expected error but got none")
			}
			if !tt.wantErr && len(collector.result.Nodes) != numOne {
				t.Errorf("nodes count = %v, want %v", len(collector.result.Nodes), numOne)
			}
		})
	}
}

func TestCollectorCollect(t *testing.T) {
	const (
		testIP       = "192.168.1.10"
		testHostname = "master-node"
	)

	client := &Client{
		ClientSet:  &kubernetes.Clientset{},
		RestConfig: &rest.Config{Host: "https://192.168.1.100:6443"},
		BKELog:     &bkev1beta1.BKELogger{},
	}
	collector := &Collector{
		client:  client,
		keyWord: NewCertCommandKeyWords(),
		result:  newResult(),
		errs:    []error{},
		warns:   []error{},
	}

	patches := gomonkey.ApplyMethod(client, "ListNodes",
		func(_ *Client, _ *metav1.ListOptions) (*corev1.NodeList, error) {
			return &corev1.NodeList{
				Items: []corev1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:   testHostname,
							Labels: map[string]string{"node-role.kubernetes.io/master": ""},
						},
						Status: corev1.NodeStatus{
							Addresses: []corev1.NodeAddress{
								{Type: corev1.NodeInternalIP, Address: testIP},
							},
							NodeInfo: corev1.NodeSystemInfo{
								ContainerRuntimeVersion: "containerd://1.5.5",
							},
						},
					},
				},
			}, nil
		})
	patches.ApplyMethod(client, "GetPod",
		func(_ *Client, _, _ string) (*corev1.Pod, error) {
			return &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Command: []string{}}},
					Volumes:    []corev1.Volume{},
				},
			}, nil
		})
	patches.ApplyMethod(client, "CheckComponentHealth",
		func(_ *Client, _ *corev1.Node) error {
			return nil
		})
	patches.ApplyMethod(&discovery.DiscoveryClient{}, "ServerVersion",
		func(_ *discovery.DiscoveryClient) (*version.Info, error) {
			return &version.Info{GitVersion: "v1.28.0"}, nil
		})
	patches.ApplyMethod(client.BKELog, "Info",
		func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
	defer patches.Reset()

	result, warns, errs := collector.Collect()

	if result == nil {
		t.Fatal("Collect() returned nil result")
	}
	if warns == nil {
		t.Error("warns should not be nil")
	}
	if errs == nil {
		t.Error("errs should not be nil")
	}
}

func TestClientCollect(t *testing.T) {
	const (
		testIP       = "192.168.1.10"
		testHostname = "master-node"
	)

	client := &Client{
		ClientSet:  &kubernetes.Clientset{},
		RestConfig: &rest.Config{Host: "https://192.168.1.100:6443"},
		BKELog:     &bkev1beta1.BKELogger{},
	}

	patches := gomonkey.ApplyMethod(client, "ListNodes",
		func(_ *Client, _ *metav1.ListOptions) (*corev1.NodeList, error) {
			return &corev1.NodeList{
				Items: []corev1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:   testHostname,
							Labels: map[string]string{"node-role.kubernetes.io/master": ""},
						},
						Status: corev1.NodeStatus{
							Addresses: []corev1.NodeAddress{
								{Type: corev1.NodeInternalIP, Address: testIP},
							},
							NodeInfo: corev1.NodeSystemInfo{
								ContainerRuntimeVersion: "containerd://1.5.5",
							},
						},
					},
				},
			}, nil
		})
	patches.ApplyMethod(client, "GetPod",
		func(_ *Client, _, _ string) (*corev1.Pod, error) {
			return &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Command: []string{}}},
					Volumes:    []corev1.Volume{},
				},
			}, nil
		})
	patches.ApplyMethod(client, "CheckComponentHealth",
		func(_ *Client, _ *corev1.Node) error {
			return nil
		})
	patches.ApplyMethod(&discovery.DiscoveryClient{}, "ServerVersion",
		func(_ *discovery.DiscoveryClient) (*version.Info, error) {
			return &version.Info{GitVersion: "v1.28.0"}, nil
		})
	patches.ApplyMethod(client.BKELog, "Info",
		func(_ *bkev1beta1.BKELogger, _ string, _ string, _ ...interface{}) {})
	defer patches.Reset()

	result, _, _ := client.Collect()

	if result == nil {
		t.Fatal("Collect() returned nil result")
	}
}
