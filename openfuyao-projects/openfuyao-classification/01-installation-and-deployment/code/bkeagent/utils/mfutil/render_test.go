package mfutil

import (
	"testing"

	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"

	"gopkg.openfuyao.cn/bkeagent/utils"
)

var (
	ip, _ = bkenet.GetExternalIP()

	cfg = &BootScope{
		BkeConfig: &bkev1beta1.BKEConfig{
			Cluster: bkev1beta1.Cluster{
				ControlPlane: bkev1beta1.ControlPlane{

					APIServer: &bkev1beta1.APIServer{
						ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
							ExtraArgs: map[string]string{
								"authorization-mode": "Node,RBAC",
							},
						},
						CertSANs: nil,
					},
				},
				Networking:        bkev1beta1.Networking{},
				HTTPRepo:          bkev1beta1.Repo{},
				ImageRepo:         bkev1beta1.Repo{},
				KubernetesVersion: "v1.24.1",
				CertificatesDir:   "",
				NTPServer:         "",
			},
			Nodes: []bkev1beta1.Node{
				bkev1beta1.Node{
					Role:     []string{bkenode.EtcdNodeRole, bkenode.MasterNodeRole},
					IP:       "172.28.100.203",
					Port:     "22",
					Username: "root",
					Password: "123123",
					Hostname: "oldcc",
					ControlPlane: bkev1beta1.ControlPlane{
						Etcd: &bkev1beta1.Etcd{
							ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
								ExtraArgs: map[string]string{
									"listen-client-urls": "https://" + ip + ":2379,https://",
									"listen-peer-urls":   "https://" + ip + ":2380",
								},
							},
						},
						APIServer: &bkev1beta1.APIServer{
							ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
								ExtraArgs: map[string]string{
									"authorization-mode": "Node,RBAC",
									"advertise-address":  ip,
								},
							},
						},
					},
					Kubelet: nil,
				},
				bkev1beta1.Node{
					Role:     []string{bkenode.EtcdNodeRole, bkenode.MasterNodeRole},
					IP:       "172.28.100.204",
					Port:     "22",
					Username: "root",
					Password: "123123",
					Hostname: "oldcc",
				},
			},
		},
		HostName: utils.HostName(),
		HostIP:   ip,
		CurrentNode: bkenode.Node{
			Role:     []string{bkenode.EtcdNodeRole, bkenode.MasterNodeRole},
			IP:       "172.28.100.203",
			Port:     "22",
			Username: "root",
			Password: "123123",
			Hostname: "master-1",
			ControlPlane: bkev1beta1.ControlPlane{
				//Etcd: bkev1beta1.Etcd{
				//	ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
				//		ExtraArgs: map[string]string{
				//			"listen-client-urls": "https://" + ip + ":2379,https://",
				//			"listen-peer-urls":   "https://" + ip + ":2380",
				//		},
				//	},
				//},
				APIServer: &bkev1beta1.APIServer{
					ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
						ExtraArgs: map[string]string{
							"authorization-mode": "Node,RBAC",
							//"advertise-address":  ip,
						},
					},
				},
			},
			Kubelet: nil,
		},
		Extra: map[string]interface{}{
			"Init":          true,
			"KubernetesDir": KubernetesDir,
		},
	}
)

func init() {
	bkeinit.SetDefaultBKEConfig((*bkeinit.BkeConfig)(cfg.BkeConfig))
}

func TestRender(t *testing.T) {
	//t.Run("TestRenderAll", testRenderAll)
	t.Run("TestRenderApiServer", testRenderApiServer)
	t.Run("TestRenderEtcd", testRenderEtcd)
	t.Run("TestRenderControllerManager", testRenderControllerManager)
	t.Run("TestRenderScheduler", testRenderScheduler)
}

func testRenderAll(t *testing.T) {
	err := GenerateManifestYaml(GetDefaultComponentList(), cfg)
	if err != nil {
		t.Error(err)
	}
}

func testRenderApiServer(t *testing.T) {

	c := APIServerComponent()
	c.MfPath = "./"

	err := GenerateManifestYaml(Components{c}, cfg)
	if err != nil {
		t.Errorf("failed to render api server static pod yaml: %v", err)
	}

}

func testRenderEtcd(t *testing.T) {
	err := GenerateManifestYaml(Components{EtcdComponent()}, cfg)
	if err != nil {
		t.Error(err)
	}
}

func testRenderControllerManager(t *testing.T) {
	c := ControllerComponent()
	c.MfPath = "./tmp/"

	err := GenerateManifestYaml(Components{c}, cfg)
	if err != nil {
		t.Error(err)
	}
}

func testRenderScheduler(t *testing.T) {
	err := GenerateManifestYaml(Components{SchedulerComponent()}, cfg)
	if err != nil {
		t.Error(err)
	}
}

func TestRenderIngressComponents(t *testing.T) {
	nodes := []HANode{
		{
			Hostname: "1",
			IP:       "1.1.1.1",
		},
		{
			Hostname: "2",
			IP:       "2.2.2.2",
		},
		{
			Hostname: "3",
			IP:       "172.28.8.102",
		},
	}

	config := map[string]interface{}{
		"vip":                      "0.0.0.0",
		"controlPlaneEndpointPort": "36443",
		"nodes":                    nodes,
		"keepAlivedConfigDir":      "./tmp",
		"haproxyConfigDir":         "./tmp",
	}
	config["interface"] = "ens33"
	config["keepalivedAdvertInt"] = "1"
	config["keepalivedAuthPass"] = "22222222"
	config["virtualRouterId"] = "51"

	cps := GetIngressHaComponentList()
	cps.SetMfPath("./tmp")

}
