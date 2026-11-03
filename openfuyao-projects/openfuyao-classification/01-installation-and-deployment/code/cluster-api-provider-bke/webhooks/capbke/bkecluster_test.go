package capbke

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"bou.ke/monkey"
	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkecommonv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

var expectedVMDefaults = map[string]string{
	"useVMSingle":                  "false",
	"vmSingleStorageSize":          "50Gi",
	"vmAgentAllowStatefulSet":      "true",
	"vmAgentCpuCount":              "4",
	"vmAgentMemorySize":            "8Gi",
	"vmAgentStorageSize":           "60Gi",
	"vmAgentReplicaCount":          "2",
	"vmAgentShareCount":            "2",
	"vmAgentScrapeInterval":        "20s",
	"vmSelectCpuCount":             "12",
	"vmSelectMemorySize":           "24Gi",
	"vmSelectStorageSize":          "35Gi",
	"vmSelectReplicaCount":         "2",
	"vmStorageCPUCount":            "5",
	"vmStorageMemorySize":          "32Gi",
	"vmStorageStorageSize":         "720Gi",
	"vmStorageReplicaCount":        "10",
	"vmInsertReplicaCount":         "6",
	"vmInsertCpuCount":             "4",
	"vmInsertMemorySize":           "8Gi",
	"vmAlertReplicaCount":          "2",
	"vmAlertCpuCount":              "4",
	"vmAlertMemorySize":            "16Gi",
	"vmAlertManagerReplicaCount":   "2",
	"vmAlertManagerCpuCount":       "4",
	"vmAlertManagerMemorySize":     "16Gi",
	"vmClusterRetentionPeriod":     "15d",
	"vmClusterReplicationFactor":   "2",
	"grafanaNodePort":              "30010",
	"kubeStateMetricsAutoSharding": "true",
	"kubeStateMetricsReplicaCount": "3",
	"kubeStateMetricsCpuCount":     "4",
	"kubeStateMetricsMemorySize":   "12Gi",
}

func newTestCluster(params map[string]string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &bkecommonv1beta1.BKEConfig{
				Addons: []bkecommonv1beta1.Product{
					{
						Name:  "victoriametrics-controller",
						Param: params,
					},
				},
			},
		},
	}
}

func TestValidateCreateParams(t *testing.T) {
	// 构造模拟节点：一个节点，包含所有需要的标签
	mockNode := bkecommonv1beta1.Node{
		IP: "192.168.1.11",
		Labels: []bkecommonv1beta1.Label{
			{Value: "vmstorage"},
			{Value: "vminsert"},
			{Value: "vmselect"},
			{Value: "vmagent"},
			{Value: "vmalert"},
			{Value: "vmalertmanager"},
			{Value: "kube-state-metrics"},
		},
		// 如果有其他必要字段（如 Role, Port, Username 等），也可根据需要设置
	}

	mockNodes := bkenode.Nodes{mockNode}

	// 辅助函数：创建 BKECluster 对象

	t.Run("节点获取失败", func(t *testing.T) {
		webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
		// 模拟 GetNodesForBKECluster 返回错误
		monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return nil, errors.New("connection refused")
			})
		defer monkey.UnpatchAll()

		cluster := newTestCluster(map[string]string{})
		err := webhook.validateCreateParams(context.Background(), cluster)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get nodes for BKECluster")
	})

	t.Run("参数合法，节点充足", func(t *testing.T) {
		webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
		monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return mockNodes, nil
			})
		defer monkey.UnpatchAll()

		params := map[string]string{
			"vmStorageReplicaCount":        "1",
			"vmInsertReplicaCount":         "1",
			"vmSelectReplicaCount":         "1",
			"vmAgentReplicaCount":          "1",
			"vmAgentShareCount":            "1",
			"vmAlertReplicaCount":          "1",
			"vmAlertManagerReplicaCount":   "1",
			"kubeStateMetricsReplicaCount": "1",
		}
		cluster := newTestCluster(params)
		err := webhook.validateCreateParams(context.Background(), cluster)
		assert.NoError(t, err)
	})

	t.Run("副本数超过可用节点", func(t *testing.T) {
		webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
		monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return mockNodes, nil
			})
		defer monkey.UnpatchAll()

		params := map[string]string{
			"vmStorageReplicaCount": "2", // 只有一个节点，超限
		}
		cluster := newTestCluster(params)
		err := webhook.validateCreateParams(context.Background(), cluster)
		assert.Error(t, err)
		// 验证错误类型为 Invalid
		statusErr, ok := err.(*apierrors.StatusError)
		assert.True(t, ok)
		assert.Equal(t, metav1.StatusReasonInvalid, statusErr.ErrStatus.Reason)
		assert.Contains(t, statusErr.Error(), "vmstorage requires 2 vmStorageReplicaCount but only 1 nodes")
	})

	t.Run("无 victoriametrics-controller addon", func(t *testing.T) {
		webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
		monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return mockNodes, nil
			})
		defer monkey.UnpatchAll()

		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &bkecommonv1beta1.BKEConfig{
					Addons: []bkecommonv1beta1.Product{
						{Name: "other-addon"}, // 不是目标 addon
					},
				},
			},
		}
		err := webhook.validateCreateParams(context.Background(), cluster)
		assert.NoError(t, err)
	})

	t.Run("useVMSingle=true 跳过校验", func(t *testing.T) {
		webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
		monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
				return mockNodes, nil
			})
		defer monkey.UnpatchAll()

		params := map[string]string{
			"useVMSingle":           "true",
			"vmStorageReplicaCount": "10", // 即使超限也不报错
		}
		cluster := newTestCluster(params)
		err := webhook.validateCreateParams(context.Background(), cluster)
		assert.NoError(t, err)
	})
}

func TestValidateBasicVMComponents(t *testing.T) {
	// 定义节点计数常量（可根据需要调整）
	nodeLabelCount := map[string]int{
		"vmstorage":          2,
		"vminsert":           2,
		"vmselect":           2,
		"vmagent":            3,
		"vmalert":            2,
		"vmalertmanager":     2,
		"kube-state-metrics": 2,
	}

	// 字段路径前缀
	paramPath := field.NewPath("spec", "clusterConfig", "addons").Index(0).Child("param")

	webhook := &BKECluster{}

	t.Run("所有参数合法", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmStorageReplicaCount":        "2",
				"vmInsertReplicaCount":         "2",
				"vmSelectReplicaCount":         "2",
				"vmAgentReplicaCount":          "2",
				"vmAgentShareCount":            "2",
				"vmAlertReplicaCount":          "2",
				"vmAlertManagerReplicaCount":   "2",
				"kubeStateMetricsReplicaCount": "2",
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)
	})

	t.Run("缺失部分字段（跳过，无错误）", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmStorageReplicaCount": "2",
				// 其他字段缺失
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)
	})

	t.Run("非整数字段", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmStorageReplicaCount": "abc",
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Equal(t, field.ErrorTypeInvalid, errs[0].Type)
			assert.Contains(t, errs[0].Detail, "vmstorage replica count must be a valid integer")
			assert.Equal(t, "spec.clusterConfig.addons[0].param.vmStorageReplicaCount", errs[0].Field)
		}
	})

	t.Run("副本数小于1", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmInsertReplicaCount": "0",
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Contains(t, errs[0].Detail, "vminsert replica count must be at least 1")
		}
	})

	t.Run("副本数超过可用节点", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmSelectReplicaCount": "3", // 只有2个节点
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Contains(t, errs[0].Detail, "vmselect requires 3 vmSelectReplicaCount but only 2 nodes can be used")
		}
	})

	t.Run("多个错误同时存在", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmStorageReplicaCount":        "abc", // 非整数
				"vmInsertReplicaCount":         "0",   // 小于1
				"vmSelectReplicaCount":         "3",   // 超限
				"vmAgentReplicaCount":          "5",   // 超限（节点3）
				"vmAgentShareCount":            "2",   // 合法
				"vmAlertReplicaCount":          "2",   // 合法
				"vmAlertManagerReplicaCount":   "xyz", // 非整数
				"kubeStateMetricsReplicaCount": "-1",  // 小于1
			},
		}
		errs := webhook.validateBasicVMComponents(addon, paramPath, nodeLabelCount)
		// 期望错误数：vmStorage(1), vmInsert(1), vmSelect(1), vmAgent(1), vmAlertManager(1), kubeStateMetrics(1) = 6
		assert.Len(t, errs, 6)
	})
}

func TestDefaultConfigContextAppliesCoreDefaults(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "cluster-a", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster:     confv1beta1.Cluster{},
				CustomExtra: map[string]string{},
				Addons: []confv1beta1.Product{
					{Name: "bocoperator", Param: map[string]string{}},
					{Name: "kubeproxy", Param: map[string]string{"proxyMode": "ipvs"}},
					{Name: "victoriametrics-controller"},
				},
			},
		},
	}
	ctx := &defaultConfigContext{
		bkeCluster: cluster,
		data: map[string]string{
			"yumRepoPort":     "40080",
			"imageRepoPort":   "5000",
			"imageRepoPrefix": "openfuyao",
			"ntpServer":       "ntp.example.com",
			"containerd":      "containerd-1.7.0-linux-amd64.tar.gz",
			"host":            "192.168.1.10",
		},
		bkeNodes: bkenode.Nodes{
			confv1beta1.Node{
				Role: []string{"master", "etcd"},
				IP:   "192.168.1.11",
			},
		},
	}

	ctx.setRepoDefaults()
	ctx.setNTPServerDefault()
	ctx.setContainerdDefault()
	ctx.setHostRelatedDefaults()
	ctx.setAddonDefaults()
	for _, patch := range ctx.patchFuncs {
		patch(cluster)
	}

	assert.Equal(t, "40080", cluster.Spec.ClusterConfig.Cluster.HTTPRepo.Port)
	assert.Equal(t, "5000", cluster.Spec.ClusterConfig.Cluster.ImageRepo.Port)
	assert.Equal(t, "openfuyao", cluster.Spec.ClusterConfig.Cluster.ImageRepo.Prefix)
	assert.Equal(t, "ntp.example.com", cluster.Spec.ClusterConfig.Cluster.NTPServer)
	assert.Equal(t, "containerd-1.7.0-linux-{.arch}.tar.gz", cluster.Spec.ClusterConfig.CustomExtra["containerd"])
	assert.Equal(t, "192.168.1.10", cluster.Spec.ClusterConfig.Cluster.HTTPRepo.Ip)
	assert.Equal(t, "192.168.1.10", cluster.Spec.ClusterConfig.Cluster.ImageRepo.Ip)
	assert.Equal(t, "192.168.1.10", cluster.Spec.ClusterConfig.CustomExtra["host"])
	assert.Equal(t, "192.168.1.10", cluster.Spec.ClusterConfig.CustomExtra["nfsServer"])
	assert.Equal(t, "/", cluster.Spec.ClusterConfig.CustomExtra["nfsRootDir"])
	assert.Equal(t, "4.1", cluster.Spec.ClusterConfig.CustomExtra["nfsVersion"])
	assert.Equal(t, "192.168.1.10", cluster.Spec.ClusterConfig.Addons[0].Param["deployServerIp"])
	assert.Equal(t, "192.168.1.11", cluster.Spec.ClusterConfig.CustomExtra["pipelineServer"])
	assert.Equal(t, "false", cluster.Spec.ClusterConfig.CustomExtra["pipelineServerEnableCleanImages"])
	assert.Equal(t, "ipvs", cluster.Spec.ClusterConfig.CustomExtra["proxyMode"])
	assert.Equal(t, expectedVMDefaults["vmAgentCpuCount"], cluster.Spec.ClusterConfig.Addons[2].Param["vmAgentCpuCount"])
}

func TestDefaultConfigContextPreservesExistingDefaults(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					HTTPRepo:  confv1beta1.Repo{Ip: "existing-http", Port: "1"},
					ImageRepo: confv1beta1.Repo{Ip: "existing-image", Port: "2", Prefix: "existing"},
					NTPServer: "existing-ntp",
				},
				CustomExtra: map[string]string{
					"containerd":                      "existing-containerd",
					"host":                            "existing-host",
					"nfsServer":                       "existing-nfs",
					"nfsRootDir":                      "/data",
					"nfsVersion":                      "3",
					"pipelineServer":                  "existing-pipeline",
					"pipelineServerEnableCleanImages": "true",
				},
				Addons: []confv1beta1.Product{{Name: "bocoperator", Param: map[string]string{}}},
			},
		},
	}
	ctx := &defaultConfigContext{
		bkeCluster: cluster,
		data: map[string]string{
			"yumRepoPort":     "40080",
			"imageRepoPort":   "5000",
			"imageRepoPrefix": "openfuyao",
			"ntpServer":       "ntp.example.com",
			"containerd":      "containerd-1.7.0-linux-amd64.tar.gz",
			"host":            "192.168.1.10",
		},
		bkeNodes: bkenode.Nodes{{IP: "192.168.1.11", Role: []string{"master"}}},
	}
	ctx.setRepoDefaults()
	ctx.setNTPServerDefault()
	ctx.setContainerdDefault()
	ctx.setHostRelatedDefaults()
	ctx.setAddonDefaults()
	for _, patch := range ctx.patchFuncs {
		patch(cluster)
	}

	assert.Equal(t, "1", cluster.Spec.ClusterConfig.Cluster.HTTPRepo.Port)
	assert.Equal(t, "2", cluster.Spec.ClusterConfig.Cluster.ImageRepo.Port)
	assert.Equal(t, "existing", cluster.Spec.ClusterConfig.Cluster.ImageRepo.Prefix)
	assert.Equal(t, "existing-ntp", cluster.Spec.ClusterConfig.Cluster.NTPServer)
	assert.Equal(t, "existing-containerd", cluster.Spec.ClusterConfig.CustomExtra["containerd"])
	assert.Equal(t, "existing-host", cluster.Spec.ClusterConfig.CustomExtra["host"])
	assert.Equal(t, "existing-nfs", cluster.Spec.ClusterConfig.CustomExtra["nfsServer"])
	assert.Equal(t, "/data", cluster.Spec.ClusterConfig.CustomExtra["nfsRootDir"])
	assert.Equal(t, "3", cluster.Spec.ClusterConfig.CustomExtra["nfsVersion"])
	assert.Equal(t, "existing-pipeline", cluster.Spec.ClusterConfig.CustomExtra["pipelineServer"])
	assert.Equal(t, "true", cluster.Spec.ClusterConfig.CustomExtra["pipelineServerEnableCleanImages"])
}

func TestValidateVMAgentSpecialRule(t *testing.T) {
	// 固定节点计数（假设有3个 vmagent 节点）
	nodeLabelCount := map[string]int{
		"vmagent": 3,
	}

	// 字段路径前缀
	paramPath := field.NewPath("spec", "clusterConfig", "addons").Index(0).Child("param")
	webhook := &BKECluster{}

	t.Run("缺少任一字段则跳过", func(t *testing.T) {
		// 缺少 share
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "2",
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)

		// 缺少 replica
		addon = confv1beta1.Product{
			Param: map[string]string{
				"vmAgentShareCount": "2",
			},
		}
		errs = webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)
	})

	t.Run("replica 非整数", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "abc",
				"vmAgentShareCount":   "2",
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Equal(t, field.ErrorTypeInvalid, errs[0].Type)
			assert.Equal(t, "spec.clusterConfig.addons[0].param.vmAgentReplicaCount", errs[0].Field)
			assert.Contains(t, errs[0].Detail, "vmAgentReplicaCount must be a valid integer")
		}
	})

	t.Run("share 非整数", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "2",
				"vmAgentShareCount":   "xyz",
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Equal(t, field.ErrorTypeInvalid, errs[0].Type)
			assert.Equal(t, "spec.clusterConfig.addons[0].param.vmAgentShareCount", errs[0].Field)
			assert.Contains(t, errs[0].Detail, "vmAgentShareCount must be a valid integer")
		}
	})

	t.Run("乘积超过可用节点", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "2",
				"vmAgentShareCount":   "2", // 2*2=4 > 3
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		if assert.Len(t, errs, 1) {
			assert.Equal(t, field.ErrorTypeInvalid, errs[0].Type)
			assert.Equal(t, "spec.clusterConfig.addons[0].param.vmAgentShareCount", errs[0].Field)
			assert.Contains(t, errs[0].Detail, "vmagent requires 4 total instances (replicas:2 × share:2) but only 3 nodes available")
		}
	})

	t.Run("乘积等于可用节点", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "3",
				"vmAgentShareCount":   "1", // 3*1=3 == 3
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)
	})

	t.Run("乘积小于可用节点", func(t *testing.T) {
		addon := confv1beta1.Product{
			Param: map[string]string{
				"vmAgentReplicaCount": "2",
				"vmAgentShareCount":   "1", // 2*1=2 < 3
			},
		}
		errs := webhook.validateVMAgentSpecialRule(addon, paramPath, nodeLabelCount)
		assert.Empty(t, errs)
	})
}

func TestSetVictoriaMetricsControllerDefaultConfig(t *testing.T) {
	t.Run("Param 为 nil，应初始化并设置所有默认值", func(t *testing.T) {
		cluster := newTestCluster(nil)
		cluster.Spec.ClusterConfig.Addons[0].Param = nil // 确保为 nil
		ctx := &defaultConfigContext{
			bkeCluster: cluster,
			patchFuncs: []func(*bkev1beta1.BKECluster){},
		}
		addon := cluster.Spec.ClusterConfig.Addons[0]
		ctx.setVictoriaMetricsControllerDefaultConfig(addon, 0)

		// 应用所有 patch
		for _, patch := range ctx.patchFuncs {
			patch(cluster)
		}

		got := cluster.Spec.ClusterConfig.Addons[0].Param
		assert.NotNil(t, got)
		for key, expected := range expectedVMDefaults {
			assert.Equal(t, expected, got[key], "key %s", key)
		}
	})

	t.Run("Param 为空 map，应设置所有默认值", func(t *testing.T) {
		cluster := newTestCluster(map[string]string{})
		ctx := &defaultConfigContext{
			bkeCluster: cluster,
			patchFuncs: []func(*bkev1beta1.BKECluster){},
		}
		addon := cluster.Spec.ClusterConfig.Addons[0]
		ctx.setVictoriaMetricsControllerDefaultConfig(addon, 0)

		for _, patch := range ctx.patchFuncs {
			patch(cluster)
		}

		got := cluster.Spec.ClusterConfig.Addons[0].Param
		for key, expected := range expectedVMDefaults {
			assert.Equal(t, expected, got[key], "key %s", key)
		}
	})

	t.Run("部分参数已存在，不应覆盖", func(t *testing.T) {
		existing := map[string]string{
			"useVMSingle":           "true",
			"vmStorageReplicaCount": "5",
			"vmAgentReplicaCount":   "3",
		}
		cluster := newTestCluster(existing)
		ctx := &defaultConfigContext{
			bkeCluster: cluster,
			patchFuncs: []func(*bkev1beta1.BKECluster){},
		}
		addon := cluster.Spec.ClusterConfig.Addons[0]
		ctx.setVictoriaMetricsControllerDefaultConfig(addon, 0)

		for _, patch := range ctx.patchFuncs {
			patch(cluster)
		}

		got := cluster.Spec.ClusterConfig.Addons[0].Param
		// 已存在的值保持不变
		assert.Equal(t, "true", got["useVMSingle"])
		assert.Equal(t, "5", got["vmStorageReplicaCount"])
		assert.Equal(t, "3", got["vmAgentReplicaCount"])
		// 其他缺失的被设置为默认
		assert.Equal(t, "50Gi", got["vmSingleStorageSize"])
		assert.Equal(t, "true", got["vmAgentAllowStatefulSet"])
		// 可继续验证其他默认键是否存在，这里只抽查几个
	})

	t.Run("所有参数都已存在，不应产生任何 patch", func(t *testing.T) {
		// 构造一个包含所有默认键的 map（值任意，只要存在即可）
		fullParams := make(map[string]string)
		for k := range expectedVMDefaults {
			fullParams[k] = "some-value"
		}
		cluster := newTestCluster(fullParams)
		ctx := &defaultConfigContext{
			bkeCluster: cluster,
			patchFuncs: []func(*bkev1beta1.BKECluster){},
		}
		addon := cluster.Spec.ClusterConfig.Addons[0]
		ctx.setVictoriaMetricsControllerDefaultConfig(addon, 0)

		// 不应该有新的 patch 被添加（因为所有键都存在）
		assert.Empty(t, ctx.patchFuncs)

		// 应用 patch（实际上没有），参数应保持不变
		for _, patch := range ctx.patchFuncs {
			patch(cluster)
		}
		got := cluster.Spec.ClusterConfig.Addons[0].Param
		for k, v := range fullParams {
			assert.Equal(t, v, got[k], "key %s should remain unchanged", k)
		}
	})
}

func TestBKECluster_ValidateCreate(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	t.Run("non-bke cluster skip validation", func(t *testing.T) {
		bkeNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master/node", "etcd"}},
		}
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test",
				Namespace:   "default",
				Annotations: map[string]string{"bke.bocloud.com/cluster-from": "import"},
			},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeNode).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		_, err := webhook.ValidateCreate(context.Background(), cluster)
		assert.NoError(t, err)
	})

	t.Run("create with associated bkenode allowed", func(t *testing.T) {
		bkeNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-master",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master/node", "etcd"}},
		}
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test-cluster",
				Namespace:   "default",
				Annotations: map[string]string{"bke.bocloud.com/cluster-from": "bocloud"},
			},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeNode).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		_, err := webhook.ValidateCreate(context.Background(), cluster)
		assert.NoError(t, err)
	})

	t.Run("create without associated bkenode not allowed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-cluster",
				Namespace: "default",
			},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		_, err := webhook.ValidateCreate(context.Background(), cluster)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requires at least one associated BKENode")
	})

	t.Run("create managed cluster without bkenode not allowed", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test-cluster",
				Namespace:   "default",
				Annotations: map[string]string{"bke.bocloud.com/cluster-from": "bocloud"},
			},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		_, err := webhook.ValidateCreate(context.Background(), cluster)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requires at least one associated BKENode")
	})

	t.Run("create with only worker bkenode not allowed", func(t *testing.T) {
		bkeNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"node"}},
		}
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test-cluster",
				Namespace:   "default",
				Annotations: map[string]string{"bke.bocloud.com/cluster-from": "bocloud"},
			},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(bkeNode).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		_, err := webhook.ValidateCreate(context.Background(), cluster)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master")
	})
}

func TestBKECluster_ValidateDelete(t *testing.T) {
	webhook := &BKECluster{}
	cluster := &bkev1beta1.BKECluster{}
	_, err := webhook.ValidateDelete(context.Background(), cluster)
	assert.NoError(t, err)
}

func TestBKECluster_ValidateUpdate(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	t.Run("pause during addon deployment not allowed", func(t *testing.T) {
		oldCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
			Spec:       confv1beta1.BKEClusterSpec{Pause: false},
			Status: confv1beta1.BKEClusterStatus{
				ClusterHealthState: bkev1beta1.Deploying,
				ClusterStatus:      bkev1beta1.ClusterDeployingAddon,
			},
		}
		newCluster := oldCluster.DeepCopy()
		newCluster.Spec.Pause = true

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldCluster, newCluster)
		assert.Error(t, err)
	})

	t.Run("precheck phase skip validation", func(t *testing.T) {
		oldCluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test",
				Namespace:   "default",
				Annotations: map[string]string{"bke.bocloud.com/at-precheck-phase": "true"},
			},
			Spec: confv1beta1.BKEClusterSpec{Pause: true},
		}
		newCluster := oldCluster.DeepCopy()

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldCluster, newCluster)
		assert.NoError(t, err)
	})
}

func TestBKECluster_Default(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	t.Run("set default config", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
			Spec:       confv1beta1.BKEClusterSpec{},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{
			Client:      client,
			APIReader:   client,
			NodeFetcher: nodeutil.NewNodeFetcher(client),
		}
		err := webhook.Default(context.Background(), cluster)
		assert.NoError(t, err)
		assert.NotNil(t, cluster.Spec.ClusterConfig)
	})

	t.Run("wrong type returns error", func(t *testing.T) {
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKECluster{Client: client}
		err := webhook.Default(context.Background(), &confv1beta1.BKENode{})
		assert.Error(t, err)
	})
}

func TestBKECluster_ValidateCreateInvalidType(t *testing.T) {
	webhook := &BKECluster{}
	_, err := webhook.ValidateCreate(context.Background(), &confv1beta1.BKENode{})
	assert.Error(t, err)
}

func TestBKECluster_ValidateUpdateInvalidType(t *testing.T) {
	webhook := &BKECluster{}
	_, err := webhook.ValidateUpdate(context.Background(), &confv1beta1.BKENode{}, &confv1beta1.BKENode{})
	assert.Error(t, err)
}

func TestValidateControlPlaneEndpoint(t *testing.T) {
	webhook := &BKECluster{}

	tests := []struct {
		name      string
		cluster   *bkev1beta1.BKECluster
		wantError bool
	}{
		{
			name: "valid endpoint",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: "192.168.1.1",
						Port: 6443,
					},
				},
			},
			wantError: false,
		},
		{
			name: "invalid IP",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: "invalid-ip",
						Port: 6443,
					},
				},
			},
			wantError: true,
		},
		{
			name: "zero endpoint",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{},
				},
			},
			wantError: true,
		},
		{
			name: "missing port",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: "192.168.1.1",
						Port: 0,
					},
				},
			},
			wantError: true,
		},
		{
			name: "invalid port",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: "192.168.1.1",
						Port: 70000,
					},
				},
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := webhook.ValidateControlPlaneEndpoint(tt.cluster)
			if tt.wantError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPathsMatch(t *testing.T) {
	tests := []struct {
		name        string
		patternPath []string
		actualPath  []string
		want        bool
	}{
		{"exact match", []string{"a", "b"}, []string{"a", "b"}, true},
		{"wildcard match", []string{"a", "*"}, []string{"a", "b", "c"}, true},
		{"no match", []string{"a", "b"}, []string{"a", "c"}, false},
		{"empty pattern", []string{}, []string{"a"}, false},
		{"empty actual", []string{"a"}, []string{}, false},
		{"pattern longer", []string{"a", "b", "c"}, []string{"a", "b"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pathsMatch(tt.patternPath, tt.actualPath)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateImmutableOSConstraints(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		nodes       bkenode.Nodes
		wantError   bool
		errContains string
	}{
		{
			name:        "非不可变模式直接通过",
			annotations: nil,
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Role: []string{"master", "node"}},
			},
			wantError: false,
		},
		{
			name:        "不可变模式-纯master节点",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Role: []string{"master"}},
			},
			wantError: false,
		},
		{
			name:        "不可变模式-纯worker节点",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.2", Role: []string{"node"}},
			},
			wantError: false,
		},
		{
			name:        "不可变模式-master和node分列数组共节点",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Role: []string{"master", "node"}},
			},
			wantError:   true,
			errContains: "10.0.0.1",
		},
		{
			name:        "不可变模式-master/node单值视为合法control-plane",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Role: []string{"master/node"}},
			},
			wantError: false,
		},
		{
			name:        "不可变模式-多节点其中一个共节点",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Role: []string{"master"}},
				{IP: "10.0.0.2", Role: []string{"master", "node"}},
				{IP: "10.0.0.3", Role: []string{"node"}},
			},
			wantError:   true,
			errContains: "10.0.0.2",
		},
		{
			name:        "不可变模式-同一IP多BKENode角色冲突",
			annotations: map[string]string{"openfuyao.io/immutable-os": "kubeos"},
			nodes: bkenode.Nodes{
				{IP: "10.0.0.1", Hostname: "node-1", Role: []string{"master"}},
				{IP: "10.0.0.1", Hostname: "node-2", Role: []string{"node"}},
			},
			wantError:   true,
			errContains: "10.0.0.1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			webhook := &BKECluster{NodeFetcher: &nodeutil.NodeFetcher{}}
			monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
				func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
					return tt.nodes, nil
				})
			defer monkey.UnpatchAll()

			cluster := &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-cluster",
					Namespace:   "default",
					Annotations: tt.annotations,
				},
			}
			err := webhook.validateImmutableOSConstraints(context.Background(), cluster)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errContains != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateLoadBalanceType(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	masterNodes := bkenode.Nodes{
		{IP: "10.0.0.10", Role: []string{bkenode.MasterNodeRole}},
		{IP: "10.0.0.11", Role: []string{bkenode.MasterNodeRole}},
	}

	newCluster := func(mutate func(*bkev1beta1.BKECluster)) *bkev1beta1.BKECluster {
		c := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{},
			},
		}
		if mutate != nil {
			mutate(c)
		}
		return c
	}
	withLB := func(lbType string) func(*bkev1beta1.BKECluster) {
		return func(c *bkev1beta1.BKECluster) { c.Spec.ClusterConfig.Cluster.LoadBalanceType = lbType }
	}

	tests := []struct {
		name    string
		cluster *bkev1beta1.BKECluster
		wantErr bool
	}{
		{name: "empty value treated as ha", cluster: newCluster(nil), wantErr: false},
		{name: "explicit ha", cluster: newCluster(withLB("ha")), wantErr: false},
		{name: "invalid enum value rejected", cluster: newCluster(withLB("slb")), wantErr: true},
		{name: "nginx without endpoint host", cluster: newCluster(withLB("nginx")), wantErr: false},
		{name: "nginx with master ip allowed", cluster: newCluster(func(c *bkev1beta1.BKECluster) {
			c.Spec.ClusterConfig.Cluster.LoadBalanceType = "nginx"
			c.Spec.ControlPlaneEndpoint = confv1beta1.APIEndpoint{Host: "10.0.0.10", Port: 6443}
		}), wantErr: false},
		{name: "nginx with vip rejected", cluster: newCluster(func(c *bkev1beta1.BKECluster) {
			c.Spec.ClusterConfig.Cluster.LoadBalanceType = "nginx"
			c.Spec.ControlPlaneEndpoint = confv1beta1.APIEndpoint{Host: "10.0.0.100", Port: 6443}
		}), wantErr: true},
		{name: "nginx with 127.0.0.1 rejected", cluster: newCluster(func(c *bkev1beta1.BKECluster) {
			c.Spec.ClusterConfig.Cluster.LoadBalanceType = "nginx"
			c.Spec.ControlPlaneEndpoint = confv1beta1.APIEndpoint{Host: "127.0.0.1", Port: 6443}
		}), wantErr: true},
		{name: "nginx with extraLoadBalanceIP rejected", cluster: newCluster(func(c *bkev1beta1.BKECluster) {
			c.Spec.ClusterConfig.Cluster.LoadBalanceType = "nginx"
			c.Spec.ClusterConfig.CustomExtra = map[string]string{"extraLoadBalanceIP": "10.0.0.200"}
		}), wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := fake.NewClientBuilder().WithScheme(scheme).Build()
			webhook := &BKECluster{Client: client, NodeFetcher: nodeutil.NewNodeFetcher(client)}
			monkey.PatchInstanceMethod(reflect.TypeOf(webhook.NodeFetcher), "GetNodesForBKECluster",
				func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
					return masterNodes, nil
				})
			defer monkey.UnpatchAll()

			err := webhook.ValidateLoadBalanceType(context.Background(), tt.cluster)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateFieldUpdateRejectsLoadBalanceType(t *testing.T) {
	webhook := &BKECluster{}
	oldCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{LoadBalanceType: "ha"}},
		},
	}
	newCluster := oldCluster.DeepCopy()
	newCluster.Spec.ClusterConfig.Cluster.LoadBalanceType = "nginx"
	assert.Error(t, webhook.validateFieldUpdate(newCluster, oldCluster, notAllowedPaths))

	sameCluster := oldCluster.DeepCopy()
	assert.NoError(t, webhook.validateFieldUpdate(sameCluster, oldCluster, notAllowedPaths))
}
