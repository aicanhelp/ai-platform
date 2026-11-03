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

package phases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/cluster-api/util"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureClusterAPIObjName confv1beta1.BKEClusterPhase = "EnsureClusterAPIObj"
	// ClusterAPIObjTimeoutMinutes 控制ClusterAPI对象操作的超时时间（分钟）
	ClusterAPIObjTimeoutMinutes = 5
	// ClusterAPIObjPollIntervalSeconds 控制ClusterAPI对象操作的轮询间隔（秒）
	ClusterAPIObjPollIntervalSeconds = 2
)

type EnsureClusterAPIObj struct {
	phaseframe.BasePhase
}

func NewEnsureClusterAPIObj(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureClusterAPIObjName)
	return &EnsureClusterAPIObj{
		BasePhase: base,
	}
}

func (e *EnsureClusterAPIObj) Execute() (ctrl.Result, error) {
	if e.Ctx.BKECluster.OwnerReferences == nil {
		if err := e.reconcileCreateClusterAPIObj(); err != nil {
			return ctrl.Result{}, err
		}
	}

	ctx, cancel := context.WithTimeout(e.Ctx.Context, ClusterAPIObjTimeoutMinutes*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(ClusterAPIObjPollIntervalSeconds*time.Second, func() (bool, error) {
		if err := e.reconcileClusterAPIObj(ctx); err != nil {
			return false, nil
		}
		return true, nil
	}, ctx.Done())

	if errors.Is(err, wait.ErrWaitTimeout) {
		return ctrl.Result{}, errors.Errorf("Wait master init failed")
	}

	if !clusterutil.FullyControlled(e.Ctx.BKECluster) {
		return ctrl.Result{Requeue: true}, nil
	}

	return ctrl.Result{}, nil
}

func (e *EnsureClusterAPIObj) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.NormalNeedExecute(old, new) {
		return false
	}
	if new.OwnerReferences != nil {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

func (e *EnsureClusterAPIObj) reconcileCreateClusterAPIObj() error {
	phaseContextFields99 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields99.Context,
		phaseContextFields99.Client,
		phaseContextFields99.BKECluster,
		phaseContextFields99.Scheme,
		phaseContextFields99.Log

	defer func() {
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			log.Warn(constant.InternalErrorReason, "failed to patch BKECluster: %v", err)
			return
		}
	}()

	// 检查ClusterAPI对象条件
	if _, ok := condition.HasCondition(bkev1beta1.ClusterAPIObjCondition, bkeCluster); ok {
		log.Info(constant.ClusterAPIObjNotReadyReason, "Waiting cluster api obj reconciled")
		return errors.New("Waiting cluster api obj reconciled")
	}

	log.Info(constant.ClusterAPIObjCreatingReason, "Start create cluster api obj")

	// 创建BKE配置
	cfg, err := bkeinit.NewBkeConfigFromClusterConfig(bkeCluster.Spec.ClusterConfig)
	if err != nil {
		return fmt.Errorf("Failed to create bke config: %w", err)
	}

	// 准备外部etcd配置（如果需要）
	externalEtcd, err := e.prepareExternalEtcdConfig(bkeCluster)
	if err != nil {
		return fmt.Errorf("Failed to prepare external etcd config: %w", err)
	}

	// 创建集群API对象
	params := CreateClusterAPIObjParams{
		Ctx:          ctx,
		Client:       c,
		BKECluster:   bkeCluster,
		Cfg:          cfg,
		ExternalEtcd: externalEtcd,
		Log:          log,
	}
	return e.createClusterAPIObj(params)
}

// prepareExternalEtcdConfig 准备外部etcd配置
func (e *EnsureClusterAPIObj) prepareExternalEtcdConfig(bkeCluster *bkev1beta1.BKECluster) (map[string]string, error) {
	if !clusterutil.IsBocloudCluster(bkeCluster) {
		return nil, nil
	}

	externalEtcd := bkeinit.NewExternalEtcdConfig()
	if externalEtcd == nil {
		return nil, nil
	}

	externalEtcd["etcdCAFile"] = "fakeCaCert"
	externalEtcd["etcdCertFile"] = "fakeCertFile"
	externalEtcd["etcdKeyFile"] = "fakeKeyFile"

	allNodes, err := e.Ctx.GetNodes()
	if err != nil {
		return nil, fmt.Errorf("failed to get nodes: %w", err)
	}

	etcdEndpoints := buildEtcdEndpoints(bkenode.Nodes(allNodes).Etcd())
	externalEtcd["etcdEndpoints"] = etcdEndpoints
	return externalEtcd, nil
}

// buildEtcdEndpoints 构建 etcd 端点列表
func buildEtcdEndpoints(etcdNodes bkenode.Nodes) string {
	endpoints := make([]string, 0, len(etcdNodes))
	for _, node := range etcdNodes {
		endpoints = append(endpoints, fmt.Sprintf("https://%s:2379", node.IP))
	}
	return strings.Join(endpoints, ",")
}

// CreateClusterAPIObjParams 包含创建集群API对象的参数
type CreateClusterAPIObjParams struct {
	Ctx          context.Context
	Client       client.Client
	BKECluster   *bkev1beta1.BKECluster
	Cfg          *bkeinit.BkeConfig
	ExternalEtcd map[string]string
	Log          *bkev1beta1.BKELogger
}

// createClusterAPIObj 创建集群API对象
func (e *EnsureClusterAPIObj) createClusterAPIObj(params CreateClusterAPIObjParams) error {
	// generate cluster api obj yaml
	yamlPath, err := params.Cfg.GenerateClusterAPIConfigFIle(params.BKECluster.Name, params.BKECluster.Namespace, params.ExternalEtcd)
	if err != nil {
		return fmt.Errorf("Failed to generate cluster api config file: %w", err)
	}

	localClient, err := kube.NewClientFromRestConfig(params.Ctx, e.Ctx.RestConfig)
	if err != nil {
		return fmt.Errorf("Failed to create kube client: %w", err)
	}

	task := kube.NewTask("cluster-api", yamlPath, nil).
		SetOperate(bkeaddon.CreateAddon).
		SetWaiter(true, bkeinit.DefaultAddonTimeout, bkeinit.DefaultAddonInterval)
	// apply cluster api obj yaml
	if err := localClient.ApplyYaml(task); err != nil {
		return fmt.Errorf("Failed to create cluster api obj: %w", err)
	}

	params.Log.Info(constant.ClusterAPIObjNotReadyReason, "Cluster api obj create success")
	condition.ConditionMark(params.BKECluster, bkev1beta1.ClusterAPIObjCondition, confv1beta1.ConditionFalse, constant.ClusterAPIObjNotReadyReason, "cluster api obj create success")
	return nil
}

func (e *EnsureClusterAPIObj) reconcileClusterAPIObj(ctx context.Context) error {
	phaseContextFields211 := e.Ctx.Untie()
	_, c, bkeCluster, _, _ := phaseContextFields211.Context,
		phaseContextFields211.Client,
		phaseContextFields211.BKECluster,
		phaseContextFields211.Scheme,
		phaseContextFields211.Log
	ns, name := bkeCluster.Namespace, bkeCluster.Name
	combined, err := mergecluster.GetCombinedBKECluster(ctx, e.Ctx.Client, ns, name)
	if err != nil {
		return fmt.Errorf("get combined bkeCluster %s/%s: %w", ns, name, err)
	}
	bkeCluster = combined
	e.Ctx.BKECluster = bkeCluster

	if condition.HasConditionStatus(bkev1beta1.ClusterAPIObjCondition, bkeCluster, confv1beta1.ConditionFalse) {
		condition.ConditionMark(bkeCluster, bkev1beta1.ClusterAPIObjCondition, confv1beta1.ConditionTrue, constant.ClusterAPIObjReadyReason, "cluster api obj ready")
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			return fmt.Errorf("failed to update bkeCluster Status: %w", err)
		}
	}
	if e.Ctx.BKECluster.OwnerReferences != nil {
		cluster, err := util.GetOwnerCluster(ctx, c, bkeCluster.ObjectMeta)
		if err != nil {
			if apierrors.IsNotFound(err) {
				return errors.New("Cluster Obj not found")
			}
			return fmt.Errorf("Failed to get owner cluster: %w", err)
		}
		if cluster == nil {
			return errors.New("Cluster Controller has not yet set OwnerRef")
		}

		e.Ctx.Cluster = cluster
	}
	return nil
}
