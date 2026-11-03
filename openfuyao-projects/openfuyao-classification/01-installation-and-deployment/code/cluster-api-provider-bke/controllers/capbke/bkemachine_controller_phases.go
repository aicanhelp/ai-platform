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

package capbke

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/blang/semver"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	kubedrain "k8s.io/kubectl/pkg/drain"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	bootstrapv1 "sigs.k8s.io/cluster-api/bootstrap/kubeadm/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/cluster-api/util/patch"
	"sigs.k8s.io/cluster-api/util/version"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	metricrecord "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics/record"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phases"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	DefaultRequeueAfterDuration = 5 * time.Second
	DefaultNodeConnectTimeout   = 4 * time.Minute
	CertificateExpiryYears      = 100
)

// CommonContextParams holds common context parameters
type CommonContextParams struct {
	Ctx context.Context
	Log *log.Logger
}

// CommonResourceParams holds common resource parameters
type CommonResourceParams struct {
	CommonContextParams
	Machine    *clusterv1.Machine
	Cluster    *clusterv1.Cluster
	BKEMachine *bkev1beta1.BKEMachine
	BKECluster *bkev1beta1.BKECluster
}

// CommonNodeParams holds common node parameters
type CommonNodeParams struct {
	CommonResourceParams
	Node *confv1beta1.Node
	Role string
}

// CommonCommandParams holds common command parameters
type CommonCommandParams struct {
	CommonResourceParams
	PatchHelper  *patch.Helper
	Cmd          *agentv1beta1.Command
	Complete     bool
	SuccessNodes []string
	FailedNodes  []string
	Res          ctrl.Result
	Errs         []error
}

// BootstrapReconcileParams holds parameters for bootstrap reconciliation
type BootstrapReconcileParams struct {
	CommonResourceParams
}

// FakeBootstrapParams holds parameters for fake bootstrap reconciliation
type FakeBootstrapParams struct {
	CommonNodeParams
}

// RealBootstrapParams holds parameters for real bootstrap reconciliation
type RealBootstrapParams struct {
	CommonNodeParams
	Phase confv1beta1.BKEClusterPhase
}

// ProcessCommandParams holds parameters for command processing
type ProcessCommandParams struct {
	CommonResourceParams
	PatchHelper *patch.Helper
	Nodes       bkenode.Nodes
	HostIp      string
	Cmd         agentv1beta1.Command
	Res         ctrl.Result
	Errs        []error
}

// ProcessBootstrapCommandParams holds parameters for bootstrap command processing
type ProcessBootstrapCommandParams struct {
	CommonResourceParams
	PatchHelper  *patch.Helper
	CurrentNode  confv1beta1.Node // 注意：这是值类型，不是指针
	Cmd          *agentv1beta1.Command
	Complete     bool
	SuccessNodes []string
	FailedNodes  []string
	Res          ctrl.Result
	Errs         []error
}

// ProcessBootstrapCommonParams holds common parameters for bootstrap processing
type ProcessBootstrapCommonParams struct {
	CommonResourceParams
	PatchHelper *patch.Helper
	CurrentNode confv1beta1.Node // 注意：这是值类型，不是指针
	Cmd         *agentv1beta1.Command
	Res         ctrl.Result
	Errs        []error
}

// ProcessBootstrapFailureParams holds parameters for bootstrap failure processing
type ProcessBootstrapFailureParams struct {
	ProcessBootstrapCommonParams
	FailedNodes []string
	Role        string
}

// ProcessBootstrapSuccessParams holds parameters for bootstrap success processing
type ProcessBootstrapSuccessParams struct {
	ProcessBootstrapCommonParams
}

// ProcessResetCommandParams holds parameters for reset command processing
type ProcessResetCommandParams struct {
	CommonCommandParams
	CurrentNode confv1beta1.Node // 注意：这是值类型，不是指针
}

// HandleClusterStateParams holds parameters for cluster state handling
type HandleClusterStateParams struct {
	CommonContextParams
	BKECluster          *bkev1beta1.BKECluster
	BKEMachine          *bkev1beta1.BKEMachine
	NodeState           confv1beta1.Node // 重命名以避免冲突
	BKEMachines         []bkev1beta1.BKEMachine
	Nodes               bkenode.Nodes
	ClusterReady        bool
	BootstrapNodeFailed bool
}

// reconcileBootstrap is the bootstrap reconcile flow for a BKEMachine.
func (r *BKEMachineReconciler) reconcileBootstrap(params BootstrapReconcileParams) (ctrl.Result, error) {
	if params.BKEMachine.Status.Bootstrapped {
		return ctrl.Result{}, nil
	}

	patchHelper, err := patch.NewHelper(params.BKEMachine, r.Client)
	if err != nil {
		return ctrl.Result{}, err
	}
	// Always attempt to Patch the bkeCluster object and Status after each reconciliation.
	defer func() {
		if err := patchBKEMachine(params.Ctx, patchHelper, params.BKEMachine); err != nil {
			params.Log.Error("failed to patch demoMachine", err)
			return
		}
	}()

	// if bkeMachine has WorkerNodeHost or MasterNodeHost label
	// means it is at bootstrap process
	// the Status.ProviderID and Status.Ready field will be processed
	if _, ok := labelhelper.CheckBKEMachineLabel(params.BKEMachine); ok {
		return ctrl.Result{}, nil
	}

	// 处理首次协调的机器
	return r.handleFirstTimeReconciliation(params)
}

// handleFirstTimeReconciliation 处理首次协调的机器
func (r *BKEMachineReconciler) handleFirstTimeReconciliation(params BootstrapReconcileParams) (ctrl.Result, error) {
	if !util.IsControlPlaneMachine(params.Machine) && !conditions.IsTrue(params.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		params.Log.Info("Waiting for the control plane to be initialized")
		conditions.MarkFalse(params.BKEMachine, bkev1beta1.BootstrapSucceededCondition,
			clusterv1.WaitingForControlPlaneAvailableReason, clusterv1.ConditionSeverityInfo, "")
		return ctrl.Result{}, nil
	}
	if util.IsControlPlaneMachine(params.Machine) {
		if err := r.syncKubeadmConfig(params.Ctx, params.Machine, params.Cluster); err != nil {
			params.Log.Warnf("Failed to sync kubeadm config: %v", err)
		}
	}
	role := r.getMachineRole(params.Machine)
	roleNodes, err := r.getRoleNodes(params.Ctx, params.BKECluster, role)
	if err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"No available nodes in bkeCluster.spec, err: %v", err)
		return ctrl.Result{}, nil
	}
	phase, err := r.getBootstrapPhase(params.Ctx, params.Machine, params.Cluster)
	if err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to get bootstrap phase: %v", err)
		return ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration}, nil
	}
	node, err := r.filterAvailableNode(params.Ctx, roleNodes, params.BKECluster, phase)
	if err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to get available node in bkeCluster.spec, err: %v", err)
		return ctrl.Result{}, nil
	}

	if phase == bkev1beta1.InitControlPlane {
		if err := r.NodeFetcher.MarkNodeStateFlagForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.MasterInitFlag); err != nil {
			params.Log.Warnf("Failed to mark node state flag: %v", err)
		}
	}

	if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeBootStrapping, "Start bootstrap"); err != nil {
		params.Log.Warnf("Failed to set node state: %v", err)
	}
	if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
		return ctrl.Result{}, err
	}

	if err := r.recordBootstrapPhaseEvent(params.Cluster, params.BKECluster, node, phase, params.Log); err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to record bootstrap phase event: %v", err)
		return ctrl.Result{}, nil
	}

	var nodeParams = CommonNodeParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{
				Ctx: params.Ctx,
				Log: params.Log,
			},
			Machine:    params.Machine,
			Cluster:    params.Cluster,
			BKEMachine: params.BKEMachine,
			BKECluster: params.BKECluster,
		},
		Node: node,
		Role: role,
	}

	if !clusterutil.FullyControlled(params.BKECluster) {
		fakeBootstrapParams := FakeBootstrapParams{CommonNodeParams: nodeParams}
		return r.handleFakeBootstrap(fakeBootstrapParams)
	}

	realBootstrapParams := RealBootstrapParams{
		CommonNodeParams: nodeParams,
		Phase:            phase,
	}
	return r.handleRealBootstrap(realBootstrapParams)
}

// syncKubeadmConfig 同步 kubeadm 配置
func (r *BKEMachineReconciler) syncKubeadmConfig(ctx context.Context, machine *clusterv1.Machine, cluster *clusterv1.Cluster) error {
	kubeadmConfig := &bootstrapv1.KubeadmConfig{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: machine.Namespace, Name: machine.Spec.Bootstrap.ConfigRef.Name}, kubeadmConfig); err == nil {
		helper, _ := patch.NewHelper(kubeadmConfig, r.Client)
		if helper != nil {
			kcp, _ := phaseutil.GetClusterAPIKubeadmControlPlane(ctx, r.Client, cluster)
			if kcp != nil {
				clusterConfiguration := kcp.Spec.KubeadmConfigSpec.ClusterConfiguration.DeepCopy()
				initConfiguration := kcp.Spec.KubeadmConfigSpec.InitConfiguration.DeepCopy()
				joinConfiguration := kcp.Spec.KubeadmConfigSpec.JoinConfiguration.DeepCopy()
				kubeadmConfig.Spec.ClusterConfiguration = clusterConfiguration
				kubeadmConfig.Spec.InitConfiguration = initConfiguration
				kubeadmConfig.Spec.JoinConfiguration = joinConfiguration
				_ = helper.Patch(ctx, kubeadmConfig)
			}
		}
	}
	return nil
}

// getMachineRole 获取机器角色
func (r *BKEMachineReconciler) getMachineRole(machine *clusterv1.Machine) string {
	role := bkenode.WorkerNodeRole
	if util.IsControlPlaneMachine(machine) {
		role = bkenode.MasterNodeRole
	}
	return role
}

// getRoleNodes 获取角色节点
func (r *BKEMachineReconciler) getRoleNodes(ctx context.Context, bkeCluster *bkev1beta1.BKECluster, role string) (bkenode.Nodes, error) {
	roleNodes, err := r.NodeFetcher.GetReadyBootstrapNodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		return nil, err
	}
	if role == bkenode.MasterNodeRole {
		roleNodes = roleNodes.Master()
	} else {
		roleNodes = roleNodes.Worker()
	}

	if len(roleNodes) == 0 {
		return nil, errors.New("no role nodes available")
	}
	return roleNodes, nil
}

// handleFakeBootstrap 处理伪引导（非完全控制集群）
func (r *BKEMachineReconciler) handleFakeBootstrap(params FakeBootstrapParams) (ctrl.Result, error) {
	defer func() {
		r.mux.Lock()
		delete(r.nodesBootRecord, params.Node.IP)
		r.mux.Unlock()
	}()

	helper, err := patch.NewHelper(params.Machine, r.Client)
	if err != nil {
		return ctrl.Result{}, err
	}
	defer helper.Patch(params.Ctx, params.Machine)

	providerID := phaseutil.GenerateProviderID(params.BKECluster, *params.Node)

	// 修补远程节点 ProviderID
	realProviderID, err := r.patchOrGetRemoteNodeProviderID(params.Ctx, params.BKECluster, params.Node, providerID)
	if err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to patch remote node providerID, retry after 5 second : %v", err)
		return ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration}, nil
	}

	// 对于主控机器，需要设置证书过期时间注释
	if util.IsControlPlaneMachine(params.Machine) {
		if err := r.handleMasterMachineCertificates(params.Ctx, params.Machine, params.BKECluster, params.Log); err != nil {
			r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
				"Failed to handle master machine certificates: %v", err)
		}
	}

	if err = r.markBKEMachineBootstrapReady(bootstrapReadyInput{
		ctx:        params.Ctx,
		cluster:    params.BKECluster,
		machine:    params.BKEMachine,
		node:       *params.Node,
		providerID: realProviderID,
		log:        params.Log,
	}); err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to mark bkeMachine bootstrap ready: %v", err)
		return ctrl.Result{}, nil
	}

	if err = r.reconcileBKEMachine(params.Ctx, params.BKECluster, params.BKEMachine, *params.Node, params.Log); err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.ReconcileErrorReason,
			"Failed to reconcile bkeMachine: %v", err)
		return ctrl.Result{}, nil
	}

	// 设置必要注释
	annotation.SetAnnotation(params.Machine, annotation.BKEMachineProviderIDAnnotationKey, providerID)
	annotation.SetAnnotation(params.BKEMachine, annotation.BKEMachineProviderIDAnnotationKey, providerID)

	// 设置 bkeMachine 标签
	labelhelper.SetBKEMachineLabel(params.BKEMachine, params.Role, params.Node.IP)
	// Save complete node info to BKEMachine.Status.Node for use during deletion
	params.BKEMachine.Status.Node = params.Node

	return ctrl.Result{}, nil
}

// handleMasterMachineCertificates 处理主控机器证书
func (r *BKEMachineReconciler) handleMasterMachineCertificates(ctx context.Context, machine *clusterv1.Machine,
	bkeCluster *bkev1beta1.BKECluster, log *log.Logger) error {
	config, err := phaseutil.GetMachineAssociateKubeadmConfig(ctx, r.Client, machine)
	if err != nil {
		r.logWarningAndEvent(log, bkeCluster, constant.ReconcileErrorReason,
			"Failed to get machine %q kubeadm config: %v", utils.ClientObjNS(machine), err)
		return err
	}

	if config != nil {
		helper, err := patch.NewHelper(config, r.Client)
		if err == nil {
			// 设置证书过期时间注释
			annotations := config.GetAnnotations()
			if annotations == nil {
				annotations = make(map[string]string)
			}
			// 100 年
			annotations[clusterv1.MachineCertificatesExpiryDateAnnotation] =
				time.Now().AddDate(CertificateExpiryYears, 0, 0).Format(time.RFC3339)
			config.SetAnnotations(annotations)
			if err := helper.Patch(ctx, config); err != nil {
				r.logWarningAndEvent(log, bkeCluster, constant.ReconcileErrorReason,
					"Failed to patch kubeadm config: %v", err)
				return err
			}
		} else {
			r.logWarningAndEvent(log, bkeCluster, constant.ReconcileErrorReason,
				"Failed to new kubeadm config patch helper: %v", err)
			return err
		}
	}
	return nil
}

// handleRealBootstrap 处理真实引导流程
func (r *BKEMachineReconciler) handleRealBootstrap(params RealBootstrapParams) (ctrl.Result, error) {
	bootstrapCommand := command.Bootstrap{
		BaseCommand: command.BaseCommand{
			Ctx:             params.Ctx,
			NameSpace:       params.BKEMachine.Namespace,
			Client:          r.Client,
			Scheme:          r.Scheme,
			OwnerObj:        params.BKEMachine,
			ClusterName:     params.BKECluster.Name,
			Unique:          true,
			RemoveAfterWait: false,
		},
		Node:      params.Node,
		BKEConfig: params.BKECluster.Name,
		Phase:     params.Phase,
	}

	if err := bootstrapCommand.New(); err != nil {
		errInfo := "Failed to create bootstrap command"
		params.Log.Errorf("%s: %v", errInfo, err)
		r.Recorder.AnnotatedEventf(params.BKECluster, annotation.BKENormalEventAnnotation(),
			corev1.EventTypeWarning, constant.CommandCreateFailedReason,
			"%s: %v", errInfo, err)
		condition.ConditionMark(params.BKECluster, bkev1beta1.TargetClusterBootCondition,
			confv1beta1.ConditionFalse, constant.CommandCreateFailedReason, errInfo)
		if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, params.Node.IP, bkev1beta1.NodeBootStrapFailed, errInfo); err != nil {
			params.Log.Warnf("Failed to set node state: %v", err)
		}

		if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
			return ctrl.Result{}, err
		}

		return ctrl.Result{}, err
	}

	params.Log.Infof("Create Bootstrap command for node %q succeeded, waiting for the command to be finished",
		phaseutil.NodeInfo(*params.Node))
	// now we need to set the label for bkeMachine, which means this node is already in used
	labelhelper.SetBKEMachineLabel(params.BKEMachine, params.Role, params.Node.IP)
	// Save complete node info to BKEMachine.Status.Node for use during deletion
	params.BKEMachine.Status.Node = params.Node

	r.logInfoAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingReason,
		"waiting node %q (role %q) finish bootstrap", phaseutil.NodeInfo(*params.Node), params.Node.Role)

	// then wait Command reconcile by bkeagent until have the Status that we want
	// this controller will catch the Command Status changes
	return ctrl.Result{}, nil
}

// reconcileCommand used to reconcile all the Command created by BkeCluster
func (r *BKEMachineReconciler) reconcileCommand(params BootstrapReconcileParams) (ctrl.Result, error) {
	patchHelper, err := patch.NewHelper(params.BKEMachine, r.Client)
	if err != nil {
		return ctrl.Result{}, err
	}
	// Always attempt to Patch the BKEMachine object and Status after each reconciliation.
	defer func(bkeMachine *bkev1beta1.BKEMachine) {
		if !controllerutil.ContainsFinalizer(bkeMachine, bkev1beta1.BKEMachineFinalizer) {
			return
		}
		if err := patchBKEMachine(params.Ctx, patchHelper, bkeMachine); err != nil {
			params.Log.Error("failed to patch demoMachine", err)
			return
		}
	}(params.BKEMachine)

	commands, err := getBKEMachineAssociateCommands(params.Ctx, r.Client, params.BKECluster, params.BKEMachine)
	if err != nil {
		params.Log.Error(err, "list commands failed")
		return ctrl.Result{}, err
	}
	if commands == nil || len(commands) == 0 {
		return ctrl.Result{}, nil
	}

	nodes, err := r.selectAppropriateNodes(params.Ctx, params.BKECluster)
	if err != nil {
		params.Log.Errorf("failed to select appropriate nodes: %v", err)
		return ctrl.Result{}, err
	}

	hostIp, found := labelhelper.CheckBKEMachineLabel(params.BKEMachine)
	if !found {
		return ctrl.Result{}, nil
	}

	res := ctrl.Result{}
	var errs []error
	for _, cmd := range commands {
		commandParams := ProcessCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{
					Ctx: params.Ctx,
					Log: params.Log,
				},
				Machine:    params.Machine,
				Cluster:    params.Cluster,
				BKEMachine: params.BKEMachine,
				BKECluster: params.BKECluster,
			},
			PatchHelper: patchHelper,
			Nodes:       nodes,
			HostIp:      hostIp,
			Cmd:         cmd,
			Res:         res,
			Errs:        errs,
		}
		res, errs = r.processCommand(commandParams)
	}
	return res, kerrors.NewAggregate(errs)
}

// selectAppropriateNodes selects the appropriate nodes based on BKENode CRD
func (r *BKEMachineReconciler) selectAppropriateNodes(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	// Fetch nodes from BKENode CRD
	nodes, err := r.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return nil, err
	}
	return nodes, nil
}

// processCommand processes a single command
func (r *BKEMachineReconciler) processCommand(params ProcessCommandParams) (ctrl.Result, []error) {

	commandNodes := params.Nodes.Filter(bkenode.FilterOptions{"IP": params.HostIp})
	if len(commandNodes) == 0 || commandNodes == nil {
		// reset command do not need to check node
		if !strings.HasPrefix(params.Cmd.Name, command.ResetNodeCommandNamePrefix) {
			params.Log.Warnf("The node %s in this command %s cannot be matched from the known Nodes",
				params.Cmd.Spec.NodeName, params.Cmd.Name)
		}
		return params.Res, params.Errs
	}
	currentNode := commandNodes[0]
	complete, successNodes, failedNodes := command.CheckCommandStatus(&params.Cmd)

	if strings.HasPrefix(params.Cmd.Name, command.BootstrapCommandNamePrefix) {
		bootstrapCommandParams := ProcessBootstrapCommandParams{
			CommonResourceParams: CommonResourceParams{
				CommonContextParams: CommonContextParams{
					Ctx: params.Ctx,
					Log: params.Log,
				},
				Machine:    params.Machine,
				Cluster:    params.Cluster,
				BKEMachine: params.BKEMachine,
				BKECluster: params.BKECluster,
			},
			PatchHelper:  params.PatchHelper,
			CurrentNode:  currentNode,
			Cmd:          &params.Cmd,
			Complete:     complete,
			SuccessNodes: successNodes,
			FailedNodes:  failedNodes,
			Res:          params.Res,
			Errs:         params.Errs,
		}
		return r.processBootstrapCommand(bootstrapCommandParams)
	}

	if strings.HasPrefix(params.Cmd.Name, command.ResetNodeCommandNamePrefix) {
		resetCommandParams := ProcessResetCommandParams{
			CommonCommandParams: CommonCommandParams{
				CommonResourceParams: CommonResourceParams{
					CommonContextParams: CommonContextParams{
						Ctx: params.Ctx,
						Log: params.Log,
					},
					Machine:    params.Machine,
					Cluster:    params.Cluster,
					BKEMachine: params.BKEMachine,
					BKECluster: params.BKECluster,
				},
				PatchHelper:  params.PatchHelper,
				Cmd:          &params.Cmd,
				Complete:     complete,
				SuccessNodes: successNodes,
				FailedNodes:  failedNodes,
				Res:          params.Res,
				Errs:         params.Errs,
			},
			CurrentNode: currentNode,
		}
		return r.processResetCommand(resetCommandParams)
	}

	return params.Res, params.Errs
}

// processBootstrapCommand processes bootstrap commands
func (r *BKEMachineReconciler) processBootstrapCommand(params ProcessBootstrapCommandParams) (ctrl.Result, []error) {

	r.mux.Lock()
	delete(r.nodesBootRecord, params.CurrentNode.IP)
	r.mux.Unlock()

	if params.BKEMachine.Status.Bootstrapped {
		return params.Res, params.Errs
	}
	if params.BKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
		params.Log.Infof("Cluster %q is deleting, skip bootstrap", params.BKECluster.Name)
		return params.Res, params.Errs
	}

	role := bkenode.WorkerNodeRole
	if util.IsControlPlaneMachine(params.Machine) {
		role = bkenode.MasterNodeRole
	}

	var bootstrapParams = ProcessBootstrapCommonParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{
				Ctx: params.Ctx,
				Log: params.Log,
			},
			Machine:    params.Machine,
			Cluster:    params.Cluster,
			BKEMachine: params.BKEMachine,
			BKECluster: params.BKECluster,
		},
		PatchHelper: params.PatchHelper,
		CurrentNode: params.CurrentNode,
		Cmd:         params.Cmd,
		Res:         params.Res,
		Errs:        params.Errs,
	}

	if params.Complete && len(params.FailedNodes) > 0 {
		failureParams := ProcessBootstrapFailureParams{
			ProcessBootstrapCommonParams: bootstrapParams,
			FailedNodes:                  params.FailedNodes,
			Role:                         role,
		}
		return r.processBootstrapFailure(failureParams)
	}

	// bootstrapCommand only contains one node
	if params.Complete && len(params.FailedNodes) == 0 && len(params.SuccessNodes) == 1 {
		successParams := ProcessBootstrapSuccessParams{
			ProcessBootstrapCommonParams: bootstrapParams,
		}
		return r.processBootstrapSuccess(successParams)
	}

	return params.Res, params.Errs
}

// markCommandReconciled marks the Command as reconciled with retry on conflict.
func (r *BKEMachineReconciler) markCommandReconciled(ctx context.Context, cmd *agentv1beta1.Command, logger *log.Logger, errs *[]error) {
	if err := phaseutil.RetryOnConflict(func() error {
		latestCommand := &agentv1beta1.Command{}
		if err := r.Client.Get(ctx, client.ObjectKey{Name: cmd.Name, Namespace: cmd.Namespace}, latestCommand); err != nil {
			return err
		}
		annotation.SetAnnotation(latestCommand, annotation.CommandReconciledAnnotationKey, "true")
		return r.Client.Update(ctx, latestCommand)
	}); err != nil {
		logger.Errorf("failed to mark command reconciled, err: %s", err.Error())
		*errs = append(*errs, err)
	}
}

// processBootstrapFailure processes bootstrap command failures
func (r *BKEMachineReconciler) processBootstrapFailure(params ProcessBootstrapFailureParams) (ctrl.Result, []error) {

	metricrecord.NodeBootstrapFailedCountRecord(params.BKECluster)

	r.logWarningAndEvent(params.Log, params.BKECluster, constant.NodeBootStrapFailedReason,
		"Failed to bootstrap node %q", phaseutil.NodeInfo(params.CurrentNode))
	output := r.LogCommandFailed(*params.Cmd, params.BKECluster, params.FailedNodes, params.Log, constant.NodeBootStrapFailedReason)

	if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, params.CurrentNode.IP, bkev1beta1.NodeBootStrapFailed, output); err != nil {
		params.Log.Warnf("Failed to set node state: %v", err)
	}

	metricrecord.NodeBootstrapDurationRecord(params.BKECluster, params.CurrentNode, params.Cmd.CreationTimestamp.Time, output)

	conditions.MarkFalse(params.BKEMachine, bkev1beta1.BootstrapSucceededCondition,
		constant.NodeBootStrapFailedReason, clusterv1.ConditionSeverityWarning,
		"Bootstrap failed err: %s", output)
	// 忽略该函数的错误，重点是将信息输出
	_ = r.reconcileBKEMachine(params.Ctx, params.BKECluster, params.BKEMachine, params.CurrentNode, params.Log)

	r.markCommandReconciled(params.Ctx, params.Cmd, params.Log, &params.Errs)

	// 集群master未初始化，后续的部署无法进行
	if !conditions.IsTrue(params.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		// if master not init pause cluster deployment
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingFailedReason,
			"The control plane initialization encountered some errors, which are beyond "+
				"the control scope of bke, the next deployment cannot proceed, "+
				"and the cluster deployment has been paused")
		// user can fix the problem in error node, and restart bkeagent to rerun command, then controller can snap command
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingFailedReason,
			"You can check the BKEAgent log on the error node (/var/log/openFuyao/bkeagent.log) "+
				"and manually resolve the problem. Then restart the BKEAgent on the node")
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingFailedReason,
			"If problem can not be resolved, you can delete the cluster by delete "+
				"BKECluster resource %q directly", utils.ClientObjNS(params.BKECluster))

		if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
			params.Errs = append(params.Errs, err)
		}
		return ctrl.Result{}, params.Errs
	}

	// 移除bkeMachine 的label,使其能够重新触发bootstrap,master init的label由ensuremasterinit phase里去控制删除
	labelhelper.RemoveBKEMachineLabel(params.BKEMachine, params.Role)

	// 集群master已初始化，后续的部署可以进行
	r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingFailedReason,
		"You can check the BKEAgent log on the error node (/var/log/openFuyao/bkeagent.log) "+
			"and manually resolve the problem. Then restart the BKEAgent on the node")
	r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingFailedReason,
		"If problem can not be resolved, you can delete the corresponding BKENode resource")
	r.logWarningAndEvent(params.Log, params.BKECluster, constant.TargetClusterNotReadyReason,
		"Cluster %q Control plane already init, the next deployment can continue", params.BKECluster.Name)

	if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
		params.Errs = append(params.Errs, err)
	}
	return ctrl.Result{}, params.Errs
}

// connectToTargetClusterNode attempts to connect to the target cluster node
func (r *BKEMachineReconciler) connectToTargetClusterNode(params ProcessBootstrapSuccessParams) error {
	return wait.PollImmediate(DefaultRequeueAfterDuration, DefaultNodeConnectTimeout, func() (bool, error) {
		bkeCluster, err := mergecluster.GetCombinedBKECluster(params.Ctx, r.Client, params.BKECluster.Namespace, params.BKECluster.Name)
		if err != nil {
			return false, err
		}
		if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
			return true, nil
		}
		targetParams := TargetClusterNodeParams{
			BKECluster:  bkeCluster,
			Cluster:     params.Cluster,
			Ctx:         params.Ctx,
			CurrentNode: params.CurrentNode,
			Log:         params.Log,
			Machine:     params.Machine,
		}
		if err := r.checkTargetClusterNode(targetParams); err != nil {
			params.Log.Warnf("(ignore) Failed to check target cluster node %q, retrying...,err: %s",
				phaseutil.NodeInfo(params.CurrentNode), err.Error())
			return false, nil
		}
		return true, nil
	})
}

// handleBootstrapSuccessFailure handles the case when bootstrap success check fails
func (r *BKEMachineReconciler) handleBootstrapSuccessFailure(params ProcessBootstrapSuccessParams, err error) (ctrl.Result, []error) {
	errInfo := fmt.Sprintf("4 Minute after, the target cluster node %q is still unavailable,err: %s",
		phaseutil.NodeInfo(params.CurrentNode), err.Error())
	if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, params.CurrentNode.IP, bkev1beta1.NodeBootStrapFailed, errInfo); err != nil {
		params.Log.Warnf("Failed to set node state: %v", err)
	}
	r.logErrorAndEvent(params.Log, params.BKECluster, constant.NodeBootStrapFailedReason, errInfo)
	r.logErrorAndEvent(params.Log, params.BKECluster, constant.NodeBootStrapFailedReason,
		"It seems that the kubernetes component of node %q did not start properly",
		phaseutil.NodeInfo(params.CurrentNode))
	condition.ConditionMark(params.BKECluster, bkev1beta1.TargetClusterBootCondition,
		confv1beta1.ConditionFalse, constant.NodeBootStrapFailedReason, errInfo)
	params.Res = util.LowestNonZeroResult(params.Res, ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration})
	if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
		params.Log.Errorf("mergecluster.SyncStatusUntilComplete failed, err: %s", err.Error())
		params.Res = util.LowestNonZeroResult(params.Res, ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration})
	}
	return params.Res, params.Errs
}

// processBootstrapSuccess processes bootstrap command success
func (r *BKEMachineReconciler) processBootstrapSuccess(params ProcessBootstrapSuccessParams) (ctrl.Result, []error) {

	r.logInfoAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingReason,
		"Attempting to connect to target cluster node %q, please wait", phaseutil.NodeInfo(params.CurrentNode))

	err := r.connectToTargetClusterNode(params)

	if err != nil {
		return r.handleBootstrapSuccessFailure(params, err)
	}

	if params.BKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
		return ctrl.Result{}, nil
	}

	providerID := phaseutil.GenerateProviderID(params.BKECluster, params.CurrentNode)
	if err := r.markBKEMachineBootstrapReady(bootstrapReadyInput{
		ctx:        params.Ctx,
		cluster:    params.BKECluster,
		machine:    params.BKEMachine,
		node:       params.CurrentNode,
		providerID: providerID,
		log:        params.Log,
	}); err != nil {
		params.Errs = append(params.Errs, err)
	}

	metricrecord.NodeBootstrapSuccessCountRecord(params.BKECluster)
	metricrecord.NodeBootstrapDurationRecord(params.BKECluster, params.CurrentNode, params.Cmd.CreationTimestamp.Time, "success")

	r.markCommandReconciled(params.Ctx, params.Cmd, params.Log, &params.Errs)

	if err = r.reconcileBKEMachine(params.Ctx, params.BKECluster, params.BKEMachine, params.CurrentNode, params.Log); err != nil {
		params.Errs = append(params.Errs, err)
	}
	return params.Res, params.Errs
}

// processResetCommand processes reset commands
func (r *BKEMachineReconciler) processResetCommand(params ProcessResetCommandParams) (ctrl.Result, []error) {

	if len(params.SuccessNodes) == 0 {
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		if err := patchBKEMachine(params.Ctx, params.PatchHelper, params.BKEMachine); err != nil {
			params.Log.Error("failed to patch demoMachine", err)
		}
		return ctrl.Result{Requeue: true}, params.Errs
	}

	if params.Complete && len(params.FailedNodes) == 0 && len(params.SuccessNodes) == 1 {
		// Delete the BKENode CRD instead of removing from in-memory status
		if err := r.NodeFetcher.DeleteBKENodeForCluster(params.Ctx, params.BKECluster, params.CurrentNode.IP); err != nil {
			params.Log.Warnf("Failed to delete BKENode: %v", err)
		}
		// Delete the Node from target cluster if it still exists
		if params.CurrentNode.Hostname != "" {
			if err := phases.DeleteTargetClusterNode(params.Ctx, r.Client, params.BKECluster, params.CurrentNode.Hostname); err != nil {
				params.Log.Warnf("Failed to delete Node %s from target cluster: %v", params.CurrentNode.Hostname, err)
			}
		}
		// sync bkeCluster Status
		if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
			params.Log.Errorf("failed to sync bkeCluster Status: %v", err)
			params.Res = util.LowestNonZeroResult(params.Res, ctrl.Result{RequeueAfter: DefaultRequeueAfterDuration})
			return params.Res, params.Errs
		}
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		if err := patchBKEMachine(params.Ctx, params.PatchHelper, params.BKEMachine); err != nil {
			params.Log.Error("failed to patch demoMachine", err)
		}
		return ctrl.Result{Requeue: true}, params.Errs
	}

	return params.Res, params.Errs
}

// reconcileBKEMachine used to reconcile all the BKEMachine Status
func (r *BKEMachineReconciler) reconcileBKEMachine(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	bkemachine *bkev1beta1.BKEMachine, n confv1beta1.Node, log *log.Logger) error {

	if condition.HasConditionStatus(bkev1beta1.TargetClusterBootCondition, bkeCluster, confv1beta1.ConditionTrue) {
		return nil
	}

	// 获取集群相关信息
	bkeMachines, nodes, err := r.getClusterInfo(ctx, bkeCluster)
	if err != nil {
		return err
	}

	// 检查引导状态
	clusterReady, bootstrapNodeFailed := r.checkBootstrapStatus(bkeMachines, bkemachine, nodes)

	// 处理不同状态
	clusterStateParams := HandleClusterStateParams{
		CommonContextParams: CommonContextParams{
			Ctx: ctx,
			Log: log,
		},
		BKECluster:          bkeCluster,
		BKEMachine:          bkemachine,
		NodeState:           n,
		BKEMachines:         bkeMachines,
		Nodes:               nodes,
		ClusterReady:        clusterReady,
		BootstrapNodeFailed: bootstrapNodeFailed,
	}
	return r.handleClusterState(clusterStateParams)
}

// getClusterInfo retrieves cluster information
func (r *BKEMachineReconciler) getClusterInfo(ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster) ([]bkev1beta1.BKEMachine, bkenode.Nodes, error) {
	// Fetch nodes from BKENode CRD
	nodes, err := r.NodeFetcher.GetNodesForBKECluster(ctx, bkeCluster)
	if err != nil {
		return nil, nil, err
	}

	bkeMachines, err := phaseutil.GetBKEClusterAssociateBKEMachines(ctx, r.Client, bkeCluster)
	if err != nil {
		return nil, nil, err
	}

	return bkeMachines, nodes, nil
}

// checkBootstrapStatus checks the bootstrap status of machines
func (r *BKEMachineReconciler) checkBootstrapStatus(bkeMachines []bkev1beta1.BKEMachine,
	bkemachine *bkev1beta1.BKEMachine, nodes bkenode.Nodes) (bool, bool) {

	bkeMachineNum := len(bkeMachines)
	nodesNum := nodes.Length()

	bootstrapNodeFailed := false

	// bkeMachines.len 为0 或者 bkeMachines.len != Nodes.length 集群没有引导结束
	clusterReady := bkeMachineNum != 0 && bkeMachineNum == nodesNum
	for i, bm := range bkeMachines {
		if bm.Name == bkemachine.Name {
			bm = *bkemachine
			bkeMachines[i] = *bkemachine
		}

		if conditions.GetReason(&bm, bkev1beta1.BootstrapSucceededCondition) == constant.NodeBootStrapFailedReason {
			bootstrapNodeFailed = true
		}
		if !bm.Status.Bootstrapped {
			clusterReady = false
			break
		}
	}

	return clusterReady, bootstrapNodeFailed
}

// handleClusterState handles different cluster states
func (r *BKEMachineReconciler) handleClusterState(params HandleClusterStateParams) error {

	bkeMachineNum := len(params.BKEMachines)
	nodesNum := params.Nodes.Length()

	params.Log.Debugf("bkeMachineNum: %d, nodesNum: %d, clusterReady: %v", bkeMachineNum, nodesNum, params.ClusterReady)

	BkeMachineExceptNum := nodesNum
	failedBootNodeNum, successBootNodeNum := phaseutil.CalculateBKEMachineBootNum(params.BKEMachines)
	allBootFlag := failedBootNodeNum+successBootNodeNum == BkeMachineExceptNum

	if allBootFlag {
		return r.handleAllNodesBootstrapped(params.Ctx, params.BKECluster, params.Log)
	} else {
		r.logInfoAndEvent(params.Log, params.BKECluster, constant.TargetClusterBootingReason, "Waiting for Cluster-API to reconcile")
	}

	if params.BootstrapNodeFailed && allBootFlag {
		return r.handleBootstrapFailure(params.BKECluster, params.Log)
	}

	if params.ClusterReady && allBootFlag {
		return r.handleClusterReady(params.Ctx, params.BKECluster, params.Log)
	}

	return r.handleClusterBooting(params.Ctx, params.BKECluster, params.NodeState, params.Log)
}

// handleAllNodesBootstrapped handles when all nodes are bootstrapped
func (r *BKEMachineReconciler) handleAllNodesBootstrapped(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	log *log.Logger) error {

	bkeCluster.Status.KubernetesVersion = bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion
	if err := mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster); err != nil {
		log.Error("failed to patch BKECluster", err)
		return err
	}
	return nil
}

// handleBootstrapFailure handles bootstrap failure
func (r *BKEMachineReconciler) handleBootstrapFailure(bkeCluster *bkev1beta1.BKECluster,
	log *log.Logger) error {

	condition.ConditionMark(bkeCluster, bkev1beta1.TargetClusterBootCondition,
		confv1beta1.ConditionFalse, constant.NodeBootStrapFailedReason, "")
	r.logWarningAndEvent(log, bkeCluster, constant.NodeBootStrapFailedReason,
		"The target cluster %q has finished booting, but some node bootstrap failed", bkeCluster.Name)
	return nil
}

// handleClusterReady handles when cluster is ready
func (r *BKEMachineReconciler) handleClusterReady(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	log *log.Logger) error {

	condition.ConditionMark(bkeCluster, bkev1beta1.TargetClusterBootCondition,
		confv1beta1.ConditionTrue, constant.TargetClusterBootReadyReason, "")
	r.logInfoAndEvent(log, bkeCluster, constant.TargetClusterBootReadyReason,
		"The target cluster %q has finished booting", bkeCluster.Name)

	metricrecord.ClusterBootstrapDurationRecord(bkeCluster)

	r.logInfoAndEvent(log, bkeCluster, constant.TargetClusterBootReadyReason,
		"The target cluster %q has finished booting", bkeCluster.Name)
	if err := mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster); err != nil {
		log.Error("failed to patch BKECluster", err)
		return err
	}
	return nil
}

// handleClusterBooting handles when cluster is booting
func (r *BKEMachineReconciler) handleClusterBooting(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	n confv1beta1.Node, log *log.Logger) error {

	condition.ConditionMark(bkeCluster, bkev1beta1.TargetClusterBootCondition,
		confv1beta1.ConditionFalse, constant.TargetClusterBootingReason,
		fmt.Sprintf("bootstrap node %q", phaseutil.NodeInfo(n)))
	if err := mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster); err != nil {
		log.Error("failed to patch BKECluster", err)
		return err
	}
	return nil
}

// filterAvailableNode return the first available node found
func (r *BKEMachineReconciler) filterAvailableNode(ctx context.Context, roleNodes bkenode.Nodes,
	bkeCluster *bkev1beta1.BKECluster, phase confv1beta1.BKEClusterPhase) (*confv1beta1.Node, error) {

	r.mux.Lock()
	defer r.mux.Unlock()

	preferIP := controlPlaneEndpointHost(bkeCluster)
	if phase == bkev1beta1.InitControlPlane {
		if preferred := r.reservePreferredNode(roleNodes, preferIP); preferred != nil {
			return preferred, nil
		}
		return r.reserveFirstNode(roleNodes), nil
	}

	bkeMachineList := &bkev1beta1.BKEMachineList{}
	if err := r.Client.List(ctx, bkeMachineList, phaseutil.GetListFiltersByBKECluster(bkeCluster)...); err != nil {
		return nil, err
	}

	availableNodes := r.unboundNodes(roleNodes, bkeMachineList.Items)
	if len(availableNodes) == 0 {
		return nil, errors.New("no available node")
	}
	if preferred := r.reservePreferredNode(availableNodes, preferIP); preferred != nil {
		return preferred, nil
	}
	return r.reserveFirstNode(availableNodes), nil
}

func controlPlaneEndpointHost(bkeCluster *bkev1beta1.BKECluster) string {
	if bkeCluster == nil {
		return ""
	}
	return bkeCluster.Spec.ControlPlaneEndpoint.Host
}

func (r *BKEMachineReconciler) reservePreferredNode(nodes bkenode.Nodes, preferredIP string) *confv1beta1.Node {
	if preferredIP == "" {
		return nil
	}
	for i := range nodes {
		if nodes[i].IP == preferredIP {
			return r.reserveNode(&nodes[i])
		}
	}
	return nil
}

func (r *BKEMachineReconciler) reserveFirstNode(nodes bkenode.Nodes) *confv1beta1.Node {
	return r.reserveNode(&nodes[0])
}

func (r *BKEMachineReconciler) reserveNode(node *confv1beta1.Node) *confv1beta1.Node {
	r.nodesBootRecord[node.IP] = struct{}{}
	return node
}

func (r *BKEMachineReconciler) unboundNodes(roleNodes bkenode.Nodes, machines []bkev1beta1.BKEMachine) bkenode.Nodes {
	availableNodes := make(bkenode.Nodes, 0, len(roleNodes))
	for _, node := range roleNodes {
		if _, reserved := r.nodesBootRecord[node.IP]; reserved || nodeIsBound(node.IP, machines) {
			continue
		}
		availableNodes = append(availableNodes, node)
	}
	return availableNodes
}

func nodeIsBound(nodeIP string, machines []bkev1beta1.BKEMachine) bool {
	for i := range machines {
		if boundIP, ok := labelhelper.CheckBKEMachineLabel(&machines[i]); ok && boundIP == nodeIP {
			return true
		}
	}
	return false
}

// checkTargetClusterNode use the remote client to check the target cluster node with provideID has been created
// set the node taint and role label
// TargetClusterNodeParams holds parameters for checking target cluster node
type TargetClusterNodeParams struct {
	Ctx         context.Context
	BKECluster  *bkev1beta1.BKECluster
	Cluster     *clusterv1.Cluster
	Machine     *clusterv1.Machine
	CurrentNode confv1beta1.Node
	Log         *log.Logger
}

// checkTargetClusterNode use the remote client to check the target cluster node with provideID has been created
// set the node taint and role label
func (r *BKEMachineReconciler) checkTargetClusterNode(params TargetClusterNodeParams) error {
	targetClusterClient, err := kube.NewRemoteClusterClient(params.Ctx, r.Client, params.BKECluster)
	if err != nil {
		return err
	}
	providerID := phaseutil.GenerateProviderID(params.BKECluster, params.CurrentNode)

	nodeList := &corev1.NodeList{}

	if err = targetClusterClient.List(params.Ctx, nodeList); err != nil {
		return err
	}

	// 当节点是独立master角色节点时需要cordon
	// 当节点是master/node角色节点时不需要cordon, 但是需要打上master node角色标签
	// 当节点是node角色节点时不需要cordon, 但是需要打上node标签
	nodeInfo := r.getNodeInfo(params.CurrentNode, params.BKECluster)

	// check providerID already exists or not
	if len(nodeList.Items) == 0 {
		return r.handleEmptyNodeList(targetClusterClient, params, providerID, nodeInfo)
	}

	return r.handleNonEmptyNodeList(targetClusterClient, params, providerID, nodeInfo, nodeList)
}

// getNodeInfo 获取节点信息和调度标记
func (r *BKEMachineReconciler) getNodeInfo(currentNode confv1beta1.Node, bkeCluster *bkev1beta1.BKECluster) nodeInformation {
	node := bkenode.Node(currentNode)
	cordon := false
	// 注解是否允许cordon，默认允许
	annotationCordonFlag := true
	nodeRole := node.Role[0]
	if node.IsMasterWorker() {
		nodeRole = bkenode.MasterWorkerNodeRole
		cordon = false
	}
	if node.IsMaster() {
		nodeRole = bkenode.MasterNodeRole
		cordon = true
	}
	if node.IsWorker() {
		nodeRole = bkenode.WorkerNodeRole
	}
	// 如果是master节点，且注解不允许cordon，则不cordon
	if v, ok := annotation.HasAnnotation(bkeCluster, annotation.MasterSchedulableAnnotationKey); ok && v == "true" {
		annotationCordonFlag = false
	}

	return nodeInformation{
		Node:                 node,
		Cordon:               cordon,
		AnnotationCordonFlag: annotationCordonFlag,
		Role:                 nodeRole,
	}
}

// nodeInformation 包含节点相关信息
type nodeInformation struct {
	Node                 bkenode.Node
	Cordon               bool
	AnnotationCordonFlag bool
	Role                 string
}

// handleEmptyNodeList 处理空节点列表的情况
func (r *BKEMachineReconciler) handleEmptyNodeList(targetClusterClient client.Client,
	params TargetClusterNodeParams, providerID string, nodeInfo nodeInformation) error {

	// If for whatever reason the index isn't registered or available, we fall back to loop over the whole list.
	var nl corev1.NodeList
	for {
		if err := targetClusterClient.List(params.Ctx, &nl, client.Continue(nl.Continue)); err != nil {
			return err
		}

		// Check if any node matches the provider ID
		found, err := r.checkNodesForProviderID(targetClusterClient, params, nl.Items, providerID, nodeInfo)
		if err != nil {
			return err
		}
		if found {
			return nil
		}

		if nl.Continue == "" {
			break
		}
	}
	return errors.Errorf("could not find node with providerID %s ", providerID)
}

// checkNodesForProviderID checks if any node in the list matches the given provider ID
func (r *BKEMachineReconciler) checkNodesForProviderID(targetClusterClient client.Client,
	params TargetClusterNodeParams, nodes []corev1.Node, providerID string, nodeInfo nodeInformation) (bool, error) {

	for _, n := range nodes {
		if n.Spec.ProviderID == "" {
			continue
		}
		if providerID == n.Spec.ProviderID {
			if err := r.processMatchingNode(targetClusterClient, params, n, nodeInfo, providerID); err != nil {
				return false, err
			}
			return true, nil
		}
	}
	return false, nil
}

// applyNodeConfiguration 应用节点配置，包括设置节点角色标签、创建模拟configmaps等
func (r *BKEMachineReconciler) applyNodeConfiguration(targetClusterClient client.Client,
	params TargetClusterNodeParams, node *corev1.Node, nodeInfo nodeInformation, providerID string) error {

	if nodeInfo.Cordon && nodeInfo.AnnotationCordonFlag {
		if err := r.cordonMasterNode(params.Ctx, params.BKECluster, node, params.Log); err != nil {
			params.Log.Warnf("failed to cordon master node %q", node.Name)
		}
	}

	if err := setTargetClusterNodeRole(params.Ctx, targetClusterClient, node, nodeInfo.Role); err != nil {
		return errors.Wrapf(err, "failed patch target cluster node, providerID %q", providerID)
	}

	if err := MockKubeadmConfigConfigmap(params.Ctx, targetClusterClient); err != nil {
		return errors.Wrap(err, "failed to create mock kubeadm-config configmap")
	}

	if err := MockKubeletConfigConfigmap(params.Ctx, targetClusterClient, params.BKECluster.Spec.ClusterConfig.Cluster.KubernetesVersion); err != nil {
		return errors.Wrap(err, "failed to create mock kubelet-config configmap")
	}

	return nil
}

// handleNonEmptyNodeList 处理非空节点列表的情况
func (r *BKEMachineReconciler) handleNonEmptyNodeList(targetClusterClient client.Client,
	params TargetClusterNodeParams, providerID string, nodeInfo nodeInformation,
	nodeList *corev1.NodeList) error {

	var ns []corev1.Node
	for _, n := range nodeList.Items {
		if n.Spec.ProviderID == "" {
			continue
		}
		if providerID == n.Spec.ProviderID {
			ns = append(ns, n)
		}
	}
	if len(ns) != 1 {
		return errors.Errorf("unexpectedly found more than one Node matching the providerID %s", providerID)
	}

	// 使用提取的公共函数
	return r.applyNodeConfiguration(targetClusterClient, params, &ns[0], nodeInfo, providerID)
}

// processMatchingNode 处理匹配到的节点
func (r *BKEMachineReconciler) processMatchingNode(targetClusterClient client.Client,
	params TargetClusterNodeParams, n corev1.Node, nodeInfo nodeInformation, providerID string) error {

	// 使用提取的公共函数
	return r.applyNodeConfiguration(targetClusterClient, params, &n, nodeInfo, providerID)
}

// markBKEMachineBootstrapReady marks the BKEMachine as ready and bootstrapped
// sets the providerID and addresses on the BKEMachine
// sets the Ready condition to true.
// sets the NodeRef on the BKEMachine.
// add node info to BKECluster.Status.ClusterStatus.Nodes
type bootstrapReadyInput struct {
	ctx        context.Context
	cluster    *bkev1beta1.BKECluster
	machine    *bkev1beta1.BKEMachine
	node       confv1beta1.Node
	providerID string
	log        *log.Logger
}

func (r *BKEMachineReconciler) markBKEMachineBootstrapReady(input bootstrapReadyInput) error {
	ctx, bkeCluster, bkeMachine := input.ctx, input.cluster, input.machine
	assocNode, providerID, log := input.node, input.providerID, input.log

	// set MachineAddress
	setMachineAddress(bkeMachine, assocNode)
	// set providerID
	setProviderID(bkeMachine, providerID)
	bkeMachine.Status.Ready = true
	bkeMachine.Status.Bootstrapped = true
	conditions.MarkTrue(bkeMachine, bkev1beta1.BootstrapSucceededCondition)

	r.logInfoAndEvent(log, bkeCluster, constant.TargetClusterBootingReason,
		"node %q, role %v bootstrap succeeded", phaseutil.NodeInfo(assocNode), assocNode.Role)

	if err := r.NodeFetcher.MarkNodeStateFlagForCluster(ctx, bkeCluster, assocNode.IP, bkev1beta1.NodeBootFlag); err != nil {
		log.Warnf("Failed to mark node state flag: %v", err)
	}
	if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(ctx, bkeCluster, assocNode.IP, bkev1beta1.NodeNotReady, "Bootstrap Succeeded"); err != nil {
		log.Warnf("Failed to set node state: %v", err)
	}

	if err := mergecluster.SyncStatusUntilComplete(r.Client, bkeCluster); err != nil {
		log.Errorf("failed to update bkeCluster Status, err: %s", err.Error())
		return errors.Errorf("failed to update bkeCluster Status, err: %s", err.Error())
	}
	return nil
}

// patchRemoteNodeProviderID patch remote node providerID
// if remote node providerID is not equal to providerID, patch it
func (r *BKEMachineReconciler) patchOrGetRemoteNodeProviderID(ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster, node *confv1beta1.Node, providerID string) (string, error) {

	// patch remote node providerID
	targetClusterClient, err := kube.NewRemoteClusterClient(ctx, r.Client, bkeCluster)
	if err != nil {
		return "", errors.Wrap(err, "failed to create target cluster client")
	}

	if err = MockKubeadmConfigConfigmap(ctx, targetClusterClient); err != nil {
		return "", errors.Wrap(err, "failed to create mock kubeadm-config configmap")
	}
	if err = MockKubeletConfigConfigmap(ctx, targetClusterClient, bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion); err != nil {
		return "", errors.Wrap(err, "failed to create mock kubelet-config configmap")
	}

	// get remote node
	remoteNode := &corev1.Node{}
	if err = targetClusterClient.Get(ctx, client.ObjectKey{Name: node.Hostname}, remoteNode); err != nil {
		return "", errors.Errorf("failed to get remote node %q", phaseutil.NodeInfo(*node))
	}
	p := client.MergeFrom(remoteNode.DeepCopy())

	bocVersion, found := os.LookupEnv(constant.BocVersionEnvKey)
	if !found {
		bocVersion = constant.DefaultBocVersion
	}
	labelhelper.SetBocVersionLabel(remoteNode, bocVersion)
	if remoteNode.Spec.ProviderID == "" {
		remoteNode.Spec.ProviderID = providerID
	}
	if err := targetClusterClient.Patch(ctx, remoteNode, p); err != nil {
		return "", errors.Errorf("failed to patch remote node %q,err: %v", phaseutil.NodeInfo(*node), err)
	}

	return remoteNode.Spec.ProviderID, nil
}

// setMachineAddress gets the address from the node and sets it on the BKEMachine object.
func setMachineAddress(bkeMachine *bkev1beta1.BKEMachine, node confv1beta1.Node) {
	bkeMachine.Status.Addresses = []bkev1beta1.MachineAddress{
		{
			Type:    bkev1beta1.MachineHostName,
			Address: node.Hostname,
		},
		{
			Type:    bkev1beta1.MachineInternalIP,
			Address: node.IP,
		},
		{
			Type:    bkev1beta1.MachineExternalIP,
			Address: node.IP,
		},
	}
}

// setProviderID sets the providerID on the BKEMachine.spec.providerID
func setProviderID(bkeMachine *bkev1beta1.BKEMachine, providerID string) {
	bkeMachine.Spec.ProviderID = &providerID
}

// recordBootstrapPhaseEvent records the bootstrap phase events for the BKEMachine
func (r *BKEMachineReconciler) recordBootstrapPhaseEvent(cluster *clusterv1.Cluster,
	bkeCluster *bkev1beta1.BKECluster, node *confv1beta1.Node,
	phase confv1beta1.BKEClusterPhase, log *log.Logger) error {

	finalPhase := phase

	switch phase {
	case bkev1beta1.JoinWorker:
		if !clusterutil.FullyControlled(bkeCluster) {
			finalPhase = bkev1beta1.FakeJoinWorker
		}
	case bkev1beta1.JoinControlPlane:
		if !clusterutil.FullyControlled(bkeCluster) {
			finalPhase = bkev1beta1.FakeJoinControlPlane
		}
	case bkev1beta1.InitControlPlane:
		if !clusterutil.FullyControlled(bkeCluster) {
			finalPhase = bkev1beta1.FakeInitControlPlane
		}
	default:
		// Keep the original phase for any other cases
		finalPhase = phase
	}

	r.logInfoAndEvent(log, bkeCluster, constant.TargetClusterBootingReason,
		"cluster %q in the %s phase, node %q (host: %q, role %q)",
		cluster.Name, finalPhase, node.Hostname, node.IP, node.Role)
	return nil
}

// locker is a struct that holds the lock information,
// cloned from cluster-api/bootstrap/kubeadm/internal/locking/control_plane_init_mutex.go information struct
type locker struct {
	MachineName string `json:"machineName"`
}

const lockKey = "lock-information"

// getBootstrapPhase returns the deployment phase by viewing the lock information generated by clusterAPI
func (r *BKEMachineReconciler) getBootstrapPhase(ctx context.Context, machine *clusterv1.Machine,
	cluster *clusterv1.Cluster) (confv1beta1.BKEClusterPhase, error) {

	if !util.IsControlPlaneMachine(machine) {
		return bkev1beta1.JoinWorker, nil
	}

	// 检查控制平面是否已初始化
	if conditions.IsFalse(cluster, clusterv1.ControlPlaneInitializedCondition) {
		return bkev1beta1.InitControlPlane, nil
	}

	// 处理锁配置图逻辑
	return r.handleLockConfigMap(ctx, machine, cluster)
}

// handleLockConfigMap 处理锁配置图逻辑
func (r *BKEMachineReconciler) handleLockConfigMap(ctx context.Context, machine *clusterv1.Machine,
	cluster *clusterv1.Cluster) (confv1beta1.BKEClusterPhase, error) {

	cmLock := &corev1.ConfigMap{}
	key := client.ObjectKey{
		Namespace: cluster.Namespace,
		Name:      fmt.Sprintf("%s-lock", cluster.Name),
	}

	err := r.Client.Get(ctx, key, cmLock)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return bkev1beta1.JoinControlPlane, nil
		}
		return "", errors.Errorf("failed to get the lock configmap %s", key.String())
	}

	if cmLock.Data == nil {
		return "", errors.Errorf("lock data is nil,lock configmap %s", cmLock.Name)
	}

	l, err := r.parseLockInfo(cmLock)
	if err != nil {
		return "", err
	}

	if l.MachineName == machine.Name {
		return bkev1beta1.InitControlPlane, nil
	} else {
		return bkev1beta1.JoinControlPlane, nil
	}
}

// parseLockInfo 解析锁信息
func (r *BKEMachineReconciler) parseLockInfo(cmLock *corev1.ConfigMap) (*locker, error) {
	l := &locker{}
	if err := json.Unmarshal([]byte(cmLock.Data[lockKey]), l); err != nil {
		return nil, errors.Wrapf(err, "failed to unmarshal lock information")
	}
	return l, nil
}

// cordonMasterNode 对主节点进行cordon操作
func (r *BKEMachineReconciler) cordonMasterNode(ctx context.Context, bkeCluster *bkev1beta1.BKECluster,
	node *corev1.Node, log *log.Logger) error {

	remoteClient, err := kube.NewRemoteClientByBKECluster(ctx, r.Client, bkeCluster)
	if err != nil {
		return err
	}
	cs, _ := remoteClient.KubeClient()
	bkeLogger := bkev1beta1.NewBKELogger(log, r.Recorder, bkeCluster)
	drainer := phaseutil.NewDrainer(ctx, cs, nil, false, bkeLogger)
	return kubedrain.RunCordonOrUncordon(drainer, node, true)
}

// setTargetClusterNodeRole patches the node with the given taints,and set role label for the target cluster node
func setTargetClusterNodeRole(ctx context.Context, c client.Client, node *corev1.Node, nodeRole string) error {
	bocVersion, found := os.LookupEnv(constant.BocVersionEnvKey)
	if !found {
		bocVersion = constant.DefaultBocVersion
	}
	return phaseutil.RetryOnConflict(func() error {
		latestNode := &corev1.Node{}
		if err := c.Get(ctx, client.ObjectKey{Name: node.Name}, latestNode); err != nil {
			return err
		}
		switch nodeRole {
		case bkenode.WorkerNodeRole:
			labelhelper.SetWorkerRoleLabel(latestNode)
		case bkenode.MasterNodeRole:
			labelhelper.SetMasterRoleLabel(latestNode)
		case bkenode.MasterWorkerNodeRole:
			labelhelper.SetMasterRoleLabel(latestNode)
			labelhelper.SetWorkerRoleLabel(latestNode)
		default:
			// 不需要特殊处理的情况，保持节点原有的标签状态
		}
		labelhelper.SetBocVersionLabel(latestNode, bocVersion)
		return c.Update(ctx, latestNode)
	})
}

// createOrUpdateConfigMap 创建或更新ConfigMap的通用函数
func createOrUpdateConfigMap(ctx context.Context, c client.Client, name, namespace string, data map[string]string) error {
	mockCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: data,
	}

	if err := c.Create(ctx, mockCM); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return phaseutil.RetryOnConflict(func() error {
				existingConfigMap := &corev1.ConfigMap{}
				if err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, existingConfigMap); err != nil {
					return err
				}
				existingConfigMap.Data = data
				return c.Update(ctx, existingConfigMap)
			})
		}
		return err
	}
	return nil
}

// MockKubeadmConfigConfigmap mocks the kubeadm-config configmap in the target cluster
// This is used to avoid the cluster-api control-plane provider to fail when trying to fetch the kubeadm-config configmap
func MockKubeadmConfigConfigmap(ctx context.Context, c client.Client) error {
	return createOrUpdateConfigMap(ctx, c, constant.KubeadmConfigKey, metav1.NamespaceSystem, map[string]string{
		"ClusterStatus":        constant.MockData,
		"ClusterConfiguration": constant.MockData,
	})
}

// MockKubeletConfigConfigmap creates a mock kubelet ConfigMap for a given Kubernetes version in the target cluster.
// This is used to avoid the cluster-api control-plane provider to fail when trying to fetch the kubelet ConfigMap.
func MockKubeletConfigConfigmap(ctx context.Context, c client.Client, currentVersion string) error {
	v, err := version.ParseMajorMinorPatch(currentVersion)
	if err != nil {
		return err
	}
	name := generateKubeletConfigName(v)

	return createOrUpdateConfigMap(ctx, c, name, metav1.NamespaceSystem, map[string]string{
		"kubelet": constant.MockData,
	})
}

// minVerUnversionedKubeletConfig is the minimum Kubernetes version that supports the unversioned kubelet configmap.
var minVerUnversionedKubeletConfig = semver.MustParse("1.24.0")

const (
	// UnversionedKubeletConfigMapName defines base kubelet configuration ConfigMap for kubeadm >= 1.24.
	UnversionedKubeletConfigMapName = "kubelet-config"
	// KubeletConfigMapName defines base kubelet configuration ConfigMap name for kubeadm < 1.24.
	KubeletConfigMapName = "kubelet-config-%d.%d"
)

// generateKubeletConfigName returns the name of the kubelet ConfigMap for a given Kubernetes version.
// used in MockKubeletConfigConfigmap which is created by kubeadm in the target cluster
func generateKubeletConfigName(version semver.Version) string {
	majorMinor := semver.Version{Major: version.Major, Minor: version.Minor}
	if majorMinor.GTE(minVerUnversionedKubeletConfig) {
		return UnversionedKubeletConfigMapName
	}
	return fmt.Sprintf(KubeletConfigMapName, version.Major, version.Minor)
}
