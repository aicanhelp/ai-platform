/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
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
	"path/filepath"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/cluster-api/util/annotations"
	"sigs.k8s.io/cluster-api/util/patch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/predicates"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// BKEMachineReconciler reconciles a BKEMachine object
type BKEMachineReconciler struct {
	client.Client
	Scheme      *runtime.Scheme
	Recorder    record.EventRecorder
	NodeFetcher *nodeutil.NodeFetcher

	nodesBootRecord map[string]struct{}
	mux             sync.Mutex
}

const (
	machineControllerName = "bkemachine"
)

func buildResetExtraCleanupArgs(extra []string) []string {
	return buildResetExtraCleanupArgsForNode(nil, extra)
}

func buildResetExtraCleanupArgsForNode(node *confv1beta1.Node, extra []string) []string {
	cleanupArgs := append([]string{}, extra...)
	cleanupArgs = append(cleanupArgs, filepath.Join("/root", ".kube", "config"))
	if node != nil && node.Username != "" && node.Username != "root" {
		cleanupArgs = append(cleanupArgs, filepath.Join("/home", node.Username, ".kube", "config"))
	}
	return cleanupArgs
}

// +kubebuilder:rbac:groups=bke.bocloud.com,resources=*,verbs=get;list;watch;create;update;patch;delete;deletecollection
// Reconcile is part of the main Kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/reconcile
func (r *BKEMachineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.ControllerLogger(machineControllerName)

	// Fetch required objects
	objects, err := r.fetchRequiredObjects(ctx, req, log)
	if err != nil || objects == nil {
		return ctrl.Result{}, err
	}

	// Setup logging context
	log = log.With("bkeMachine", objects.BKEMachine.Name)
	log = log.With("machine", objects.Machine.Name)
	log = log.With("cluster", objects.Cluster.Name)

	// Handle pause check only
	result, shouldReturn := r.handlePauseAndFinalizer(objects, log)
	if shouldReturn {
		return result, nil
	}

	// Initialize the patch helper early
	patchHelper, err := patch.NewHelper(objects.BKEMachine, r.Client)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Add finalizer first if not exist to avoid the race condition between init and delete
	if !controllerutil.ContainsFinalizer(objects.BKEMachine, bkev1beta1.BKEMachineFinalizer) {
		controllerutil.AddFinalizer(objects.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		// Immediately patch to ensure finalizer is persisted
		if err := patchBKEMachine(ctx, patchHelper, objects.BKEMachine); err != nil {
			if apierrors.IsNotFound(err) {
				return ctrl.Result{}, nil
			}
			log.Errorf("failed to patch bkeMachine after adding finalizer: %v", err)
			return ctrl.Result{}, err
		}
	}

	// Fetch BKE Cluster
	bkeCluster, err := mergecluster.GetCombinedBKECluster(ctx, r.Client, objects.BKEMachine.Namespace, objects.Cluster.Spec.InfrastructureRef.Name)
	if err != nil {
		return ctrl.Result{}, nil
	}

	log = log.With("bkeCluster", bkeCluster.Name)

	// Create params for subsequent calls
	params := BootstrapReconcileParams{
		CommonResourceParams: CommonResourceParams{
			CommonContextParams: CommonContextParams{
				Ctx: ctx,
				Log: log,
			},
			Machine:    objects.Machine,
			Cluster:    objects.Cluster,
			BKEMachine: objects.BKEMachine,
			BKECluster: bkeCluster,
		},
	}

	// Handle main reconciliation logic
	return r.handleMainReconcile(params)
}

// RequiredObjects holds the objects needed for reconciliation
type RequiredObjects struct {
	BKEMachine *bkev1beta1.BKEMachine
	Machine    *clusterv1.Machine
	Cluster    *clusterv1.Cluster
}

// fetchRequiredObjects fetches all required objects for reconciliation
func (r *BKEMachineReconciler) fetchRequiredObjects(
	ctx context.Context, req ctrl.Request, log *log.Logger,
) (*RequiredObjects, error) {
	// Fetch the BKEMachine instance
	bkeMachine := &bkev1beta1.BKEMachine{}
	if err := r.Client.Get(ctx, req.NamespacedName, bkeMachine); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}

	// Fetch the Machine
	machine, err := util.GetOwnerMachine(ctx, r.Client, bkeMachine.ObjectMeta)
	if machine == nil {
		log.Info("Waiting for Machine Controller to set OwnerRef on BKEMachine")
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	// Fetch the Cluster
	cluster, err := util.GetClusterFromMetadata(ctx, r.Client, machine.ObjectMeta)
	if err != nil {
		log.Info("BKEMachine owner Machine is missing cluster label or cluster does not exist")
		return nil, err
	}
	if cluster == nil {
		log.Info(fmt.Sprintf("Please associate this machine with a cluster using the label %s: <name of cluster>",
			clusterv1.ClusterNameLabel))
		return nil, nil
	}

	return &RequiredObjects{
		BKEMachine: bkeMachine,
		Machine:    machine,
		Cluster:    cluster,
	}, nil
}

// handlePauseAndFinalizer handles pause checks only
func (r *BKEMachineReconciler) handlePauseAndFinalizer(objects *RequiredObjects, log *log.Logger) (ctrl.Result, bool) {
	// Return early if the object or Cluster is paused
	if annotations.IsPaused(objects.Cluster, objects.BKEMachine) {
		log.Info("Reconciliation is paused for this object")
		return ctrl.Result{}, true
	}

	return ctrl.Result{}, false
}

// handleMainReconcile handles the main reconciliation logic
func (r *BKEMachineReconciler) handleMainReconcile(params BootstrapReconcileParams) (ctrl.Result, error) {
	// Initialize the patch helper
	patchHelper, err := patch.NewHelper(params.BKEMachine, r.Client)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Always attempt to Patch the bkeMachine object and Status after each reconciliation.
	defer func() {
		if err := patchBKEMachine(params.Ctx, patchHelper, params.BKEMachine); err != nil {
			if apierrors.IsNotFound(err) {
				return
			}
			params.Log.Errorf("failed to patch bkeMachine: %v", err)
		}
	}()

	// Handle deleted bke machines
	if !params.BKEMachine.ObjectMeta.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(params)
	}

	// Check if the infrastructure is ready, otherwise return and wait for the cluster object to be updated
	if !params.Cluster.Status.InfrastructureReady {
		params.Log.Info("Waiting for BKECluster Controller to create cluster infrastructure")
		return ctrl.Result{}, nil
	}

	// 查询 bkemachine 关联的 node，如果关联的节点目前状态被标记了失败状态码，则直接返回
	hostIp, found := labelhelper.CheckBKEMachineLabel(params.BKEMachine)
	if found {
		hasFailedFlag, err := r.NodeFetcher.GetNodeStateFlagForCluster(params.Ctx, params.BKECluster, hostIp, bkev1beta1.NodeFailedFlag)
		if err == nil && hasFailedFlag {
			return ctrl.Result{}, nil
		}
	}

	return r.reconcile(params)
}

func (r *BKEMachineReconciler) reconcile(params BootstrapReconcileParams) (ctrl.Result, error) {
	if params.BKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
		params.Log.Info("bkeCluster is in deleting phase, waiting for bkeCluster to be deleted")
		return ctrl.Result{}, nil
	}

	var res ctrl.Result
	var errs []error

	// Call reconcileCommand
	commandResult, err := r.reconcileCommand(params)
	if err != nil {
		errs = append(errs, err)
	}
	if len(errs) == 0 {
		res = util.LowestNonZeroResult(res, commandResult)
	}

	// Call reconcileBootstrap
	if len(errs) == 0 {
		bootstrapResult, err := r.reconcileBootstrap(params)
		if err != nil {
			errs = append(errs, err)
		} else {
			res = util.LowestNonZeroResult(res, bootstrapResult)
		}
	}

	return res, kerrors.NewAggregate(errs)
}

// reconcileDelete handles BKEMachine deletion.
// reconcileDelete handles BKEMachine deletion.
func (r *BKEMachineReconciler) reconcileDelete(params BootstrapReconcileParams) (ctrl.Result, error) {
	params.Log = params.Log.With("name", "reconcileBKEMachineDelete", "bkeMachine", params.BKEMachine.Name)

	// Handle pre-deletion cleanup
	defer r.handlePreDeletionCleanup(params)

	// Check if already marked for deletion
	if isMarkDeletion(params.BKEMachine) {
		return r.handleAlreadyMarkedDeletion(params)
	}

	// Setup deletion process
	patchHelper, err := r.setupDeletionProcess(params)
	if err != nil {
		return ctrl.Result{}, err
	}
	defer patchBKEMachine(params.Ctx, patchHelper, params.BKEMachine)

	// Get node for deletion
	node, err := r.getNodeForDeletion(params)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Set node state
	if err := r.NodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeDeleting, "Deleting"); err != nil {
		params.Log.Warnf("Failed to set node state: %v", err)
	}
	if err := mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster); err != nil {
		return ctrl.Result{}, err
	}

	// Handle post-node-setup cleanup
	defer r.handlePostNodeSetupCleanup(params, node)

	// Check if should skip deletion
	if r.shouldSkipDeletion(params, node) {
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		return ctrl.Result{}, nil
	}

	// Execute reset command
	return r.executeResetCommand(params, node)
}

// handlePreDeletionCleanup handles pre-deletion cleanup tasks
func (r *BKEMachineReconciler) handlePreDeletionCleanup(params BootstrapReconcileParams) {
	if !controllerutil.ContainsFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer) {
		nodeIP, found := labelhelper.CheckBKEMachineLabel(params.BKEMachine)
		if found && nodeIP != "" {
			r.mux.Lock()
			delete(r.nodesBootRecord, nodeIP)
			r.mux.Unlock()

			// remove node from BKENode CRD
			if err := r.NodeFetcher.DeleteBKENodeForCluster(params.Ctx, params.BKECluster, nodeIP); err != nil {
				params.Log.Warnf("Failed to delete BKENode for IP %s: %v", nodeIP, err)
			}
			// remove node from AppointmentDeletedNodesAnnotationKey
			patchFunc := func(cluster *bkev1beta1.BKECluster) {
				phaseutil.RemoveAppointmentDeletedNodes(cluster, nodeIP)
			}
			_ = mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster, patchFunc)
		}
	}
}

// handleAlreadyMarkedDeletion handles case where BKEMachine is already marked for deletion
func (r *BKEMachineReconciler) handleAlreadyMarkedDeletion(params BootstrapReconcileParams) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer) {
		return ctrl.Result{}, nil
	}
	params.Log.Debug("BKEMachine is already marked for deletion")
	return r.reconcileCommand(params)
}

// setupDeletionProcess sets up the deletion process
func (r *BKEMachineReconciler) setupDeletionProcess(params BootstrapReconcileParams) (*patch.Helper, error) {
	patchHelper, err := patch.NewHelper(params.BKEMachine, r.Client)
	if err != nil {
		params.Log.Warnf("failed to create patch helper for bkeMachine %s, requeue", params.BKEMachine.Name)
		return nil, err
	}

	// Mark the bkeMachine as deleted to avoid re-entry
	if err := markBKEMachineDeletion(params.Ctx, params.BKEMachine, patchHelper); err != nil {
		r.logWarningAndEvent(params.Log, params.BKECluster, constant.CommandCreateFailedReason,
			"failed to mark BKEMachine %s as deleted, retry after 10 second", params.BKEMachine.Name)
		return nil, err
	}

	return patchHelper, nil
}

// getNodeForDeletion gets the node for deletion
func (r *BKEMachineReconciler) getNodeForDeletion(params BootstrapReconcileParams) (*confv1beta1.Node, error) {
	// Fetch nodes from BKENode CRD
	nodes, err := r.NodeFetcher.GetNodesForBKECluster(params.Ctx, params.BKECluster)
	if err != nil {
		params.Log.Warnf("failed to get nodes from BKENode CRD: %v", err)
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		return nil, err
	}
	params.Log.Debug("step 1 get bke node from BKENode CRD")
	node, err := bkeMachineToNode(params.BKEMachine, nodes)
	if err != nil {
		params.Log.Warnf("failed to get node from BKENode CRD, force delete: %v", err)
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		return nil, err
	}
	return node, nil
}

// handlePostNodeSetupCleanup handles post-node-setup cleanup tasks
func (r *BKEMachineReconciler) handlePostNodeSetupCleanup(params BootstrapReconcileParams, node *confv1beta1.Node) {
	if !controllerutil.ContainsFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer) {
		r.mux.Lock()
		delete(r.nodesBootRecord, node.IP)
		r.mux.Unlock()

		// remove node from BKENode CRD
		if err := r.NodeFetcher.DeleteBKENodeForCluster(params.Ctx, params.BKECluster, node.IP); err != nil {
			params.Log.Warnf("Failed to delete BKENode for IP %s: %v", node.IP, err)
		}
		// remove node from AppointmentDeletedNodesAnnotationKey
		patchFunc := func(cluster *bkev1beta1.BKECluster) {
			phaseutil.RemoveAppointmentDeletedNodes(cluster, node.IP)
		}
		_ = mergecluster.SyncStatusUntilComplete(r.Client, params.BKECluster, patchFunc)
	}
}

// shouldSkipDeletion checks if deletion should be skipped
func (r *BKEMachineReconciler) shouldSkipDeletion(params BootstrapReconcileParams, node *confv1beta1.Node) bool {
	params.Log = params.Log.With("node", phaseutil.NodeInfo(*node))

	if condition.HasConditionStatus(bkev1beta1.SwitchBKEAgentCondition, params.BKECluster, confv1beta1.ConditionTrue) {
		params.Log.Info("agent not listening current cluster, delete BKEMachine directly")
		return true
	}

	// 不删除目标集群，如果有注解，且注解值为true或没有注释时
	if params.BKECluster.Status.ClusterStatus == bkev1beta1.ClusterDeleting {
		if v, ok := annotation.HasAnnotation(params.BKECluster, annotation.DeleteIgnoreTargetClusterAnnotationKey); (ok && v == "true") || !ok {
			params.Log.Info("ignore delete target cluster, delete BKEMachine directly")
			return true
		}
	}

	params.Log.Debug("step 3 check BKEAgent is running")
	// 没有推送过agent，直接删除不做清理了
	hasAgentReadyFlag, err := r.NodeFetcher.GetNodeStateFlagForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeAgentReadyFlag)
	if err != nil {
		params.Log.Debugf("failed to get agent ready flag (BKENode may be deleted): %v, will try reset", err)
		return false
	}
	if !hasAgentReadyFlag {
		// Agent was never deployed, skip reset
		params.Log.Debug("agent was never deployed, skip reset")
		return true
	}

	return false
}

// executeResetCommand executes the reset command
func (r *BKEMachineReconciler) executeResetCommand(params BootstrapReconcileParams, node *confv1beta1.Node) (ctrl.Result, error) {
	params.Log = params.Log.With("node", phaseutil.NodeInfo(*node))
	params.Log.Debugf("step 4 create reset command for node %s", phaseutil.NodeInfo(*node))

	// extra is used to store extra ip in node interface、file or directory to remove
	var extra []string

	// Fetch nodes from BKENode CRD to check load balancer endpoint
	nodes, err := r.NodeFetcher.GetNodesForBKECluster(params.Ctx, params.BKECluster)
	if err != nil {
		params.Log.Warnf("Failed to fetch nodes for cluster: %v", err)
	}
	if err == nil && clusterutil.AvailableLoadBalancerEndPoint(params.BKECluster.Spec.ControlPlaneEndpoint, nodes) {
		extra = append(extra, params.BKECluster.Spec.ControlPlaneEndpoint.Host)
	}
	ingressVip, _ := clusterutil.GetIngressConfig(params.BKECluster.Spec.ClusterConfig.Addons)
	if ingressVip != "" && ingressVip != params.BKECluster.Spec.ControlPlaneEndpoint.Host {
		extra = append(extra, ingressVip)
	}
	extra = buildResetExtraCleanupArgsForNode(node, extra)

	v, ok := annotation.HasAnnotation(params.BKECluster, annotation.DeepRestoreNodeAnnotationKey)
	deepRestore := (ok && v == "true") || !ok
	reset := command.Reset{
		BaseCommand: command.BaseCommand{
			Ctx:             params.Ctx,
			NameSpace:       params.BKECluster.Namespace,
			Client:          r.Client,
			Scheme:          r.Scheme,
			OwnerObj:        params.BKEMachine,
			ClusterName:     params.BKECluster.Name,
			Unique:          true,
			RemoveAfterWait: true,
		},
		Node:        node,
		BKEConfig:   params.BKECluster.Name,
		Extra:       extra,
		DeepRestore: deepRestore,
	}
	if err := reset.New(); err != nil {
		if apierrors.HasStatusCause(err, corev1.NamespaceTerminatingCause) {
			controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
			return ctrl.Result{}, nil
		}

		errInfo := "failed to create reset command"
		r.logErrorAndEvent(params.Log, params.BKECluster, constant.CommandCreateFailedReason, "%s: %v", errInfo, err)
		return ctrl.Result{}, nil
	}

	r.logInfoAndEvent(params.Log, params.BKECluster, constant.CommandCreateSuccessReason, "reset command created, node  %q", phaseutil.NodeInfo(*node))
	// wait for reset command to be completed
	params.Log.Debug("step 5 wait for reset command to be completed")
	err, _, failed := reset.Wait()
	if err != nil || len(failed) > 0 {
		params.Log.Infof("failed to wait for reset command to be completed, delete directly, err: %v", err)
		// remove finalizer
		controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
		return ctrl.Result{}, nil
	}
	r.logInfoAndEvent(params.Log, params.BKECluster, constant.WorkerDeletedReason, "reset command completed, remove node %q from cluster %q", node.IP, params.Cluster.Name)

	// Agent uninstall is Reset scope "shutdown"; do not issue a separate Shutdown Command.
	controllerutil.RemoveFinalizer(params.BKEMachine, bkev1beta1.BKEMachineFinalizer)
	return ctrl.Result{}, nil
}

// patchBKEMachine will patch the BKEMachine
func patchBKEMachine(ctx context.Context, patchHelper *patch.Helper, bkeMachine *bkev1beta1.BKEMachine) error {
	return patchHelper.Patch(ctx, bkeMachine)
}

// SetupWithManager sets up the controller with the Manager.
func (r *BKEMachineReconciler) SetupWithManager(mgr ctrl.Manager, options controller.Options) error {
	r.nodesBootRecord = make(map[string]struct{})
	r.mux = sync.Mutex{}
	r.NodeFetcher = nodeutil.NewNodeFetcher(mgr.GetClient())

	clusterToBKEMachines, err := util.ClusterToObjectsMapper(mgr.GetClient(), &bkev1beta1.BKEMachineList{}, mgr.GetScheme())
	if err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&bkev1beta1.BKEMachine{}).
		WithOptions(options).
		Watches(
			&agentv1beta1.Command{},
			handler.EnqueueRequestForOwner(mgr.GetScheme(), mgr.GetRESTMapper(), &bkev1beta1.BKEMachine{}, handler.OnlyControllerOwner()),
			builder.WithPredicates(predicates.CommandUpdateCompleted()),
		).
		Watches(
			&clusterv1.Machine{},
			handler.EnqueueRequestsFromMapFunc(util.MachineToInfrastructureMapFunc(bkev1beta1.GroupVersion.WithKind("BKEMachine"))),
		).
		Watches(
			&clusterv1.Cluster{},
			handler.EnqueueRequestsFromMapFunc(clusterToBKEMachines),
			builder.WithPredicates(predicates.ClusterUnPause()),
		).
		Watches(
			&bkev1beta1.BKECluster{},
			handler.EnqueueRequestsFromMapFunc(r.BKEClusterToBKEMachines),
			builder.WithPredicates(predicates.BKEAgentReady(), predicates.BKEClusterUnPause()),
		).
		Complete(r)
}

// BKEClusterToBKEMachines is a handler.ToRequestsFunc to be used to enqeue
// requests for reconciliation of BKEMachines.
func (r *BKEMachineReconciler) BKEClusterToBKEMachines(ctx context.Context, o client.Object) []ctrl.Request {
	var result []ctrl.Request
	c, ok := o.(*bkev1beta1.BKECluster)
	if !ok {
		panic(fmt.Sprintf("Expected a BKECluster but got a %T", o))
	}

	cluster, err := util.GetOwnerCluster(context.TODO(), r.Client, c.ObjectMeta)
	switch {
	case apierrors.IsNotFound(err) || cluster == nil:
		return result
	case err != nil:
		return result
	default:
	}

	labels := map[string]string{clusterv1.ClusterNameLabel: cluster.Name}
	machineList := &clusterv1.MachineList{}
	if err := r.Client.List(context.TODO(), machineList, client.InNamespace(c.Namespace), client.MatchingLabels(labels)); err != nil {
		return nil
	}
	for _, m := range machineList.Items {
		if m.Spec.InfrastructureRef.Name == "" || m.Status.BootstrapReady {
			continue
		}
		name := client.ObjectKey{Namespace: m.Spec.InfrastructureRef.Namespace, Name: m.Spec.InfrastructureRef.Name}
		result = append(result, ctrl.Request{NamespacedName: name})
	}

	return result
}

// LogCommandFailed log command failed message
func (r *BKEMachineReconciler) LogCommandFailed(
	cmd agentv1beta1.Command, bkeCluster *bkev1beta1.BKECluster,
	failedNods []string, log *log.Logger, reson string,
) string {
	for _, node := range failedNods {
		nodeStatus := cmd.Status[node]
		if nodeStatus == nil {
			continue
		}
		for _, condition := range nodeStatus.Conditions {
			if condition.Status == metav1.ConditionFalse && (condition.StdErr != nil || len(condition.StdErr) > 0) {
				// 输出最后一次运行的错误信息
				r.logWarningAndEvent(log, bkeCluster, reson, "Node %q, Command %s, sub ID %q, err: %s", node, utils.ClientObjNS(&cmd), condition.ID, condition.StdErr[len(condition.StdErr)-1])
				return condition.StdErr[len(condition.StdErr)-1]
			}
		}
	}
	return ""
}

func (r *BKEMachineReconciler) logInfoAndEvent(
	log *log.Logger, bkeCluster *bkev1beta1.BKECluster,
	reason, msg string, args ...interface{},
) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	r.Recorder.AnnotatedEventf(bkeCluster, annotation.BKENormalEventAnnotation(), corev1.EventTypeNormal, reason, msg, args...)
	if log != nil {
		log.Infof(msg, args...)
		return
	}
	log.Infof(msg, args...)
}

func (r *BKEMachineReconciler) logErrorAndEvent(
	log *log.Logger, bkeCluster *bkev1beta1.BKECluster,
	reason, msg string, args ...interface{},
) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	r.Recorder.AnnotatedEventf(bkeCluster, annotation.BKENormalEventAnnotation(), corev1.EventTypeWarning, reason, msg, args...)
	if log != nil {
		log.Errorf(msg, args...)
		return
	}
	log.Errorf(msg, args...)
}

func (r *BKEMachineReconciler) logWarningAndEvent(
	log *log.Logger, bkeCluster *bkev1beta1.BKECluster,
	reason, msg string, args ...interface{},
) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	r.Recorder.AnnotatedEventf(bkeCluster, annotation.BKENormalEventAnnotation(), corev1.EventTypeWarning, reason, msg, args...)
	if log != nil {
		log.Warnf(msg, args...)
		return
	}
	log.Warnf(msg, args...)
}

func (r *BKEMachineReconciler) logFinishAndEvent(
	log *log.Logger, bkeCluster *bkev1beta1.BKECluster,
	reason, msg string, args ...interface{},
) {
	tameStamp := time.Now().Unix()
	msg = fmt.Sprintf("(%d) %s", tameStamp, msg)
	r.Recorder.AnnotatedEventf(bkeCluster, annotation.BKEFinishEventAnnotation(), corev1.EventTypeNormal, reason, msg, args...)
	if log != nil {
		log.Infof(msg, args...)
		return
	}
	log.Infof(msg, args...)
}

// bkeMachineToNode convert bke machine to node
func bkeMachineToNode(bkeMachine *bkev1beta1.BKEMachine, bkeNodes bkenode.Nodes) (*confv1beta1.Node, error) {
	if bkeMachine.Status.Node != nil {
		return bkeMachine.Status.Node, nil
	}

	hostIP, ok := labelhelper.CheckBKEMachineLabel(bkeMachine)
	if !ok {
		return nil, fmt.Errorf("bke machine %s label is not found", bkeMachine.Name)
	}
	nodes := bkeNodes.Filter(bkenode.FilterOptions{"IP": hostIP})
	if nodes.Length() == 0 {
		log.Warnf("node %s is not found in BKECluster.Status.Nodes, maybe already deleted", hostIP)
		// still try to create node
		return &confv1beta1.Node{
			IP: hostIP,
		}, nil
	}
	return &nodes[0], nil
}

// markBKEMachineDeletion mark bke machine deletion
func markBKEMachineDeletion(ctx context.Context, bkeMachine *bkev1beta1.BKEMachine, patchHelper *patch.Helper) error {
	as := bkeMachine.GetAnnotations()
	if as == nil {
		as = map[string]string{}
	}
	as[clusterv1.DeleteMachineAnnotation] = ""
	bkeMachine.SetAnnotations(as)
	return patchHelper.Patch(ctx, bkeMachine)
}

// isMarkDeletion check if bke machine is marked deletion
func isMarkDeletion(bkeMachine *bkev1beta1.BKEMachine) bool {
	as := bkeMachine.GetAnnotations()
	if as == nil {
		return false
	}
	_, ok := as[clusterv1.DeleteMachineAnnotation]
	return ok
}

// getBKEMachineAssociateCommands get bke machine associate commands
func getBKEMachineAssociateCommands(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, bkeMachine *bkev1beta1.BKEMachine) ([]agentv1beta1.Command, error) {
	commandsLi := agentv1beta1.CommandList{}
	filters := phaseutil.GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, &commandsLi, filters...); err != nil {
		return nil, err
	}
	var commands []agentv1beta1.Command
	for _, cmdItem := range commandsLi.Items {
		if !command.IsOwnerRefCommand(bkeMachine, cmdItem) {
			continue
		}
		if _, ok := cmdItem.Annotations[annotation.CommandReconciledAnnotationKey]; ok {
			continue
		}
		if err := command.ValidateCommand(&cmdItem); err != nil {
			log.Error(cmdItem.Name, err)
			continue
		}
		commands = append(commands, cmdItem)
	}

	return commands, nil
}
