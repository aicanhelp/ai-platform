/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"k8s.io/apimachinery/pkg/api/meta"
	"os"
	"strconv"
	"time"

	karmadav1alpha1 "github.com/karmada-io/karmada/operator/pkg/apis/operator/v1alpha1"
	karmadaclusterv1alpha1 "github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	karmadaclientset "github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"openfuyao.com/clusters-computing-operator/internal/config"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clusterv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// TenantClusterReconciler reconciles a TenantCluster object
type TenantClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

type TenantClusterStatus struct {
	SuperClusterDReady               bool
	KarmadaReady                     bool
	TotalManagedClustersCount        int
	InstallationReadyManagedClusters []string // 集群安装成功
	JoinedManagedClusters            []string
	ReadyManagedClusters             []string // 集群Ready，所有组件均已安装完成
	NotReadyManagedClusters          []string
	FinalState                       bool
}

const (
	deploymentAPIVersion = "apps/v1"
	deploymentKind       = "Deployment"

	ManagedLabel           = "app.kubernetes.io/managed-by"
	ManagedLabelValue      = "cluster.operator.huawei.com"
	TenantClusterLabelName = "huawei.com/tenantcluster-name"
	NameLabel              = "app.kubernetes.io/name"

	VersionLabel = "app.kubernetes.io/version"

	karmadaConfigFile = "config/app/cr/karmada.yaml"

	karmadaDefaultName = "tenantcluster-karmada"

	tenantClusterKubeconfigKey = "tenant.cluster.admin.config"

	managedClusterConfigSuffix = "-managed-kubeconfig"

	karmadaConfigSuffix = "-admin-config"

	karmadaJoinJobPrefix = "karmada-join-"

	DefaultReconcileInterval = time.Second * 15

	UnManagedMode = "unmanaged"

	// the master cluster's Condition types
	SuperClusterDCondition string = "SuperClusterD"
	KarmadaCondition       string = "Karmada"

	// Condition reasons
	ReasonReady      = "Ready"
	ReasonInProgress = "InProgress"
	ReasonFailed     = "Failed"

	TenantClusterFinalizer = "tenantcluster.finalizers.openfuyao.com"
)

// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=operator.karmada.io,resources=karmadas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the TenantCluster object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *TenantClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Reconciling TenantCluster")

	tenantCluster := &clusterv1alpha1.TenantCluster{}
	if err := r.Get(ctx, req.NamespacedName, tenantCluster); err != nil {
		logger.Error(err, "Cannot fetch TenantCluster")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	//  Check if the resource is marked for deletion
	if !tenantCluster.DeletionTimestamp.IsZero() {
		logger.Info("TenantCluster is being deleted, handling deletion", "name", tenantCluster.Name)
		if err := r.handleDeletion(ctx, tenantCluster); err != nil {
			logger.Error(err, "Failed to handle deletion of TenantCluster")
			return ctrl.Result{}, err
		}
		logger.Info("TenantCluster deleted successfully", "name", tenantCluster.Name)
		return ctrl.Result{}, nil
	}

	// The resource is not marked for deletion, ensure to add our finalizer
	if !controllerutil.ContainsFinalizer(tenantCluster, TenantClusterFinalizer) {
		controllerutil.AddFinalizer(tenantCluster, TenantClusterFinalizer)
		if err := r.Update(ctx, tenantCluster); err != nil {
			return ctrl.Result{Requeue: true}, err
		}
		// Return immediately after adding the finalizer, wait for the next reconciliation
		logger.Info("Added finalizer", "name", tenantCluster.Name)
	}

	// Build the current cluster's status to prevent calling related functions repeatedly
	clusterStatus := r.buildClusterStatus(ctx, tenantCluster)

	// Create a kubeconfig for the management cluster, which will be used in the job for creating the managed cluster
	if err := r.reconcileKubeconfigSecret(ctx, tenantCluster); err != nil {
		logger.Error(err, "Cannot create or update kubeconfig secret")
		return ctrl.Result{}, err
	}

	// Create SuperClusterD Deployment
	if err := r.reconcileSuperClusterD(ctx, tenantCluster); err != nil {
		logger.Error(err, "reconcileSuperClusterD failed")
		return ctrl.Result{}, err
	}

	// Create Karmada instance
	if err := r.reconcileKarmada(ctx, tenantCluster); err != nil {
		logger.Error(err, "reconcileKarmada failed")
		return ctrl.Result{}, err
	}

	if tenantCluster.Spec.Mode == UnManagedMode {
		return r.reconcileUnmanagedMode(ctx, tenantCluster, clusterStatus)
	}

	return r.reconcileManagedMode(ctx, tenantCluster, clusterStatus)
}

func (r *TenantClusterReconciler) handleDeletion(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	logger := log.FromContext(ctx)
	// Resource has been marked for deletion, check if we have our finalizer
	if controllerutil.ContainsFinalizer(tc, TenantClusterFinalizer) {
		// First, check if all TenantManagedClusters have been cleaned up
		if tmcExist, err := r.checkAndDeleteTenantManagedClusters(ctx, tc); err != nil || tmcExist {
			return fmt.Errorf("waiting for TenantManagedClusters to be fully deleted")
		}

		// Clean up Karmada
		if err := r.cleanupKarmada(ctx, tc); err != nil {
			return fmt.Errorf("failed to cleanup Karmada: %w", err)
		}

		// Clean up completed, remove finalizer
		controllerutil.RemoveFinalizer(tc, TenantClusterFinalizer)
		if err := r.Update(ctx, tc); err != nil {
			return fmt.Errorf("failed to remove finalizer: %w", err)
		}
		logger.Info("Successfully removed finalizer", "name", tc.Name)
	}
	return nil
}

func (r *TenantClusterReconciler) checkAndDeleteTenantManagedClusters(ctx context.Context, tc *clusterv1alpha1.TenantCluster) (bool, error) {
	logger := log.FromContext(ctx)

	tmcList := &clusterv1alpha1.TenantManagedClusterList{}
	if err := r.List(ctx, tmcList, client.InNamespace(tc.Namespace)); err != nil {
		return false, fmt.Errorf("failed to list tenantmanaged clusters: %w", err)
	}

	if len(tmcList.Items) == 0 {
		return false, nil
	}

	if len(tmcList.Items) > 0 {
		logger.Info("TenantManagedClusters still exist, waiting for them to be fully deleted", "count", len(tmcList.Items))
		// Delete any TenantManagedCluster that is not marked for deletion
		for i := range tmcList.Items {
			tmc := &tmcList.Items[i]
			if tmc.DeletionTimestamp.IsZero() {
				if err := r.Delete(ctx, tmc); err != nil && !errors.IsNotFound(err) {
					return true, fmt.Errorf("failed to delete tenantManagedCluster %s: %w", tmc.Name, err)
				}
			}
		}
	}
	return true, nil
}

func (r *TenantClusterReconciler) buildClusterStatus(ctx context.Context, tc *clusterv1alpha1.TenantCluster) *TenantClusterStatus {
	logger := log.FromContext(ctx)
	status := &TenantClusterStatus{}

	// Check the control plane
	status.SuperClusterDReady = r.isTenantClusterComponentReady(tc, SuperClusterDCondition)
	status.KarmadaReady = r.isTenantClusterComponentReady(tc, KarmadaCondition)

	// Get the managed cluster status
	var tmcList clusterv1alpha1.TenantManagedClusterList
	if err := r.List(ctx, &tmcList, client.InNamespace(tc.Namespace)); err != nil {
		logger.Error(err, "Failed to list TenantManagedClusters")
		return status
	}

	for _, tmc := range tmcList.Items {
		for _, condition := range tmc.Status.Conditions {
			switch {
			case condition.Type == ClusterInstallationCondition && condition.Status == metav1.ConditionTrue:
				status.InstallationReadyManagedClusters = append(status.InstallationReadyManagedClusters, tmc.Name)
			case condition.Type == JoinedKarmadaCondition && condition.Status == metav1.ConditionTrue:
				status.JoinedManagedClusters = append(status.JoinedManagedClusters, tmc.Name)
			case condition.Type == ClusterCondition:
				if condition.Status == metav1.ConditionTrue {
					status.ReadyManagedClusters = append(status.ReadyManagedClusters, tmc.Name)
				} else {
					status.NotReadyManagedClusters = append(status.NotReadyManagedClusters, tmc.Name)
				}
			}
		}
	}

	// Unmanaged mode requires waiting for the user to join the already existing managed cluster to Karmada, set FinalState to false
	if tc.Spec.Mode == UnManagedMode {
		status.FinalState = false
		return status
	}

	managedClusters := r.buildManagedClusters(tc)
	status.TotalManagedClustersCount = len(managedClusters)

	// Set the final state
	status.FinalState = status.SuperClusterDReady &&
		status.KarmadaReady &&
		status.TotalManagedClustersCount == len(status.ReadyManagedClusters)

	return status
}

func (r *TenantClusterReconciler) isTenantClusterComponentReady(tc *clusterv1alpha1.TenantCluster, name string) bool {
	for _, condition := range tc.Status.Conditions {
		if condition.Type == name && condition.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func (r *TenantClusterReconciler) reconcileSuperClusterD(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	// Check if it already exists
	existingDeployment := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      fmt.Sprintf("%s-superclusterd", tc.Name),
		Namespace: tc.Namespace,
	}, existingDeployment)

	if err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("failed to get SuperClusterD deployment %w", err)
	}

	// If not, create it
	if errors.IsNotFound(err) {
		newDeployment := r.buildSuperClusterdDeployment(tc)
		if err = controllerutil.SetControllerReference(tc, newDeployment, r.Scheme); err != nil {
			return fmt.Errorf("failed to set SuperClusterD deployment owner reference %w", err)
		}
		if err = r.Create(ctx, newDeployment); err != nil {
			return fmt.Errorf("failed to create SuperClusterD deployment %w", err)
		}
		return nil
	}

	// Update the status
	if existingDeployment.Status.AvailableReplicas >= *existingDeployment.Spec.Replicas {
		updateTenantClusterCondition(tc, SuperClusterDCondition, metav1.ConditionTrue, ReasonReady, "SuperClusterD is ready")
	} else {
		updateTenantClusterCondition(tc, SuperClusterDCondition, metav1.ConditionFalse, ReasonInProgress, "SuperClusterD is being installed")
	}
	return nil
}

func (r *TenantClusterReconciler) reconcileKarmada(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	karmada, err := config.LoadKarmadaConfig(karmadaConfigFile, tc)
	if err != nil {
		return fmt.Errorf("failed to load Karmada configuration: %w", err)
	}

	karmada.Name = getKarmadaName(tc)

	// if Karmada does not exist, install it
	err = r.Get(ctx, types.NamespacedName{
		Name:      karmada.Name,
		Namespace: karmada.Namespace,
	}, karmada)

	if err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("failed to get Karmada: %w", err)
	}

	// If not, create it
	if errors.IsNotFound(err) {
		if err = r.Create(ctx, karmada); err != nil {
			return fmt.Errorf("failed to create Karmada: %w", err)
		}
		return nil
	}

	// Update the status
	r.updateKarmadaStatus(tc, karmada)
	return nil
}

func getKarmadaName(tc *clusterv1alpha1.TenantCluster) string {
	if tc.Spec.ControlPlane.Karmada.Name == "" {
		return karmadaDefaultName
	}
	return tc.Spec.ControlPlane.Karmada.Name
}

func (r *TenantClusterReconciler) updateKarmadaStatus(tc *clusterv1alpha1.TenantCluster, karmada *karmadav1alpha1.Karmada) {
	state := metav1.ConditionFalse
	for _, condition := range karmada.Status.Conditions {
		if condition.Type == "Ready" && condition.Status == metav1.ConditionTrue {
			state = metav1.ConditionTrue
			break
		}
	}
	if state == metav1.ConditionTrue {
		updateTenantClusterCondition(tc, KarmadaCondition, state, ReasonReady, "Karmada is ready")
	} else {
		updateTenantClusterCondition(tc, KarmadaCondition, state, ReasonInProgress, "Karmada is being installed")
	}
}

func (r *TenantClusterReconciler) reconcileManagedClusters(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	managedClusters := r.buildManagedClusters(tc)
	for _, tmc := range managedClusters {
		if err := controllerutil.SetControllerReference(tc, tmc, r.Scheme); err != nil {
			return fmt.Errorf("failed to set TenantManagedCluster owner reference: %w", err)
		}
		if err := r.Create(ctx, tmc); err != nil && !errors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create TenantManagedCluster: %w", err)
		}
	}
	return nil
}

func updateTenantClusterCondition(tc *clusterv1alpha1.TenantCluster, conditionType string, status metav1.ConditionStatus, reason, message string) {
	updateCondition(&tc.Status.Conditions, conditionType, status, reason, message)
}

func updateCondition(conditions *[]metav1.Condition, conditionType string, status metav1.ConditionStatus, reason, message string) {
	// Check if there is an existing condition
	existingCondition := meta.FindStatusCondition(*conditions, conditionType)

	// Create new condition
	newCondition := metav1.Condition{
		Type:               conditionType,
		Status:             status,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
	}
	// Update only when the status or reason changes
	if existingCondition == nil || existingCondition.Status != status || existingCondition.Reason != reason {
		meta.SetStatusCondition(conditions, newCondition)
	}
}

func (r *TenantClusterReconciler) reconcileUnmanagedMode(ctx context.Context, tc *clusterv1alpha1.TenantCluster, clusterStatus *TenantClusterStatus) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Karmada installation successful, obtain Karmada Client
	if clusterStatus.KarmadaReady {
		if err := r.getKarmadaClient(ctx, tc.Namespace, tc); err != nil {
			logger.Error(err, "getKarmadaClient failed")
			return ctrl.Result{}, err
		}

		logger.Info("Please join the cluster into karmada!")
		logger.Info("I will be here waiting for you!")
		// Get tenantManagedCluster and Karmada sub-clusters.
		// If the tenantManagedCluster already contains Karmada managed clusters, no need to process. If new ones exist, they need processing
		// clusterStatus.JoinedManagedClusters - already processed managed clusters
		memberClusters, err := r.getMemberClusters(ctx)
		if err != nil {
			logger.Error(err, "Failed to get member clusters")
			return ctrl.Result{}, err
		}

		// Filter out unclaimed clusters, if processed, tenantManagedCluster will mark them as joined to Karmada
		needingAdoptionClusters := r.filterClustersNeedingAdoption(memberClusters, clusterStatus.JoinedManagedClusters)
		for _, cluster := range needingAdoptionClusters {
			if err = r.adoptCluster(ctx, tc, cluster); err != nil {
				logger.Error(err, "Failed to adopt cluster", "cluster", cluster)
			}
		}
	}

	if err := r.updateClusterStatus(ctx, tc, clusterStatus); err != nil {
		logger.Error(err, "update tc status error")
		return ctrl.Result{}, err
	}

	// Unmanaged mode requires waiting for the user to join the already created sub-clusters to Karmada
	return ctrl.Result{RequeueAfter: DefaultReconcileInterval}, nil
}

func (r *TenantClusterReconciler) reconcileManagedMode(ctx context.Context, tc *clusterv1alpha1.TenantCluster, clusterStatus *TenantClusterStatus) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Create managed cluster
	if err := r.reconcileManagedClusters(ctx, tc); err != nil {
		logger.Error(err, "reconcileManagedClusters failed")
		return ctrl.Result{}, err
	}

	// Karmada installation successful, obtain Karmada Client
	if clusterStatus.KarmadaReady {
		if err := r.getKarmadaClient(ctx, tc.Namespace, tc); err != nil {
			logger.Error(err, "getKarmadaClient failed")
			return ctrl.Result{}, err
		}
	}

	// Karmada installation successful, and managed clusters are already installed, create job to add the managed cluster to Karmada
	if clusterStatus.KarmadaReady && len(clusterStatus.InstallationReadyManagedClusters) > 0 {
		if err := r.reconcileJoinClusterToKarmadaJob(ctx, tc, clusterStatus); err != nil {
			logger.Error(err, "reconcileJoinClusterToKarmadaJob failed")
			return ctrl.Result{}, err
		}
	}

	if err := r.updateClusterStatus(ctx, tc, clusterStatus); err != nil {
		logger.Error(err, "update tc status error")
		return ctrl.Result{}, err
	}

	// Final state check
	if clusterStatus.FinalState {
		logger.Info("TenantCluster complete, all components are ready")
		return ctrl.Result{}, nil
	}

	return ctrl.Result{RequeueAfter: DefaultReconcileInterval}, nil
}

func (r *TenantClusterReconciler) updateClusterStatus(ctx context.Context, tc *clusterv1alpha1.TenantCluster, clusterStatus *TenantClusterStatus) error {
	tc.Status.MemberClusterState = clusterv1alpha1.MemberClusterState{
		ReadyClusterCount:    len(clusterStatus.ReadyManagedClusters),
		ReadyClusterList:     clusterStatus.ReadyManagedClusters,
		NotReadyClusterCount: len(clusterStatus.NotReadyManagedClusters),
		NotReadyClusterList:  clusterStatus.NotReadyManagedClusters,
	}
	return r.Status().Update(ctx, tc)
}

func (r *TenantClusterReconciler) reconcileJoinClusterToKarmadaJob(ctx context.Context, tc *clusterv1alpha1.TenantCluster, clusterStatus *TenantClusterStatus) error {
	// Filter out clusters not yet added to Karmada
	clustersToJoin := r.filterNotJoinedClusters(clusterStatus.InstallationReadyManagedClusters, clusterStatus.JoinedManagedClusters)

	for _, tmcClusterName := range clustersToJoin {
		job := &batchv1.Job{}
		nn := types.NamespacedName{
			Name:      getJoinClusterToKarmadaJobName(tmcClusterName),
			Namespace: tc.Namespace,
		}
		err := r.Get(ctx, nn, job)
		if err != nil && !errors.IsNotFound(err) {
			return fmt.Errorf("failed to get cluster to join Karmada job: %w", err)
		}

		if errors.IsNotFound(err) {
			job = r.buildJoinClusterToKarmadaJob(tc, tmcClusterName, getKarmadaName(tc))
			if err = controllerutil.SetControllerReference(tc, job, r.Scheme); err != nil {
				return fmt.Errorf("failed to set cluster to join Karmada job owner reference %w", err)
			}
			if err = r.Create(ctx, job); err != nil {
				return fmt.Errorf("failed to create cluster to join Karmada job %w", err)
			}
		}
	}
	return nil
}

func getJoinClusterToKarmadaJobName(clusterName string) string {
	return fmt.Sprintf("%s%s", karmadaJoinJobPrefix, clusterName)
}

// Filter out clusters not yet added to Karmada
func (r *TenantClusterReconciler) filterNotJoinedClusters(readyClusters []string, joinedClusters []string) []string {
	return filterItemsNotInSecondList(readyClusters, joinedClusters)
}

// Filter clusters that need to be adopted
func (r *TenantClusterReconciler) filterClustersNeedingAdoption(memberClusters []string, processedClusters []string) []string {
	return filterItemsNotInSecondList(memberClusters, processedClusters)
}

// A general filtering function that returns items that exist in the first list but not in the second list
func filterItemsNotInSecondList(firstList []string, secondList []string) []string {
	var result []string

	itemsMap := make(map[string]bool)
	for _, name := range secondList {
		itemsMap[name] = true
	}

	for _, cluster := range firstList {
		if !itemsMap[cluster] {
			result = append(result, cluster)
		}
	}

	return result
}

func (r *TenantClusterReconciler) buildSuperClusterdDeployment(cluster *clusterv1alpha1.TenantCluster) *appsv1.Deployment {
	labels := map[string]string{ManagedLabel: clusterv1alpha1.GroupVersion.Group}
	name := fmt.Sprintf("%s-superclusterd", cluster.Name)
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:    name,
				Image:   cluster.Spec.ControlPlane.SuperClusterd.Image,
				Command: cluster.Spec.ControlPlane.SuperClusterd.Command,
			},
		},
	}
	replicas := cluster.Spec.ControlPlane.SuperClusterd.Replicas
	deploymentSpec := appsv1.DeploymentSpec{
		Replicas: &replicas,
		Selector: &metav1.LabelSelector{
			MatchLabels: labels,
		},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: labels,
			},
			Spec: podSpec,
		},
	}

	superClusterd := &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: deploymentAPIVersion,
			Kind:       deploymentKind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			Labels:    labels,
		},
		Spec: deploymentSpec,
	}
	return superClusterd
}

func (r *TenantClusterReconciler) buildManagedClusters(cluster *clusterv1alpha1.TenantCluster) (memberClusters []*clusterv1alpha1.TenantManagedCluster) {
	var nodeList []string
	for _, nodeExpr := range cluster.Spec.NodeList {
		nodes := ExpandNode(nodeExpr)
		nodeList = append(nodeList, nodes...)
	}

	log.Log.Info("The node list is expanded", "num nodes", len(nodeList))
	numNodes := len(nodeList)

	managedClusterConfig := config.GetManagedClusterConfig()
	clusterSize := managedClusterConfig.ClusterSize
	minClusterSize := managedClusterConfig.MinClusterSize
	numManagementNodes := managedClusterConfig.NumManagementNodes

	// simple scheduling: all computing nodes are partitioned into predefined cluster size
	numClusters := numNodes / clusterSize
	remainNodes := numNodes % clusterSize

	if numClusters <= 0 {
		return
	}

	// construct cluster with full single cluster size
	var clusters []*clusterv1alpha1.TenantManagedCluster
	clusterID := 1
	for i := 0; i < numClusters-1; i++ {
		index := i * clusterSize
		master := nodeList[index : index+numManagementNodes]
		member := nodeList[index+numManagementNodes : (i+1)*clusterSize]

		managedCluster := r.constructManagedCluster(cluster, clusterID, master, member)
		clusterID++
		clusters = append(clusters, managedCluster)
	}

	// create left 1 or 2 clusters
	iLastFullCluster := numClusters - 1
	index := iLastFullCluster * clusterSize
	if remainNodes <= minClusterSize {
		// merge left nodes into the last cluster
		master := nodeList[index : index+numManagementNodes]
		member := nodeList[index+numManagementNodes:]
		managedCluster := r.constructManagedCluster(cluster, clusterID, master, member)
		clusterID++
		clusters = append(clusters, managedCluster)
	} else {
		nextIndex := (iLastFullCluster + 1) * clusterSize

		master := nodeList[index : index+numManagementNodes]
		member := nodeList[index+numManagementNodes : nextIndex]
		managedCluster := r.constructManagedCluster(cluster, clusterID, master, member)
		clusterID++
		clusters = append(clusters, managedCluster)

		master = nodeList[nextIndex : nextIndex+numManagementNodes]
		member = nodeList[nextIndex+numManagementNodes:]
		managedCluster = r.constructManagedCluster(cluster, clusterID, master, member)
		clusters = append(clusters, managedCluster)
	}

	memberClusters = clusters

	log.Log.Info("The managed clusters was constructed", "num clusters", len(memberClusters))
	return
}

func (r *TenantClusterReconciler) constructManagedCluster(cluster *clusterv1alpha1.TenantCluster, index int, master []string, member []string) *clusterv1alpha1.TenantManagedCluster {
	managedCluster := &clusterv1alpha1.TenantManagedCluster{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "clusters-computing-operator.openfuyao.com/v1alpha1",
			Kind:       "TenantManagedCluster",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cluster.Name + "-member-" + strconv.Itoa(index),
			Namespace: cluster.Namespace,
			Labels: map[string]string{
				ManagedLabel:           ManagedLabelValue,
				TenantClusterLabelName: cluster.Name,
			},
		},
		Spec: clusterv1alpha1.TenantManagedClusterSpec{
			Mode:       "managed",
			MasterList: master,
			WorkerList: member,
		},
	}

	return managedCluster
}

// SetupWithManager sets up the controller with the Manager.
func (r *TenantClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	scheme := mgr.GetScheme()
	karmadav1alpha1.AddToScheme(scheme)
	err := ctrl.NewControllerManagedBy(mgr).
		For(&clusterv1alpha1.TenantCluster{}).
		Named("tenantcluster").
		WithEventFilter(predicate.GenerationChangedPredicate{}).
		Complete(r)

	if err != nil {
		return err
	}

	return nil
}

func (r *TenantClusterReconciler) reconcileKubeconfigSecret(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	secret, err := r.buildKubeconfigSecret(tc)
	if err != nil {
		return fmt.Errorf("failed to build Kubeconfig Secret: %w", err)
	}
	if err = controllerutil.SetControllerReference(tc, secret, r.Scheme); err != nil {
		return fmt.Errorf("failed to set kubeconfig owner reference %w", err)
	}
	err = r.Create(ctx, secret)
	if err != nil {
		if !errors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create kubeconfig %w", err)
		}
		return r.Update(ctx, secret)
	}
	return nil
}

func (r *TenantClusterReconciler) buildKubeconfigSecret(tc *clusterv1alpha1.TenantCluster) (*corev1.Secret, error) {
	kubeconfig, err := generateKubeconfig(tc.Namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to generate kubeconfig: %w", err)
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      tenantClusterAdminConfigName,
			Namespace: tc.Namespace,
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte{
			tenantClusterKubeconfigKey: kubeconfig,
		},
	}
	return secret, nil
}

func generateKubeconfig(namespace string) ([]byte, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		log.Log.Error(err, "Cannot get InClusterConfig")
		return nil, err
	}

	caData := config.CAData
	if len(caData) == 0 && config.CAFile != "" {
		var err error
		caData, err = os.ReadFile(config.CAFile)
		if err != nil {
			return nil, fmt.Errorf("read CA file failed: %w", err)
		}
	}

	kubeconfig := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"default-cluster": {
				Server:                   config.Host,
				CertificateAuthorityData: caData,
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"default-user": {
				Token:                 config.BearerToken,
				ClientCertificateData: config.CertData,
				ClientKeyData:         config.KeyData,
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"default-context": {
				Cluster:   "default-cluster",
				AuthInfo:  "default-user",
				Namespace: namespace,
			},
		},
		CurrentContext: "default-context",
	}

	return clientcmd.Write(kubeconfig)
}

func (r *TenantClusterReconciler) buildJoinClusterToKarmadaJob(tenantCluster *clusterv1alpha1.TenantCluster, clusterName string, karmadaName string) *batchv1.Job {
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getJoinClusterToKarmadaJobName(clusterName),
			Namespace: tenantCluster.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by-cr": "clusters-computing-operator",
			},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "join-cluster-to-karmada-container",
							Image:           tenantCluster.Spec.ManagedClusters.Installer.Image,
							ImagePullPolicy: corev1.PullIfNotPresent,
							Command: []string{
								"/bin/sh",
								"-c",
								fmt.Sprintf(`karmadactl join %s --kubeconfig=/etc/kubernetes/kubeconfig --cluster-kubeconfig=/etc/cluster/kubeconfig`, clusterName),
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "karmada-kubeconfig",
									MountPath: "/etc/kubernetes",
									ReadOnly:  true,
								},
								{
									Name:      "cluster-kubeconfig",
									MountPath: "/etc/cluster",
									ReadOnly:  true,
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "karmada-kubeconfig",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName: fmt.Sprintf("%s%s", karmadaName, karmadaConfigSuffix),
								},
							},
						},
						{
							Name: "cluster-kubeconfig",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName: fmt.Sprintf("%s%s", clusterName, managedClusterConfigSuffix),
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
			BackoffLimit: new(int32),
		},
	}
	return job
}

// adoptSingleCluster adopts a single cluster
func (r *TenantClusterReconciler) adoptCluster(ctx context.Context, tc *clusterv1alpha1.TenantCluster, clusterName string) error {
	logger := log.FromContext(ctx)

	// collect detailed cluster metrics
	subCluster, err := karmadaClient.ClusterV1alpha1().Clusters().Get(ctx, clusterName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get Karmada sub cluster, %s: %w", clusterName, err)
	}

	secret, err := karmadaK8sClient.CoreV1().Secrets("karmada-cluster").Get(ctx, subCluster.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get sub cluster secret, %s: %w", clusterName, err)
	}
	var conf *rest.Config
	if kubeconfigData, ok := secret.Data["kubeconfig"]; ok {
		// use secret.Data["kubeconfig"] if it exists
		conf, err = clientcmd.RESTConfigFromKubeConfig(kubeconfigData)
		if err != nil {
			return fmt.Errorf("failed to get sub cluster rest config from kubeconfig data, %s: %w", clusterName, err)
		}
	} else {
		// construct RESTConfig
		conf = &rest.Config{
			Host:        subCluster.Spec.APIEndpoint, // it is actual Kubernetes API server address
			BearerToken: string(secret.Data["token"]),
			TLSClientConfig: rest.TLSClientConfig{
				CAData: secret.Data["caBundle"],
			},
		}
	}

	clientSet, err := kubernetes.NewForConfig(conf)
	if err != nil {
		return fmt.Errorf("failed to get kubernetes client from rest config, %s: %w", clusterName, err)
	}

	if !isClusterReady(subCluster) {
		return fmt.Errorf("cluster %s is not ready", clusterName)
	}

	nodes, err := clientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to get nodes from kubernetes client, %s: %w", clusterName, err)
	}
	var masterList []string
	var workerList []string
	for _, node := range nodes.Items {
		identifier := getNodeIdentifier(node)
		if _, ok := node.Labels["node-role.kubernetes.io/control-plane"]; ok {
			// append it into masterList if the label "node-role.kubernetes.io/control-plane" exists
			masterList = append(masterList, identifier)
		} else {
			// append it into workerList if doesn't exists
			workerList = append(workerList, identifier)
		}
	}
	tenantManagedCluster := &clusterv1alpha1.TenantManagedCluster{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "clusters-computing-operator.openfuyao.com/v1alpha1",
			Kind:       "TenantManagedCluster",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      subCluster.Name,
			Namespace: tc.Namespace,
		},
		Spec: clusterv1alpha1.TenantManagedClusterSpec{
			Mode:       UnManagedMode,
			MasterList: masterList,
			WorkerList: workerList,
		},
	}

	if err = controllerutil.SetControllerReference(tc, tenantManagedCluster, r.Scheme); err != nil {
		return fmt.Errorf("failed to set TenantManagedCluster owner reference,%s, %w", tenantManagedCluster.Name, err)
	}

	// create tenantManagedCluster
	if err = r.Create(ctx, tenantManagedCluster); err != nil && !errors.IsAlreadyExists(err) {
		return fmt.Errorf("failed to create TenantManagedCluster,%s, %w", tenantManagedCluster.Name, err)
	}

	logger.Info("Successfully adopted cluster", "cluster", clusterName)
	return nil
}

func createClusterClient(ctx context.Context, clusterName string) (*kubernetes.Clientset, error) {
	// Get detailed cluster information
	subCluster, err := karmadaClient.ClusterV1alpha1().Clusters().Get(ctx, clusterName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get cluster %s: %w", clusterName, err)
	}

	secret, err := karmadaK8sClient.CoreV1().Secrets("karmada-cluster").Get(ctx, subCluster.Name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get secret %s: %w", clusterName, err)
	}
	var conf *rest.Config
	if kubeconfigData, ok := secret.Data["kubeconfig"]; ok {
		// If kubeconfig exists, use it directly
		conf, err = clientcmd.RESTConfigFromKubeConfig(kubeconfigData)
		if err != nil {
			return nil, fmt.Errorf("failed to get rest config from kubeconfig data, %s: %w", clusterName, err)
		}
	} else {
		// Build RESTConfig
		conf = &rest.Config{
			Host:        subCluster.Spec.APIEndpoint, // 替换为实际的 Kubernetes API 服务器地址
			BearerToken: string(secret.Data["token"]),
			TLSClientConfig: rest.TLSClientConfig{
				CAData: secret.Data["caBundle"],
			},
		}
	}

	clientSet, err := kubernetes.NewForConfig(conf)
	if err != nil {
		return nil, fmt.Errorf("failed to get kubernetes client from rest config, %s: %w", clusterName, err)
	}
	return clientSet, nil
}

// Get the identifier for the node (IP or name), if IP is available, use IP; otherwise, use the name
func getNodeIdentifier(node corev1.Node) string {
	for _, addr := range node.Status.Addresses {
		if addr.Type == "InternalIP" {
			return addr.Address
		}
	}
	return node.Name
}

// Check if the Karmada sub-cluster is successfully
func isClusterReady(cluster *karmadaclusterv1alpha1.Cluster) bool {
	for _, condition := range cluster.Status.Conditions {
		if condition.Type == karmadaclusterv1alpha1.ClusterConditionReady && condition.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

// Get the list of newly added clusters
func (r *TenantClusterReconciler) getMemberClusters(ctx context.Context) ([]string, error) {
	memberClusters, err := karmadaClient.ClusterV1alpha1().Clusters().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list member clusters: %w", err)
	}
	var clusterNames []string
	for _, cluster := range memberClusters.Items {
		clusterNames = append(clusterNames, cluster.Name)
	}
	return clusterNames, nil
}

var karmadaClient *karmadaclientset.Clientset
var karmadaK8sClient *kubernetes.Clientset

func (r *TenantClusterReconciler) getKarmadaClient(ctx context.Context, namespace string, tc *clusterv1alpha1.TenantCluster) error {
	secret := &corev1.Secret{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: getKarmadaName(tc) + karmadaConfigSuffix}, secret); err != nil {
		return fmt.Errorf("failed to get Karmada secret: %w", err)
	}

	kubeconfig, exist := secret.Data["kubeconfig"]
	if !exist {
		return fmt.Errorf("failed to find kubeconfig")
	}

	config, err := clientcmd.NewClientConfigFromBytes(kubeconfig)
	if err != nil {
		return fmt.Errorf("failed to create Karmada config: %w", err)
	}

	restConfig, err := config.ClientConfig()
	if err != nil {
		return fmt.Errorf("failed to create Karmada restConfig: %w", err)
	}

	karmadaClient, err = karmadaclientset.NewForConfig(restConfig)
	if err != nil {
		return fmt.Errorf("failed to create Karmada client: %w", err)
	}

	karmadaK8sClient, err = kubernetes.NewForConfig(restConfig)

	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return nil
}

func (r *TenantClusterReconciler) cleanupKarmada(ctx context.Context, tc *clusterv1alpha1.TenantCluster) error {
	karmada := &karmadav1alpha1.Karmada{}

	karmada.Name = getKarmadaName(tc)

	if err := r.Get(ctx, types.NamespacedName{Name: karmada.Name, Namespace: tc.Namespace}, karmada); err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to get Karmada: %w", err)
		}
	}

	if err := r.Delete(ctx, karmada); err != nil {
		return fmt.Errorf("failed to delete Karmada: %w", err)
	}

	return nil
}
