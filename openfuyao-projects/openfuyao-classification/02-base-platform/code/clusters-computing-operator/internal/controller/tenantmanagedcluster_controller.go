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
	"sigs.k8s.io/controller-runtime/pkg/controller"

	"gopkg.in/yaml.v2"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"openfuyao.com/clusters-computing-operator/internal/config"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	policyv1alpha1 "github.com/karmada-io/karmada/pkg/apis/policy/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	clusterv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
)

// TenantManagedClusterReconciler reconciles a TenantManagedCluster object
type TenantManagedClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

type TenantManagedClusterStatus struct {
	ClusterInstallation bool
	JoinedToKarmada     bool
	ClusterDReady       bool
	AscendReady         bool
	FinalState          bool
}

const (
	createManagedClusterJobConfigFile = "config/app/job/fuyao-create-cluster.yaml"
	// admin config means accessing the cluster
	managedClusterConfigYamlSuffix = ".yaml"
	tenantClusterAdminConfigName   = "tenant-cluster-admin-config"
	createManagedClusterConfigName = "create-cluster-config"

	ClusterInstallationCondition string = "ClusterInstallation"
	JoinedKarmadaCondition       string = "JoinedKarmada"
	ClusterDCondition            string = "ClusterD"
	AscendCondition              string = "Ascend"
	ClusterCondition             string = "Cluster"

	ClusterInstallationSuccessMsg    string = "Cluster installation successful"
	ClusterInstallationFailedMsg     string = "Cluster installation failed"
	ClusterInstallationInProgressMsg string = "Cluster is being installed"

	ClusterJoinedToKarmadaSuccessMsg    string = "The managed cluster has been joined to Karmada"
	ClusterJoinedToKarmadaFailedMsg     string = "Failed to join the managed cluster to Karmada"
	ClusterJoinedToKarmadaInProgressMsg string = "The managed cluster is being joined to Karmada"

	AscendSuccessMsg    string = "Ascend Job executed successfully"
	AscendFailedMsg     string = "Ascend Job execution failed"
	AscendInProgressMsg string = "Ascend Job is in progress"

	TenantManagedClusterFinalizer = "tenantmanagedcluster.finalizers.openfuyao.com"
)

// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantmanagedclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantmanagedclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=clusters-computing-operator.openfuyao.com,resources=tenantmanagedclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the TenantManagedCluster object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *TenantManagedClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Reconciling TenantManagedCluster")

	tenantManagedCluster := &clusterv1alpha1.TenantManagedCluster{}
	if err := r.Get(ctx, req.NamespacedName, tenantManagedCluster); err != nil {
		logger.Error(err, "Cannot fetch TenantManagedCluster")
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Check if the resource is marked for deletion
	if !tenantManagedCluster.DeletionTimestamp.IsZero() {
		logger.Info("TenantManagedCluster is being deleted, handling deletion", "name", tenantManagedCluster.Name)
		if err := r.handleDeletion(ctx, tenantManagedCluster); err != nil {
			logger.Error(err, "Failed to handle deletion of TenantManagedCluster")
			return ctrl.Result{}, err
		}
		logger.Info("TenantManagedCluster deleted successfully", "name", tenantManagedCluster.Name)
		return ctrl.Result{}, nil
	}

	// The resource is not marked for deletion, ensure to add our finalizer
	if !controllerutil.ContainsFinalizer(tenantManagedCluster, TenantManagedClusterFinalizer) {
		controllerutil.AddFinalizer(tenantManagedCluster, TenantManagedClusterFinalizer)
		if err := r.Update(ctx, tenantManagedCluster); err != nil {
			return ctrl.Result{Requeue: true}, err
		}
		logger.Info("Added finalizer", "name", tenantManagedCluster.Name)
	}

	tenantCluster := &clusterv1alpha1.TenantCluster{}
	if err := r.getOwnerTenantCluster(ctx, tenantCluster, tenantManagedCluster); err != nil {
		logger.Error(err, "Failed to get owner TenantCluster")
		return ctrl.Result{}, err
	}

	// Build the current cluster's status to prevent calling related functions repeatedly
	clusterStatus := r.buildTenantManagedClusterStatus(tenantManagedCluster)

	if tenantManagedCluster.Spec.Mode == UnManagedMode {
		return r.reconcileUnmanagedMode(ctx, tenantManagedCluster, tenantCluster, clusterStatus)
	}
	return r.reconcileManagedMode(ctx, tenantManagedCluster, tenantCluster, clusterStatus)
}

func (r *TenantManagedClusterReconciler) handleDeletion(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster) error {
	logger := log.FromContext(ctx)

	if controllerutil.ContainsFinalizer(tmc, TenantManagedClusterFinalizer) {
		// Clean up managed cluster's kubeconfig secret
		if err := r.cleanupClusterKubeconfigSecret(ctx, tmc); err != nil {
			return err
		}

		// Clean up managed cluster's ClusterD
		if err := r.cleanupClusterdDeployment(ctx, tmc); err != nil {
			return err
		}

		// Clean-up completed, remove finalizer
		controllerutil.RemoveFinalizer(tmc, TenantManagedClusterFinalizer)
		if err := r.Update(ctx, tmc); err != nil {
			return fmt.Errorf("failed to remove finalizer: %w", err)
		}
		logger.Info("Successfully removed finalizer", "name", tmc.Name)
	}

	return nil
}

func (r *TenantManagedClusterReconciler) reconcileUnmanagedMode(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster, tc *clusterv1alpha1.TenantCluster,
	clusterStatus *TenantManagedClusterStatus) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Unmanaged mode reaching here indicates that the cluster is ready and has been joined to Karmada
	updateTenantManagedClusterConditions(tmc, ClusterInstallationCondition, metav1.ConditionTrue, ReasonReady, ClusterInstallationSuccessMsg)
	updateTenantManagedClusterConditions(tmc, JoinedKarmadaCondition, metav1.ConditionTrue, ReasonReady, ClusterJoinedToKarmadaSuccessMsg)
	updateTenantManagedClusterConditions(tmc, AscendCondition, metav1.ConditionTrue, ReasonReady, AscendSuccessMsg)

	if err := r.reconcileClusterDDeployment(ctx, tc, tmc); err != nil {
		logger.Error(err, "Failed to create clusterD setup")
		return ctrl.Result{}, err
	}

	if err := r.updateClusterStatesAndNodeSummary(ctx, tmc, clusterStatus); err != nil {
		logger.Error(err, "Update tenantManagedCluster status error")
		return ctrl.Result{}, err
	}

	// Final state check
	if clusterStatus.FinalState {
		logger.Info("In unmanaged mode,tenantManagedCluster complete, all components are ready")
		return ctrl.Result{}, nil
	}
	return ctrl.Result{RequeueAfter: DefaultReconcileInterval}, nil
}

func (r *TenantManagedClusterReconciler) reconcileManagedMode(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster,
	tc *clusterv1alpha1.TenantCluster, clusterStatus *TenantManagedClusterStatus) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Install managed cluster
	if err := r.reconcileManagedClusterSetupJob(ctx, tc, tmc); err != nil {
		logger.Error(err, "Failed to create cluster setup job")
		return ctrl.Result{}, err
	}

	// Check if the cluster has joined Karmada and update the status
	if err := r.updateClusterJoinedToKarmada(ctx, tc, tmc); err != nil {
		logger.Error(err, "Failed to update cluster joined to Karmada")
		return ctrl.Result{}, err
	}

	if clusterStatus.JoinedToKarmada {
		//  Deploy ClusterD Deployment for the managed cluster
		if err := r.reconcileClusterDDeployment(ctx, tc, tmc); err != nil {
			logger.Error(err, "Failed to create clusterD setup")
			return ctrl.Result{}, err
		}

		// Deploy Ascend Job for the managed cluster
		if err := r.reconcileAscendJob(ctx, tc, tmc); err != nil {
			logger.Error(err, "Failed to create Ascend setup job")
			return ctrl.Result{}, err
		}
	}

	if err := r.updateClusterStatesAndNodeSummary(ctx, tmc, clusterStatus); err != nil {
		logger.Error(err, "Update tenantManagedCluster status error")
		return ctrl.Result{}, err
	}

	// Final state check
	if clusterStatus.FinalState {
		logger.Info("In managed mode,tenantManagedCluster complete, all components are ready")
		return ctrl.Result{}, nil
	}

	return ctrl.Result{RequeueAfter: DefaultReconcileInterval}, nil
}

func (r *TenantManagedClusterReconciler) buildTenantManagedClusterStatus(tmc *clusterv1alpha1.TenantManagedCluster) *TenantManagedClusterStatus {
	status := &TenantManagedClusterStatus{}

	status.ClusterInstallation = r.isTenantManagedComponentReady(tmc, ClusterInstallationCondition)
	status.JoinedToKarmada = r.isTenantManagedComponentReady(tmc, JoinedKarmadaCondition)
	status.ClusterDReady = r.isTenantManagedComponentReady(tmc, ClusterDCondition)
	status.AscendReady = r.isTenantManagedComponentReady(tmc, AscendCondition)

	status.FinalState = status.ClusterDReady && status.AscendReady

	return status
}

func (r *TenantManagedClusterReconciler) getOwnerTenantCluster(ctx context.Context, tc *clusterv1alpha1.TenantCluster, tmc *clusterv1alpha1.TenantManagedCluster) error {
	ownerReferences := tmc.GetOwnerReferences()
	for _, owner := range ownerReferences {
		if owner.Kind == "TenantCluster" {
			if err := r.Get(ctx, types.NamespacedName{
				Name:      owner.Name,
				Namespace: tmc.Namespace,
			}, tc); err != nil {
				return fmt.Errorf("tenantCluster %s not found: %w", owner.Name, err)
			}
			return nil
		}
	}
	return fmt.Errorf("no TenantCluster owner found for %s", tmc.Name)
}

func (r *TenantManagedClusterReconciler) reconcileManagedClusterSetupJob(ctx context.Context, tc *clusterv1alpha1.TenantCluster, tmc *clusterv1alpha1.TenantManagedCluster) error {
	if len(tmc.Spec.MasterList) == 0 {
		return fmt.Errorf("no masters specified for tmc %s", tmc.Name)
	}
	job := &batchv1.Job{}
	nn := types.NamespacedName{
		Name:      getManagedClusterJobName(tmc.Name),
		Namespace: tmc.Namespace,
	}
	err := r.Get(ctx, nn, job)
	// Find the Job, update the status
	if err == nil {
		updateConditionBasedOnJobStatus(tmc, job, ClusterInstallationCondition,
			ClusterInstallationSuccessMsg,
			ClusterInstallationFailedMsg,
			ClusterInstallationInProgressMsg)
		return nil
	}

	// Return immediately on other errors
	if !errors.IsNotFound(err) {
		return fmt.Errorf("failed to get managed cluster setup job: %w", err)
	}

	// If no Job is found, create a job to install the managed cluster
	configMap, err := r.createClusterTemporaryConfigMap(ctx, tmc)
	if err != nil {
		return fmt.Errorf("failed to create temporary ConfigMap: %w", err)
	}
	job, err = r.buildManagedClusterJob(configMap, tc, tmc)
	if err != nil {
		return fmt.Errorf("failed to build tmc job: %w", err)
	}
	if err = controllerutil.SetControllerReference(tmc, job, r.Scheme); err != nil {
		return fmt.Errorf("failed to set job owner reference: %w", err)
	}
	if err = r.Create(ctx, job); err != nil {
		return fmt.Errorf("failed to create install tmc job: %w", err)
	}
	return nil
}

func (r *TenantManagedClusterReconciler) updateClusterJoinedToKarmada(ctx context.Context, tc *clusterv1alpha1.TenantCluster, tmc *clusterv1alpha1.TenantManagedCluster) error {
	job := &batchv1.Job{}
	nn := types.NamespacedName{
		Name:      getJoinClusterToKarmadaJobName(tmc.Name),
		Namespace: tc.Namespace,
	}
	err := r.Get(ctx, nn, job)
	// No job found means the managed cluster or Karmada has not been successfully installed yet, and the JoinJob has not been created, return nil;
	// return error for other errors
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to get cluster joined to Karmada job: %w", err)
		}
		return nil
	}
	updateConditionBasedOnJobStatus(tmc, job, JoinedKarmadaCondition,
		ClusterJoinedToKarmadaSuccessMsg,
		ClusterJoinedToKarmadaFailedMsg,
		ClusterJoinedToKarmadaInProgressMsg)
	return nil
}

func (r *TenantManagedClusterReconciler) reconcileClusterDDeployment(ctx context.Context, tc *clusterv1alpha1.TenantCluster, tmc *clusterv1alpha1.TenantManagedCluster) error {
	if karmadaClient == nil || karmadaK8sClient == nil {
		return nil
	}
	deployment := r.buildClusterdDeployment(tmc, tc)
	// Create PropagationPolicy
	ppName := deployment.Name + "-propagation"
	if _, err := karmadaClient.PolicyV1alpha1().PropagationPolicies(deployment.Namespace).Get(ctx, ppName, metav1.GetOptions{}); err != nil {
		if errors.IsNotFound(err) {
			pp := r.newPropagationPolicy(deployment, tmc.Name)
			if _, err = karmadaClient.PolicyV1alpha1().PropagationPolicies(deployment.Namespace).Create(ctx, pp, metav1.CreateOptions{}); err != nil {
				return fmt.Errorf("failed to create PropagationPolicies: %w", err)
			}
		} else {
			return fmt.Errorf("failed to get PropagationPolicies: %w", err)
		}
	}

	// Check if the Deployment exists
	// Submit the Job to the Karmada API Server using the Kubernetes client
	existDeployment, err := karmadaK8sClient.AppsV1().Deployments(deployment.Namespace).Get(ctx, deployment.Name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			if _, err = karmadaK8sClient.AppsV1().Deployments(deployment.Namespace).Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
				return fmt.Errorf("failed to create clusterD Deployment: %w", err)
			}
		} else {
			return fmt.Errorf("failed to get clusterD Deployment: %w", err)
		}
	}
	// Find the Job, update the status
	if existDeployment.Status.AvailableReplicas >= *deployment.Spec.Replicas {
		updateTenantManagedClusterConditions(tmc, ClusterDCondition, metav1.ConditionTrue, ReasonReady, "ClusterD deployment completed")
	} else {
		updateTenantManagedClusterConditions(tmc, ClusterDCondition, metav1.ConditionFalse, ReasonInProgress, "ClusterD is being deployed")
	}
	return nil
}

func (r *TenantManagedClusterReconciler) reconcileAscendJob(ctx context.Context, tc *clusterv1alpha1.TenantCluster, tmc *clusterv1alpha1.TenantManagedCluster) error {
	if karmadaClient == nil || karmadaK8sClient == nil {
		return nil
	}
	// construct job
	job := r.buildAscendJob(tmc, tc)

	if err := controllerutil.SetControllerReference(tmc, job, r.Scheme); err != nil {
		return fmt.Errorf("failed to set Ascend job owner reference: %w", err)
	}

	existingJob := &batchv1.Job{}

	// check if the job exists
	err := r.Get(ctx, client.ObjectKey{Name: job.Name, Namespace: job.Namespace}, existingJob)
	if err != nil {
		if errors.IsNotFound(err) {
			if err = r.Create(ctx, job); err != nil {
				return fmt.Errorf("failed to create Ascend job: %w", err)
			}
		} else {
			// other error
			return fmt.Errorf("failed to get Ascend job: %w", err)
		}
	}
	// if the job exist, update its conditions
	updateConditionBasedOnJobStatus(tmc, existingJob, AscendCondition, AscendSuccessMsg, AscendFailedMsg, AscendInProgressMsg)
	return nil
}

func updateConditionBasedOnJobStatus(tmc *clusterv1alpha1.TenantManagedCluster, job *batchv1.Job, conditionType, successMsg, failureMsg, progressMsg string) {
	switch {
	case job.Status.Succeeded > 0:
		updateTenantManagedClusterConditions(tmc, conditionType, metav1.ConditionTrue, ReasonReady, successMsg)
	case job.Spec.BackoffLimit != nil && job.Status.Failed > *job.Spec.BackoffLimit:
		updateTenantManagedClusterConditions(tmc, conditionType, metav1.ConditionFalse, ReasonFailed, failureMsg)
	default:
		updateTenantManagedClusterConditions(tmc, conditionType, metav1.ConditionFalse, ReasonInProgress, progressMsg)
	}
}

func updateTenantManagedClusterConditions(tmc *clusterv1alpha1.TenantManagedCluster, conditionType string, status metav1.ConditionStatus, reason, message string) {
	updateCondition(&tmc.Status.Conditions, conditionType, status, reason, message)
}

func (r *TenantManagedClusterReconciler) isTenantManagedComponentReady(tmc *clusterv1alpha1.TenantManagedCluster, name string) bool {
	for _, condition := range tmc.Status.Conditions {
		if condition.Type == name && condition.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

func (r *TenantManagedClusterReconciler) buildAscendJob(tmc *clusterv1alpha1.TenantManagedCluster, tc *clusterv1alpha1.TenantCluster) *batchv1.Job {
	labels := map[string]string{
		ManagedLabel:                      clusterv1alpha1.GroupVersion.Group,
		NameLabel:                         "ascend",
		VersionLabel:                      "v0.0.1",
		"app.kubernetes.io/managed-by-cr": "clusters-computing-operator",
	}
	name := fmt.Sprintf("%s-ascend", tmc.Name)

	var ttlSecond int32 = 100

	job := &batchv1.Job{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "batch/v1",
			Kind:       "Job",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: tmc.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "install-ascend",
							Image:   tc.Spec.ManagedClusters.Installer.Image,
							Command: tc.Spec.ManagedClusters.Installer.AscendInstaller.Command,
							Env: []corev1.EnvVar{
								{
									Name:  "INI_FILE",
									Value: generateAscendIniFile(tmc),
								},
								{
									Name:  "TARGET_IP",
									Value: tmc.Spec.WorkerList[0],
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "ssh-keys",
									MountPath: "/root/.ssh",
								},
							},
							ImagePullPolicy: corev1.PullIfNotPresent,
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "ssh-keys",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: "/root/.ssh",
									Type: &[]corev1.HostPathType{corev1.HostPathDirectoryOrCreate}[0],
								},
							}},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
			TTLSecondsAfterFinished: &ttlSecond,
		},
	}

	return job
}

func (r *TenantManagedClusterReconciler) newPropagationPolicy(deployment *appsv1.Deployment, clusterName string) *policyv1alpha1.PropagationPolicy {
	return &policyv1alpha1.PropagationPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deployment.Name + "-propagation",
			Namespace: deployment.Namespace,
		},
		Spec: policyv1alpha1.PropagationSpec{
			ResourceSelectors: []policyv1alpha1.ResourceSelector{
				{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       deployment.Name,
					Namespace:  deployment.Namespace,
				},
			},
			Placement: policyv1alpha1.Placement{
				ClusterAffinity: &policyv1alpha1.ClusterAffinity{
					ClusterNames: []string{clusterName},
				},
				ReplicaScheduling: &policyv1alpha1.ReplicaSchedulingStrategy{
					ReplicaSchedulingType: policyv1alpha1.ReplicaSchedulingTypeDuplicated,
				},
			},
		},
	}
}

func (r *TenantManagedClusterReconciler) buildClusterdDeployment(managedCluster *clusterv1alpha1.TenantManagedCluster, tenantCluster *clusterv1alpha1.TenantCluster) *appsv1.Deployment {
	labels := map[string]string{
		ManagedLabel:                      clusterv1alpha1.GroupVersion.Group,
		NameLabel:                         "clusterd",
		VersionLabel:                      "v0.0.1",
		"app.kubernetes.io/managed-by-cr": "clusters-computing-operator",
	}

	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-clusterd", managedCluster.Name),
			Namespace: managedCluster.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &tenantCluster.Spec.ControlPlane.Clusterd.Replicas, // 从CRD获取副本数
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "clusterd",
							Image:   tenantCluster.Spec.ControlPlane.Clusterd.Image,
							Command: tenantCluster.Spec.ControlPlane.Clusterd.Command,
							Ports: []corev1.ContainerPort{ // 添加必要的端口暴露
								{ContainerPort: 8080},
							},
						},
					},
				},
			},
		},
	}
}

// Set the final state of the entire cluster and node details
func (r *TenantManagedClusterReconciler) updateClusterStatesAndNodeSummary(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster,
	clusterStatus *TenantManagedClusterStatus) error {
	logger := log.FromContext(ctx)
	if clusterStatus.FinalState {
		updateTenantManagedClusterConditions(tmc, ClusterCondition, metav1.ConditionTrue, ReasonReady, "Cluster all components are ready")
	} else {
		updateTenantManagedClusterConditions(tmc, ClusterCondition, metav1.ConditionFalse, ReasonInProgress, "Cluster components are being installed")
	}

	// Update node details, only after joining Karmada, can node details be retrieved
	if clusterStatus.JoinedToKarmada {
		// Check if the node status has already been updated
		if tmc.Status.NodeSummary.ReadyCount == 0 && tmc.Status.NodeSummary.NotReadyCount == 0 {
			// If the node status is empty, it means it hasn't been updated yet, perform an update.
			if err := r.updateNodeSummary(ctx, tmc); err != nil {
				// Record errors but continue execution, ensuring the status update is not affected
				logger.Error(err, "Failed to update node summary")
			}
		}
	}

	return r.Status().Update(ctx, tmc)
}

func (r *TenantManagedClusterReconciler) updateNodeSummary(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster) error {
	// Create managed cluster client
	clusterClient, err := createClusterClient(ctx, tmc.Name)
	if err != nil {
		return fmt.Errorf("failed to create cluster client: %w", err)
	}

	// Get the node list
	nodes, err := clusterClient.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list nodes: %w", err)
	}
	var readList []string
	var notReadyList []string
	for _, node := range nodes.Items {
		identifier := getNodeIdentifier(node)
		if nodeReady(node) {
			readList = append(readList, identifier)
		} else {
			notReadyList = append(notReadyList, identifier)
		}
	}
	tmc.Status.NodeSummary = clusterv1alpha1.NodeSummary{
		ReadyCount:    len(readList),
		ReadyList:     readList,
		NotReadyCount: len(notReadyList),
		NotReadyList:  notReadyList,
	}
	return nil
}

func nodeReady(node corev1.Node) bool {
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady {
			if condition.Status == corev1.ConditionTrue {
				return true
			} else {
				return false
			}
		}
	}
	return false
}

// SetupWithManager sets up the controller with the Manager.
func (r *TenantManagedClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&clusterv1alpha1.TenantManagedCluster{}).
		Named("tenantmanagedcluster").
		WithEventFilter(predicate.GenerationChangedPredicate{}).
		WithOptions(controller.Options{MaxConcurrentReconciles: 10}).
		Complete(r)
}

func getManagedClusterJobName(clusterName string) string {
	return fmt.Sprintf("install-%s", clusterName)
}

func (r *TenantManagedClusterReconciler) buildManagedClusterJob(configMap *corev1.ConfigMap, tenantCluster *clusterv1alpha1.TenantCluster, tenantManagedCluster *clusterv1alpha1.TenantManagedCluster) (*batchv1.Job, error) {
	createClusterConfig, err := config.LoadConfig(createManagedClusterJobConfigFile)
	if err != nil {
		return nil, err
	}
	privileged := true
	volumes := r.buildVolumes(configMap, createClusterConfig)
	volumeMounts := r.buildVolumeMounts(createClusterConfig)

	command := tenantCluster.Spec.ManagedClusters.Installer.FuyaoInstaller.Command
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getManagedClusterJobName(tenantManagedCluster.Name),
			Namespace: tenantManagedCluster.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by-cr": "clusters-computing-operator",
			},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Affinity: &corev1.Affinity{
						NodeAffinity: &corev1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "install-managed-cluster-node",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"true"},
											},
										},
									},
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:            "install-managed-cluster-container",
							Image:           tenantCluster.Spec.ManagedClusters.Installer.Image,
							ImagePullPolicy: corev1.PullAlways,
							Command: append(command[:2],
								fmt.Sprintf(tenantCluster.Spec.ManagedClusters.Installer.FuyaoInstaller.Command[2],
									configMap.Name, tenantManagedCluster.Spec.MasterList[0], tenantManagedCluster.Name)),
							SecurityContext: &corev1.SecurityContext{
								Privileged: &privileged,
							},
							VolumeMounts: volumeMounts,
						},
					},
					Volumes:       volumes,
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
			BackoffLimit: new(int32),
		},
	}
	return job, nil
}

func (r *TenantManagedClusterReconciler) buildVolumeMounts(config *config.FuyaoCreateClusterConfig) []corev1.VolumeMount {
	volumeMounts := make([]corev1.VolumeMount, 0, len(config.VolumeMounts))
	for _, volumeMount := range config.VolumeMounts {
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      volumeMount.Name,
			MountPath: volumeMount.MountPath,
		})
	}
	return volumeMounts
}

func (r *TenantManagedClusterReconciler) buildVolumes(configMap *corev1.ConfigMap, config *config.FuyaoCreateClusterConfig) []corev1.Volume {
	volumes := make([]corev1.Volume, 0, len(config.Volumes)+2)
	for _, volume := range config.Volumes {
		volumes = append(volumes, corev1.Volume{
			Name: volume.Name,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: volume.HostPath,
					Type: &[]corev1.HostPathType{corev1.HostPathDirectory}[0],
				},
			},
		})
	}

	volumes = append(volumes, corev1.Volume{
		Name: tenantClusterAdminConfigName,
		VolumeSource: corev1.VolumeSource{
			Secret: &corev1.SecretVolumeSource{
				SecretName: tenantClusterAdminConfigName,
			},
		},
	})

	volumes = append(volumes, corev1.Volume{
		Name: createManagedClusterConfigName,
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: configMap.Name,
				},
			},
		},
	})
	return volumes
}

func generateClusterYaml(cluster *clusterv1alpha1.TenantManagedCluster) (string, error) {
	fuyaoCluster := config.FuyaoCluster{
		APIVersion: "openfuyao.io/v1beta1",
		Kind:       "cluster",
		Metadata: config.FuyaoMeta{
			Name: cluster.Name,
		},
		Spec: config.FuyaoSpec{
			Nodes: config.FuyaoNodes{
				Masters: make([]config.FuyaoNode, 0, len(cluster.Spec.MasterList)),
				Workers: make([]config.FuyaoNode, 0, len(cluster.Spec.WorkerList)),
			},
		},
	}

	for i, ip := range cluster.Spec.MasterList {
		fuyaoCluster.Spec.Nodes.Masters = append(fuyaoCluster.Spec.Nodes.Masters, createFuyaoNode("master", i, ip))
	}

	for i, ip := range cluster.Spec.WorkerList {
		fuyaoCluster.Spec.Nodes.Workers = append(fuyaoCluster.Spec.Nodes.Workers, createFuyaoNode("worker", i, ip))
	}

	yamlBytes, err := yaml.Marshal(fuyaoCluster)
	if err != nil {
		return "", err
	}

	return string(yamlBytes), nil
}

// createFuyaoNode creates a FuyaoNode configuration
func createFuyaoNode(role string, index int, ip string) config.FuyaoNode {
	return config.FuyaoNode{
		Hostname:       fmt.Sprintf("%s-%d", role, index),
		User:           "root",
		IP:             ip,
		Port:           22,
		PrivateKeyPath: "/root/.ssh/id_rsa",
		RemoveTaints:   true,
	}
}

func (r *TenantManagedClusterReconciler) createClusterTemporaryConfigMap(ctx context.Context, cluster *clusterv1alpha1.TenantManagedCluster) (*corev1.ConfigMap, error) {
	clusterYaml, err := generateClusterYaml(cluster)
	if err != nil {
		return nil, fmt.Errorf("failed to generate cluster yaml: %w", err)
	}

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s", cluster.Name),
			Namespace: cluster.Namespace,
		},
		Data: map[string]string{
			fmt.Sprintf("%s%s", cluster.Name, managedClusterConfigYamlSuffix): clusterYaml,
		},
	}

	controllerutil.SetControllerReference(cluster, configMap, r.Scheme)

	var existing corev1.ConfigMap
	err = r.Get(ctx, client.ObjectKey{Name: configMap.Name, Namespace: configMap.Namespace}, &existing)

	if err != nil && errors.IsNotFound(err) {
		return configMap, r.Create(ctx, configMap)
	} else if err != nil {
		return configMap, err
	}

	existing.Data = configMap.Data
	return configMap, r.Update(ctx, &existing)
}

func (r *TenantManagedClusterReconciler) cleanupClusterdDeployment(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster) error {
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-clusterd", tmc.Name),
			Namespace: tmc.Namespace,
		},
	}

	// If Karmada does not exist, it could be that Karmada has not been created yet
	// Or the user manually deleted Karmada, in both cases, managed cluster resources cannot be deleted, return directly, otherwise TenantManagedCluster cannot be deleted
	if karmadaK8sClient == nil {
		return nil
	}

	if err := karmadaK8sClient.AppsV1().Deployments(deployment.Namespace).Delete(ctx, deployment.Name, metav1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("failed to delete clusterd deployment: %w", err)
	}
	return nil
}

func (r *TenantManagedClusterReconciler) cleanupClusterKubeconfigSecret(ctx context.Context, tmc *clusterv1alpha1.TenantManagedCluster) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s%s", tmc.Name, managedClusterConfigSuffix),
			Namespace: tmc.Namespace,
		},
	}

	if err := r.Delete(ctx, secret); err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("failed to delete secret: %w", err)
	}

	return nil
}
