package utils

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
)

var lwsControllerInstalledBySuite bool

const (
	transientWebhookRetryTimeout  = 2 * time.Minute
	transientWebhookRetryInterval = 2 * time.Second
)

func resourceScalingGroupHelmInstallArgs() ([]string, string) {
	chart := resolveHelmChart("RSG_HELM_CHART", RSGHelmChart)
	args := []string{
		RSGHelmReleaseName,
		chart,
	}
	args = appendHelmVersionArg(args, "RSG_HELM_VERSION", RSGHelmVersion, chart)
	args = appendHelmTimeoutArg(args, RSGHelmTimeout)
	args = append(args,
		"-n", RSGTestNamespace,
		"--create-namespace",
		"--set", "elastic-scaler.enabled=false",
		"--set", "tidal.enabled=false",
		"--set", "resourcescalinggroup.enabled=true",
		"--set", "resourcescalinggroup.namespace.name="+RSGTestNamespace,
		"--set", "resourcescalinggroup.namespace.create=false",
		"--set", "resourcescalinggroup.instanceConfig.enabled=false",
		"--set", "resourcescalinggroup.webhook.failurePolicy=Fail",
	)
	return args, chart
}

// InstallResourceScalingGroupController installs only the ResourceScalingGroup controller.
func InstallResourceScalingGroupController() error {
	args, _ := resourceScalingGroupHelmInstallArgs()
	cmd := exec.Command("helm", append([]string{"install"}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm install resourcescalinggroup failed: %v\nOutput: %s", err, string(out))
	}

	GinkgoWriter.Printf("resourcescalinggroup helm install output:\n%s\n", string(out))
	return nil
}

// InstallOrUpgradeResourceScalingGroupController 幂等安装/升级 RSG 控制器（metrics-framework ES-E2E-016+ 复用）。
func InstallOrUpgradeResourceScalingGroupController() error {
	args, _ := resourceScalingGroupHelmInstallArgs()
	cmd := exec.Command("helm", append([]string{"upgrade", "--install"}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm upgrade --install resourcescalinggroup failed: %v\nOutput: %s", err, string(out))
	}

	GinkgoWriter.Printf("resourcescalinggroup helm upgrade --install output:\n%s\n", string(out))
	return nil
}

// VerifyResourceScalingGroupControllerReady waits until deployment, service and webhook configs are ready.
func VerifyResourceScalingGroupControllerReady(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("Verifying ResourceScalingGroup controller and webhook readiness...")

	Eventually(func() error {
		deployment, err := clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, RSGControllerDeploymentName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if !deploymentReady(deployment) {
			return fmt.Errorf("deployment %s/%s is not ready", RSGTestNamespace, RSGControllerDeploymentName)
		}

		if _, err := clientset.CoreV1().Services(RSGTestNamespace).Get(ctx, RSGWebhookServiceName, metav1.GetOptions{}); err != nil {
			return err
		}
		if !webhookServiceReady(clientset, ctx) {
			return fmt.Errorf("webhook service %s/%s has no ready endpoints", RSGTestNamespace, RSGWebhookServiceName)
		}
		if _, err := clientset.AdmissionregistrationV1().ValidatingWebhookConfigurations().Get(ctx, RSGValidatingWebhookName, metav1.GetOptions{}); err != nil {
			return err
		}
		if _, err := clientset.AdmissionregistrationV1().MutatingWebhookConfigurations().Get(ctx, RSGMutatingWebhookName, metav1.GetOptions{}); err != nil {
			return err
		}
		return nil
	}, 5*time.Minute, 10*time.Second).Should(Succeed(),
		"ResourceScalingGroup controller and webhook should be ready within timeout")
}

func deploymentReady(deployment *appsv1.Deployment) bool {
	return deployment.Status.ReadyReplicas >= 1
}

func webhookServiceReady(clientset *kubernetes.Clientset, ctx context.Context) bool {
	endpointSlices, err := clientset.DiscoveryV1().EndpointSlices(RSGTestNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: "kubernetes.io/service-name=" + RSGWebhookServiceName,
	})
	if err != nil {
		return false
	}
	for _, slice := range endpointSlices.Items {
		if slice.AddressType != discoveryv1.AddressTypeIPv4 && slice.AddressType != discoveryv1.AddressTypeIPv6 && slice.AddressType != discoveryv1.AddressTypeFQDN {
			continue
		}
		for _, endpoint := range slice.Endpoints {
			if endpoint.Conditions.Ready == nil || *endpoint.Conditions.Ready {
				if len(endpoint.Addresses) > 0 {
					return true
				}
			}
		}
	}
	return false
}

// EnsureLeaderWorkerSetController makes LWS prerequisites explicit for real e2e runs.
func EnsureLeaderWorkerSetController(clientset *kubernetes.Clientset, ctx context.Context) error {
	if err := ConfigureLeaderWorkerSetAPI(clientset); err == nil && hasReadyLeaderWorkerSetController(clientset, ctx) {
		GinkgoWriter.Printf("LeaderWorkerSet API available at %s and controller already ready, skip install\n", LeaderWorkerSetAPIVersion)
		return nil
	}

	if err := EnsureNamespace(clientset, ctx, RSGLWSNamespace); err != nil {
		return err
	}
	if err := HelmUninstall(RSGLWSHelmReleaseName, RSGLWSNamespace); err != nil {
		return err
	}
	if err := cleanupLeaderWorkerSetInstallArtifacts(clientset, ctx); err != nil {
		return err
	}
	if err := EnsureNamespace(clientset, ctx, RSGLWSNamespace); err != nil {
		return err
	}
	if err := InstallLeaderWorkerSetController(); err != nil {
		return err
	}
	lwsControllerInstalledBySuite = true
	VerifyHelmReleaseStatus(RSGLWSHelmReleaseName, RSGLWSNamespace)
	VerifyLeaderWorkerSetControllerReady(clientset, ctx)
	return ConfigureLeaderWorkerSetAPI(clientset)
}

func cleanupLeaderWorkerSetInstallArtifacts(clientset *kubernetes.Clientset, ctx context.Context) error {
	if err := DeleteNamespaceIfExists(clientset, ctx, RSGLWSNamespace); err != nil {
		return err
	}
	if err := deleteLWSMutatingWebhookIfExists(clientset, ctx, RSGLWSMutatingWebhook); err != nil {
		return err
	}
	if err := deleteLWSValidatingWebhookIfExists(clientset, ctx, RSGLWSValidatingWebhook); err != nil {
		return err
	}
	return nil
}

func deleteLWSMutatingWebhookIfExists(clientset *kubernetes.Clientset, ctx context.Context, name string) error {
	err := clientset.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil {
		GinkgoWriter.Printf("已删除 LWS MutatingWebhookConfiguration %s\n", name)
	}
	return nil
}

func deleteLWSValidatingWebhookIfExists(clientset *kubernetes.Clientset, ctx context.Context, name string) error {
	err := clientset.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil {
		GinkgoWriter.Printf("已删除 LWS ValidatingWebhookConfiguration %s\n", name)
	}
	return nil
}

// CleanupLeaderWorkerSetController removes the controller release only when installed by this suite.
func CleanupLeaderWorkerSetController(clientset *kubernetes.Clientset, ctx context.Context) error {
	if !lwsControllerInstalledBySuite {
		return nil
	}
	if err := HelmUninstall(RSGLWSHelmReleaseName, RSGLWSNamespace); err != nil {
		return err
	}
	return DeleteNamespaceIfExists(clientset, ctx, RSGLWSNamespace)
}

// ConfigureLeaderWorkerSetAPI picks the served LWS group/version exposed by the cluster.
func ConfigureLeaderWorkerSetAPI(clientset *kubernetes.Clientset) error {
	candidates := []schemaCandidate{
		{
			gvr: schema.GroupVersionResource{
				Group:    "leaderworkerset.x-k8s.io",
				Version:  "v1",
				Resource: "leaderworkersets",
			},
			apiVersion: "leaderworkerset.x-k8s.io/v1",
		},
		{
			gvr:        LeaderWorkerSetFallbackGVR,
			apiVersion: "scheduling.k8s.io/v1alpha1",
		},
	}
	for _, candidate := range candidates {
		if _, err := clientset.Discovery().ServerResourcesForGroupVersion(candidate.apiVersion); err == nil {
			LeaderWorkerSetGVR = candidate.gvr
			LeaderWorkerSetAPIVersion = candidate.apiVersion
			GinkgoWriter.Printf("Using LeaderWorkerSet apiVersion %s\n", LeaderWorkerSetAPIVersion)
			return nil
		}
	}
	return fmt.Errorf("no served LeaderWorkerSet apiVersion detected, tried %s and %s", candidates[0].apiVersion, candidates[1].apiVersion)
}

type schemaCandidate struct {
	gvr        schema.GroupVersionResource
	apiVersion string
}

// InstallLeaderWorkerSetController installs the upstream LWS controller, version can be overridden by env.
func InstallLeaderWorkerSetController() error {
	chart := resolveHelmChart("RSG_LWS_HELM_CHART", RSGLWSHelmChart)
	args := []string{
		"install",
		RSGLWSHelmReleaseName,
		chart,
		"-n", RSGLWSNamespace,
		"--create-namespace",
		"--set-string", "resources.requests.cpu=500m",
	}
	args = appendHelmVersionArg(args, "RSG_LWS_HELM_VERSION", RSGLWSHelmVersion, chart)
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm install lws failed: %v\nOutput: %s", err, string(out))
	}

	GinkgoWriter.Printf("lws helm install output:\n%s\n", string(out))
	return nil
}

// VerifyLeaderWorkerSetControllerReady waits for the managed LWS release deployment and CRD api.
func VerifyLeaderWorkerSetControllerReady(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("Verifying LeaderWorkerSet controller readiness...")

	Eventually(func() error {
		if err := ConfigureLeaderWorkerSetAPI(clientset); err != nil {
			return err
		}
		deployments, err := clientset.AppsV1().Deployments(RSGLWSNamespace).List(ctx, metav1.ListOptions{
			LabelSelector: "app.kubernetes.io/instance=" + RSGLWSHelmReleaseName,
		})
		if err != nil {
			return err
		}
		if len(deployments.Items) == 0 {
			return fmt.Errorf("no LWS deployment found for release %s/%s", RSGLWSNamespace, RSGLWSHelmReleaseName)
		}
		for i := range deployments.Items {
			if deploymentReady(&deployments.Items[i]) {
				return nil
			}
		}
		return fmt.Errorf("no ready LWS deployment found in namespace %s", RSGLWSNamespace)
	}, 5*time.Minute, 10*time.Second).Should(Succeed(),
		"LeaderWorkerSet controller should be ready within timeout")
}

func hasReadyLeaderWorkerSetController(clientset *kubernetes.Clientset, ctx context.Context) bool {
	deployments, err := clientset.AppsV1().Deployments("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return false
	}
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		if !deploymentReady(deployment) {
			continue
		}
		if looksLikeLeaderWorkerSetController(deployment) {
			return true
		}
	}
	return false
}

func looksLikeLeaderWorkerSetController(deployment *appsv1.Deployment) bool {
	name := strings.ToLower(deployment.Name)
	if strings.Contains(name, "leaderworker") || strings.Contains(name, "leader-worker") || strings.Contains(name, "leaderworkerset") {
		return true
	}
	if strings.Contains(name, "lws") && strings.Contains(name, "controller") {
		return true
	}
	if deployment.Labels["app.kubernetes.io/name"] == "lws" {
		return true
	}
	if deployment.Labels["app.kubernetes.io/component"] == "controller-manager" &&
		(strings.Contains(strings.ToLower(deployment.Labels["app.kubernetes.io/name"]), "lws") ||
			strings.Contains(strings.ToLower(deployment.Labels["app.kubernetes.io/part-of"]), "lws")) {
		return true
	}
	if strings.Contains(strings.ToLower(deployment.Labels["app.kubernetes.io/part-of"]), "leader") &&
		strings.Contains(strings.ToLower(deployment.Labels["app.kubernetes.io/part-of"]), "worker") {
		return true
	}
	return false
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func resolveHelmChart(envKey, fallback string) string {
	return envOrDefault(envKey, fallback)
}

func appendHelmVersionArg(args []string, envKey, fallbackVersion, chart string) []string {
	version := envOrDefault(envKey, fallbackVersion)
	if version == "" || !helmChartUsesVersionArg(chart) {
		return args
	}
	return append(args, "--version", version)
}

// appendHelmTimeoutArg 显式传入 helm --timeout：helm 默认 5m，且它同时是"等待 hook 完成"的上限，
// 会在 chart 内 pre/post-install hook（生成 webhook 证书、注册 webhook）跑完前超时，
// 报 "timed out waiting for the condition"。
func appendHelmTimeoutArg(args []string, timeout string) []string {
	return append(args, "--timeout", timeout)
}

func helmChartUsesVersionArg(chart string) bool {
	normalized := strings.ToLower(strings.TrimSpace(chart))
	if normalized == "" {
		return false
	}
	if strings.HasPrefix(normalized, "oci://") || strings.HasPrefix(normalized, "http://") || strings.HasPrefix(normalized, "https://") {
		return true
	}
	if _, err := os.Stat(chart); err == nil {
		return false
	}
	if strings.HasPrefix(chart, ".\\") || strings.HasPrefix(chart, "./") || strings.HasPrefix(chart, "..\\") || strings.HasPrefix(chart, "../") {
		return false
	}
	if strings.HasPrefix(chart, `/`) {
		return false
	}
	if strings.Contains(chart, `:\`) {
		return false
	}
	if strings.HasSuffix(normalized, ".tgz") || strings.HasSuffix(normalized, ".tar.gz") {
		return false
	}
	return !strings.Contains(chart, `\`)
}

// CleanupRSGWebhookTestResources deletes webhook test CRs created by the suite.
func CleanupRSGWebhookTestResources(dynamicClient dynamic.Interface, ctx context.Context) {
	list, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).List(
		ctx,
		metav1.ListOptions{LabelSelector: RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue},
	)
	if err != nil && !apierrors.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}

	for i := range list.Items {
		obj := list.Items[i].DeepCopy()
		err = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Delete(
			ctx,
			obj.GetName(),
			metav1.DeleteOptions{},
		)
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}

	Eventually(func() ([]unstructured.Unstructured, error) {
		list, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).List(
			ctx,
			metav1.ListOptions{LabelSelector: RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue},
		)
		if err != nil && apierrors.IsNotFound(err) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		for i := range list.Items {
			obj := &list.Items[i]
			if obj.GetDeletionTimestamp() != nil && len(obj.GetFinalizers()) > 0 {
				Expect(forceRemoveRSGFinalizers(dynamicClient, ctx, obj.GetName())).NotTo(HaveOccurred())
			}
		}
		return list.Items, nil
	}, 1*time.Minute, 2*time.Second).Should(BeEmpty())
}

// PrepareRSGForDeletion shrinks GroupReplication RSGs before deleting them so generated resources are reconciled down first.
func PrepareRSGForDeletion(dynamicClient dynamic.Interface, ctx context.Context, obj *unstructured.Unstructured) {
	strategyType, found, err := unstructured.NestedString(obj.Object, "spec", "scalingStrategy", "type")
	if err != nil || !found || strategyType != "GroupReplication" {
		return
	}

	replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "scalingStrategy", "groupConfig", "replicas")
	if err != nil || !found || replicas == 1 {
		return
	}

	Expect(unstructured.SetNestedField(obj.Object, int64(1), "spec", "scalingStrategy", "groupConfig", "replicas")).To(Succeed())

	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, getErr := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, obj.GetName(), metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}
		if setErr := unstructured.SetNestedField(latest.Object, int64(1), "spec", "scalingStrategy", "groupConfig", "replicas"); setErr != nil {
			return setErr
		}
		_, updateErr := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Update(ctx, latest, metav1.UpdateOptions{})
		return updateErr
	})
	if err != nil && !apierrors.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}

	Eventually(func() (int64, error) {
		current, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		replicas, _, err := unstructured.NestedInt64(current.Object, "spec", "scalingStrategy", "groupConfig", "replicas")
		return replicas, err
	}, 30*time.Second, 2*time.Second).Should(Equal(int64(1)))
}

func forceRemoveRSGFinalizers(dynamicClient dynamic.Interface, ctx context.Context, name string) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}
		if len(obj.GetFinalizers()) == 0 {
			return nil
		}
		obj.SetFinalizers(nil)
		_, err = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Update(ctx, obj, metav1.UpdateOptions{}, "finalizers")
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	})
}

// CreateRSG creates a ResourceScalingGroup custom resource.
func CreateRSG(dynamicClient dynamic.Interface, ctx context.Context, obj *unstructured.Unstructured) error {
	return retryTransientWebhookError(ctx, func(operationCtx context.Context) error {
		_, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Create(operationCtx, obj, metav1.CreateOptions{})
		return err
	})
}

// CreateRSGExpectAdmissionFailureEventually retries create until admission returns the expected rejection.
func CreateRSGExpectAdmissionFailureEventually(dynamicClient dynamic.Interface, ctx context.Context, obj *unstructured.Unstructured, substrings ...string) error {
	name := obj.GetName()
	var finalErr error

	Eventually(func() error {
		err := CreateRSG(dynamicClient, ctx, obj.DeepCopy())
		if err == nil {
			_ = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Delete(ctx, name, metav1.DeleteOptions{})
			return fmt.Errorf("create unexpectedly succeeded before admission cache observed the conflict")
		}
		if apierrors.IsAlreadyExists(err) {
			_ = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Delete(ctx, name, metav1.DeleteOptions{})
			return fmt.Errorf("resource %s unexpectedly exists while waiting for admission failure", name)
		}

		message := err.Error()
		for _, part := range substrings {
			if !strings.Contains(message, part) {
				return fmt.Errorf("expected error %q to contain %q", message, part)
			}
		}
		finalErr = err
		return nil
	}, 1*time.Minute, 2*time.Second).Should(Succeed(), "admission should eventually reject ResourceScalingGroup %s", name)

	return finalErr
}

// VerifyRSGWasNotCreated verifies the invalid ResourceScalingGroup was rejected by webhook.
func VerifyRSGWasNotCreated(dynamicClient dynamic.Interface, ctx context.Context, name string) {
	Consistently(func() bool {
		_, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 5*time.Second, 1*time.Second).Should(BeTrue(),
		"invalid ResourceScalingGroup %s should not exist", name)
}

// AssertErrorContains checks the webhook rejection error text.
func AssertErrorContains(err error, substrings ...string) {
	Expect(err).To(HaveOccurred())

	message := err.Error()
	for _, part := range substrings {
		Expect(strings.Contains(message, part)).To(BeTrue(),
			"expected error %q to contain %q", message, part)
	}
}

// NewGroupReplicationRSG returns a minimal valid GroupReplication ResourceScalingGroup.
func NewGroupReplicationRSG(name string) *unstructured.Unstructured {
	return NewResourceScalingGroup(name, map[string]any{
		"targetResources": []any{
			NewTargetResource("primary", "apps/v1", "Deployment", name+"-primary", ""),
		},
		"scalingStrategy": NewGroupReplicationStrategy(1),
	})
}

// NewResourceScalingGroup constructs an unstructured ResourceScalingGroup object.
func NewResourceScalingGroup(name string, spec map[string]any) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "autoscaling.openfuyao.com/v1alpha1",
			"kind":       "ResourceScalingGroup",
			"metadata": map[string]any{
				"name":      name,
				"namespace": RSGTestNamespace,
				"labels": map[string]any{
					RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
				},
			},
			"spec": spec,
		},
	}
}

// NewGroupReplicationStrategy returns a basic GroupReplication strategy payload.
func NewGroupReplicationStrategy(replicas int64) map[string]any {
	return map[string]any{
		"type": "GroupReplication",
		"groupConfig": map[string]any{
			"groupName": "pd",
			"replicas":  replicas,
		},
	}
}

// NewTargetResource builds a targetResources entry.
func NewTargetResource(name, apiVersion, kind, resourceName, namespace string) map[string]any {
	resourceRef := map[string]any{
		"apiVersion": apiVersion,
		"kind":       kind,
		"name":       resourceName,
	}
	if namespace != "" {
		resourceRef["namespace"] = namespace
	}

	return map[string]any{
		"name":        name,
		"resourceRef": resourceRef,
	}
}

// CleanupRSGBusinessResources removes target workloads, cloned workloads and control-plane artifacts created by e2e cases.
func CleanupRSGBusinessResources(clientset *kubernetes.Clientset, ctx context.Context) {
	deleteDeploymentsByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteStatefulSetsByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteServicesByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteConfigMapsByPrefix(clientset, ctx, RSGTestNamespace, "rsg-")
	deleteControllerRevisionsByPrefix(clientset, ctx, RSGTestNamespace, "rsg-")
	waitForNoDeploymentsByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	waitForNoStatefulSetsByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	waitForNoServicesByLabel(clientset, ctx, RSGTestNamespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
}

// CleanupRSGBusinessResourcesInNamespace removes labeled RSG test resources from the specified namespace.
func CleanupRSGBusinessResourcesInNamespace(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteDeploymentsByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteStatefulSetsByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteServicesByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	deleteConfigMapsByPrefix(clientset, ctx, namespace, "rsg-")
	deleteControllerRevisionsByPrefix(clientset, ctx, namespace, "rsg-")
	waitForNoDeploymentsByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	waitForNoStatefulSetsByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
	waitForNoServicesByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
}

// CleanupRSGLWSResources removes LeaderWorkerSet resources created by e2e cases.
func CleanupRSGLWSResources(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string) {
	list, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
	})
	if err != nil && !apierrors.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		err = dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Delete(ctx, list.Items[i].GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
	waitForNoLWSByLabel(dynamicClient, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
}

func WaitForNoRSGTestPods(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	waitForNoPodsByLabel(clientset, ctx, namespace, RSGWebhookTestLabelKey+"="+RSGWebhookTestLabelValue)
}

func deleteDeploymentsByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	list, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		err = clientset.AppsV1().Deployments(namespace).Delete(ctx, list.Items[i].Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

func deleteStatefulSetsByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	list, err := clientset.AppsV1().StatefulSets(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		err = clientset.AppsV1().StatefulSets(namespace).Delete(ctx, list.Items[i].Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

func deleteServicesByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	list, err := clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		err = clientset.CoreV1().Services(namespace).Delete(ctx, list.Items[i].Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

func deleteConfigMapsByPrefix(clientset *kubernetes.Clientset, ctx context.Context, namespace, prefix string) {
	list, err := clientset.CoreV1().ConfigMaps(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		if !strings.HasPrefix(list.Items[i].Name, prefix) {
			continue
		}
		err = clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, list.Items[i].Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

func deleteControllerRevisionsByPrefix(clientset *kubernetes.Clientset, ctx context.Context, namespace, prefix string) {
	list, err := clientset.AppsV1().ControllerRevisions(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	for i := range list.Items {
		if !strings.HasPrefix(list.Items[i].Name, prefix) {
			continue
		}
		err = clientset.AppsV1().ControllerRevisions(namespace).Delete(ctx, list.Items[i].Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

func waitForNoDeploymentsByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	Eventually(func() int {
		list, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if apierrors.IsNotFound(err) {
			return 0
		}
		if err != nil {
			Expect(err).NotTo(HaveOccurred())
		}
		return len(list.Items)
	}, 2*time.Minute, 2*time.Second).Should(BeZero(), "deployments with selector %s in namespace %s should be deleted", labelSelector, namespace)
}

func waitForNoStatefulSetsByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	Eventually(func() int {
		list, err := clientset.AppsV1().StatefulSets(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if apierrors.IsNotFound(err) {
			return 0
		}
		if err != nil {
			Expect(err).NotTo(HaveOccurred())
		}
		return len(list.Items)
	}, 2*time.Minute, 2*time.Second).Should(BeZero(), "statefulsets with selector %s in namespace %s should be deleted", labelSelector, namespace)
}

func waitForNoServicesByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	Eventually(func() int {
		list, err := clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if apierrors.IsNotFound(err) {
			return 0
		}
		if err != nil {
			Expect(err).NotTo(HaveOccurred())
		}
		return len(list.Items)
	}, 2*time.Minute, 2*time.Second).Should(BeZero(), "services with selector %s in namespace %s should be deleted", labelSelector, namespace)
}

func waitForNoPodsByLabel(clientset *kubernetes.Clientset, ctx context.Context, namespace, labelSelector string) {
	Eventually(func() int {
		list, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if apierrors.IsNotFound(err) {
			return 0
		}
		if err != nil {
			Expect(err).NotTo(HaveOccurred())
		}
		return len(list.Items)
	}, 3*time.Minute, 2*time.Second).Should(BeZero(), "pods with selector %s in namespace %s should be deleted", labelSelector, namespace)
}

func waitForNoLWSByLabel(dynamicClient dynamic.Interface, ctx context.Context, namespace, labelSelector string) {
	Eventually(func() int {
		list, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if apierrors.IsNotFound(err) {
			return 0
		}
		if err != nil {
			Expect(err).NotTo(HaveOccurred())
		}
		return len(list.Items)
	}, 2*time.Minute, 2*time.Second).Should(BeZero(), "leaderworkersets with selector %s in namespace %s should be deleted", labelSelector, namespace)
}

// CreateRSGTestDeployment creates a test Deployment used by group replication cases.
func CreateRSGTestDeployment(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, appName, role string, replicas int32) error {
	labels := map[string]string{
		"app":                  appName,
		"role":                 role,
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
	}
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nginx",
						Image: "nginx:alpine",
					}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{})
	return err
}

// CreateRSGTestService creates a Service matching the target workload labels.
func CreateRSGTestService(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, appName, role string) error {
	selector := map[string]string{
		"app":                  appName,
		"role":                 role,
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
	}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: selector,
			Ports: []corev1.ServicePort{{
				Name: "http",
				Port: 80,
			}},
		},
	}
	_, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{})
	return err
}

// CreateRSGTestHeadlessService creates a headless Service for StatefulSet workloads.
func CreateRSGTestHeadlessService(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, appName, role string) error {
	selector := map[string]string{
		"app":                  appName,
		"role":                 role,
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
	}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
			},
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  selector,
			Ports: []corev1.ServicePort{{
				Name: "http",
				Port: 80,
			}},
		},
	}
	_, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{})
	return err
}

// CreateRSGTestStatefulSet creates a test StatefulSet used by group replication cases.
func CreateRSGTestStatefulSet(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, appName, role, serviceName string, replicas int32) error {
	labels := map[string]string{
		"app":                  appName,
		"role":                 role,
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
	}
	statefulSet := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: appsv1.StatefulSetSpec{
			ServiceName: serviceName,
			Replicas:    &replicas,
			Selector:    &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nginx",
						Image: "nginx:alpine",
					}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().StatefulSets(namespace).Create(ctx, statefulSet, metav1.CreateOptions{})
	return err
}

// CreateRSGTestLeaderWorkerSet creates a test LeaderWorkerSet used by group replication cases.
func CreateRSGTestLeaderWorkerSet(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, appName, role string, replicas int64) error {
	labels := map[string]any{
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
		"app":                  appName,
		"role":                 role,
	}
	obj := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": LeaderWorkerSetAPIVersion,
			"kind":       "LeaderWorkerSet",
			"metadata": map[string]any{
				"name":      name,
				"namespace": namespace,
				"labels":    labels,
			},
			"spec": map[string]any{
				"replicas": replicas,
				"leaderWorkerTemplate": map[string]any{
					"size": int64(1),
					"leaderTemplate": map[string]any{
						"metadata": map[string]any{
							"labels": labels,
						},
						"spec": map[string]any{
							"containers": []any{
								map[string]any{
									"name":    "leader",
									"image":   "busybox:1.36",
									"command": []any{"sh", "-c", "sleep 3600"},
								},
							},
						},
					},
					"workerTemplate": map[string]any{
						"metadata": map[string]any{
							"labels": labels,
						},
						"spec": map[string]any{
							"containers": []any{
								map[string]any{
									"name":    "worker",
									"image":   "busybox:1.36",
									"command": []any{"sh", "-c", "sleep 3600"},
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// VerifyRSGControllerRevisionExists checks the per-target snapshot exists.
func VerifyRSGControllerRevisionExists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().ControllerRevisions(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "controller revision %s/%s should exist", namespace, name)
}

// VerifyRSGControllerRevisionSnapshot checks the snapshot content is sanitized and captures the expected workload spec.
func VerifyRSGControllerRevisionSnapshot(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, rsgName, rsgUID, targetName, expectedAPIVersion, expectedKind, expectedWorkloadName, expectedWorkloadNamespace string, expectedReplicas int32, expectedServiceName string) {
	Eventually(func() bool {
		revision, err := clientset.AppsV1().ControllerRevisions(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		if revision.Labels["rsg.io/name"] != rsgName || revision.Labels["rsg.io/target-name"] != targetName {
			return false
		}
		if rsgUID != "" && revision.Labels["rsg-uid"] != rsgUID {
			return false
		}
		if len(revision.OwnerReferences) == 0 {
			return false
		}
		ref := revision.OwnerReferences[0]
		if ref.Kind != "ResourceScalingGroup" || ref.Name != rsgName {
			return false
		}
		if rsgUID != "" && string(ref.UID) != rsgUID {
			return false
		}

		var snapshot map[string]any
		if err := json.Unmarshal(revision.Data.Raw, &snapshot); err != nil {
			return false
		}
		if snapshot["apiVersion"] != expectedAPIVersion || snapshot["kind"] != expectedKind {
			return false
		}
		if _, exists := snapshot["status"]; exists {
			return false
		}

		metadata, ok := snapshot["metadata"].(map[string]any)
		if !ok {
			return false
		}
		if metadata["name"] != expectedWorkloadName || metadata["namespace"] != expectedWorkloadNamespace {
			return false
		}
		if rawUID, exists := metadata["uid"]; exists && rawUID != "" {
			return false
		}
		if rawResourceVersion, exists := metadata["resourceVersion"]; exists && rawResourceVersion != "" {
			return false
		}
		if finalizers, exists := metadata["finalizers"]; exists {
			if values, ok := finalizers.([]any); !ok || len(values) != 0 {
				return false
			}
		}
		if ownerReferences, exists := metadata["ownerReferences"]; exists {
			if values, ok := ownerReferences.([]any); !ok || len(values) != 0 {
				return false
			}
		}

		spec, ok := snapshot["spec"].(map[string]any)
		if !ok {
			return false
		}
		replicasValue, ok := spec["replicas"]
		if !ok {
			return false
		}
		replicas, ok := replicasValue.(float64)
		if !ok || int32(replicas) != expectedReplicas {
			return false
		}
		if expectedServiceName != "" {
			serviceName, _ := spec["serviceName"].(string)
			if serviceName != expectedServiceName {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "controller revision %s/%s should contain the expected snapshot content", namespace, name)
}

// VerifyRSGControllerRevisionDeleted checks the snapshot is deleted.
func VerifyRSGControllerRevisionDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().ControllerRevisions(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "controller revision %s/%s should be deleted", namespace, name)
}

// VerifyRSGReplicaConfigMapData checks the ConfigMap exists and contains expected entries.
func VerifyRSGReplicaConfigMapData(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		cm, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for key, value := range expected {
			if cm.Data[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "configmap %s/%s should contain expected data", namespace, name)
}

// VerifyRSGReplicaConfigMapContent checks labels, owner reference and exact data entries.
func VerifyRSGReplicaConfigMapContent(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, rsgName, rsgUID string, expectedData map[string]string) {
	Eventually(func() bool {
		configMap, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		if configMap.Labels["rsg.io/name"] != rsgName {
			return false
		}
		if rsgUID != "" && configMap.Labels["rsg-uid"] != rsgUID {
			return false
		}
		if len(configMap.OwnerReferences) == 0 {
			return false
		}
		ref := configMap.OwnerReferences[0]
		if ref.Kind != "ResourceScalingGroup" || ref.Name != rsgName {
			return false
		}
		if rsgUID != "" && string(ref.UID) != rsgUID {
			return false
		}
		if len(configMap.Data) != len(expectedData) {
			return false
		}
		for key, value := range expectedData {
			if configMap.Data[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "configmap %s/%s should contain the expected managed content", namespace, name)
}

// VerifyRSGReplicaConfigMapDeleted checks the replica ConfigMap is deleted.
func VerifyRSGReplicaConfigMapDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "configmap %s/%s should be deleted", namespace, name)
}

// VerifyDeploymentExists checks a Deployment exists.
func VerifyDeploymentExists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s should exist", namespace, name)
}

// VerifyDeploymentPersists checks a Deployment keeps existing during the observation window.
func VerifyDeploymentPersists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Consistently(func() bool {
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "deployment %s/%s should persist", namespace, name)
}

// VerifyStatefulSetExists checks a StatefulSet exists.
func VerifyStatefulSetExists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s should exist", namespace, name)
}

// VerifyStatefulSetPersists checks a StatefulSet keeps existing during the observation window.
func VerifyStatefulSetPersists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Consistently(func() bool {
		_, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "statefulset %s/%s should persist", namespace, name)
}

// VerifyLWSExists checks a LeaderWorkerSet exists.
func VerifyLWSExists(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s should exist", namespace, name)
}

// VerifyLWSPersists checks a LeaderWorkerSet keeps existing during the observation window.
func VerifyLWSPersists(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) {
	Consistently(func() bool {
		_, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "leaderworkerset %s/%s should persist", namespace, name)
}

// VerifyServiceExists checks a Service exists.
func VerifyServiceExists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "service %s/%s should exist", namespace, name)
}

// VerifyServicePersists checks a Service keeps existing during the observation window.
func VerifyServicePersists(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Consistently(func() bool {
		_, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "service %s/%s should persist", namespace, name)
}

// VerifyDeploymentDeleted checks a Deployment is deleted.
func VerifyDeploymentDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s should be deleted", namespace, name)
}

// VerifyStatefulSetDeleted checks a StatefulSet is deleted.
func VerifyStatefulSetDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s should be deleted", namespace, name)
}

// VerifyLWSDeleted checks a LeaderWorkerSet is deleted.
func VerifyLWSDeleted(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s should be deleted", namespace, name)
}

// VerifyServiceDeleted checks a Service is deleted.
func VerifyServiceDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		_, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "service %s/%s should be deleted", namespace, name)
}

// VerifyDeploymentHasNoOwnerReferences checks orphaning completed on a retained Deployment.
func VerifyDeploymentHasNoOwnerReferences(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return len(dep.OwnerReferences) == 0
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s should have no owner references", namespace, name)
}

// VerifyStatefulSetHasNoOwnerReferences checks orphaning completed on a retained StatefulSet.
func VerifyStatefulSetHasNoOwnerReferences(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return len(statefulSet.OwnerReferences) == 0
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s should have no owner references", namespace, name)
}

// VerifyLWSHasNoOwnerReferences checks orphaning completed on a retained LeaderWorkerSet.
func VerifyLWSHasNoOwnerReferences(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return len(obj.GetOwnerReferences()) == 0
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s should have no owner references", namespace, name)
}

// VerifyServiceHasNoOwnerReferences checks orphaning completed on a retained Service.
func VerifyServiceHasNoOwnerReferences(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	Eventually(func() bool {
		svc, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return len(svc.OwnerReferences) == 0
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "service %s/%s should have no owner references", namespace, name)
}

// VerifyDeploymentReplicas checks the Deployment spec replicas value.
func VerifyDeploymentReplicas(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected int32) {
	Eventually(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil || dep.Spec.Replicas == nil {
			return false
		}
		return *dep.Spec.Replicas == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s replicas should be %d", namespace, name, expected)
}

// VerifyStatefulSetReplicas checks the StatefulSet spec replicas value.
func VerifyStatefulSetReplicas(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected int32) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil || statefulSet.Spec.Replicas == nil {
			return false
		}
		return *statefulSet.Spec.Replicas == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s replicas should be %d", namespace, name, expected)
}

// VerifyStatefulSetSelectorContains checks StatefulSet selector matchLabels include expected labels.
func VerifyStatefulSetSelectorContains(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil || statefulSet.Spec.Selector == nil {
			return false
		}
		for key, value := range expected {
			if statefulSet.Spec.Selector.MatchLabels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s selector should contain expected values", namespace, name)
}

// VerifyStatefulSetTemplateLabelsContain checks StatefulSet pod template labels include expected values.
func VerifyStatefulSetTemplateLabelsContain(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for key, value := range expected {
			if statefulSet.Spec.Template.Labels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s template labels should contain expected values", namespace, name)
}

// VerifyLWSReplicas checks the LeaderWorkerSet spec replicas value.
func VerifyLWSReplicas(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, expected int64) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas")
		return err == nil && found && replicas == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s replicas should be %d", namespace, name, expected)
}

// VerifyStatefulSetServiceName checks the StatefulSet spec.serviceName.
func VerifyStatefulSetServiceName(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, expected string) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return statefulSet.Spec.ServiceName == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s serviceName should be %s", namespace, name, expected)
}

// VerifyLWSLabelsContain checks LeaderWorkerSet metadata labels include expected values.
func VerifyLWSLabelsContain(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		labels := obj.GetLabels()
		for key, value := range expected {
			if labels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s labels should contain expected values", namespace, name)
}

// VerifyLWSTemplateLabelsContain checks leader and worker template labels include expected values.
func VerifyLWSTemplateLabelsContain(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		leaderLabels, _, err := unstructured.NestedStringMap(obj.Object, "spec", "leaderWorkerTemplate", "leaderTemplate", "metadata", "labels")
		if err != nil {
			return false
		}
		workerLabels, _, err := unstructured.NestedStringMap(obj.Object, "spec", "leaderWorkerTemplate", "workerTemplate", "metadata", "labels")
		if err != nil {
			return false
		}
		for key, value := range expected {
			if leaderLabels[key] != value || workerLabels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s templates should contain expected labels", namespace, name)
}

// VerifyDeploymentLabelsContain checks the Deployment labels include all expected keys.
func VerifyDeploymentLabelsContain(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for key, value := range expected {
			if dep.Labels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s labels should contain expected values", namespace, name)
}

// VerifyDeploymentSelectorContains checks Deployment selector matchLabels include expected labels.
func VerifyDeploymentSelectorContains(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil || dep.Spec.Selector == nil {
			return false
		}
		for key, value := range expected {
			if dep.Spec.Selector.MatchLabels[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s selector should contain expected values", namespace, name)
}

// VerifyServiceSelectorContains checks Service selector labels include expected labels.
func VerifyServiceSelectorContains(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, expected map[string]string) {
	Eventually(func() bool {
		svc, err := clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for key, value := range expected {
			if svc.Spec.Selector[key] != value {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "service %s/%s selector should contain expected values", namespace, name)
}

// VerifyRSGStatusReplicas checks currentReplicas and lastDesiredReplicas.
func VerifyRSGStatusReplicas(dynamicClient dynamic.Interface, ctx context.Context, name string, currentExpected, desiredExpected int64) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		current, foundCurrent, _ := unstructured.NestedInt64(obj.Object, "status", "currentReplicas")
		desired, foundDesired, _ := unstructured.NestedInt64(obj.Object, "status", "lastDesiredReplicas")
		return foundCurrent && foundDesired && current == currentExpected && desired == desiredExpected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(),
		"ResourceScalingGroup %s status replicas should be current=%d desired=%d", name, currentExpected, desiredExpected)
}

// VerifyRSGStatusSelectorContains checks the status.selector text contains expected fragments.
func VerifyRSGStatusSelectorContains(dynamicClient dynamic.Interface, ctx context.Context, name string, expected []string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		selector, found, _ := unstructured.NestedString(obj.Object, "status", "selector")
		if !found {
			return false
		}
		for _, item := range expected {
			if !strings.Contains(selector, item) {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ResourceScalingGroup %s status.selector should contain expected fragments", name)
}

// VerifyRSGReadyCondition checks Ready condition status and reason.
func VerifyRSGReadyCondition(dynamicClient dynamic.Interface, ctx context.Context, name, expectedStatus, expectedReason string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
		if err != nil || !found {
			return false
		}
		for _, item := range conditions {
			condition, ok := item.(map[string]any)
			if !ok {
				continue
			}
			conditionType, _ := condition["type"].(string)
			status, _ := condition["status"].(string)
			reason, _ := condition["reason"].(string)
			if conditionType == "Ready" && status == expectedStatus && reason == expectedReason {
				return true
			}
		}
		return false
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ResourceScalingGroup %s Ready condition should be %s/%s", name, expectedStatus, expectedReason)
}

// UpdateRSGGroupReplicas updates GroupReplication replicas on an existing CR.
func UpdateRSGGroupReplicas(dynamicClient dynamic.Interface, ctx context.Context, name string, replicas int64) error {
	return retryTransientWebhookError(ctx, func(operationCtx context.Context) error {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(operationCtx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if err := unstructured.SetNestedField(obj.Object, replicas, "spec", "scalingStrategy", "groupConfig", "replicas"); err != nil {
				return err
			}
			_, err = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Update(operationCtx, obj, metav1.UpdateOptions{})
			return err
		})
	})
}

func retryTransientWebhookError(ctx context.Context, operation func(context.Context) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	operationCtx, cancel := context.WithTimeout(context.Background(), transientWebhookRetryTimeout)
	defer cancel()

	var lastErr error
	err := wait.PollUntilContextTimeout(operationCtx, transientWebhookRetryTimeout, transientWebhookRetryInterval, true, func(context.Context) (bool, error) {
		lastErr = operation(operationCtx)
		if lastErr == nil {
			return true, nil
		}
		if isTransientWebhookTransportError(lastErr) {
			GinkgoWriter.Printf("retrying transient webhook error: %v\n", lastErr)
			return false, nil
		}
		return false, lastErr
	})
	if err == nil {
		return nil
	}
	if errors.Is(err, context.DeadlineExceeded) && lastErr != nil {
		return lastErr
	}
	return err
}

func isTransientWebhookTransportError(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	if !strings.Contains(message, "failed calling webhook") {
		return false
	}
	return strings.Contains(message, "connect: connection refused") ||
		strings.Contains(message, "no endpoints available") ||
		strings.Contains(message, "context deadline exceeded") ||
		strings.Contains(message, "i/o timeout") ||
		strings.Contains(message, "EOF")
}

// UpdateRSGReplicaConfigMapData updates one or more replica-config entries.
func UpdateRSGReplicaConfigMapData(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, updates map[string]string) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Minute, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			cm, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if cm.Data == nil {
				cm.Data = map[string]string{}
			}
			for key, value := range updates {
				cm.Data[key] = value
			}
			_, err = clientset.CoreV1().ConfigMaps(namespace).Update(ctx, cm, metav1.UpdateOptions{})
			return err
		})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
}

// DeleteRSGAndWait deletes a CR and waits for it to disappear.
func DeleteRSGAndWait(dynamicClient dynamic.Interface, ctx context.Context, name string) {
	err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}

	Eventually(func() bool {
		_, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ResourceScalingGroup %s should be deleted", name)
}

// GetRSGUID returns the UID of an existing CR.
func GetRSGUID(dynamicClient dynamic.Interface, ctx context.Context, name string) string {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(RSGTestNamespace).Get(ctx, name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return string(obj.GetUID())
}

// AnnotateRSGTestDeployment patches annotations on an existing Deployment.
func AnnotateRSGTestDeployment(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, annotations map[string]string) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Minute, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			deployment, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if deployment.Annotations == nil {
				deployment.Annotations = map[string]string{}
			}
			for key, value := range annotations {
				deployment.Annotations[key] = value
			}
			_, err = clientset.AppsV1().Deployments(namespace).Update(ctx, deployment, metav1.UpdateOptions{})
			return err
		})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
}

// AnnotateRSGTestStatefulSet patches annotations on an existing StatefulSet.
func AnnotateRSGTestStatefulSet(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string, annotations map[string]string) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Minute, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if statefulSet.Annotations == nil {
				statefulSet.Annotations = map[string]string{}
			}
			for key, value := range annotations {
				statefulSet.Annotations[key] = value
			}
			_, err = clientset.AppsV1().StatefulSets(namespace).Update(ctx, statefulSet, metav1.UpdateOptions{})
			return err
		})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
}

// AnnotateRSGTestLWS patches annotations on an existing LeaderWorkerSet.
func AnnotateRSGTestLWS(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, annotations map[string]string) error {
	return wait.PollUntilContextTimeout(ctx, 2*time.Minute, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			current := obj.GetAnnotations()
			if current == nil {
				current = map[string]string{}
			}
			for key, value := range annotations {
				current[key] = value
			}
			obj.SetAnnotations(current)
			_, err = dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
			return err
		})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
}

// VerifyDeploymentLabelValue checks one Deployment label exact value.
func VerifyDeploymentLabelValue(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, key, expected string) {
	Eventually(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return dep.Labels[key] == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "deployment %s/%s label %s should be %s", namespace, name, key, expected)
}

// VerifyStatefulSetLabelValue checks one StatefulSet label exact value.
func VerifyStatefulSetLabelValue(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, key, expected string) {
	Eventually(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return statefulSet.Labels[key] == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "statefulset %s/%s label %s should be %s", namespace, name, key, expected)
}

// VerifyLWSLabelValue checks one LeaderWorkerSet label exact value.
func VerifyLWSLabelValue(dynamicClient dynamic.Interface, ctx context.Context, namespace, name, key, expected string) {
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return obj.GetLabels()[key] == expected
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "leaderworkerset %s/%s label %s should be %s", namespace, name, key, expected)
}

// VerifyDeploymentLabelValueNotEqual checks one Deployment label does not equal the unexpected value.
func VerifyDeploymentLabelValueNotEqual(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, key, unexpected string) {
	Consistently(func() bool {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		value := dep.Labels[key]
		return value != "" && value != unexpected
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "deployment %s/%s label %s should not become %s", namespace, name, key, unexpected)
}

// VerifyStatefulSetLabelValueNotEqual checks one StatefulSet label does not equal the unexpected value.
func VerifyStatefulSetLabelValueNotEqual(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, key, unexpected string) {
	Consistently(func() bool {
		statefulSet, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		value := statefulSet.Labels[key]
		return value != "" && value != unexpected
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "statefulset %s/%s label %s should not become %s", namespace, name, key, unexpected)
}

// VerifyLWSLabelValueNotEqual checks one LeaderWorkerSet label does not equal the unexpected value.
func VerifyLWSLabelValueNotEqual(dynamicClient dynamic.Interface, ctx context.Context, namespace, name, key, unexpected string) {
	Consistently(func() bool {
		obj, err := dynamicClient.Resource(LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false
		}
		value := obj.GetLabels()[key]
		return value != "" && value != unexpected
	}, 15*time.Second, 2*time.Second).Should(BeTrue(), "leaderworkerset %s/%s label %s should not become %s", namespace, name, key, unexpected)
}

// VerifyRSGControllerLogsContain checks controller logs since the given time contain the expected fragments.
func VerifyRSGControllerLogsContain(clientset *kubernetes.Clientset, ctx context.Context, since time.Time, substrings ...string) {
	var lastLogs string
	var lastErr error
	Eventually(func() bool {
		lastLogs, lastErr = getRSGControllerLogsSince(clientset, ctx, since)
		if lastErr != nil {
			GinkgoWriter.Printf("get controller logs failed: %v\n", lastErr)
			return false
		}
		for _, part := range substrings {
			if !strings.Contains(lastLogs, part) {
				return false
			}
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "controller logs should contain expected fragments: %v\nlast error: %v\ncaptured logs tail:\n%s", substrings, lastErr, tailString(lastLogs, 8000))
}

// VerifyRSGControllerLogsContainForRSG checks expected fragments appear on log lines for the specified RSG.
func VerifyRSGControllerLogsContainForRSG(clientset *kubernetes.Clientset, ctx context.Context, since time.Time, rsgName string, substrings ...string) {
	var lastLogs string
	var lastErr error
	Eventually(func() bool {
		lastLogs, lastErr = getRSGControllerLogsSince(clientset, ctx, since)
		if lastErr != nil {
			GinkgoWriter.Printf("get controller logs failed: %v\n", lastErr)
			return false
		}
		return logsContainRSGFragments(lastLogs, rsgName, substrings)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "controller logs for %s should contain expected fragments: %v\nlast error: %v\ncaptured logs tail:\n%s", rsgName, substrings, lastErr, tailString(lastLogs, 8000))
}

func logsContainRSGFragments(logs, rsgName string, substrings []string) bool {
	namespacedName := RSGTestNamespace + "/" + rsgName
	lines := strings.Split(logs, "\n")
	for _, part := range substrings {
		matched := false
		for _, line := range lines {
			if !strings.Contains(line, part) {
				continue
			}
			if strings.Contains(line, namespacedName) || strings.Contains(line, rsgName) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

// rsgControllerLogSinceSkewMargin 是读取控制器日志时给 SinceTime 留的时钟偏差余量。
//
// kubelet 的 --since-time 过滤比较的是**节点时钟**打的日志时间戳（containerd 写入，时间戳来自节点），
// 而 since 由**测试机时钟**计算。节点时钟落后于测试机时，刚写入的日志会被判为"早于 since"而丢弃，
// 于是日志断言（"controller logs should contain ..."）在 120s 内永远等不到那一行而假失败。
// 实测：n3 落后测试机 m1 约 13s，RSG 套件里依赖首轮 reconcile 日志的用例因此失败。
// 这里留出足够余量（RSG 的 RSG 名逐用例唯一，放宽窗口不会误匹配其它用例的日志）。
const rsgControllerLogSinceSkewMargin = 60 * time.Second

func getRSGControllerLogsSince(clientset *kubernetes.Clientset, ctx context.Context, since time.Time) (string, error) {
	deployment, err := clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, RSGControllerDeploymentName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	selector := metav1.FormatLabelSelector(deployment.Spec.Selector)
	pods, err := clientset.CoreV1().Pods(RSGTestNamespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return "", err
	}
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no controller pods found")
	}
	var builder strings.Builder
	sinceTime := metav1.NewTime(since.Add(-rsgControllerLogSinceSkewMargin))
	for i := range pods.Items {
		builder.WriteString("\n=== pod " + pods.Items[i].Name + " current ===\n")
		data, err := readPodLogs(ctx, clientset, pods.Items[i].Name, sinceTime, false)
		if err != nil {
			return "", err
		}
		builder.WriteString(data)

		previousData, err := readPodLogs(ctx, clientset, pods.Items[i].Name, sinceTime, true)
		if err == nil && len(previousData) > 0 {
			builder.WriteString("\n=== pod " + pods.Items[i].Name + " previous ===\n")
			builder.WriteString(previousData)
		}
	}
	return builder.String(), nil
}

func readPodLogs(ctx context.Context, clientset *kubernetes.Clientset, podName string, sinceTime metav1.Time, previous bool) (string, error) {
	stream, err := clientset.CoreV1().Pods(RSGTestNamespace).GetLogs(podName, &corev1.PodLogOptions{
		SinceTime: &sinceTime,
		Previous:  previous,
	}).Stream(ctx)
	if err != nil {
		if previous && strings.Contains(err.Error(), "previous terminated container") {
			return "", nil
		}
		return "", err
	}
	data, readErr := io.ReadAll(stream)
	_ = stream.Close()
	if readErr != nil {
		return "", readErr
	}
	return string(data), nil
}

func tailString(value string, max int) string {
	if len(value) <= max {
		return value
	}
	return value[len(value)-max:]
}

// VerifyRSGCreateEventuallySucceeds waits for create to succeed after targets appear.
func VerifyRSGCreateEventuallySucceeds(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, name string, currentExpected, desiredExpected int64) {
	VerifyRSGStatusReplicas(dynamicClient, ctx, name, currentExpected, desiredExpected)
}

// VerifyConfigMapDataKeyValue checks one key in the ConfigMap data.
func VerifyConfigMapDataKeyValue(clientset *kubernetes.Clientset, ctx context.Context, namespace, name, key string, expected int32) {
	VerifyRSGReplicaConfigMapData(clientset, ctx, namespace, name, map[string]string{
		key: strconv.Itoa(int(expected)),
	})
}
