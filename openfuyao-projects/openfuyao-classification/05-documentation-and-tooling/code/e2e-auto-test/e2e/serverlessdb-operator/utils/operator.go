package utils

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// DefaultLeaderElectionID matches the operator's --leader-election-id default.
const DefaultLeaderElectionID = "serverlessdb-operator.openfuyao.cn"

// GetOperatorDeploymentName resolves the Deployment name owning the operator pod
// matching labelSelector in ns, by following pod -> ReplicaSet -> Deployment
// ownerReferences. Falls back to listing deployments by the same selector.
func GetOperatorDeploymentName(ctx context.Context, k8sClient *k8s.K8SClient, ns, labelSelector string) (string, error) {
	pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return "", fmt.Errorf("list operator pods: %w", err)
	}
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no operator pod found with selector %q in %q", labelSelector, ns)
	}
	pod := &pods.Items[0]
	for _, ref := range pod.OwnerReferences {
		if ref.Kind == "ReplicaSet" && ref.Controller != nil && *ref.Controller {
			rs, rerr := k8sClient.Clientset.AppsV1().ReplicaSets(ns).Get(ctx, ref.Name, metav1.GetOptions{})
			if rerr != nil {
				return "", fmt.Errorf("get replicaset %s: %w", ref.Name, rerr)
			}
			for _, rref := range rs.OwnerReferences {
				if rref.Kind == "Deployment" && rref.Controller != nil && *rref.Controller {
					return rref.Name, nil
				}
			}
		}
		if ref.Kind == "Deployment" && ref.Controller != nil && *ref.Controller {
			return ref.Name, nil
		}
	}
	deploys, derr := k8sClient.Clientset.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if derr != nil {
		return "", fmt.Errorf("list deployments by selector %q: %w", labelSelector, derr)
	}
	if len(deploys.Items) == 0 {
		return "", fmt.Errorf("no deployment found owning operator pod %q", pod.Name)
	}
	return deploys.Items[0].Name, nil
}

// ScaleDeployment patches spec.replicas on the named Deployment.
func ScaleDeployment(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string, replicas int32) error {
	patch := []byte(fmt.Sprintf(`{"spec":{"replicas":%d}}`, replicas))
	_, err := k8sClient.Clientset.AppsV1().Deployments(ns).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{})
	return err
}

// GetLeaseHolder returns the holderIdentity of the named Lease in ns, or "" if
// the lease has no holder. The identity format from controller-runtime is
// "<podName>_<uuid>"; use LeaderPodName to extract the pod name.
func GetLeaseHolder(ctx context.Context, k8sClient *k8s.K8SClient, ns, leaseName string) (string, error) {
	lease, err := k8sClient.Clientset.CoordinationV1().Leases(ns).Get(ctx, leaseName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	if lease.Spec.HolderIdentity == nil {
		return "", nil
	}
	return *lease.Spec.HolderIdentity, nil
}

// LeaderPodName extracts the pod name from a controller-runtime leader identity
// of the form "<podName>_<uuid>". If the identity does not contain "_" the whole
// string is returned unchanged.
func LeaderPodName(identity string) string {
	if i := strings.IndexByte(identity, '_'); i >= 0 {
		return identity[:i]
	}
	return identity
}

// ListReadyPodNames returns the names of Ready pods matching labelSelector in ns.
func ListReadyPodNames(ctx context.Context, k8sClient *k8s.K8SClient, ns, labelSelector string) ([]string, error) {
	pods, err := k8sClient.ListPods(ctx, ns, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return nil, err
	}
	var names []string
	for i := range pods.Items {
		if IsPodReady(&pods.Items[i]) {
			names = append(names, pods.Items[i].Name)
		}
	}
	return names, nil
}

// DeletePod deletes a single pod by name in ns.
func DeletePod(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) error {
	return k8sClient.Clientset.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

// CreateHeadlessService creates a ClusterIP=None service named svcName in ns with
// the given labels and port. Used to simulate orphaned services for TTL tests.
func CreateHeadlessService(ctx context.Context, k8sClient *k8s.K8SClient, ns, svcName string, labels map[string]string, portName string, portNum int32) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      svcName,
			Namespace: ns,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Ports: []corev1.ServicePort{{
				Name:     portName,
				Port:     portNum,
				Protocol: corev1.ProtocolTCP,
			}},
		},
	}
	_, err := k8sClient.Clientset.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
	return err
}

// AnnotateService merge-patches the given annotations onto the named service.
func AnnotateService(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string, annotations map[string]string) error {
	parts := make([]string, 0, len(annotations))
	for k, v := range annotations {
		parts = append(parts, fmt.Sprintf("%q:%q", k, v))
	}
	patch := []byte(fmt.Sprintf(`{"metadata":{"annotations":{%s}}}`, strings.Join(parts, ",")))
	_, err := k8sClient.Clientset.CoreV1().Services(ns).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{})
	return err
}

// GetService returns the named service in ns.
func GetService(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) (*corev1.Service, error) {
	return k8sClient.Clientset.CoreV1().Services(ns).Get(ctx, name, metav1.GetOptions{})
}

// DeleteService deletes the named service in ns (ignore not-found).
func DeleteService(ctx context.Context, k8sClient *k8s.K8SClient, ns, name string) error {
	return k8sClient.Clientset.CoreV1().Services(ns).Delete(ctx, name, metav1.DeleteOptions{})
}
