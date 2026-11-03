/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package framework

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// ServiceAccountName is the SA used by the e2e agent and watcher pods.
	ServiceAccountName = "flux-sandbox"
	// ClusterRoleName is the shared e2e ClusterRole.
	ClusterRoleName = "flux-sandbox-e2e"
	// AgentPodName is the fixed name of the e2e agent pod. A bare pod (not a
	// DaemonSet) keeps the name across RestartAgentPod recreations, so the
	// agent's name-based reconcile re-adoption mirrors the production
	// in-place-restart path where scheduler-managed agent pod names are stable.
	AgentPodName = "flux-sandbox-agent"
	// WatcherDaemonSetName is the e2e watcher DaemonSet name.
	WatcherDaemonSetName = "sandbox-watcher"
)

// DeployComponents deploys the Namespace, ServiceAccount, RBAC, the
// fixed-name agent Pod, and the Watcher DaemonSet. All creates are idempotent.
func DeployComponents(k8s *K8s) error {
	ctx := context.Background()
	cfg := k8s.Cfg

	objects := []client.Object{
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: cfg.Namespace}},
		&corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: ServiceAccountName, Namespace: cfg.Namespace}},
		&rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: ClusterRoleName},
			Rules: []rbacv1.PolicyRule{
				{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"get", "list", "watch"}},
				{APIGroups: []string{"sandbox.flux.io"}, Resources: []string{"sandboxes"}, Verbs: []string{"get", "list", "watch", "create", "delete"}},
			},
		},
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: ClusterRoleName},
			Subjects:   []rbacv1.Subject{{Kind: "ServiceAccount", Name: ServiceAccountName, Namespace: cfg.Namespace}},
			RoleRef:    rbacv1.RoleRef{Kind: "ClusterRole", Name: ClusterRoleName, APIGroup: "rbac.authorization.k8s.io"},
		},
		buildAgentPod(cfg),
		buildWatcherDaemonSet(cfg),
	}

	for _, obj := range objects {
		if err := k8s.Client.Create(ctx, obj); err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("deploy %s %q: %w", obj.GetObjectKind().GroupVersionKind().Kind, obj.GetName(), err)
		}
	}
	return nil
}

// CleanupComponents deletes the e2e-deployed components.
func CleanupComponents(k8s *K8s) error {
	ctx := context.Background()
	var errs []error

	for _, obj := range []client.Object{
		&appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Name: WatcherDaemonSetName, Namespace: k8s.Cfg.Namespace}},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: AgentPodName, Namespace: k8s.Cfg.Namespace}},
		&rbacv1.ClusterRoleBinding{ObjectMeta: metav1.ObjectMeta{Name: ClusterRoleName}},
		&rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: ClusterRoleName}},
		&corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: ServiceAccountName, Namespace: k8s.Cfg.Namespace}},
	} {
		if err := k8s.Client.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}

// buildAgentPod builds the fixed-name bare agent pod used by the e2e suite.
// The agent talks to the node containerd, injects execd into every sandbox,
// and exposes its HTTP control API on the configured port.
func buildAgentPod(cfg *Config) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      AgentPodName,
			Namespace: cfg.Namespace,
			Labels:    map[string]string{"app": AgentPodName},
		},
		Spec: agentPodSpec(cfg),
	}
}

// agentPodSpec builds the agent pod spec: node-level containerd access and
// execd injection for every sandbox it creates.
func agentPodSpec(cfg *Config) corev1.PodSpec {
	return agentPodSpecWithCapacity(cfg, "10")
}

// agentPodSpecWithCapacity builds an agent pod spec with an explicit sandbox
// capacity (used by the suite-managed small agent with capacity=2).
func agentPodSpecWithCapacity(cfg *Config, capacity string) corev1.PodSpec {
	return corev1.PodSpec{
		ServiceAccountName: ServiceAccountName,
		HostPID:            true,
		InitContainers:     []corev1.Container{injectExecdInitContainer(cfg)},
		Containers:         []corev1.Container{agentContainer(cfg, capacity)},
		Volumes:            agentVolumes(),
	}
}

// injectExecdInitContainer copies the execd binary onto the hostPath volume
// shared with the sandboxes.
func injectExecdInitContainer(cfg *Config) corev1.Container {
	return corev1.Container{
		Name:            "inject-execd",
		Image:           cfg.ExecdImage,
		ImagePullPolicy: corev1.PullIfNotPresent,
		Command:         []string{"/bin/sh", "-c"},
		Args: []string{
			"mkdir -p /opt/opensandbox && cp /execd /opt/opensandbox/execd && cp /bootstrap.sh /opt/opensandbox/bootstrap.sh && chmod +x /opt/opensandbox/execd /opt/opensandbox/bootstrap.sh",
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "execd-bin", MountPath: "/opt/opensandbox"},
		},
	}
}

// agentContainer builds the agent container with the given sandbox capacity.
func agentContainer(cfg *Config, capacity string) corev1.Container {
	privileged := true
	return corev1.Container{
		Name:            "agent",
		Image:           cfg.AgentImage,
		ImagePullPolicy: corev1.PullIfNotPresent,
		SecurityContext: &corev1.SecurityContext{Privileged: &privileged},
		Env:             agentEnvVars(cfg, capacity),
		Ports:           []corev1.ContainerPort{{Name: "http", ContainerPort: int32(cfg.AgentPort)}},
		ReadinessProbe:  agentReadinessProbe(cfg),
		VolumeMounts:    agentVolumeMounts(),
	}
}

// agentEnvVars builds the agent container environment.
func agentEnvVars(cfg *Config, capacity string) []corev1.EnvVar {
	return []corev1.EnvVar{
		EnvFromField("NODE_NAME", "spec.nodeName"),
		EnvFromField("POD_NAME", "metadata.name"),
		EnvFromField("POD_IP", "status.podIP"),
		EnvFromField("POD_UID", "metadata.uid"),
		EnvFromField("POD_NAMESPACE", "metadata.namespace"),
		{Name: "RUNTIME_TYPE", Value: "containerd"},
		{Name: "RUNTIME_SOCKET", Value: "/run/containerd/containerd.sock"},
		{Name: "EXECD_BIN_PATH", Value: "/opt/opensandbox"},
		{Name: "FLUXSANDBOXAGENT_CONTROL_PORT", Value: fmt.Sprintf(":%d", cfg.AgentPort)},
		{Name: "FLUXSANDBOXAGENT_CAPACITY", Value: capacity},
	}
}

// agentReadinessProbe probes the agent /healthz endpoint.
func agentReadinessProbe(cfg *Config) *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromInt(cfg.AgentPort)},
		},
		InitialDelaySeconds: 3, PeriodSeconds: 5,
	}
}

// agentVolumeMounts builds the agent container volume mounts.
func agentVolumeMounts() []corev1.VolumeMount {
	return []corev1.VolumeMount{
		{Name: "containerd-socket", MountPath: "/run/containerd/containerd.sock"},
		{Name: "containerd-data", MountPath: "/var/lib/containerd"},
		{Name: "containerd-run", MountPath: "/run/containerd"},
		{Name: "log-dir", MountPath: "/var/log/flux-sandbox"},
		{Name: "execd-bin", MountPath: "/opt/opensandbox", ReadOnly: true},
		// The agent image is built FROM scratch and has no /tmp, while the
		// containerd client creates in-process temp mounts under os.TempDir()
		// during container creation. Mount an emptyDir at /tmp to restore the
		// behavior of the previous distroless-based image.
		{Name: "tmp", MountPath: "/tmp"},
	}
}

// agentVolumes builds the hostPath volumes backing the agent container.
func agentVolumes() []corev1.Volume {
	return []corev1.Volume{
		hostPathVolume("containerd-socket", "/run/containerd/containerd.sock", corev1.HostPathSocket),
		hostPathVolume("containerd-data", "/var/lib/containerd", corev1.HostPathDirectory),
		hostPathVolume("containerd-run", "/run/containerd", corev1.HostPathDirectory),
		hostPathVolume("log-dir", "/var/log/flux-sandbox", corev1.HostPathDirectoryOrCreate),
		hostPathVolume("execd-bin", "/opt/opensandbox", corev1.HostPathDirectoryOrCreate),
		{Name: "tmp", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	}
}

// buildWatcherDaemonSet builds the watcher DaemonSet with e2e-friendly timing
// flags (short grace period and scan interval).
func buildWatcherDaemonSet(cfg *Config) *appsv1.DaemonSet {
	labels := map[string]string{"app": "sandboxwatcher"}
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      WatcherDaemonSetName,
			Namespace: cfg.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec:       watcherPodSpec(cfg),
			},
		},
	}
}

// watcherPodSpec builds the watcher pod spec: node-level containerd access
// with e2e-tuned timing flags.
func watcherPodSpec(cfg *Config) corev1.PodSpec {
	privileged := true
	return corev1.PodSpec{
		ServiceAccountName: ServiceAccountName,
		HostPID:            true,
		Containers: []corev1.Container{
			{
				Name:            "sandbox-watcher",
				Image:           cfg.WatcherImage,
				ImagePullPolicy: corev1.PullIfNotPresent,
				SecurityContext: &corev1.SecurityContext{Privileged: &privileged},
				Args: []string{
					"--containerd-socket=/run/containerd/containerd.sock",
					"--cri-socket=",
					"--orphan-timeout=" + OrphanTimeout.String(),
					"--orphan-grace-period=" + cfg.GracePeriod.String(),
					"--scan-interval=" + cfg.ScanInterval.String(),
				},
				Env: []corev1.EnvVar{
					EnvFromField("NODE_NAME", "spec.nodeName"),
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "containerd-socket", MountPath: "/run/containerd/containerd.sock"},
					{Name: "containerd-run", MountPath: "/run/containerd"},
					{Name: "fifo-dir", MountPath: "/run/containerd/fifo"},
				},
			},
		},
		Volumes: []corev1.Volume{
			hostPathVolume("containerd-socket", "/run/containerd/containerd.sock", corev1.HostPathSocket),
			hostPathVolume("containerd-run", "/run/containerd", corev1.HostPathDirectory),
			hostPathVolume("fifo-dir", "/run/containerd/fifo", corev1.HostPathDirectoryOrCreate),
		},
	}
}

// hostPathVolume builds a named hostPath volume of the given type.
func hostPathVolume(name, path string, typ corev1.HostPathType) corev1.Volume {
	return corev1.Volume{
		Name: name,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{Path: path, Type: &typ},
		},
	}
}

// EnvFromField builds an env var sourced from a downward-API field path.
func EnvFromField(name, fieldPath string) corev1.EnvVar {
	return corev1.EnvVar{
		Name: name,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
		},
	}
}

// SmallAgentPodName is the fixed name of the suite-managed capacity=2 small
// agent pod used by DP-E2E-005/029.
const SmallAgentPodName = "flux-sandbox-agent-small"

// SmallAgentPodLabel selects the suite-managed small agent pod. It differs
// from AgentPodLabel so the main-agent discovery never picks the small pod.
const SmallAgentPodLabel = "app=flux-sandbox-agent-small"

// SmallAgentLocalPort is the local port of the kubectl port-forward to the
// small agent control port; the test machine reaches the small agent through
// it via E2E_SMALL_CAPACITY_AGENT_URL.
const SmallAgentLocalPort = 17508

// DeploySmallAgentPod creates the suite-managed capacity=2 small agent pod
// (create-only, AlreadyExists tolerated) so DP-E2E-005/029 can run without
// an externally deployed agent.
func DeploySmallAgentPod(k8s *K8s) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pod := buildSmallAgentPod(k8s.Cfg)
	if err := k8s.Client.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create small agent pod %s: %w", SmallAgentPodName, err)
	}
	return nil
}

// CleanupSmallAgentPod deletes the suite-managed small agent pod.
func CleanupSmallAgentPod(k8s *K8s) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: SmallAgentPodName, Namespace: k8s.Cfg.Namespace}}
	if err := k8s.Client.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("delete small agent pod %s: %w", SmallAgentPodName, err)
	}
	return nil
}

// buildSmallAgentPod builds the fixed-name capacity=2 small agent pod. It
// carries a dedicated label for suite discovery (the watcher detects orphan
// sandboxes by agent-pod UID lookup, so the label choice does not affect
// orphan handling). Access from the test machine goes through a local
// kubectl port-forward, so no node port is needed.
func buildSmallAgentPod(cfg *Config) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      SmallAgentPodName,
			Namespace: cfg.Namespace,
			Labels:    map[string]string{"app": "flux-sandbox-agent-small"},
		},
		Spec: agentPodSpecWithCapacity(cfg, "2"),
	}
}
