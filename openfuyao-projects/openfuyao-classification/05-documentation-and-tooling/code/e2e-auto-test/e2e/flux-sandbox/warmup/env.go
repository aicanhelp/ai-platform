//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package warmup contains end-to-end specs for the snapshot warm-up subsystem
// embedded in the flux-sandbox controller (WARM-E2E-001~029, see
// warmup-test.md). It attaches to an already-deployed controller and creates
// its own test dependencies: PostgreSQL, a fake MoonCake etcd, and a
// programmable fake E2B API-Server that also simulates MoonCakeStore reports.
package warmup

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	ginkgo "github.com/onsi/ginkgo/v2"
	clientv3 "go.etcd.io/etcd/client/v3"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
)

// Environment-driven configuration (WARM_E2E_* to avoid clashing with the
// dataplane/controlplane suites).
var (
	wKubeconfig   = envOrDefault("E2E_KUBECONFIG", defaultKubeconfigPath())
	wNamespace    = envOrDefault("WARM_E2E_NAMESPACE", "flux-warm-e2e")
	wControllerNS = envOrDefault("WARM_E2E_CONTROLLER_NAMESPACE", defaultControllerNamespace())
	wFakeE2BIm    = envOrDefault("WARM_E2E_FAKE_E2B_IMAGE", "flux-sandbox/fake-e2b:latest")
	wPGImage      = envOrDefault("WARM_E2E_PG_IMAGE", "postgres:16-alpine")
	wEtcdImage    = envOrDefault("WARM_E2E_ETCD_IMAGE", "quay.io/coreos/etcd:v3.5.10")
	wSkipDeploy   = os.Getenv("E2E_SKIP_DEPLOY") != ""

	// Static-PV fallback for clusters without a default StorageClass: the
	// postgres PVC would otherwise stay Pending forever and the suite would
	// time out in BeforeSuite. The PV is cluster-scoped, so it survives the
	// AfterSuite namespace deletion and every next run re-binds it.
	wPGPVName  = envOrDefault("WARM_E2E_PG_PV_NAME", "warm-postgres-static-pv")
	wPGPVPath  = envOrDefault("WARM_E2E_PG_HOSTPATH", "/var/lib/warm-e2e-pv")
	wPGPVSize  = "2Gi"
	wPGPVCName = "warm-postgres-data"

	// SuperPod topology used across the suite. Capacities are distinct and
	// strictly descending so load-balancing cases (WARM-E2E-008) get a
	// deterministic ordering.
	wSuperPodIDs   = []string{"superpod1", "superpod2", "superpod3"}
	wSuperPodSpecs = []string{"superpod1:4000", "superpod2:3000", "superpod3:2000"}

	// Warm timings as deployed. Retries use the controller defaults
	// (WarmRetryMax=3, WarmRetryInterval=10s); the stale timeout and etcd
	// staleness are shortened to keep WARM-E2E-007/020 tractable.
	wPollInterval = 1 * time.Second
	wStaleTimeout = 30 * time.Second
	wStaleness    = 10 * time.Second

	// Effective retry behavior (controller defaults; no flags exist for
	// these). Used by specs that compute wait windows.
	wRetryMax      = 3
	wRetryAttempts = wRetryMax + 1 // 4 total attempts
	wRetryInterval = 10 * time.Second
)

// Local port-forward ports (distinct from dataplane 5758 and controlplane
// 18080/19091).
const (
	wCtrlPort = 18085 // controller HTTP (warm API)
	wPGPort   = 15432 // PostgreSQL
	wEtcdPort = 12379 // fake MoonCake etcd
	wE2BAdmin = 12380 // fake E2B admin API
)

// wPGDSN addresses the warm database through the postgres port-forward.
const wPGDSN = "postgres://postgres:warm@127.0.0.1:15432/warm?sslmode=disable"

// wEnv is the suite environment, initialized in BeforeSuite.
var wEnv *warmEnv

type warmEnv struct {
	ctx        context.Context
	kube       *kubernetes.Clientset
	crClient   client.Client
	ctrlPF     *framework.PortForward
	pgPF       *framework.PortForward
	etcdPF     *framework.PortForward
	e2bPF      *framework.PortForward
	etcdClient *clientv3.Client
	dbConn     *warmDB

	// origController captures the controller Deployment as found at attach
	// time so teardown can restore the pre-run configuration.
	origController *appsv1.Deployment

	// staticPVCreated records that ensurePostgresStorage fell back to the
	// hostPath static PV, so teardown can remove it and leave no residue.
	staticPVCreated bool
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func defaultKubeconfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kube", "config")
}

// suiteRoot locates the warmup suite directory so the suite can read its
// testdata fixtures regardless of the test working directory.
func suiteRoot(tb ginkgo.GinkgoTInterface) string {
	tb.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("cannot locate source file")
	}
	return filepath.Dir(thisFile)
}

// setupWarmEnv deploys the full warm stack and establishes port-forwards. It
// runs once in BeforeSuite.
func setupWarmEnv(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	ctx := context.Background()

	config, err := clientcmd.BuildConfigFromFlags("", wKubeconfig)
	if err != nil {
		tb.Fatalf("build kubeconfig: %v", err)
	}
	kube, err := kubernetes.NewForConfig(config)
	if err != nil {
		tb.Fatalf("create clientset: %v", err)
	}
	scheme := k8sruntime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = appsv1.AddToScheme(scheme)
	crClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		tb.Fatalf("create controller-runtime client: %v", err)
	}

	e := &warmEnv{ctx: ctx, kube: kube, crClient: crClient}
	// Register the environment early so a failed setup still runs the
	// AfterSuite teardown (controller restore + mock namespace cleanup).
	wEnv = e

	if !wSkipDeploy {
		e.deployNamespace(tb)
		e.ensurePostgresStorage(tb)
		e.deployPostgres(tb)
		e.deployEtcd(tb)
		e.deployFakeE2B(tb)
		e.runMigrations(tb)
		e.attachController(tb)
	} else {
		tb.Log("E2E_SKIP_DEPLOY set, assuming warm stack already deployed")
	}

	e.startControllerPortForward(tb)
	e.startPGPortForward(tb)
	e.startEtcdPortForward(tb)
	e.startE2BAdminPortForward(tb)

	// Cancel any non-terminal tasks a killed previous run left behind, so
	// their workers do not interfere with this run.
	e.db(tb).purgeLeftoverTasks(tb)

	etcdCli, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{fmt.Sprintf("127.0.0.1:%d", wEtcdPort)},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		tb.Fatalf("connect etcd via port-forward: %v", err)
	}
	e.etcdClient = etcdCli
}

// teardownWarmEnv releases port-forwards and the etcd client (AfterSuite).
func teardownWarmEnv() {
	if wEnv == nil {
		return
	}
	tb := ginkgo.GinkgoT()
	// Delete runs last via defer; restore first so the controller never
	// references the mock services after the suite exits.
	defer wEnv.deleteMockNamespace(tb)
	wEnv.restoreController(tb)
	if wEnv.etcdClient != nil {
		_ = wEnv.etcdClient.Close()
	}
	if wEnv.ctrlPF != nil {
		wEnv.ctrlPF.Stop()
	}
	if wEnv.pgPF != nil {
		wEnv.pgPF.Stop()
	}
	if wEnv.etcdPF != nil {
		wEnv.etcdPF.Stop()
	}
	if wEnv.e2bPF != nil {
		wEnv.e2bPF.Stop()
	}
}

// restartControllerPortForward drops and re-establishes the controller
// port-forward (used after deleting the controller pod).
func (e *warmEnv) restartControllerPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if e.ctrlPF != nil {
		e.ctrlPF.Stop()
	}
	e.startControllerPortForward(tb)
}

// restartPGPortForward drops and re-establishes the PostgreSQL port-forward
// (needed after the outage specs replace the warm-postgres pod).
func (e *warmEnv) restartPGPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if e.pgPF != nil {
		e.pgPF.Stop()
	}
	e.startPGPortForward(tb)
}

// reconnectEtcd re-establishes the etcd port-forward and client (needed after
// the outage specs replace the warm-etcd pod).
func (e *warmEnv) reconnectEtcd(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if e.etcdPF != nil {
		e.etcdPF.Stop()
	}
	e.startEtcdPortForward(tb)
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{fmt.Sprintf("127.0.0.1:%d", wEtcdPort)},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		tb.Fatalf("reconnect etcd via port-forward: %v", err)
	}
	if e.etcdClient != nil {
		_ = e.etcdClient.Close()
	}
	e.etcdClient = cli
}

// setControllerArgs replaces the controller container args. With wait=true
// it blocks until the deployment is available again and re-establishes the
// controller port-forward; fail-fast scenarios pass wait=false because the
// new args must crash the pod.
func (e *warmEnv) setControllerArgs(tb ginkgo.GinkgoTInterface, args []string, wait bool) {
	tb.Helper()
	dep, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Fatalf("get controller deployment %s/%s: %v", wControllerNS, wControllerDeployName, err)
	}
	dep.Spec.Template.Spec.Containers[0].Args = args
	if _, err := e.kube.AppsV1().Deployments(wControllerNS).Update(e.ctx, dep, metav1.UpdateOptions{}); err != nil {
		tb.Fatalf("update controller args: %v", err)
	}
	if wait {
		e.waitForControllerAvailable(tb, 5*time.Minute)
		e.restartControllerPortForward(tb)
	}
}

// restoreControllerArgs restores previously captured controller args and
// waits for the deployment to become available again. Intended for
// DeferCleanup: restore failures are logged, not fatal, so the remaining
// cleanups still run.
func (e *warmEnv) restoreControllerArgs(tb ginkgo.GinkgoTInterface, origArgs []string) {
	tb.Helper()
	dep, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Logf("restore controller args: %v", err)
		return
	}
	dep.Spec.Template.Spec.Containers[0].Args = origArgs
	if _, err := e.kube.AppsV1().Deployments(wControllerNS).Update(e.ctx, dep, metav1.UpdateOptions{}); err != nil {
		tb.Logf("restore controller args: %v", err)
		return
	}
	e.waitForControllerAvailable(tb, 5*time.Minute)
	e.restartControllerPortForward(tb)
}

// healthzReady returns a port-forward readiness probe that GETs /healthz on
// the forwarded local port. Shared by the controller and fake-e2b forwards.
func healthzReady(localPort int) func() error {
	return func() error {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/healthz", localPort))
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("healthz status %d", resp.StatusCode)
		}
		return nil
	}
}

func (e *warmEnv) startControllerPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	ns, sel := e.controllerPodSelector(tb)
	pod := e.waitForPod(tb, ns, sel, 5*time.Minute)
	pf, err := framework.StartPortForward(ns, pod, wCtrlPort, 8080, healthzReady(wCtrlPort))
	if err != nil {
		tb.Fatalf("port-forward controller: %v", err)
	}
	e.ctrlPF = pf
}

func (e *warmEnv) startPGPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	pod := e.waitForPod(tb, wNamespace, "app=warm-postgres", 5*time.Minute)
	pf, err := framework.StartPortForward(wNamespace, pod, wPGPort, 5432, nil)
	if err != nil {
		tb.Fatalf("port-forward postgres: %v", err)
	}
	e.pgPF = pf
}

func (e *warmEnv) startEtcdPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	pod := e.waitForPod(tb, wNamespace, "app=warm-etcd", 5*time.Minute)
	pf, err := framework.StartPortForward(wNamespace, pod, wEtcdPort, 2379, nil)
	if err != nil {
		tb.Fatalf("port-forward etcd: %v", err)
	}
	e.etcdPF = pf
}

func (e *warmEnv) startE2BAdminPortForward(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	pod := e.waitForPod(tb, wNamespace, "app=fake-e2b", 5*time.Minute)
	pf, err := framework.StartPortForward(wNamespace, pod, wE2BAdmin, 8080, healthzReady(wE2BAdmin))
	if err != nil {
		tb.Fatalf("port-forward fake-e2b: %v", err)
	}
	e.e2bPF = pf
}

// ─── deployment helpers ───

func (e *warmEnv) deployNamespace(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	_, err := e.kube.CoreV1().Namespaces().Get(e.ctx, wNamespace, metav1.GetOptions{})
	if err == nil {
		return
	}
	_, err = e.kube.CoreV1().Namespaces().Create(e.ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: wNamespace},
	}, metav1.CreateOptions{})
	if err != nil {
		tb.Fatalf("create namespace: %v", err)
	}
}

// defaultStorageClassExists reports whether the cluster has any StorageClass
// (a bare-PVC provisioner exists, so a plain PVC can bind).
func (e *warmEnv) defaultStorageClassExists(tb ginkgo.GinkgoTInterface) bool {
	tb.Helper()
	scs, err := e.kube.StorageV1().StorageClasses().List(e.ctx, metav1.ListOptions{})
	if err != nil {
		tb.Logf("list storage classes: %v (assuming one exists)", err)
		return true
	}
	return len(scs.Items) > 0
}

// ensurePostgresStorage guarantees warm-postgres-data can bind before the
// postgres Deployment is created. On clusters with a StorageClass it is a
// no-op (the upsertPVC in deployPostgres then auto-provisions). Without one,
// a Retain hostPath PV is created and pre-bound to the PVC. Leftovers from a
// previous run are cleared PVC-first (PVs block in Terminating while still
// bound), and runMigrations truncates the tables so every run starts clean.
func (e *warmEnv) ensurePostgresStorage(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if e.defaultStorageClassExists(tb) {
		return
	}
	tb.Logf("no StorageClass found; falling back to static hostPath PV %s (%s)", wPGPVName, wPGPVPath)

	// Clear leftovers from a previous run: the PVC must go first — a PV that
	// is still bound (or Released with a claimRef) blocks in Terminating and
	// never disappears, which used to time the suite out here.
	err := e.kube.CoreV1().PersistentVolumeClaims(wNamespace).Delete(e.ctx, wPGPVCName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		tb.Fatalf("delete stale pvc %s: %v", wPGPVCName, err)
	}
	waitFor(tb, "stale PVC deleted", 2*time.Minute, func() bool {
		_, err := e.kube.CoreV1().PersistentVolumeClaims(wNamespace).Get(e.ctx, wPGPVCName, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	})
	_ = e.kube.CoreV1().PersistentVolumes().Delete(e.ctx, wPGPVName, metav1.DeleteOptions{})
	waitFor(tb, "stale PV deleted", 2*time.Minute, func() bool {
		_, err := e.kube.CoreV1().PersistentVolumes().Get(e.ctx, wPGPVName, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	})

	pv := &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: wPGPVName,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "warm-e2e-suite",
			},
		},
		Spec: corev1.PersistentVolumeSpec{
			Capacity:                      corev1.ResourceList{corev1.ResourceStorage: resource.MustParse(wPGPVSize)},
			AccessModes:                   []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: wPGPVPath,
					Type: ptr(corev1.HostPathDirectoryOrCreate),
				},
			},
		},
	}
	if _, err := e.kube.CoreV1().PersistentVolumes().Create(e.ctx, pv, metav1.CreateOptions{}); err != nil {
		tb.Fatalf("create static PV %s: %v", wPGPVName, err)
	}
	e.staticPVCreated = true

	// Pre-bind the PVC: upsertPVC skips when it already exists (also with a
	// stale binding, which would otherwise point at a dead PV).
	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: wPGPVCName, Namespace: wNamespace},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1Gi")},
			},
			VolumeName: wPGPVName,
		},
	}
	if _, err := e.kube.CoreV1().PersistentVolumeClaims(wNamespace).Create(e.ctx, pvc, metav1.CreateOptions{}); err != nil {
		tb.Fatalf("create pre-bound pvc %s: %v", wPGPVCName, err)
	}
	waitFor(tb, fmt.Sprintf("PVC %s bound", wPGPVCName), 2*time.Minute, func() bool {
		got, err := e.kube.CoreV1().PersistentVolumeClaims(wNamespace).Get(e.ctx, wPGPVCName, metav1.GetOptions{})
		return err == nil && got.Status.Phase == corev1.ClaimBound
	})
	tb.Logf("PVC %s bound to hostPath PV %s", wPGPVCName, wPGPVName)
}

func (e *warmEnv) waitForPod(tb ginkgo.GinkgoTInterface, ns, labelSelector string, timeout time.Duration) string {
	tb.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := e.kube.CoreV1().Pods(ns).List(e.ctx, metav1.ListOptions{LabelSelector: labelSelector})
		if err != nil {
			tb.Fatalf("list pods %s: %v", labelSelector, err)
		}
		// Prefer the newest ready pod and skip terminating pods: during a
		// rollout (arg swaps in the node-backend / fail-fast specs) both the
		// old and new pods can be Ready, and a port-forward aimed at the old
		// one dies as soon as it terminates.
		var best string
		var bestCreated metav1.Time
		for _, p := range pods.Items {
			if p.DeletionTimestamp != nil {
				continue
			}
			ready := false
			for _, c := range p.Status.ContainerStatuses {
				if c.Ready && c.State.Running != nil {
					ready = true
					break
				}
			}
			if !ready {
				continue
			}
			if best == "" || p.CreationTimestamp.After(bestCreated.Time) {
				best = p.Name
				bestCreated = p.CreationTimestamp
			}
		}
		if best != "" {
			return best
		}
		time.Sleep(2 * time.Second)
	}
	tb.Fatalf("timed out waiting for pod %s in %s", labelSelector, ns)
	return ""
}

// upsert applies obj if it does not exist yet (create-only, mirroring the
// dataplane deploy convention).
func (e *warmEnv) upsert(tb ginkgo.GinkgoTInterface, obj client.Object, name string) {
	tb.Helper()
	err := e.crClient.Get(e.ctx, client.ObjectKey{Namespace: wNamespace, Name: name}, obj)
	if err == nil {
		return // already deployed
	}
	if err := e.crClient.Create(e.ctx, obj); err != nil {
		tb.Fatalf("create %s %s: %v", obj.GetObjectKind().GroupVersionKind().Kind, name, err)
	}
}

func (e *warmEnv) deployPostgres(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	labels := map[string]string{"app": "warm-postgres"}
	e.upsert(tb, &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "warm-postgres", Namespace: wNamespace, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr(int32(1)),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "postgres",
						Image: wPGImage,
						Env: []corev1.EnvVar{
							{Name: "PGDATA", Value: "/var/lib/postgresql/data/pgdata"},
							{Name: "POSTGRES_PASSWORD", Value: "warm"},
							{Name: "POSTGRES_DB", Value: "warm"},
						},
						Ports:           []corev1.ContainerPort{{ContainerPort: 5432}},
						ReadinessProbe:  pgProbe(),
						ImagePullPolicy: corev1.PullIfNotPresent,
						VolumeMounts: []corev1.VolumeMount{{
							// Persistent data: outage specs scale the
							// Deployment to 0 and back; without a PVC the
							// recreated pod would lose the warm tables.
							Name:      "pgdata",
							MountPath: "/var/lib/postgresql/data",
						}},
					}},
					Volumes: []corev1.Volume{{
						Name: "pgdata",
						VolumeSource: corev1.VolumeSource{
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
								ClaimName: "warm-postgres-data",
							},
						},
					}},
				},
			},
		},
	}, "warm-postgres")
	e.upsertPVC(tb, "warm-postgres-data", "1Gi")
	e.upsertService(tb, "warm-postgres", 5432, labels)
}

func (e *warmEnv) upsertPVC(tb ginkgo.GinkgoTInterface, name, size string) {
	tb.Helper()
	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: wNamespace},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse(size),
				},
			},
		},
	}
	err := e.crClient.Get(e.ctx, client.ObjectKey{Namespace: wNamespace, Name: name}, pvc)
	if err == nil {
		return
	}
	if err := e.crClient.Create(e.ctx, pvc); err != nil {
		tb.Fatalf("create pvc %s: %v", name, err)
	}
}

func pgProbe() *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: []string{"pg_isready", "-U", "postgres"}},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       5,
	}
}

func (e *warmEnv) deployEtcd(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	labels := map[string]string{"app": "warm-etcd"}
	e.upsert(tb, &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "warm-etcd", Namespace: wNamespace, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr(int32(1)),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "etcd",
						Image: wEtcdImage,
						Args: []string{
							"etcd",
							"--name", "warm-etcd",
							"--listen-client-urls", "http://0.0.0.0:2379",
							"--advertise-client-urls", "http://0.0.0.0:2379",
							"--listen-peer-urls", "http://0.0.0.0:2380",
							"--initial-advertise-peer-urls", "http://0.0.0.0:2380",
							"--initial-cluster", "warm-etcd=http://0.0.0.0:2380",
						},
						Ports:           []corev1.ContainerPort{{ContainerPort: 2379}},
						ReadinessProbe:  etcdProbe(),
						ImagePullPolicy: corev1.PullIfNotPresent,
					}},
				},
			},
		},
	}, "warm-etcd")
	e.upsertService(tb, "warm-etcd", 2379, labels)
}

func etcdProbe() *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt32(2379)},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       5,
	}
}

func (e *warmEnv) deployFakeE2B(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	labels := map[string]string{"app": "fake-e2b"}
	e.upsert(tb, &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fake-e2b", Namespace: wNamespace, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr(int32(1)),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "fake-e2b",
						Image: wFakeE2BIm,
						Env: []corev1.EnvVar{
							{Name: "FAKE_E2B_PORT", Value: "8080"},
							{Name: "FAKE_ETCD_ENDPOINTS", Value: "http://warm-etcd:2379"},
							{Name: "FAKE_SUPERPODS", Value: strings.Join(wSuperPodSpecs, ",")},
							{Name: "FAKE_REPORT_INTERVAL", Value: "3s"},
							{Name: "FAKE_DISPATCH_DELAY", Value: "2s"},
						},
						Ports:           []corev1.ContainerPort{{ContainerPort: 8080}},
						ReadinessProbe:  e2bProbe(),
						ImagePullPolicy: corev1.PullIfNotPresent,
					}},
				},
			},
		},
	}, "fake-e2b")
	e.upsertService(tb, "fake-e2b", 8080, labels)
}

func e2bProbe() *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromInt32(8080)},
		},
		InitialDelaySeconds: 3,
		PeriodSeconds:       5,
	}
}

func (e *warmEnv) upsertService(tb ginkgo.GinkgoTInterface, name string, port int32, labels map[string]string) {
	tb.Helper()
	ns := wNamespace
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns, Labels: labels},
		Spec: corev1.ServiceSpec{
			Selector: labels,
			Ports:    []corev1.ServicePort{{Port: port, TargetPort: intstr.FromInt32(port)}},
		},
	}
	err := e.crClient.Get(e.ctx, client.ObjectKey{Namespace: ns, Name: name}, svc)
	if err == nil {
		return
	}
	if err := e.crClient.Create(e.ctx, svc); err != nil {
		tb.Fatalf("create service %s: %v", name, err)
	}
}

// runMigrations applies config/controller/warm-migrations/0001_warm.sql via
// psql exec in the postgres pod.
func (e *warmEnv) runMigrations(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	pod := e.waitForPod(tb, wNamespace, "app=warm-postgres", 3*time.Minute)
	sql, err := os.ReadFile(filepath.Join(suiteRoot(tb),
		"testdata/config/controller/warm-migrations/0001_warm.sql"))
	if err != nil {
		tb.Fatalf("read migration: %v", err)
	}
	cmd := exec.Command("kubectl", "--kubeconfig", wKubeconfig, "exec", "-i", "-n", wNamespace, pod, "--",
		"psql", "-U", "postgres", "-d", "warm", "-v", "ON_ERROR_STOP=1")
	cmd.Stdin = strings.NewReader(string(sql))
	if out, err := cmd.CombinedOutput(); err != nil {
		tb.Fatalf("run migration: %v\n%s", err, out)
	}
	// The static-PV fallback reuses the hostPath across runs, so wipe the
	// warm tables: every run must start from an empty database.
	trunc := "TRUNCATE TABLE snapshot_warm_task, superpod_warm_detail CASCADE;"
	trim := exec.Command("kubectl", "--kubeconfig", wKubeconfig, "exec", "-i", "-n", wNamespace, pod, "--",
		"psql", "-U", "postgres", "-d", "warm", "-v", "ON_ERROR_STOP=1", "-c", trunc)
	if out, err := trim.CombinedOutput(); err != nil {
		tb.Fatalf("truncate warm tables: %v\n%s", err, out)
	}
	tb.Log("warm migration applied, tables truncated")
}

// ─── warm API HTTP helpers (via the controller port-forward) ───

type warmAPIError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

type errorEnvelope struct {
	Error warmAPIError `json:"error"`
}

type createTaskResponse struct {
	TaskID             string `json:"task_id"`
	State              string `json:"state"`
	AcceptedSnapshotCt int    `json:"accepted_snapshot_count"`
	CreatedAt          string `json:"created_at"`
}

// wHTTPClient bounds every warm API call so a hung handler fails the spec
// quickly instead of deadlocking the suite (mirrors the controller-side
// etcdQueryTimeout fix).
var wHTTPClient = &http.Client{Timeout: 30 * time.Second}

func (e *warmEnv) httpDo(tb ginkgo.GinkgoTInterface, method, path string, body any) (*http.Response, []byte) {
	tb.Helper()
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			tb.Fatalf("marshal body: %v", err)
		}
		rdr = strings.NewReader(string(b))
	}
	req, err := http.NewRequest(method, wURL(path), rdr)
	if err != nil {
		tb.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return e.sendHTTP(tb, req, method, path)
}

// wURL builds a controller API URL.
func wURL(path string) string {
	return fmt.Sprintf("http://127.0.0.1:%d%s", wCtrlPort, path)
}

// sendHTTP performs the request on the shared client and drains the body.
// Shared by httpDo and httpDoRaw.
func (e *warmEnv) sendHTTP(tb ginkgo.GinkgoTInterface, req *http.Request, method, path string) (*http.Response, []byte) {
	tb.Helper()
	resp, err := wHTTPClient.Do(req)
	if err != nil {
		tb.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp, raw
}

// httpDoRaw sends a raw string body (for malformed-JSON cases).
func (e *warmEnv) httpDoRaw(tb ginkgo.GinkgoTInterface, method, path, body string) (*http.Response, []byte) {
	tb.Helper()
	req, err := http.NewRequest(method, wURL(path), strings.NewReader(body))
	if err != nil {
		tb.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return e.sendHTTP(tb, req, method, path)
}

// createWarmTask posts a warm task creation request and returns the decoded
// response.
func (e *warmEnv) createWarmTask(tb ginkgo.GinkgoTInterface, snapshotIDs []string) (createTaskResponse, int, []byte) {
	tb.Helper()
	resp, raw := e.httpDo(tb, http.MethodPost, "/api/v1/snapshot-warm-tasks", map[string]any{"snapshot_ids": snapshotIDs})
	var out createTaskResponse
	if resp.StatusCode/100 == 2 {
		if err := json.Unmarshal(raw, &out); err != nil {
			tb.Fatalf("decode create response %s: %v", raw, err)
		}
	}
	return out, resp.StatusCode, raw
}

func decodeError(tb ginkgo.GinkgoTInterface, raw []byte) warmAPIError {
	tb.Helper()
	var env errorEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		tb.Fatalf("decode error body %s: %v", raw, err)
	}
	return env.Error
}

func ptr[T any](v T) *T { return &v }

// ─── fake E2B admin helpers (via the fake-e2b port-forward) ───

type fakeE2BRequest struct {
	SnapshotID       string `json:"snapshot_id"`
	WarmedSnapshotID string `json:"warmed_snapshot_id"`
	TargetSuperPodID string `json:"target_superpod_id"`
	At               string `json:"at"`
}

type fakeE2BRequests struct {
	Requests []fakeE2BRequest `json:"requests"`
}

func (e *warmEnv) e2bAdmin(tb ginkgo.GinkgoTInterface, method, path string, body any) []byte {
	tb.Helper()
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			tb.Fatalf("marshal body: %v", err)
		}
		rdr = strings.NewReader(string(b))
	}
	req, err := http.NewRequest(method, fmt.Sprintf("http://127.0.0.1:%d%s", wE2BAdmin, path), rdr)
	if err != nil {
		tb.Fatalf("build admin request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		tb.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		tb.Fatalf("%s %s: status %d, body %s", method, path, resp.StatusCode, raw)
	}
	return raw
}

func (e *warmEnv) e2bRequests(tb ginkgo.GinkgoTInterface) fakeE2BRequests {
	tb.Helper()
	raw := e.e2bAdmin(tb, http.MethodGet, "/_admin/requests", nil)
	var out fakeE2BRequests
	if err := json.Unmarshal(raw, &out); err != nil {
		tb.Fatalf("decode requests: %v", err)
	}
	return out
}

func (e *warmEnv) e2bClearRequests(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	e.e2bAdmin(tb, http.MethodDelete, "/_admin/requests", nil)
}

// e2bInjectFault arms/disarms a fault rule for warmed_snapshot_id
// (mode: always | once | off).
func (e *warmEnv) e2bInjectFault(tb ginkgo.GinkgoTInterface, warmedSnapshotID, mode string) {
	tb.Helper()
	e.e2bAdmin(tb, http.MethodPost, "/_admin/fault?id="+warmedSnapshotID, map[string]string{"mode": mode})
}

// e2bSetDelay adjusts the fake E2B per-dispatch delay (milliseconds).
func (e *warmEnv) e2bSetDelay(tb ginkgo.GinkgoTInterface, delayMS int64) {
	tb.Helper()
	e.e2bAdmin(tb, http.MethodPost, "/_admin/delay", map[string]any{"delay_ms": delayMS})
}

// e2bDispatchCount returns how many dispatches the fake E2B received for the
// given warmed_snapshot_id (snapshot_id:superpod_id).
func (e *warmEnv) e2bDispatchCount(tb ginkgo.GinkgoTInterface, warmedSnapshotID string) int {
	tb.Helper()
	reqs := e.e2bRequests(tb)
	n := 0
	for _, r := range reqs.Requests {
		if r.WarmedSnapshotID == warmedSnapshotID {
			n++
		}
	}
	return n
}

// ─── polling helpers (mirroring the dataplane harness conventions) ───

// waitFor polls cond every second until it returns true or timeout elapses.
func waitFor(tb ginkgo.GinkgoTInterface, what string, timeout time.Duration, cond func() bool) {
	tb.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(1 * time.Second)
	}
	tb.Fatalf("timed out after %s waiting for %s", timeout, what)
}

// wObjectKey builds a controller-runtime object key for spec files.
func wObjectKey(ns, name string) client.ObjectKey {
	return client.ObjectKey{Namespace: ns, Name: name}
}
