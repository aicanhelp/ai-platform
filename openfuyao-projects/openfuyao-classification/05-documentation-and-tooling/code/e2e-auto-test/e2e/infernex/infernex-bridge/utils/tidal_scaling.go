package utils

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
)

const (
	// DefaultBridgeScaleTimeout waits for tidal cron + NPU rollout during scale up/down (each direction).
	DefaultBridgeScaleTimeout = 15 * time.Minute
	// rsgReplicationGroupName matches fixtures/scaling/fit_pd_rsg.yaml groupConfig.groupName.
	rsgReplicationGroupName = "pd"
)

// RequireNPUWorkload skips when SKIP_NPU_WORKLOAD_E2E is set (scaling tests need real vLLM/NPU).
func RequireNPUWorkload(fitID string) {
	if SkipNPUWorkloadE2E() {
		Skip("SKIP_NPU_WORKLOAD_E2E is set; scaling FITs require NPU workload")
	}
}

// BridgeTidalScaleUpRules triggers scale-out to 2 replicas (every 2 minutes).
var BridgeTidalScaleUpRules = []esUtils.TidalRule{
	{Name: "scale-up-2", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 2, Description: "scale up to 2"},
}

// BridgeTidalScaleDownRules triggers scale-in back to 1 replica.
var BridgeTidalScaleDownRules = []esUtils.TidalRule{
	{Name: "scale-down-1", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 1, Description: "scale down to 1"},
}

func ScalingESName(owner string) string    { return owner + "-es" }
func ScalingTidalName(owner string) string { return owner + "-tidal" }
func ScalingRSGName(owner string) string   { return owner + "-rsg" }

func InsvcAggregateScaleDeploy(owner string) string {
	return DeployName(owner, "engine-aggregate")
}

func InsvcPDScalePrefillDeploy(owner string) string {
	return DeployName(owner, "engine-pd-prefill")
}

func InsvcPDScaleDecodeDeploy(owner string) string {
	return DeployName(owner, "engine-pd-decode")
}

func LLMISVCPDScalePrefillDeploy(owner string) string {
	return KServePrefillWorkloadDeployName(owner)
}

func LLMISVCPDScaleDecodeDeploy(owner string) string {
	return KServeAggregateWorkloadDeployName(owner)
}

// RequireScalingCRDs skips when elastic-scaler / tidal (and optionally RSG) CRDs are missing.
func RequireScalingCRDs(fitID string, pd bool) {
	if !CRDExists(fitID, "elasticscaler.io", "elasticscalers") {
		Skip("elasticscaler.io CRD not installed")
	}
	if !CRDExists(fitID, "tidal.io", "tidals") {
		Skip("tidal.io CRD not installed")
	}
	if pd && !CRDExists(fitID, "autoscaling.openfuyao.com", "resourcescalinggroups") {
		Skip("autoscaling.openfuyao.com ResourceScalingGroup CRD not installed")
	}
}

func verifyBridgeTidalStatusReady(ctx context.Context, dyn dynamic.Interface, fitID, name, ns string) {
	phase := fmt.Sprintf("tidal %s status Ready=True", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := esUtils.GetTidalCR(dyn, ctx, name, ns)
		g.Expect(err).NotTo(HaveOccurred())
		ready, degraded := tidalReadyDegraded(obj)
		if !ready || degraded {
			msg, _, _ := unstructured.NestedString(obj.Object, "status", "message")
			LogStep(fitID, "tidal %s ready=%v degraded=%v message=%q", name, ready, degraded, msg)
		}
		g.Expect(ready).To(BeTrue(), "tidal %s Ready should be True", name)
		g.Expect(degraded).To(BeFalse(), "tidal %s Degraded should not be True", name)
	}).WithTimeout(5 * time.Minute).WithPolling(5 * time.Second).Should(Succeed())
}

func tidalReadyDegraded(obj *unstructured.Unstructured) (ready, degraded bool) {
	conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil || !found {
		return false, false
	}
	for _, c := range conditions {
		m, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		ctype, _ := m["type"].(string)
		status, _ := m["status"].(string)
		if ctype == "Ready" && status == "True" {
			ready = true
		}
		if ctype == "Degraded" && status == "True" {
			degraded = true
		}
	}
	return ready, degraded
}

func createAggregateTidalESChain(ctx context.Context, dyn dynamic.Interface, fitID, ns, owner, targetDeploy string) error {
	target := esUtils.TidalTargetRef{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       targetDeploy,
		Namespace:  ns,
	}
	tidalName := ScalingTidalName(owner)
	esName := ScalingESName(owner)
	LogStep(fitID, "create tidal->deployment %s and es->tidal %s", targetDeploy, tidalName)
	if err := esUtils.CreateTidalCRWithTargetRulesAndESRef(
		dyn, ctx, tidalName, ns, target, BridgeTidalScaleUpRules, esName, ns,
	); err != nil {
		return err
	}
	esUtils.VerifyTidalCRExists(dyn, ctx, tidalName, ns)
	verifyBridgeTidalStatusReady(ctx, dyn, fitID, tidalName, ns)
	if err := esUtils.CreateStateTriggerESForTidal(dyn, ctx, esName, ns, tidalName, ns); err != nil {
		return err
	}
	esUtils.VerifyESCRExists(dyn, ctx, esName, ns)
	return nil
}

func applyPDResourceScalingGroup(fitID, owner, prefillDeploy, decodeDeploy string) error {
	path, err := WriteRenderedFixtureOpts([]string{"scaling", "fit_pd_rsg.yaml"}, RenderOptions{
		Namespace:     E2ENS(),
		Name:          owner,
		PrefillDeploy: prefillDeploy,
		DecodeDeploy:  decodeDeploy,
	})
	if err != nil {
		return err
	}
	stdout, stderr, err := ApplyManifest(path)
	LogStep(fitID, "apply RSG prefill=%s decode=%s: stdout=%s stderr=%s err=%v",
		prefillDeploy, decodeDeploy, stdout, stderr, err)
	return err
}

func createPDTidalESChain(ctx context.Context, dyn dynamic.Interface, fitID, ns, owner, prefillDeploy, decodeDeploy string) error {
	if err := applyPDResourceScalingGroup(fitID, owner, prefillDeploy, decodeDeploy); err != nil {
		return err
	}
	rsgName := ScalingRSGName(owner)
	target := esUtils.TidalTargetRef{
		APIVersion: "autoscaling.openfuyao.com/v1alpha1",
		Kind:       "ResourceScalingGroup",
		Name:       rsgName,
		Namespace:  ns,
	}
	tidalName := ScalingTidalName(owner)
	esName := ScalingESName(owner)
	LogStep(fitID, "create tidal->rsg %s (prefill=%s decode=%s) es->tidal %s",
		rsgName, prefillDeploy, decodeDeploy, tidalName)
	if err := esUtils.CreateTidalCRWithTargetRulesAndESRef(
		dyn, ctx, tidalName, ns, target, BridgeTidalScaleUpRules, esName, ns,
	); err != nil {
		return err
	}
	esUtils.VerifyTidalCRExists(dyn, ctx, tidalName, ns)
	verifyBridgeTidalStatusReady(ctx, dyn, fitID, tidalName, ns)
	if err := esUtils.CreateStateTriggerESForTidal(dyn, ctx, esName, ns, tidalName, ns); err != nil {
		return err
	}
	esUtils.VerifyESCRExists(dyn, ctx, esName, ns)
	return nil
}

// SwitchBridgeTidalScaleDownRule replaces Tidal rules with scale-down-to-1 schedule.
func SwitchBridgeTidalScaleDownRule(ctx context.Context, dyn dynamic.Interface, fitID, ns, owner string) error {
	tidalName := ScalingTidalName(owner)
	LogStep(fitID, "switch tidal %s to scale-down rules", tidalName)
	return esUtils.UpdateTidalCRRules(dyn, ctx, tidalName, ns, BridgeTidalScaleDownRules)
}

// CleanupBridgeScalingChain deletes ES → Tidal → RSG (when pd) before insvc/llmisvc teardown.
func CleanupBridgeScalingChain(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, owner string, pd bool) {
	esName := ScalingESName(owner)
	tidalName := ScalingTidalName(owner)
	rsgName := ScalingRSGName(owner)
	LogPhaseBegin(fitID, fmt.Sprintf("scaling chain cleanup owner=%s pd=%v", owner, pd))
	defer LogPhaseEnd(fitID, fmt.Sprintf("scaling chain cleanup owner=%s", owner))
	if err := esUtils.DeleteElasticScalerCR(dyn, ctx, esName, ns); err != nil && !apierrors.IsNotFound(err) {
		LogStep(fitID, "delete ES %s: %v", esName, err)
	} else {
		LogStep(fitID, "delete ES %s ok", esName)
	}
	if err := esUtils.DeleteTidalCR(dyn, ctx, tidalName, ns); err != nil && !apierrors.IsNotFound(err) {
		LogStep(fitID, "delete Tidal %s: %v", tidalName, err)
	} else {
		LogStep(fitID, "delete Tidal %s ok", tidalName)
	}
	if pd {
		if err := dyn.Resource(esUtils.ResourceScalingGroupGVR).Namespace(ns).Delete(ctx, rsgName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			LogStep(fitID, "delete RSG %s: %v", rsgName, err)
		} else {
			LogStep(fitID, "delete RSG %s ok", rsgName)
		}
		CleanupPDRSGCloneDeploymentsFallback(ctx, client, fitID, ns, owner, 3*time.Minute)
	}
}

// CleanupPDRSGCloneDeploymentsFallback deletes RSG GroupReplication clones (pd-{base}-{groupID}, groupID>=1).
// InferNexService/LLMISVC delete does not remove these Deployments; call after scale-down and on teardown.
func CleanupPDRSGCloneDeploymentsFallback(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	clones := listPDRSGCloneDeploymentNames(ctx, client, ns, owner)
	if len(clones) == 0 {
		return
	}
	LogStep(fitID, "fallback delete %d RSG clone deployment(s) for owner=%s: %v", len(clones), owner, clones)
	for _, name := range clones {
		if err := client.AppsV1().Deployments(ns).Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			LogStep(fitID, "delete clone deploy %s: %v", name, err)
		}
	}
	WaitPDRSGCloneDeploymentsAbsent(ctx, client, fitID, ns, owner, timeout)
}

func runAggregateTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner, targetDeploy string, scaleTimeout time.Duration) {
	ExpectWithOffset(1, createAggregateTidalESChain(ctx, dyn, fitID, ns, owner, targetDeploy)).To(Succeed())
	LogPhaseBegin(fitID, "aggregate scale-up "+targetDeploy)
	waitDeploymentScaleReplicas(ctx, client, fitID, ns, targetDeploy, 2, false, scaleTimeout)
	LogPhaseEnd(fitID, "aggregate scale-up "+targetDeploy)
	LogStep(fitID, "aggregate scale-up verified: %s readyReplicas>=2", targetDeploy)

	ExpectWithOffset(1, SwitchBridgeTidalScaleDownRule(ctx, dyn, fitID, ns, owner)).To(Succeed())
	LogPhaseBegin(fitID, "aggregate scale-down "+targetDeploy)
	waitDeploymentScaleReplicas(ctx, client, fitID, ns, targetDeploy, 1, true, scaleTimeout)
	LogPhaseEnd(fitID, "aggregate scale-down "+targetDeploy)
	LogStep(fitID, "aggregate scale-down verified: %s readyReplicas==1", targetDeploy)
}

func runPDTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner, prefillDeploy, decodeDeploy string, scaleTimeout time.Duration) {
	ExpectWithOffset(1, createPDTidalESChain(ctx, dyn, fitID, ns, owner, prefillDeploy, decodeDeploy)).To(Succeed())

	LogPhaseBegin(fitID, "PD RSG scale-up owner="+owner)
	waitPDGroupReplicationBackendReady(ctx, client, fitID, ns, owner, prefillDeploy, decodeDeploy, 2, false, scaleTimeout)
	assertPDRSGCloneDeploymentsPresent(ctx, client, fitID, ns, owner, 2, scaleTimeout)
	LogPhaseEnd(fitID, "PD RSG scale-up owner="+owner)
	LogStep(fitID, "pd scale-up: owner=%s groups=2 with RSG clone deploys present", owner)

	ExpectWithOffset(1, SwitchBridgeTidalScaleDownRule(ctx, dyn, fitID, ns, owner)).To(Succeed())
	waitPDGroupReplicationScaleDown(ctx, client, fitID, ns, owner, prefillDeploy, decodeDeploy, scaleTimeout)
	LogStep(fitID, "pd scale-down: owner=%s base groups=1, RSG clone deploys removed", owner)
}

// waitPDGroupReplicationScaleDown waits for base (non-clone) backends ==1, then deletes RSG clones.
// RSG/Tidal often do not remove pd-*-1 Deployments on scale-in; passive wait would spin until timeout.
func waitPDGroupReplicationScaleDown(ctx context.Context, client kubernetes.Interface, fitID, ns, owner, prefillDeploy, decodeDeploy string, scaleTimeout time.Duration) {
	LogPhaseBegin(fitID, "PD RSG scale-down owner="+owner)
	defer LogPhaseEnd(fitID, "PD RSG scale-down owner="+owner)

	// Best-effort wait for controller scale-in (may never delete clones).
	waitPDGroupReplicationBackendReady(ctx, client, fitID, ns, owner, prefillDeploy, decodeDeploy, 1, true, scaleTimeout)

	clones := listPDRSGCloneDeploymentNames(ctx, client, ns, owner)
	if len(clones) > 0 {
		LogStep(fitID, "pd scale-down: RSG clones still present after tidal wait, fallback delete: %v", clones)
		CleanupPDRSGCloneDeploymentsFallback(ctx, client, fitID, ns, owner, 5*time.Minute)
	} else {
		LogStep(fitID, "pd scale-down: no RSG clone deploys after tidal wait")
	}
	WaitPDRSGCloneDeploymentsAbsent(ctx, client, fitID, ns, owner, 2*time.Minute)
}

// RunInsvcAggregateTidalScaleRoundTrip scales Bridge engine-aggregate (FIT 27).
func RunInsvcAggregateTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner string, scaleTimeout time.Duration) {
	runAggregateTidalScaleRoundTrip(ctx, client, dyn, fitID, ns, owner, InsvcAggregateScaleDeploy(owner), scaleTimeout)
}

// RunLLMISvcAggregateTidalScaleRoundTrip scales KServe aggregate workload (FIT 26).
func RunLLMISvcAggregateTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner string, scaleTimeout time.Duration) {
	runAggregateTidalScaleRoundTrip(ctx, client, dyn, fitID, ns, owner, KServeAggregateWorkloadDeployName(owner), scaleTimeout)
}

// RunInsvcPDTidalScaleRoundTrip scales Bridge PD prefill+decode via RSG (FIT 29).
func RunInsvcPDTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner string, scaleTimeout time.Duration) {
	runPDTidalScaleRoundTrip(ctx, client, dyn, fitID, ns, owner,
		InsvcPDScalePrefillDeploy(owner), InsvcPDScaleDecodeDeploy(owner), scaleTimeout)
}

// RunLLMISvcPDTidalScaleRoundTrip scales KServe PD prefill+decode via RSG (FIT 28).
func RunLLMISvcPDTidalScaleRoundTrip(ctx context.Context, client kubernetes.Interface, dyn dynamic.Interface, fitID, ns, owner string, scaleTimeout time.Duration) {
	runPDTidalScaleRoundTrip(ctx, client, dyn, fitID, ns, owner,
		LLMISVCPDScalePrefillDeploy(owner), LLMISVCPDScaleDecodeDeploy(owner), scaleTimeout)
}

// waitDeploymentScaleReplicas waits for Deployment readyReplicas during tidal scale (up: >=want, down: ==want).
func waitDeploymentScaleReplicas(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string, want int32, exact bool, timeout time.Duration) {
	op := ">="
	if exact {
		op = "=="
	}
	phase := fmt.Sprintf("deployment %s scale readyReplicas%s%d", deployName, op, want)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		d, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "deployment %s", deployName)
		if exact {
			g.Expect(d.Status.ReadyReplicas).To(Equal(want), "deployment %s readyReplicas", deployName)
			if d.Spec.Replicas != nil && *d.Spec.Replicas != want {
				LogPollThrottled(fitID, "deploy-scale-spec-"+deployName, defaultPollLogInterval,
					"deployment %s spec.replicas=%d (ready=%d want=%d)", deployName, *d.Spec.Replicas, d.Status.ReadyReplicas, want)
			}
		} else {
			g.Expect(d.Status.ReadyReplicas).To(BeNumerically(">=", want))
		}
		if d.Status.ReadyReplicas < want || (exact && d.Status.ReadyReplicas != want) {
			LogDeploymentStatusThrottled(ctx, client, fitID, ns, deployName)
		}
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

// waitPDGroupReplicationBackendReady waits until RSG GroupReplication has enough ready backends.
// RSG clones new Deployments (e.g. pd-{groupName}-{baseDeploy}-{groupID}) instead of raising
// spec.replicas on the original prefill/decode Deployments, so we sum readyReplicas across all
// owner-related prefill/decode Deployments in the namespace.
func waitPDGroupReplicationBackendReady(ctx context.Context, client kubernetes.Interface, fitID, ns, owner, prefillDeploy, decodeDeploy string, wantGroups int32, exact bool, timeout time.Duration) {
	op := ">="
	if exact {
		op = "=="
	}
	phase := fmt.Sprintf("PD group replication owner=%s prefill/decode ready groups%s%d", owner, op, wantGroups)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		prefillReady, decodeReady := sumPDRoleReadyReplicas(list.Items, owner, decodeDeploy, exact)
		clones := listPDRSGCloneDeploymentNamesFromList(list.Items, owner)
		if prefillReady < wantGroups || decodeReady < wantGroups {
			LogPollThrottled(fitID, "pd-scale-"+owner, defaultPollLogInterval,
				"PD scale owner=%s prefill=%d/%d decode=%d/%d clones=%v (bases %s / %s)",
				owner, prefillReady, wantGroups, decodeReady, wantGroups, clones, prefillDeploy, decodeDeploy)
			LogOwnerDeploymentsThrottled(ctx, client, fitID, ns, owner)
		}
		if exact {
			g.Expect(prefillReady).To(Equal(wantGroups), "prefill backends for owner %s", owner)
			g.Expect(decodeReady).To(Equal(wantGroups), "decode backends for owner %s", owner)
		} else {
			g.Expect(prefillReady).To(BeNumerically(">=", wantGroups),
				"prefill backends for owner %s (includes RSG clones)", owner)
			g.Expect(decodeReady).To(BeNumerically(">=", wantGroups),
				"decode backends for owner %s base %s (includes RSG clones)", owner, decodeDeploy)
		}
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

func assertPDRSGCloneDeploymentsPresent(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, minClones int, timeout time.Duration) {
	phase := fmt.Sprintf("PD RSG clone deploys present owner=%s min=%d", owner, minClones)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		clones := listPDRSGCloneDeploymentNames(ctx, client, ns, owner)
		if len(clones) < minClones {
			LogPollThrottled(fitID, "pd-clone-up-"+owner, defaultPollLogInterval,
				"owner=%s RSG clone deploys=%v want>=%d", owner, clones, minClones)
		}
		g.Expect(len(clones)).To(BeNumerically(">=", minClones),
			"expected RSG clone deployments pd-*-%d for owner %s", minClones, owner)
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
	LogStep(fitID, "PD RSG clone deployments present: %v", listPDRSGCloneDeploymentNames(ctx, client, ns, owner))
}

func WaitPDRSGCloneDeploymentsAbsent(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	phase := fmt.Sprintf("PD RSG clone deploys absent owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		clones := listPDRSGCloneDeploymentNames(ctx, client, ns, owner)
		if len(clones) > 0 {
			LogPollThrottled(fitID, "pd-clone-down-"+owner, defaultPollLogInterval,
				"owner=%s remaining RSG clone deploys: %v", owner, clones)
		}
		g.Expect(clones).To(BeEmpty(), "RSG clone deployments should be deleted for owner %s", owner)
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
	LogStep(fitID, "PD RSG clone deployments absent for owner=%s", owner)
}

func listPDRSGCloneDeploymentNames(ctx context.Context, client kubernetes.Interface, ns, owner string) []string {
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil
	}
	return listPDRSGCloneDeploymentNamesFromList(list.Items, owner)
}

func listPDRSGCloneDeploymentNamesFromList(deploys []appsv1.Deployment, owner string) []string {
	var out []string
	for _, d := range deploys {
		if isPDRSGCloneDeploymentName(d.Name, owner) {
			out = append(out, d.Name)
		}
	}
	return out
}

// isPDRSGCloneDeploymentName reports RSG GroupReplication clones: pd-{baseDeploy}-{groupID}, groupID>=1.
func isPDRSGCloneDeploymentName(name, owner string) bool {
	if !strings.HasPrefix(name, rsgReplicationGroupName+"-") {
		return false
	}
	if !strings.Contains(name, owner) {
		return false
	}
	idx := strings.LastIndex(name, "-")
	if idx < 0 || idx >= len(name)-1 {
		return false
	}
	n, err := strconv.Atoi(name[idx+1:])
	return err == nil && n >= 1
}

func pdDeploymentBelongsToOwner(name, owner string) bool {
	return strings.Contains(name, owner)
}

// sumPDRoleReadyReplicas sums readyReplicas for prefill/decode. When exact is true (scale-down),
// RSG clone Deployments (pd-*-{id>=1}) are excluded so base group ready==1 is required.
func sumPDRoleReadyReplicas(deploys []appsv1.Deployment, owner, decodeDeploy string, excludeClones bool) (prefillReady, decodeReady int32) {
	for _, d := range deploys {
		if !pdDeploymentBelongsToOwner(d.Name, owner) {
			continue
		}
		if excludeClones && isPDRSGCloneDeploymentName(d.Name, owner) {
			continue
		}
		if strings.Contains(d.Name, "prefill") {
			prefillReady += d.Status.ReadyReplicas
			continue
		}
		if isPDDecodeDeployment(d.Name, decodeDeploy) {
			decodeReady += d.Status.ReadyReplicas
		}
	}
	return prefillReady, decodeReady
}

func isPDDecodeDeployment(name, decodeBase string) bool {
	if strings.Contains(name, "prefill") {
		return false
	}
	if name == decodeBase {
		return true
	}
	// RSG GroupReplication clone: {groupName}-{baseDeploy}-{groupID}, e.g. pd-fit28-pd-llm-kserve-1
	if strings.HasPrefix(name, "pd-") && strings.Contains(name, decodeBase) {
		return true
	}
	return false
}
