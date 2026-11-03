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

package controller

import (
	"context"
	"fmt"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const workloadKindLeaderWorkerSet = "LeaderWorkerSet"

// applyLWSReplicas sets lws.Spec.Replicas from plan. Inference engine workloads omit replicas in
// merged spec (nil) to allow external autoscalers; first create defaults to 1.
func applyLWSReplicas(lws *lwsv1.LeaderWorkerSet, component string, plan componentPlan) {
	if plan.Replicas != nil {
		lws.Spec.Replicas = plan.Replicas
		return
	}
	if !isInferenceEngineComponent(component) {
		lws.Spec.Replicas = ptr.To(int32(1))
		return
	}
	if lws.UID == "" {
		lws.Spec.Replicas = ptr.To(int32(1))
	}
}

func lwsRolloutReady(lws *lwsv1.LeaderWorkerSet) (bool, string) {
	if lws.Spec.Replicas == nil || *lws.Spec.Replicas == 0 {
		return true, ""
	}
	want := *lws.Spec.Replicas
	if lws.Status.ReadyReplicas < want {
		return false, fmt.Sprintf("ready groups %d/%d", lws.Status.ReadyReplicas, want)
	}
	for i := range lws.Status.Conditions {
		c := lws.Status.Conditions[i]
		if c.Type == string(lwsv1.LeaderWorkerSetAvailable) && c.Status != metav1.ConditionTrue {
			msg := strings.TrimSpace(c.Message)
			if msg == "" {
				msg = "LeaderWorkerSet not available"
			}
			return false, msg
		}
	}
	return true, ""
}

func (r *InferNexServiceReconciler) reconcileComponentLeaderWorkerSet(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) error {
	name := fmt.Sprintf("%s-%s", owner.Name, component)
	resourceLabels := deploymentResourceLabels(owner.Name, component, plan.AppKubernetesIOName)
	matchLabels := deploymentPodMatchLabels(owner, component, plan)
	lws := &lwsv1.LeaderWorkerSet{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, lws, func() error {
		lws.Labels = resourceLabels
		applyLWSReplicas(lws, component, plan)
		if plan.GroupSize < 2 {
			return fmt.Errorf("component %q: internal error: LWS reconcile with groupSize %d (expected >=2)", component, plan.GroupSize)
		}
		leaderTpl, tplErr := r.buildManagedComponentPodTemplate(ctx, owner, component, plan, resourceLabels, matchLabels, effectiveEngine)
		if tplErr != nil {
			return tplErr
		}
		workerPlan := plan
		if plan.WorkerTemplate != nil {
			workerPlan.Template = plan.WorkerTemplate
		}
		workerTpl, tplErr := r.buildManagedComponentPodTemplate(ctx, owner, component, workerPlan, resourceLabels, matchLabels, effectiveEngine)
		if tplErr != nil {
			return tplErr
		}
		lws.Spec.StartupPolicy = lwsv1.LeaderCreatedStartupPolicy
		lws.Spec.LeaderWorkerTemplate = lwsv1.LeaderWorkerTemplate{
			Size:           ptr.To(plan.GroupSize),
			LeaderTemplate: leaderTpl,
			WorkerTemplate: *workerTpl,
			RestartPolicy:  lwsv1.RecreateGroupOnPodRestart,
		}
		return controllerutil.SetControllerReference(owner, lws, r.Scheme)
	})
	return err
}

func (r *InferNexServiceReconciler) pruneComponentLeaderWorkerSets(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	var sets lwsv1.LeaderWorkerSetList
	if err := r.List(
		ctx,
		&sets,
		client.InNamespace(owner.Namespace),
		client.MatchingLabels{"infernex.io/owner": owner.Name},
	); err != nil {
		return err
	}
	for i := range sets.Items {
		lws := &sets.Items[i]
		component := lws.Labels["infernex.io/component"]
		if plan, ok := desired[component]; ok && componentWorkloadKind(plan) == workloadKindLeaderWorkerSet {
			continue
		}
		if err := r.Delete(ctx, lws); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}
