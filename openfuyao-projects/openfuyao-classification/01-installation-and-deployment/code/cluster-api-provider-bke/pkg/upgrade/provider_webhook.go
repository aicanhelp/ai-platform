/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgrade

import (
	"context"
	"fmt"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
)

const (
	bkeMutatingWebhookConfigurationName   = "bke-mutating-webhook-configuration"
	bkeValidatingWebhookConfigurationName = "bke-validating-webhook-configuration"
	bkeMutatingWebhookClusterName         = "mbkecluster.kb.io"
	bkeMutatingWebhookNodeName            = "mbkenode.kb.io"
	bkeValidatingWebhookClusterName       = "vbkecluster.kb.io"
	bkeValidatingWebhookNodeName          = "vbkenode.kb.io"
	bkeWebhookServiceName                 = "bke-webhook-service"
	bkeWebhookServiceNamespace            = "cluster-system"
	bkeMutateBKENodePath                  = "/mutate-bke-bocloud-com-v1beta1-bkenode"
	bkeValidateBKENodePath                = "/validate-bke-bocloud-com-v1beta1-bkenode"

	providerWebhookNamespace    = "cluster-system"
	providerWebhookDeployment   = "bke-controller-manager"
	providerWebhookContainer    = "manager"
	providerWebhookReadyTimeout = 5 * time.Minute
)

// EnsureBKENodeWebhooks waits for the provider Deployment to be Ready, then
// registers mbkenode / vbkenode on the live Mutating/Validating webhook
// configurations using caBundle from the existing cluster webhooks.
// Idempotent when the node webhooks already exist.
func EnsureBKENodeWebhooks(ctx context.Context, c client.Client) error {
	if c == nil {
		return fmt.Errorf("client is nil")
	}
	if err := waitProviderDeploymentReady(ctx, c); err != nil {
		return err
	}
	if err := ensureMBKENodeWebhook(ctx, c); err != nil {
		return err
	}
	return ensureVBKENodeWebhook(ctx, c)
}

func waitProviderDeploymentReady(ctx context.Context, c client.Client) error {
	target := phaseutil.DeploymentTarget{
		Namespace: providerWebhookNamespace,
		Name:      providerWebhookDeployment,
		Container: providerWebhookContainer,
	}
	image, err := phaseutil.GetDeploymentImage(ctx, c, target)
	if err != nil {
		return fmt.Errorf("get provider deployment image before webhook ensure: %w", err)
	}
	if err := phaseutil.WaitDeploymentReady(ctx, c, target, image, providerWebhookReadyTimeout); err != nil {
		return fmt.Errorf("wait provider ready before webhook ensure: %w", err)
	}
	return nil
}

func ensureMBKENodeWebhook(ctx context.Context, c client.Client) error {
	mwc := &admissionregistrationv1.MutatingWebhookConfiguration{}
	if err := c.Get(ctx, types.NamespacedName{Name: bkeMutatingWebhookConfigurationName}, mwc); err != nil {
		return fmt.Errorf("get %s: %w", bkeMutatingWebhookConfigurationName, err)
	}
	if mutatingWebhookHasName(mwc.Webhooks, bkeMutatingWebhookNodeName) {
		return nil
	}
	caBundle, err := mutatingWebhookCABundle(mwc.Webhooks, bkeMutatingWebhookClusterName)
	if err != nil {
		return err
	}
	mwc.Webhooks = append(mwc.Webhooks, newMBKENodeMutatingWebhook(caBundle))
	if err := c.Update(ctx, mwc); err != nil {
		return fmt.Errorf("update %s with mbkenode: %w", bkeMutatingWebhookConfigurationName, err)
	}
	return nil
}

func ensureVBKENodeWebhook(ctx context.Context, c client.Client) error {
	vwc := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	if err := c.Get(ctx, types.NamespacedName{Name: bkeValidatingWebhookConfigurationName}, vwc); err != nil {
		return fmt.Errorf("get %s: %w", bkeValidatingWebhookConfigurationName, err)
	}
	if validatingWebhookHasName(vwc.Webhooks, bkeValidatingWebhookNodeName) {
		return nil
	}
	caBundle, err := validatingWebhookCABundle(vwc.Webhooks, bkeValidatingWebhookClusterName)
	if err != nil {
		return err
	}
	vwc.Webhooks = append(vwc.Webhooks, newVBKENodeValidatingWebhook(caBundle))
	if err := c.Update(ctx, vwc); err != nil {
		return fmt.Errorf("update %s with vbkenode: %w", bkeValidatingWebhookConfigurationName, err)
	}
	return nil
}

func mutatingWebhookHasName(webhooks []admissionregistrationv1.MutatingWebhook, name string) bool {
	for _, wh := range webhooks {
		if wh.Name == name {
			return true
		}
	}
	return false
}

func validatingWebhookHasName(webhooks []admissionregistrationv1.ValidatingWebhook, name string) bool {
	for _, wh := range webhooks {
		if wh.Name == name {
			return true
		}
	}
	return false
}

func mutatingWebhookCABundle(webhooks []admissionregistrationv1.MutatingWebhook, name string) ([]byte, error) {
	for _, wh := range webhooks {
		if wh.Name != name {
			continue
		}
		return cloneCABundle(name, wh.ClientConfig.CABundle)
	}
	return nil, fmt.Errorf("webhook %s not found on %s", name, bkeMutatingWebhookConfigurationName)
}

func validatingWebhookCABundle(webhooks []admissionregistrationv1.ValidatingWebhook, name string) ([]byte, error) {
	for _, wh := range webhooks {
		if wh.Name != name {
			continue
		}
		return cloneCABundle(name, wh.ClientConfig.CABundle)
	}
	return nil, fmt.Errorf("webhook %s not found on %s", name, bkeValidatingWebhookConfigurationName)
}

func cloneCABundle(name string, caBundle []byte) ([]byte, error) {
	if len(caBundle) == 0 {
		return nil, fmt.Errorf("webhook %s has empty caBundle", name)
	}
	out := make([]byte, len(caBundle))
	copy(out, caBundle)
	return out, nil
}

func newMBKENodeMutatingWebhook(caBundle []byte) admissionregistrationv1.MutatingWebhook {
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone
	return admissionregistrationv1.MutatingWebhook{
		Name:                    bkeMutatingWebhookNodeName,
		AdmissionReviewVersions: []string{"v1"},
		ClientConfig:            bkeNodeWebhookClientConfig(caBundle, bkeMutateBKENodePath),
		FailurePolicy:           &fail,
		SideEffects:             &none,
		Rules: bkeNodeWebhookRules(
			admissionregistrationv1.Create,
			admissionregistrationv1.Update,
		),
	}
}

func newVBKENodeValidatingWebhook(caBundle []byte) admissionregistrationv1.ValidatingWebhook {
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone
	return admissionregistrationv1.ValidatingWebhook{
		Name:                    bkeValidatingWebhookNodeName,
		AdmissionReviewVersions: []string{"v1"},
		ClientConfig:            bkeNodeWebhookClientConfig(caBundle, bkeValidateBKENodePath),
		FailurePolicy:           &fail,
		SideEffects:             &none,
		Rules: bkeNodeWebhookRules(
			admissionregistrationv1.Create,
			admissionregistrationv1.Update,
			admissionregistrationv1.Delete,
		),
	}
}

func bkeNodeWebhookClientConfig(caBundle []byte, path string) admissionregistrationv1.WebhookClientConfig {
	p := path
	return admissionregistrationv1.WebhookClientConfig{
		CABundle: caBundle,
		Service: &admissionregistrationv1.ServiceReference{
			Name:      bkeWebhookServiceName,
			Namespace: bkeWebhookServiceNamespace,
			Path:      &p,
		},
	}
}

func bkeNodeWebhookRules(ops ...admissionregistrationv1.OperationType) []admissionregistrationv1.RuleWithOperations {
	return []admissionregistrationv1.RuleWithOperations{{
		Operations: ops,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{"bke.bocloud.com"},
			APIVersions: []string{"v1beta1"},
			Resources:   []string{"bkenodes"},
		},
	}}
}
