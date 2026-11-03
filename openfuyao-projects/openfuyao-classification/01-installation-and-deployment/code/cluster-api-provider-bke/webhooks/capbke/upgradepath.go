/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for the more details.
 ******************************************************************/

package capbke

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	confv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

type UpgradePath struct {
	Client client.Client
	// APIReader lists UpgradePath CRs without informer cache during admission.
	APIReader client.Reader
}

//+kubebuilder:webhook:verbs=create;update;delete,path=/validate-config-openfuyao-com-v1alpha1-upgradepath,mutating=false,failurePolicy=fail,sideEffects=None,groups=config.openfuyao.com,resources=upgradepaths,versions=v1alpha1,name=vupgradepath.kb.io,admissionReviewVersions=v1

var _ webhook.CustomValidator = &UpgradePath{}

func (webhook *UpgradePath) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(&confv1alpha1.UpgradePath{}).
		WithValidator(webhook).
		Complete()
}

func (webhook *UpgradePath) getReader() client.Reader {
	if webhook.APIReader != nil {
		return webhook.APIReader
	}
	return webhook.Client
}

func (webhook *UpgradePath) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	upList := &confv1alpha1.UpgradePathList{}
	if err := webhook.getReader().List(ctx, upList); err != nil {
		return nil, fmt.Errorf("failed to list existing UpgradePath CRs: %w", err)
	}
	if len(upList.Items) > 0 {
		return nil, fmt.Errorf("only one UpgradePath CR is allowed per cluster, existing: %s", upList.Items[0].Name)
	}
	return nil, nil
}

func (webhook *UpgradePath) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (webhook *UpgradePath) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}
