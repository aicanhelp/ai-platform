/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/utils"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

const (
	readyCondition = "Ready"
	errorCondition = "Error"
)

// statusUpdateParams 封装状态更新所需的参数
type statusUpdateParams struct {
	readyStatus metav1.ConditionStatus
	errorStatus metav1.ConditionStatus
	reason      string
	message     string
}

// updateKAEPolicyState 跟新 CR State 字段
func updateKAEPolicyState(ctx context.Context, r *KAEPolicyReconciler, namespacedName types.NamespacedName,
	state kaev1.State) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		instance := &kaev1.KAEPolicy{}
		err := r.Client.Get(ctx, namespacedName, instance)
		if err != nil {
			zlog.Error("failed to get KAEPolicy instance for status update")
			return err
		}
		if instance.Status.State == state {
			zlog.Infof("consistent status: %s, no need to update", state)
			return nil
		}
		utils.SetStatus(instance, state, kaePolicyManager.namespace)
		return r.Client.Status().Update(ctx, instance)
	})
	if err != nil {
		zlog.Errorf("failed to update KAEPolicy State: %s", state)
	}
}

// updateStatusCondition 根据条件更新 CR State condition 字段
func updateStatusCondition(client client.Client, ctx context.Context, cr *kaev1.KAEPolicy,
	params statusUpdateParams) error {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// 刷新实例以确保我们使用的是最新状态
		instance := &kaev1.KAEPolicy{}
		if err := client.Get(ctx, types.NamespacedName{Name: cr.Name}, instance); err != nil {
			zlog.Errorf("get CR: %s hai err: %v", cr.Name, err)
			return err
		}

		// 更新 Ready 条件
		meta.SetStatusCondition(&instance.Status.Conditions, metav1.Condition{
			Type:    readyCondition,
			Status:  params.readyStatus,
			Reason:  params.reason,
			Message: params.message,
		})

		// 更新 Error 条件
		meta.SetStatusCondition(&instance.Status.Conditions, metav1.Condition{
			Type:   errorCondition,
			Status: params.errorStatus,
			Reason: params.reason,
			Message: func() string {
				if params.errorStatus == metav1.ConditionTrue {
					return params.message
				}
				return ""
			}(),
		})

		// 更新状态
		return client.Status().Update(ctx, instance)
	})
	if err != nil {
		zlog.Errorf("failed to update CR: %s status condition hai err: %v", cr.Name, err)
		return err
	}
	return nil
}

// setConditionsReady 将策略设置为就绪状态，并给出原因和消息。
func setConditionsReady(client client.Client, ctx context.Context, cr *kaev1.KAEPolicy, reason, message string) error {
	params := statusUpdateParams{
		readyStatus: metav1.ConditionTrue,
		errorStatus: metav1.ConditionFalse,
		reason:      reason,
		message:     message,
	}
	return updateStatusCondition(client, ctx, cr, params)
}

// setConditionsError 将策略设置为有错误状态，并给出原因和消息。
func setConditionsError(client client.Client, ctx context.Context, cr *kaev1.KAEPolicy, reason, message string) error {
	params := statusUpdateParams{
		readyStatus: metav1.ConditionFalse,
		errorStatus: metav1.ConditionTrue,
		reason:      reason,
		message:     message,
	}
	return updateStatusCondition(client, ctx, cr, params)
}
