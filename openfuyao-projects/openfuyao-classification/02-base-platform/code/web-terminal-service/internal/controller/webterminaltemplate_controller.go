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
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"openfuyao.com/web-terminal-service/api/v1beta1"
	terminalv1beta1 "openfuyao.com/web-terminal-service/api/v1beta1"
	"openfuyao.com/web-terminal-service/pkg/zlog"
)

const (
	finalizer              = "openfuyao.com.finalizer.webterminal"
	checkTimePeriod        = 26 * time.Minute
	defaultUpdateFrequency = 20 * time.Second
)

// WebterminalTemplateReconciler reconciles a WebterminalTemplate object
type WebterminalTemplateReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=terminal.openfuyao.com,resources=webterminaltemplates,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=terminal.openfuyao.com,resources=webterminaltemplates/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=terminal.openfuyao.com,resources=webterminaltemplates/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the WebterminalTemplate object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
func (r *WebterminalTemplateReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	wtTemplate := &v1beta1.WebterminalTemplate{}
	if err := r.Get(ctx, req.NamespacedName, wtTemplate); err != nil {
		if errors.IsNotFound(err) {
			zlog.LogWarnf("NotFound wt template to reconcile.")
			return ctrl.Result{}, client.IgnoreNotFound(err)
		}
		zlog.LogErrorf("LogError Retrieving wttemplate object : %v", err)
		return ctrl.Result{}, err
	}
	_ = log.FromContext(ctx)

	if shouldReturn, result, err := r.handleFinalizer(ctx, wtTemplate); shouldReturn {
		return result, err
	}

	if shouldReturn, result, err := r.ensurePodRunning(ctx, wtTemplate); shouldReturn {
		return result, err
	}

	if shouldReturn, result, err := r.checkTTL(ctx, wtTemplate); shouldReturn {
		return result, err
	}

	return r.syncStatus(ctx, wtTemplate)
}

// handleFinalizer 处理资源的 Finalizer 逻辑
func (r *WebterminalTemplateReconciler) handleFinalizer(ctx context.Context, wtTemplate *v1beta1.WebterminalTemplate) (bool, ctrl.Result, error) {
	if wtTemplate.ObjectMeta.DeletionTimestamp.IsZero() {
		// 对象未被删除，确保 Finalizer 存在
		if !controllerutil.ContainsFinalizer(wtTemplate, finalizer) {
			if !controllerutil.AddFinalizer(wtTemplate, finalizer) {
				zlog.LogErrorf(" Adding Finalizer failed !")
				return true, ctrl.Result{}, nil
			}
			if err := r.Update(ctx, wtTemplate, &client.UpdateOptions{}); err != nil {
				zlog.LogErrorf("Failed to add finalizer : %v", err)
				return true, ctrl.Result{}, err
			}
		}
	} else {
		// 对象正在删除中，执行清理
		if controllerutil.ContainsFinalizer(wtTemplate, finalizer) {
			if err := r.deletePodTemplate(ctx, wtTemplate); err != nil {
				return true, ctrl.Result{}, err
			}

			if !controllerutil.RemoveFinalizer(wtTemplate, finalizer) {
				zlog.LogErrorf(" Removing Finalizer failed !")
				return true, ctrl.Result{}, nil
			}

			if err := r.Update(ctx, wtTemplate, &client.UpdateOptions{}); err != nil {
				zlog.LogErrorf("Failed to remove finalizer : %v", err)
				return true, ctrl.Result{}, err
			}
		}
		return true, ctrl.Result{}, nil
	}

	return false, ctrl.Result{}, nil
}

// ensurePodRunning 检查是否需要创建 Pod 并更新相关时间戳和状态
func (r *WebterminalTemplateReconciler) ensurePodRunning(ctx context.Context, wtTemplate *v1beta1.WebterminalTemplate) (bool, ctrl.Result, error) {
	// 如果未初始化或状态为 Stopped，则启动 Pod
	if wtTemplate.Spec.ExistsTime.IsZero() || wtTemplate.Status.Phase == v1beta1.WebTerminalTemplateStopped {
		if err := r.createPodTemplate(ctx, wtTemplate); err != nil {
			return true, ctrl.Result{}, err
		}

		wtTemplate.Spec.ExistsTime = metav1.NewTime(time.Now())
		wtTemplate.Spec.RenewTime = metav1.NewTime(time.Now())

		zlog.LogInfoln("after create pod existtime is: ", wtTemplate.Spec.ExistsTime.Time)
		zlog.LogInfoln("after create pod renewtime is : ", wtTemplate.Spec.RenewTime.Time)

		if err := r.Update(ctx, wtTemplate); err != nil {
			zlog.LogErrorf("Failed to update create time : %v", err)
			return true, ctrl.Result{}, err
		}

		wtTemplate.Status.Phase = v1beta1.WebTerminalTemplateRunning
		if err := r.Status().Update(ctx, wtTemplate); err != nil {
			zlog.LogErrorf("Failed to update pod status : %v", err)
			return true, ctrl.Result{}, err
		}

		return true, ctrl.Result{}, nil
	}

	return false, ctrl.Result{}, nil
}

// checkTTL
func (r *WebterminalTemplateReconciler) checkTTL(ctx context.Context, wtTemplate *v1beta1.WebterminalTemplate) (bool, ctrl.Result, error) {
	currentTime := time.Now().Add(-checkTimePeriod)

	if !wtTemplate.Spec.RenewTime.After(currentTime) {
		zlog.LogInfof(" start to delete cr ! \n")

		if err := r.Delete(ctx, wtTemplate, &client.DeleteOptions{}); err != nil {
			zlog.LogErrorf("Failed to delete wrbterminal template : %v", err)
			return true, ctrl.Result{}, err
		}
		return true, ctrl.Result{}, nil
	}

	return false, ctrl.Result{}, nil
}

func (r *WebterminalTemplateReconciler) syncStatus(ctx context.Context, obj *v1beta1.WebterminalTemplate) (ctrl.Result, error) {
	currentStatus := *obj.Status.DeepCopy()
	currentStatus.Phase = v1beta1.WebTerminalTemplateRunning

	podtpl := &corev1.Pod{}
	err := r.Get(ctx, types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}, podtpl)
	if err != nil {
		if errors.IsNotFound(err) {
			currentStatus.Phase = v1beta1.WebTerminalTemplateStopped
			return r.updateStatus(ctx, obj, currentStatus)

		}
		zlog.LogErrorf("LogError Retrieving user pod: %v", err)
		return ctrl.Result{}, err
	}

	var tplCondition []v1beta1.WebTerminalTemplateCondition
	for _, c := range podtpl.Status.Conditions {
		tplCondition = append(tplCondition, v1beta1.WebTerminalTemplateCondition{
			Status:             c.Status,
			Reason:             c.Reason,
			Message:            c.Message,
			LastTransitionTime: metav1.Time{Time: c.LastTransitionTime.Time},
		})
	}
	currentStatus.Conditions = tplCondition

	return r.updateStatus(ctx, obj, currentStatus) // 更新当前Pod状态到webterminaltemplate的状态
}

func (r *WebterminalTemplateReconciler) deletePodTemplate(ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
	currentPod := &corev1.Pod{}

	err := wait.PollUntilContextTimeout(context.TODO(), time.Second, time.Minute, false,
		func(context.Context) (done bool, err error) {
			err = r.Get(ctx, types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}, currentPod)
			if err != nil {
				if errors.IsNotFound(err) {
					return true, nil
				}
				return false, err
			}

			if currentPod.DeletionTimestamp.IsZero() {
				err = r.Client.Delete(ctx, currentPod)
				if err != nil {
					zlog.LogErrorf("Deleting user pod failed : %v", err)
					return false, err
				}
			}

			zlog.LogInfof("Deleting the %s pod.", obj.Name)
			return false, nil
		})
	if err != nil {
		zlog.LogErrorf("LogError Deleting user pod, timeout : %v", err)
		return err
	}

	return nil

}

func (r *WebterminalTemplateReconciler) createPodTemplate(ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
	podtpl := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      obj.Spec.PodTemplate.ObjectMeta.Name,
			Namespace: obj.Spec.PodTemplate.ObjectMeta.Namespace,
			Labels:    obj.Spec.PodTemplate.ObjectMeta.Labels,
		},
		Spec: corev1.PodSpec{
			InitContainers: obj.Spec.PodTemplate.Spec.InitContainers,
			Containers:     obj.Spec.PodTemplate.Spec.Containers,
			Volumes:        obj.Spec.PodTemplate.Spec.Volumes,
			RestartPolicy:  obj.Spec.PodTemplate.Spec.RestartPolicy,
		},
	}

	err := wait.PollUntilContextTimeout(context.TODO(), time.Second, time.Minute, false,
		func(context.Context) (done bool, err error) {
			currentPod := &corev1.Pod{}
			err = r.Get(ctx, types.NamespacedName{Name: obj.Name, Namespace: obj.Namespace}, currentPod)
			if err != nil {
				if errors.IsNotFound(err) {
					if err = r.Create(ctx, podtpl); err != nil {
						zlog.LogErrorf("Creating %s Pod failed !", obj.Name)
						return false, err
					}
					zlog.LogInfof("Create %s pod sucess !", obj.Name)
					return false, err
				}
				return false, err
			}

			// check pod status
			if !isPodReady(currentPod) {
				zlog.LogWarnf("%s pod status is not ready !", obj.Name)
				return false, nil
			}

			zlog.LogInfof("%s pod status now is ready !", obj.Name)
			return true, nil
		})
	if err != nil {
		zlog.LogErrorf("LogError Creating user pod, timeout : %v", err)
		return err
	}

	return nil
}

func (r *WebterminalTemplateReconciler) updateStatus(ctx context.Context, obj *v1beta1.WebterminalTemplate,
	newStatus v1beta1.WebterminalTemplateStatus) (ctrl.Result, error) {
	// retry avoid conflict update
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		wbTpl := v1beta1.WebterminalTemplate{}
		if err := r.Get(ctx, client.ObjectKeyFromObject(obj), &wbTpl); err != nil {
			zlog.LogErrorf("LogError Retrieving web-terminal template object : %v", err)
			return err
		}

		wbTpl.Status.Phase = newStatus.Phase
		wbTpl.Status.Conditions = newStatus.Conditions

		return r.Status().Update(ctx, &wbTpl)
	})

	if err != nil {
		zlog.LogErrorf("LogError Updating wbtemplate status : %v", err)
		return ctrl.Result{}, nil
	}

	return ctrl.Result{RequeueAfter: defaultUpdateFrequency}, nil // 更新 WebterminalTemplate 对象的状态。它通过重试机制避免更新冲突 并在20s定时调谐
}

func isPodReady(p *corev1.Pod) bool {
	for _, c := range p.Status.Conditions {
		zlog.LogInfof("pod type %s, status %s", c.Type, c.Status)
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// SetupWithManager sets up the controller with the Manager.
func (r *WebterminalTemplateReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&terminalv1beta1.WebterminalTemplate{}).
		Complete(r)
}
