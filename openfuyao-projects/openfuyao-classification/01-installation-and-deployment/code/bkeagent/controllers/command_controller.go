/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"
	apierr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	agentv1beta1 "gopkg.openfuyao.cn/bkeagent/api/v1beta1"
	"gopkg.openfuyao.cn/bkeagent/pkg/job"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

// CommandReconciler reconciles a Command object
type CommandReconciler struct {
	client.Client
	APIReader client.Reader
	Scheme    *runtime.Scheme
	Ctx       context.Context
	Job       job.Job
	NodeName  string
	NodeIP    string
}

//+kubebuilder:rbac:groups=bkeagent.bocloud.com,resources=commands,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=bkeagent.bocloud.com,resources=commands/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=bkeagent.bocloud.com,resources=commands/finalizers,verbs=update

// Reconcile command
func (r *CommandReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {

	command := new(agentv1beta1.Command)
	if err := r.Get(ctx, req.NamespacedName, command); err != nil {
		if apierr.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Errorf("unable to fetch Command, %s", err.Error())
		return ctrl.Result{}, err
	}
	// If command is empty
	if len(command.Spec.Commands) == 0 {
		log.Warnf("command is not configured, %s/%s", command.Namespace, command.Name)
		return ctrl.Result{}, nil
	}
	log.Infof("reconcile %s/%s resourceVersion: %s, generation: %d",
		command.Namespace, command.Name, command.GetResourceVersion(), command.GetGeneration())
	// Initialize status
	if command.Status == nil {
		command.Status = map[string]*agentv1beta1.CommandStatus{}
	}
	if _, ok := command.Status[r.commandStatusKey()]; !ok {
		command.Status[r.commandStatusKey()] = &agentv1beta1.CommandStatus{
			Conditions:     []*agentv1beta1.Condition{},
			LastStartTime:  &metav1.Time{Time: time.Now()},
			CompletionTime: nil,
			Succeeded:      -1,
			Failed:         -1,
			Phase:          agentv1beta1.CommandRunning,
			Status:         metav1.ConditionUnknown,
		}
		if err := r.syncStatusUntilComplete(command); err != nil {
			log.Errorf("unable to update command status resourceVersion: %s error: %s",
				command.GetResourceVersion(), err.Error())
			return ctrl.Result{}, nil
		}
	}

	currentStatus := command.Status[r.commandStatusKey()]
	// Global Task Key
	gid := fmt.Sprintf("%s/%s", command.Namespace, command.Name)

	// Set the finalizers
	commandFinalizerName := "command.bkeagent.bocloud.com/finalizers"
	if command.ObjectMeta.DeletionTimestamp.IsZero() {
		if !controllerutil.ContainsFinalizer(command, commandFinalizerName) {
			controllerutil.AddFinalizer(command, commandFinalizerName)
			if err := r.Update(ctx, command); err != nil {
				if apierr.IsConflict(err) {
					return ctrl.Result{Requeue: true}, nil
				}
				if apierr.IsNotFound(err) {
					return ctrl.Result{}, nil
				}
				return ctrl.Result{}, err
			}
		}
	} else {
		if controllerutil.ContainsFinalizer(command, commandFinalizerName) {
			if v, ok := r.Job.Task[gid]; ok {
				v.SafeClose()
				delete(r.Job.Task, gid)
			}
			controllerutil.RemoveFinalizer(command, commandFinalizerName)
			if err := r.Update(ctx, command); err != nil {
				if apierr.IsConflict(err) {
					log.Warnf("conflict when update command, %s", err.Error())
					return ctrl.Result{Requeue: true}, nil
				}
				if apierr.IsNotFound(err) {
					return ctrl.Result{}, nil
				}
				return ctrl.Result{}, err
			}
			log.Infof("The Finalizer is removed %s", gid)
		}
		return ctrl.Result{}, nil
	}

	// Skip commands that have already been processed
	if currentStatus.Phase == agentv1beta1.CommandComplete {
		return ctrl.Result{}, nil
	}
	// Process suspended services
	if command.Spec.Suspend {
		if currentStatus.Phase == agentv1beta1.CommandSuspend {
			return ctrl.Result{}, nil
		}
		log.Infof("%s has been suspended", gid)
		if v, ok := r.Job.Task[gid]; ok {
			v.SafeClose()
			v.Phase = agentv1beta1.CommandSuspend
		}
		// Statistical state
		currentStatus.Succeeded, currentStatus.Failed, currentStatus.Status, currentStatus.Phase =
			agentv1beta1.ConditionCount(currentStatus.Conditions, len(command.Spec.Commands))
		currentStatus.Phase = agentv1beta1.CommandSuspend
		if err := r.syncStatusUntilComplete(command); err != nil {
			log.Errorf("unable to update command status resourceVersion: %s error: %s",
				command.GetResourceVersion(), err.Error())
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	if v, ok := r.Job.Task[gid]; ok {
		// This value is increased by a spec change
		if command.GetGeneration() <= v.Generation {
			log.Infof("A later version task is being executed, command:%s resourceVersion:%s-%s, generation:%d<=%d",
				gid, command.GetResourceVersion(), v.ResourceVersion, command.GetGeneration(), v.Generation)
			return ctrl.Result{}, nil
		}
		v.SafeClose()
	}
	r.Job.Task[gid] = &job.Task{
		StopChan:                make(chan struct{}),
		Phase:                   agentv1beta1.CommandRunning,
		ResourceVersion:         command.ResourceVersion,
		Generation:              command.GetGeneration(),
		TTLSecondsAfterFinished: command.Spec.TTLSecondsAfterFinished,
		HasAddTimer:             false,
		Once:                    &sync.Once{},
	}
	// The start time is reset each time the Reconcile function is entered
	currentStatus.LastStartTime = &metav1.Time{Time: time.Now()}
	currentStatus.CompletionTime = nil
	if err := r.syncStatusUntilComplete(command); err != nil {
		log.Errorf("unable to update command status resourceVersion: %s error: %s",
			command.GetResourceVersion(), err.Error())
		return ctrl.Result{}, nil
	}
	go r.startTask(ctx, r.Job.Task[gid].StopChan, command)
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *CommandReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Clears resource objects with TTL set
	go r.ttlSecondAfterFinished()

	return ctrl.NewControllerManagedBy(mgr).
		For(&agentv1beta1.Command{}, r.commandPredicateFn()).
		WithOptions(controller.Options{
			RateLimiter: workqueue.NewItemFastSlowRateLimiter(10*time.Second, 60*time.Second, 5),
		}).Complete(r)
}

func (r *CommandReconciler) commandPredicateFn() builder.Predicates {
	commandPredicateFn := builder.WithPredicates(
		predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool {
				o := e.Object.(*agentv1beta1.Command)
				if o != nil {
					gid := fmt.Sprintf("%s/%s", o.Namespace, o.Name)
					if v, ok := r.Job.Task[gid]; ok {
						// The value of this field will increase only if you update the spec
						if o.Generation <= v.Generation || o.ResourceVersion <= v.ResourceVersion {
							log.Debugf("CreateFunc: Update status without Reconcile %s", gid)
							return false
						}
					}
					// todo 废弃Spec.NodeName，可承载信息太少
					if o.Spec.NodeName == r.NodeName {
						return true
					}
					return r.nodeMatchNodeSelector(o.Spec.NodeSelector)
				}
				return false
			},
			UpdateFunc: func(e event.UpdateEvent) bool {
				o := e.ObjectNew.(*agentv1beta1.Command)
				if o != nil {
					gid := fmt.Sprintf("%s/%s", o.Namespace, o.Name)
					if v, ok := r.Job.Task[gid]; ok {
						// The value of this field will increase only if you update the spec
						if o.Generation <= v.Generation || o.ResourceVersion <= v.ResourceVersion {
							log.Debugf("UpdateFunc: %d<=%d Only update status without Reconcile %s",
								o.Generation, v.Generation, gid)
							return false
						}
					}
					// todo 废弃Spec.NodeName，可承载信息太少
					if o.Spec.NodeName == r.NodeName {
						return true
					}
					return r.nodeMatchNodeSelector(o.Spec.NodeSelector)
				}
				return false
			},
			DeleteFunc: func(e event.DeleteEvent) bool {
				return false
			},
			GenericFunc: func(e event.GenericEvent) bool {
				return false
			},
		},
	)
	return commandPredicateFn
}

func (r *CommandReconciler) startTask(ctx context.Context, stopChan chan struct{}, command *agentv1beta1.Command) {
	gid := fmt.Sprintf("%s/%s", command.Namespace, command.Name)
	currentStatus := command.Status[r.commandStatusKey()]
	// The cut-off time of the entire instruction
	stopTime := currentStatus.LastStartTime.Time
	if command.Spec.ActiveDeadlineSecond > 0 {
		stopTime = stopTime.Add(time.Duration(command.Spec.ActiveDeadlineSecond) * time.Second)
	} else {
		stopTime = stopTime.Add(time.Duration(agentv1beta1.DefaultActiveDeadlineSecond) * time.Second)
	}
	// Execute each instruction in order
	for _, execCommand := range command.Spec.Commands {
		// You can uncomment this annotation when testing SUSPEND
		// time.Sleep(30 * time.Second)
		select {
		case <-stopChan:
			log.Warnf("Execution command terminated %s", gid)
			return
		default:
		}
		// Determine whether timeout occurs
		if stopTime.Before(time.Now()) {
			break
		}
		// Condition determines whether the command has been executed
		existsCondition := agentv1beta1.GetCondition(currentStatus.Conditions, &agentv1beta1.Condition{ID: execCommand.ID})
		if existsCondition != nil && existsCondition.Phase == agentv1beta1.CommandComplete {
			// If it has already been executed, skip it and proceed to the next instruction
			continue
		}
		// If the subcommand is in the failed and skipped state, allow it to be re-executed
		currentCondition := &agentv1beta1.Condition{
			ID:            execCommand.ID,
			Status:        metav1.ConditionUnknown,
			Phase:         agentv1beta1.CommandRunning,
			LastStartTime: &metav1.Time{Time: time.Now()},
			StdErr:        []string{},
			StdOut:        []string{},
			Count:         0,
		}
		// Conditions will also be added if conditions are not present
		currentStatus.Conditions = agentv1beta1.ReplaceCondition(currentStatus.Conditions, currentCondition)

		// 每个命令执行结束后都需要更新command.status.condition字段信息
		// 最少执行一次，当命令执行失败后，判断最大重试次数是否达到，未达到继续执行
		for command.Spec.BackoffLimit >= 0 && currentCondition.Count <= command.Spec.BackoffLimit {
			// Determine whether timeout occurs
			if stopTime.Before(time.Now()) {
				break
			}
			// delay when retrying
			if execCommand.BackoffDelay != 0 && currentCondition.Count > 0 {
				time.Sleep(time.Duration(execCommand.BackoffDelay) * time.Second)
			}
			// Update execution time
			currentCondition.LastStartTime = &metav1.Time{Time: time.Now()}
			currentCondition.Count += 1

			var result []string
			var err error
			// Assign to different actuators according to the instruction type
			switch execCommand.Type {
			case agentv1beta1.CommandBuiltIn:
				result, err = r.Job.BuiltIn.Execute(execCommand.Command)
			case agentv1beta1.CommandKubernetes:
				result, err = r.Job.K8s.Execute(execCommand.Command)
			case agentv1beta1.CommandShell:
				result, err = r.Job.Shell.Execute(execCommand.Command)
			default:
				log.Errorf("Unsupported command type: %s", execCommand.Type)
				break
			}
			// Command execution failure
			if err != nil {
				currentCondition.Status = metav1.ConditionFalse
				currentCondition.Phase = agentv1beta1.CommandFailed
				currentCondition.StdErr = append(currentCondition.StdErr, err.Error())
				log.Errorf("Command exec failed: %s %s", currentCondition.ID, err.Error())
				continue
			} else {
				// Command executed successfully
				currentCondition.Status = metav1.ConditionTrue
				currentCondition.Phase = agentv1beta1.CommandComplete
				currentCondition.StdOut = append(currentCondition.StdOut, result...)
				break
			}
		}
		// If ignore is configured, you can skip even if the command fails
		if currentCondition.Status == metav1.ConditionFalse && execCommand.BackoffIgnore {
			currentCondition.Phase = agentv1beta1.CommandSkip
		}
		// Update the state of each sub instruction after execution
		if err := r.syncStatusUntilComplete(command); err != nil {
			log.Errorf("unable to update command status resourceVersion: %s error: %s",
				command.ResourceVersion, err.Error())
			return
		}
		// If the final status is execution failure, the execution of subsequent instructions is stopped
		if currentCondition.Phase == agentv1beta1.CommandFailed {
			break
		}
	}
	// Statistical state
	currentStatus.Succeeded, currentStatus.Failed, currentStatus.Status, currentStatus.Phase =
		agentv1beta1.ConditionCount(currentStatus.Conditions, len(command.Spec.Commands))
	currentStatus.CompletionTime = &metav1.Time{Time: time.Now()}
	// 更新command status - 当status中的各项状态信息配置完成后，执行更新状态操作
	command.Status[r.commandStatusKey()] = currentStatus
	if err := r.syncStatusUntilComplete(command); err != nil {
		log.Errorf("unable to update command status resourceVersion: %s error: %s",
			command.ResourceVersion, err.Error())
		return
	}
	// When deleting command, the global task is being cleaned up
	if v, ok := r.Job.Task[gid]; ok {
		v.Phase = currentStatus.Phase
		v.SafeClose()
	}
	return
}

func (r *CommandReconciler) commandStatusKey() string {
	if r.NodeIP == "" {
		return r.NodeName
	}
	return fmt.Sprintf("%s/%s", r.NodeName, r.NodeIP)
}

func (r *CommandReconciler) ttlSecondAfterFinished() {
	for {
		select {
		case <-r.Ctx.Done():
			return
		default:
		}
		// Periodically scan local tasks
		time.Sleep(time.Duration(rand.IntnRange(30, 60)) * time.Second)
		for key, value := range r.Job.Task {
			if value.HasAddTimer || value.TTLSecondsAfterFinished == 0 || value.Phase != agentv1beta1.CommandComplete {
				continue
			}
			namespace, name, _ := cache.SplitMetaNamespaceKey(key)
			obj := &agentv1beta1.Command{}
			err := r.Client.Get(r.Ctx, client.ObjectKey{Namespace: namespace, Name: name}, obj)
			if err != nil {
				if apierr.IsNotFound(err) {
					delete(r.Job.Task, key)
					continue
				}
				log.Warnf("unable fetch command %s", err.Error())
				continue
			}
			// It will not be deleted until everything is complete
			needDelete := true
			for n, v := range obj.Status {
				if v.Status != metav1.ConditionTrue {
					log.Warnf("Instruction %s not completed on node %s, refused to delete", key, n)
					needDelete = false
					break
				}
			}
			if needDelete {
				value.HasAddTimer = true
				newTTL := value.TTLSecondsAfterFinished - int(time.Now().Sub(obj.Status[r.commandStatusKey()].CompletionTime.Time).Seconds())
				if newTTL <= 0 {
					newTTL = rand.IntnRange(0, 3)
				}
				log.Infof("Command Resource %s will be deleted in %d seconds", key, newTTL)
				time.AfterFunc(time.Duration(newTTL)*time.Second, func() {
					log.Infof("Deleting the Command resource %s/%s", obj.Namespace, obj.Name)
					err = r.Delete(r.Ctx, obj)
					if err != nil {
						log.Warnf("Command resource %s/%s deletion failed. %s", obj.Namespace, obj.Name, err.Error())
						return
					}
				})
			}
		}
	}
}

func (r *CommandReconciler) syncStatusUntilComplete(cmd *agentv1beta1.Command) (err error) {
	ctx, cancel := context.WithTimeout(r.Ctx, 5*time.Minute)
	defer cancel()
	for {
		select {
		case <-r.Ctx.Done():
			return
		case <-ctx.Done():
			return errors.New("The update failed to complete after 5 minutes. ")
		default:
		}
		// Execute concurrent tasks at different peaks.
		// When the number of concurrent tasks is greater than 100, a random value of 1-15 is preferred
		time.Sleep(time.Duration(rand.IntnRange(1, 2)) * time.Second)
		obj := &agentv1beta1.Command{}
		// This refresh is a direct request to the Kube-apiserver, which is equivalent to refreshing the local Client-go cache
		err = r.APIReader.Get(r.Ctx, client.ObjectKey{Namespace: cmd.Namespace, Name: cmd.Name}, obj)
		if err != nil {
			if apierr.IsNotFound(err) {
				log.Warnf("Command resource %s-%s not found, skip sync", cmd.Namespace, cmd.Name)
				return nil
			}
			log.Errorf("Get command resource failed %s-%s error: %v", cmd.Namespace, cmd.Name, err)
			continue
		}
		if obj.Status == nil {
			obj.Status = map[string]*agentv1beta1.CommandStatus{}
		}
		objCopy := obj.DeepCopy()
		objCopy.Status[r.commandStatusKey()] = cmd.Status[r.commandStatusKey()]
		//patch status
		err = r.Client.Status().Patch(r.Ctx, objCopy, client.MergeFrom(obj))
		if err != nil {
			log.Warnf("Update command resource failed %s-%s error: %v", cmd.Namespace, cmd.Name, err)
			continue
		}
		break
	}
	return nil
}

func (r *CommandReconciler) nodeMatchNodeSelector(s *metav1.LabelSelector) bool {
	if s == nil {
		return false
	}
	selector, err := metav1.LabelSelectorAsSelector(s)
	if err != nil {
		return false
	}
	// get node name from NodeSelector
	nodeName, found := selector.RequiresExactMatch(r.NodeName)
	if !found {
		nodeName = ""
	}
	if nodeName == r.NodeName {
		return true
	}

	// check ip exit in node interface ip
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		return false
	}
	for _, p := range ips {
		tmpIP, _, err := net.ParseCIDR(p)
		if err != nil {
			continue
		}
		// skip localhost

		if ip, found := selector.RequiresExactMatch(tmpIP.String()); found {
			if tmpIP.String() == "127.0.0.1" || tmpIP.String() == "::1" {
				return false
			}
			if ip == tmpIP.String() {
				r.NodeIP = ip
				return true
			}
		}
	}
	return false
}
