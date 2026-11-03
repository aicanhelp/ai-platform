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

package sandbox_checkpoint_restore

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"

	sandboxv1alpha1 "gitcode.com/openFuyao/openfuyao-sandbox/api/v1alpha1"
)

var _ = Describe("Restore", Label("restore"), func() {
	var (
		sandbox    *sandboxv1alpha1.Sandbox
		checkpoint *sandboxv1alpha1.Checkpoint
		ctx        = context.Background()
		namespace  string
	)

	BeforeEach(func() {
		namespace = createNamespace(ctx)

		sandboxName := fmt.Sprintf("test-restore-sandbox-%d", time.Now().UnixNano())

		var runtimeClassName *string
		if hasKataRuntime {
			runtimeClassName = ptrString("kata-qemu")
		}

		sandbox = &sandboxv1alpha1.Sandbox{
			ObjectMeta: metav1.ObjectMeta{
				Name:      sandboxName,
				Namespace: namespace,
			},
			Spec: sandboxv1alpha1.SandboxSpec{
				EmbeddedSandboxTemplate: sandboxv1alpha1.EmbeddedSandboxTemplate{
					Template: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							RuntimeClassName: runtimeClassName,
							Containers: []corev1.Container{
								{
									Name:  "sandbox",
									Image: "nginx:stable-alpine3.20",
									Ports: []corev1.ContainerPort{
										{
											Name:          "http",
											ContainerPort: 80,
										},
									},
								},
							},
							RestartPolicy: corev1.RestartPolicyNever,
						},
					},
				},
			},
		}

		checkpointName := fmt.Sprintf("test-restore-checkpoint-%d", time.Now().UnixNano())
		checkpoint = &sandboxv1alpha1.Checkpoint{
			ObjectMeta: metav1.ObjectMeta{
				Name:      checkpointName,
				Namespace: namespace,
			},
		}
	})

	AfterEach(func() {
		_ = k8sClient.Delete(ctx, sandbox)
		_ = k8sClient.Delete(ctx, checkpoint)

		checkpointList := &sandboxv1alpha1.CheckpointList{}
		if err := k8sClient.List(ctx, checkpointList, client.InNamespace(namespace)); err == nil {
			for i := range checkpointList.Items {
				cp := &checkpointList.Items[i]
				if len(cp.Finalizers) > 0 {
					patch := client.MergeFrom(cp.DeepCopy())
					cp.Finalizers = nil
					_ = k8sClient.Patch(ctx, cp, patch)
				}
			}
		}

		_ = k8sClient.Delete(ctx, &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: namespace},
		})

		Eventually(func() error {
			return k8sClient.Get(ctx, types.NamespacedName{Name: namespace}, &corev1.Namespace{})
		}, time.Minute, time.Second).ShouldNot(Succeed())
	})

	It("RESTORE-E2E-001: 从Succeeded的Checkpoint恢复Sandbox到Running", func() {
		By("Creating a checkpoint and setting it to Succeeded with ArchiveLocation")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/local/test-archive.tar.gz"
			cp.Status.RuntimeType = "kata-qemu"
			cp.Status.ArchiveSize = 1024
			cp.Status.NodeName = "test-node"
			cp.Status.CompletionTime = &metav1.Time{Time: time.Now()}
		})).To(Succeed())

		By("Verifying checkpoint is Succeeded with ArchiveLocation")
		Eventually(func() sandboxv1alpha1.CheckpointPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      checkpoint.Name,
				Namespace: checkpoint.Namespace,
			}, checkpoint)
			return checkpoint.Status.Phase
		}, time.Second*30, time.Millisecond*500).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

		By("Creating a sandbox with restore-from annotation referencing the checkpoint")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Restoring phase (Pod created)")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			klog.InfoS("sandbox status", "phase", restoreSandbox.Status.Phase, "message", restoreSandbox.Status.Message)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRestoring))

		By("Verifying sandbox has Restoring condition")
		restoringCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Status).To(Equal(metav1.ConditionTrue))

		By("Verifying restore Pod has correct annotations")
		podList := &corev1.PodList{}
		Expect(k8sClient.List(ctx, podList, client.InNamespace(namespace))).To(Succeed())
		Expect(podList.Items).NotTo(BeEmpty())
		restorePod := &podList.Items[0]
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodArchiveLocation]).NotTo(BeEmpty())
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodRestoreCheckpointRef]).To(Equal(checkpoint.Name))
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodRestoreRuntimeType]).To(Equal("kata-qemu"))

		By("Waiting for sandbox to reach Running phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			klog.InfoS("sandbox status", "phase", restoreSandbox.Status.Phase,
				"message", restoreSandbox.Status.Message)
			return restoreSandbox.Status.Phase
		}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.SandboxRunning))

		By("Verifying sandbox restore status fields")
		Expect(restoreSandbox.Status.RestoreFromCheckpoint).To(Equal(checkpoint.Name))
		Expect(restoreSandbox.Status.RestoreCompletionTime).NotTo(BeNil())
		Expect(restoreSandbox.Status.NodeName).NotTo(BeEmpty())

		By("Verifying Restoring condition is False with RestoreCompleted reason")
		restoringCond = findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Status).To(Equal(metav1.ConditionFalse))
		Expect(restoringCond.Reason).To(Equal("RestoreCompleted"))

		By("Verifying Ready condition is True")
		readyCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionReady))
		Expect(readyCond).NotTo(BeNil())
		Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))
	})

	It("RESTORE-E2E-002: 引用Failed的Checkpoint导致Sandbox失败", func() {
		By("Creating a checkpoint and setting it to Failed phase")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointFailed
			cp.Status.Message = "checkpoint creation failed"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation referencing the failed checkpoint")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Failed phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxFailed))

		By("Verifying failure message mentions checkpoint phase")
		Expect(restoreSandbox.Status.Message).To(ContainSubstring("not Succeeded"))
		Expect(restoreSandbox.Status.Message).To(ContainSubstring(checkpoint.Name))
	})

	It("RESTORE-E2E-003: 引用不存在的Checkpoint导致Sandbox失败", func() {
		By("Creating a sandbox with restore-from annotation referencing nonexistent checkpoint")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: "nonexistent-checkpoint-xyz",
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Failed phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxFailed))

		By("Verifying failure message mentions checkpoint not found")
		Expect(restoreSandbox.Status.Message).To(ContainSubstring("not found"))
	})

	It("RESTORE-E2E-004: Checkpoint缺少ArchiveLocation导致Sandbox失败", func() {
		By("Creating a Succeeded checkpoint without ArchiveLocation")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.RuntimeType = "kata-qemu"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Failed phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxFailed))

		By("Verifying failure message mentions no ArchiveLocation")
		Expect(restoreSandbox.Status.Message).To(ContainSubstring("no ArchiveLocation"))
	})

	It("RESTORE-E2E-005: Checkpoint缺少RuntimeType导致Sandbox失败", func() {
		By("Creating a Succeeded checkpoint with ArchiveLocation but no RuntimeType")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/local/test-archive.tar.gz"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Failed phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxFailed))

		By("Verifying failure message mentions no RuntimeType")
		Expect(restoreSandbox.Status.Message).To(ContainSubstring("no RuntimeType"))
	})

	It("RESTORE-E2E-006: RestorePod注解注入验证（含vmm-type）", func() {
		By("Creating a checkpoint with kata-qemu runtime")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/local/kata-qemu-archive.tar.gz"
			cp.Status.RuntimeType = "kata-qemu"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Restoring phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRestoring))

		By("Verifying Pod annotations include vmm-type derived from runtime-type")
		podList := &corev1.PodList{}
		Expect(k8sClient.List(ctx, podList, client.InNamespace(namespace))).To(Succeed())
		Expect(podList.Items).NotTo(BeEmpty())
		restorePod := &podList.Items[0]
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodArchiveLocation]).NotTo(BeEmpty())
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodRestoreCheckpointRef]).To(Equal(checkpoint.Name))
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodRestoreRuntimeType]).To(Equal("kata-qemu"))
		Expect(restorePod.Annotations[sandboxv1alpha1.AnnotationPodRestoreVMMType]).To(Equal("qemu"))
	})

	It("RESTORE-E2E-007: 无restore注解的Sandbox正常创建", func() {
		By("Creating a sandbox without restore-from annotation")
		Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

		By("Waiting for sandbox to reach Running phase (normal creation)")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      sandbox.Name,
				Namespace: sandbox.Namespace,
			}, sandbox)
			return sandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

		By("Verifying sandbox has no restore-related status fields")
		Expect(sandbox.Status.RestoreFromCheckpoint).To(BeEmpty())
		Expect(sandbox.Status.RestoreCompletionTime).To(BeNil())

		By("Verifying sandbox has no Restoring condition")
		restoringCond := findCondition(sandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).To(BeNil())
	})

	It("RESTORE-E2E-008: DownloadSnapshot失败阻断sandbox创建", func() {
		By("Creating a checkpoint with non-existent archive location")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/does-not-exist.tar.gz"
			cp.Status.RuntimeType = "kata-qemu"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation and kata-qemu runtime")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		kataRuntime := "kata-qemu"
		restoreSandbox.Spec.Template.Spec.RuntimeClassName = &kataRuntime
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Restoring phase (Pod created)")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRestoring))

		By("Verifying FailedCreatePodSandBox event with detectRestoreSnapshot error")
		Eventually(func() bool {
			events := &corev1.EventList{}
			_ = k8sClient.List(ctx, events, client.InNamespace(namespace))
			for _, event := range events.Items {
				if event.Reason == "FailedCreatePodSandBox" &&
					containsSubstring(event.Message, "detectRestoreSnapshot failed") {
					return true
				}
			}
			return false
		}, time.Second*60, time.Millisecond*500).Should(BeTrue())

		By("Verifying Pod stays in Pending phase (sandbox creation blocked)")
		podList := &corev1.PodList{}
		Expect(k8sClient.List(ctx, podList, client.InNamespace(namespace))).To(Succeed())
		Expect(podList.Items).NotTo(BeEmpty())
		Expect(podList.Items[0].Status.Phase).To(Equal(corev1.PodPending))

		By("Verifying sandbox does NOT transition to Running within observation window")
		Consistently(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*30, time.Second*5).Should(Equal(sandboxv1alpha1.SandboxRestoring))

		By("Verifying Restoring condition remains True (no completion)")
		restoringCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Status).To(Equal(metav1.ConditionTrue))
		Expect(restoringCond.Reason).To(Equal("PodCreated"))

		By("Verifying Ready condition is NOT set")
		readyCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionReady))
		Expect(readyCond).To(BeNil())
	})

	It("RESTORE-E2E-009: 有效archive路径下Sandbox恢复成功", func() {
		By("Creating a checkpoint with valid archive location")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/valid-archive.tar.gz"
			cp.Status.RuntimeType = "kata-qemu"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Running phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.SandboxRunning))

		By("Verifying sandbox restore status fields")
		Expect(restoreSandbox.Status.RestoreFromCheckpoint).To(Equal(checkpoint.Name))
		Expect(restoreSandbox.Status.RestoreCompletionTime).NotTo(BeNil())

		By("Verifying Restoring condition is False with RestoreCompleted reason")
		restoringCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Status).To(Equal(metav1.ConditionFalse))
		Expect(restoringCond.Reason).To(Equal("RestoreCompleted"))

		By("Verifying Ready condition is True")
		readyCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionReady))
		Expect(readyCond).NotTo(BeNil())
		Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))
	})

	It("RESTORE-E2E-010: Restore完成状态转换（Restoring→Running）", func() {
		By("Creating a checkpoint with valid archive location")
		Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())
		Expect(updateCheckpointStatus(ctx, checkpoint, func(cp *sandboxv1alpha1.Checkpoint) {
			cp.Status.Phase = sandboxv1alpha1.CheckpointSucceeded
			cp.Status.ArchiveLocation = "local:///var/lib/checkpoints/valid-archive.tar.gz"
			cp.Status.RuntimeType = "kata-qemu"
		})).To(Succeed())

		By("Creating a sandbox with restore-from annotation")
		restoreSandbox := sandbox.DeepCopy()
		restoreSandbox.Annotations = map[string]string{
			sandboxv1alpha1.AnnotationRestoreFrom: checkpoint.Name,
		}
		Expect(k8sClient.Create(ctx, restoreSandbox)).To(Succeed())

		By("Waiting for sandbox to reach Restoring phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRestoring))

		By("Verifying Restoring condition is True")
		restoringCond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Status).To(Equal(metav1.ConditionTrue))
		Expect(restoringCond.Reason).To(Equal("PodCreated"))

		By("Waiting for sandbox to reach Running phase")
		Eventually(func() sandboxv1alpha1.SandboxPhase {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			return restoreSandbox.Status.Phase
		}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.SandboxRunning))

		By("Verifying Restoring condition transitions to False")
		Eventually(func() metav1.ConditionStatus {
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      restoreSandbox.Name,
				Namespace: restoreSandbox.Namespace,
			}, restoreSandbox)
			cond := findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
			if cond != nil {
				return cond.Status
			}
			return metav1.ConditionUnknown
		}, time.Minute*2, time.Second*5).Should(Equal(metav1.ConditionFalse))

		By("Verifying Restoring condition reason is RestoreCompleted")
		restoringCond = findCondition(restoreSandbox.Status.Conditions, string(sandboxv1alpha1.SandboxConditionRestoring))
		Expect(restoringCond).NotTo(BeNil())
		Expect(restoringCond.Reason).To(Equal("RestoreCompleted"))
		Expect(restoringCond.Message).To(ContainSubstring("Sandbox restored from checkpoint"))
	})
})

// updateCheckpointStatus updates Checkpoint status using retry-on-conflict
// to handle concurrent controller updates.
func updateCheckpointStatus(ctx context.Context, checkpoint *sandboxv1alpha1.Checkpoint, modify func(*sandboxv1alpha1.Checkpoint)) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest := &sandboxv1alpha1.Checkpoint{}
		if err := k8sClient.Get(ctx, types.NamespacedName{
			Name:      checkpoint.Name,
			Namespace: checkpoint.Namespace,
		}, latest); err != nil {
			return err
		}
		modify(latest)
		return k8sClient.Status().Update(ctx, latest)
	})
}

// containsSubstring checks if a string contains a substring
func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || containsSubstring(s[1:], substr)))
}
