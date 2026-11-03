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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"

	sandboxv1alpha1 "gitcode.com/openFuyao/openfuyao-sandbox/api/v1alpha1"
	castorage "gitcode.com/openFuyao/openfuyao-sandbox/pkg/node-agent/storage"
)

var (
	hasKataRuntime        bool
	originalNodeAgentArgs []string
	nodeAgentPatched      bool
)

var _ = BeforeSuite(func() {
	ctx := context.Background()
	// Detect kata-qemu RuntimeClass
	rc := &nodev1.RuntimeClass{}
	err := k8sClient.Get(ctx, types.NamespacedName{Name: "kata-qemu"}, rc)
	hasKataRuntime = err == nil
	klog.InfoS("Environment detection", "hasKataRuntime", hasKataRuntime)

	// Patch node-agent DaemonSet to mock mode
	dsObj := &appsv1.DaemonSet{}
	err = k8sClient.Get(ctx, types.NamespacedName{
		Name:      "sandbox-node-agent",
		Namespace: "sandbox-system",
	}, dsObj)
	Expect(err).NotTo(HaveOccurred(), "node-agent DaemonSet must be deployed before running E2E tests")

	// Record original args
	originalNodeAgentArgs = make([]string, len(dsObj.Spec.Template.Spec.Containers[0].Args))
	copy(originalNodeAgentArgs, dsObj.Spec.Template.Spec.Containers[0].Args)

	// Check if already in mock mode
	alreadyMock := false
	for _, arg := range originalNodeAgentArgs {
		if arg == "--containerd-mode=mock" {
			alreadyMock = true
			break
		}
	}
	if alreadyMock {
		klog.InfoS("node-agent already in mock mode, skipping patch")
		return
	}

	// Patch to mock mode
	newArgs := make([]string, 0, len(originalNodeAgentArgs)+1)
	hasMode := false
	for _, arg := range originalNodeAgentArgs {
		if strings.HasPrefix(arg, "--containerd-mode=") {
			newArgs = append(newArgs, "--containerd-mode=mock")
			hasMode = true
		} else {
			newArgs = append(newArgs, arg)
		}
	}
	if !hasMode {
		newArgs = append(newArgs, "--containerd-mode=mock")
	}
	dsObj.Spec.Template.Spec.Containers[0].Args = newArgs
	Expect(k8sClient.Update(ctx, dsObj)).To(Succeed())
	nodeAgentPatched = true
	klog.InfoS("Patched node-agent DaemonSet to mock mode")

	// Wait for rollout
	Eventually(func() bool {
		d := &appsv1.DaemonSet{}
		_ = k8sClient.Get(ctx, types.NamespacedName{
			Name:      "sandbox-node-agent",
			Namespace: "sandbox-system",
		}, d)
		return d.Status.NumberReady == d.Status.DesiredNumberScheduled &&
			d.Status.UpdatedNumberScheduled == d.Status.DesiredNumberScheduled
	}, time.Minute*2, time.Second*5).Should(BeTrue(), "node-agent DaemonSet rollout should complete")
})

var _ = AfterSuite(func() {
	if !nodeAgentPatched {
		return
	}
	ctx := context.Background()
	dsObj := &appsv1.DaemonSet{}
	err := k8sClient.Get(ctx, types.NamespacedName{
		Name:      "sandbox-node-agent",
		Namespace: "sandbox-system",
	}, dsObj)
	if err != nil {
		return
	}
	dsObj.Spec.Template.Spec.Containers[0].Args = originalNodeAgentArgs
	_ = k8sClient.Update(ctx, dsObj)
	klog.InfoS("Restored node-agent DaemonSet to original args")

	Eventually(func() bool {
		d := &appsv1.DaemonSet{}
		_ = k8sClient.Get(ctx, types.NamespacedName{
			Name:      "sandbox-node-agent",
			Namespace: "sandbox-system",
		}, d)
		return d.Status.NumberReady == d.Status.DesiredNumberScheduled
	}, time.Minute*2, time.Second*5).Should(BeTrue(), "node-agent DaemonSet should be ready after restore")
})

var _ = Describe("Checkpoint", func() {
	var (
		sandbox    *sandboxv1alpha1.Sandbox
		checkpoint *sandboxv1alpha1.Checkpoint
		ctx        = context.Background()
		namespace  string
	)

	BeforeEach(func() {
		namespace = createNamespace(ctx)

		sandboxName := fmt.Sprintf("test-sandbox-%d", time.Now().UnixNano())

		// Conditionally set RuntimeClassName based on environment
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

		checkpointName := fmt.Sprintf("test-checkpoint-%d", time.Now().UnixNano())
		checkpoint = &sandboxv1alpha1.Checkpoint{
			ObjectMeta: metav1.ObjectMeta{
				Name:      checkpointName,
				Namespace: namespace,
				Annotations: map[string]string{
					sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
				},
			},
			Spec: sandboxv1alpha1.CheckpointSpec{
				SandboxName:        &sandboxName,
				KeepRunning:        ptrBool(true),
				PersistentContents: []string{"memory", "filesystem"},
			},
		}
	})

	AfterEach(func() {
		_ = k8sClient.Delete(ctx, checkpoint)
		_ = k8sClient.Delete(ctx, sandbox)
		_ = k8sClient.Delete(ctx, &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: namespace},
		})
	})

	Context("CP-E2E-001: checkpoint end-to-end lifecycle", func() {
		It("should create checkpoint for sandbox and complete lifecycle", func() {
			By("Creating a sandbox with kata-qemu runtime")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Verifying sandbox has pod information")
			Expect(sandbox.Status.PodInfo.PodIP).NotTo(BeEmpty())
			Expect(sandbox.Status.NodeName).NotTo(BeEmpty())

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint transitions to Creating phase")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return checkpoint.Status.Phase
			}, time.Second*30, time.Millisecond*500).Should(Or(
				Equal(sandboxv1alpha1.CheckpointCreating),
				Equal(sandboxv1alpha1.CheckpointSucceeded),
			))

			By("Verifying checkpoint has runtime type kata-qemu")
			Eventually(func() string {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return checkpoint.Status.RuntimeType
			}, time.Second*30, time.Millisecond*500).Should(Equal("kata-qemu"))

			By("Verifying checkpoint eventually succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				klog.InfoS("checkpoint status", "phase", checkpoint.Status.Phase, "message", checkpoint.Status.Message)
				return checkpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has archive location")
			Expect(checkpoint.Status.ArchiveLocation).NotTo(BeEmpty())
			Expect(checkpoint.Status.ArchiveLocation).To(ContainSubstring("local://"))

			By("Verifying checkpoint has archive size")
			Expect(checkpoint.Status.ArchiveSize).To(BeNumerically(">", 0))

			By("Verifying sandbox is still running (keepRunning=true)")
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      sandbox.Name,
				Namespace: sandbox.Namespace,
			}, sandbox)
			Expect(sandbox.Status.Phase).To(Equal(sandboxv1alpha1.SandboxRunning))
		})
	})

	Context("CP-E2E-002: checkpoint with non-kata runtime failure", func() {
		It("should fail when sandbox uses unsupported runtime", func() {
			By("Creating a sandbox without kata runtime (no RuntimeClassName)")
			sandbox.Spec.Template.Spec.RuntimeClassName = nil
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint without kata runtime")
			noKataCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &sandbox.Name,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
				},
			}
			Expect(k8sClient.Create(ctx, noKataCheckpoint)).To(Succeed())
			DeferCleanup(func() { _ = k8sClient.Delete(ctx, noKataCheckpoint) })

			By("Verifying checkpoint transitions to Failed")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      noKataCheckpoint.Name,
					Namespace: noKataCheckpoint.Namespace,
				}, noKataCheckpoint)
				klog.InfoS("checkpoint status", "phase", noKataCheckpoint.Status.Phase, "message", noKataCheckpoint.Status.Message)
				return noKataCheckpoint.Status.Phase
			}, time.Minute*1, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointFailed))

			By("Verifying failure message mentions runtime detection")
			Expect(noKataCheckpoint.Status.Message).To(SatisfyAny(
				ContainSubstring("RuntimeDetectionFailed"),
				ContainSubstring("kata-qemu runtime required"),
			))
		})
	})

	Context("CP-E2E-003: checkpoint with nonexistent sandbox failure", func() {
		It("should fail when referenced sandbox does not exist", func() {
			By("Creating a checkpoint referencing a nonexistent sandbox")
			nonexistent := "nonexistent-sandbox-xyz"
			noExistCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &nonexistent,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
				},
			}
			Expect(k8sClient.Create(ctx, noExistCheckpoint)).To(Succeed())
			DeferCleanup(func() { _ = k8sClient.Delete(ctx, noExistCheckpoint) })

			By("Verifying checkpoint transitions to Failed")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      noExistCheckpoint.Name,
					Namespace: noExistCheckpoint.Namespace,
				}, noExistCheckpoint)
				klog.InfoS("checkpoint status", "phase", noExistCheckpoint.Status.Phase, "message", noExistCheckpoint.Status.Message)
				return noExistCheckpoint.Status.Phase
			}, time.Minute*1, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointFailed))

			By("Verifying failure message mentions sandbox not found")
			Expect(noExistCheckpoint.Status.Message).To(ContainSubstring("SourceSandboxNotFound"))
		})
	})

	Context("CP-E2E-004: checkpoint without sandboxName failure", func() {
		It("should fail when sandboxName is not specified", func() {
			By("Creating a checkpoint without sandboxName")
			noNameCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
				},
			}
			Expect(k8sClient.Create(ctx, noNameCheckpoint)).To(Succeed())
			DeferCleanup(func() { _ = k8sClient.Delete(ctx, noNameCheckpoint) })

			By("Verifying checkpoint transitions to Failed")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      noNameCheckpoint.Name,
					Namespace: noNameCheckpoint.Namespace,
				}, noNameCheckpoint)
				klog.InfoS("checkpoint status", "phase", noNameCheckpoint.Status.Phase, "message", noNameCheckpoint.Status.Message)
				return noNameCheckpoint.Status.Phase
			}, time.Minute*1, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointFailed))

			By("Verifying failure message mentions sandboxName is required")
			Expect(noNameCheckpoint.Status.Message).To(ContainSubstring("sandboxName is required"))
		})
	})

	Context("CP-E2E-005: checkpoint with TTL completion time", func() {
		It("should record completion time when checkpoint succeeds with TTL", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint with short TTL")
			ttl := "30s"
			ttlCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &sandbox.Name,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
					TtlAfterFinished:   &ttl,
				},
			}
			Expect(k8sClient.Create(ctx, ttlCheckpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				return ttlCheckpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has completion time")
			Expect(ttlCheckpoint.Status.CompletionTime).NotTo(BeNil())
		})
	})

	Context("CP-E2E-006: checkpoint TTL auto-deletion", func() {
		It("should delete checkpoint after TTL expires", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint with short TTL (15s)")
			ttl := "15s"
			ttlCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &sandbox.Name,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
					TtlAfterFinished:   &ttl,
				},
			}
			Expect(k8sClient.Create(ctx, ttlCheckpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				klog.InfoS("checkpoint status", "phase", ttlCheckpoint.Status.Phase, "message", ttlCheckpoint.Status.Message)
				return ttlCheckpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has completion time")
			Expect(ttlCheckpoint.Status.CompletionTime).NotTo(BeNil())
			completionTime := ttlCheckpoint.Status.CompletionTime.Time
			klog.InfoS("checkpoint completed", "completionTime", completionTime)

			By("Verifying checkpoint still exists before TTL expires (within 10s of completion)")
			Consistently(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				if err != nil {
					klog.InfoS("checkpoint unexpectedly deleted before TTL", "elapsed", time.Since(completionTime))
					return false
				}
				klog.InfoS("checkpoint still exists", "elapsed", time.Since(completionTime).Truncate(time.Second))
				return true
			}, time.Second*10, time.Second*2).Should(BeTrue(), "Checkpoint should still exist within 10s (TTL=15s not expired)")

			By("Waiting for GC controller to delete the checkpoint after TTL expires (within 45s of completion)")
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				if err != nil {
					klog.InfoS("checkpoint deleted by GC", "elapsed", time.Since(completionTime).Truncate(time.Second))
					return true
				}
				var deletionTS interface{}
				if ttlCheckpoint.DeletionTimestamp != nil {
					deletionTS = ttlCheckpoint.DeletionTimestamp.Time
				}
				klog.InfoS("checkpoint still exists", "phase", ttlCheckpoint.Status.Phase,
					"elapsed", time.Since(completionTime).Truncate(time.Second),
					"deletionTimestamp", deletionTS)
				return false
			}, time.Second*45, time.Second*2).Should(BeTrue(), "Checkpoint should be deleted by GC controller within 45s (TTL=15s + buffer)")
		})
	})

	Context("CP-E2E-007: manual delete succeeded checkpoint finalizer cleanup", func() {
		It("should clean up archive and remove finalizer on deletion", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return checkpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has finalizer")
			Expect(checkpoint.Finalizers).To(ContainElement(sandboxv1alpha1.CheckpointFinalizer))

			By("Deleting the checkpoint")
			Expect(k8sClient.Delete(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint is fully deleted (finalizer removed, object gone)")
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				if err != nil && apierrors.IsNotFound(err) {
					return true
				}
				if err == nil {
					klog.InfoS("checkpoint still exists", "phase", checkpoint.Status.Phase,
						"finalizers", checkpoint.Finalizers, "deletionTimestamp", checkpoint.DeletionTimestamp)
				}
				return false
			}, time.Minute*1, time.Second*5).Should(BeTrue(), "Checkpoint should be fully deleted after manual delete")
		})
	})

	Context("CP-E2E-008: checkpoint with target container", func() {
		It("should checkpoint specific container and verify containerID", func() {
			By("Creating a sandbox with multiple containers")
			sandbox.Spec.Template.Spec.Containers = append(sandbox.Spec.Template.Spec.Containers, corev1.Container{
				Name:    "sidecar",
				Image:   "busybox:latest",
				Command: []string{"sh", "-c", "while true; do sleep 3600; done"},
			})
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Getting the sidecar containerID from pod status")
			pod := &corev1.Pod{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      sandbox.Name,
				Namespace: sandbox.Namespace,
			}, pod)).To(Succeed())
			sidecarContainerID := ""
			for _, cs := range pod.Status.ContainerStatuses {
				if cs.Name == "sidecar" {
					sidecarContainerID = cs.ContainerID
					break
				}
			}
			Expect(sidecarContainerID).NotTo(BeEmpty())

			By("Creating a checkpoint targeting specific container")
			targetContainer := "sidecar"
			targetCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &sandbox.Name,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
					TargetContainer:    &targetContainer,
				},
			}
			Expect(k8sClient.Create(ctx, targetCheckpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      targetCheckpoint.Name,
					Namespace: targetCheckpoint.Namespace,
				}, targetCheckpoint)
				return targetCheckpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint containerID corresponds to sidecar container")
			Expect(targetCheckpoint.Status.ContainerID).To(Equal(sidecarContainerID))
		})
	})

	Context("CP-E2E-009: checkpoint conditions state transition", func() {
		It("should set Archived and Ready conditions and remove Progressing on success", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return checkpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying Archived condition is True")
			archivedCond := findCondition(checkpoint.Status.Conditions, string(sandboxv1alpha1.CheckpointConditionArchived))
			Expect(archivedCond).NotTo(BeNil())
			Expect(archivedCond.Status).To(Equal(metav1.ConditionTrue))
			Expect(archivedCond.Reason).To(Equal("UploadComplete"))

			By("Verifying Ready condition is True")
			readyCond := findCondition(checkpoint.Status.Conditions, string(sandboxv1alpha1.CheckpointConditionReady))
			Expect(readyCond).NotTo(BeNil())
			Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))
			Expect(readyCond.Reason).To(Equal("CheckpointComplete"))

			By("Verifying Progressing condition is removed after success")
			progressingCond := findCondition(checkpoint.Status.Conditions, string(sandboxv1alpha1.CheckpointConditionProgressing))
			Expect(progressingCond).To(BeNil())
		})
	})

	Context("CP-E2E-010: succeeded checkpoint idempotency", func() {
		It("should not modify a succeeded checkpoint on subsequent reconciliation", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return checkpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Recording the checkpoint status after first success")
			originalPhase := checkpoint.Status.Phase
			originalCompletionTime := checkpoint.Status.CompletionTime.DeepCopy()
			originalArchiveLocation := checkpoint.Status.ArchiveLocation
			originalArchiveSize := checkpoint.Status.ArchiveSize

			By("Triggering a Reconcile by adding a no-op annotation")
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      checkpoint.Name,
				Namespace: checkpoint.Namespace,
			}, checkpoint)
			if checkpoint.Annotations == nil {
				checkpoint.Annotations = make(map[string]string)
			}
			checkpoint.Annotations["agents.kruise.io/idempotency-test"] = "trigger"
			Expect(k8sClient.Update(ctx, checkpoint)).To(Succeed())

			By("Waiting 5 seconds for controller to process the triggered Reconcile")
			time.Sleep(time.Second * 5)

			By("Verifying checkpoint status remains unchanged (idempotent)")
			_ = k8sClient.Get(ctx, types.NamespacedName{
				Name:      checkpoint.Name,
				Namespace: checkpoint.Namespace,
			}, checkpoint)
			Expect(checkpoint.Status.Phase).To(Equal(originalPhase))
			Expect(checkpoint.Status.CompletionTime.Time).To(BeTemporally("==", originalCompletionTime.Time))
			Expect(checkpoint.Status.ArchiveLocation).To(Equal(originalArchiveLocation))
			Expect(checkpoint.Status.ArchiveSize).To(Equal(originalArchiveSize))

			By("Verifying conditions remain stable")
			archivedCond := findCondition(checkpoint.Status.Conditions, string(sandboxv1alpha1.CheckpointConditionArchived))
			Expect(archivedCond).NotTo(BeNil())
			Expect(archivedCond.Status).To(Equal(metav1.ConditionTrue))
			readyCond := findCondition(checkpoint.Status.Conditions, string(sandboxv1alpha1.CheckpointConditionReady))
			Expect(readyCond).NotTo(BeNil())
			Expect(readyCond.Status).To(Equal(metav1.ConditionTrue))
		})
	})

	Context("CP-E2E-011: checkpoint TTL with day format", func() {
		It("should accept and process checkpoint with day-based TTL (e.g. 1d)", func() {
			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint with day-format TTL (1d)")
			ttl := "1d"
			ttlCheckpoint := &sandboxv1alpha1.Checkpoint{
				ObjectMeta: metav1.ObjectMeta{
					Name:      checkpoint.Name,
					Namespace: namespace,
					Annotations: map[string]string{
						sandboxv1alpha1.CheckpointMockRuntimeAnnotation: "true",
					},
				},
				Spec: sandboxv1alpha1.CheckpointSpec{
					SandboxName:        &sandbox.Name,
					KeepRunning:        ptrBool(true),
					PersistentContents: []string{"memory", "filesystem"},
					TtlAfterFinished:   &ttl,
				},
			}
			Expect(k8sClient.Create(ctx, ttlCheckpoint)).To(Succeed())

			By("Verifying checkpoint succeeds (TTL parsing did not reject '1d')")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				return ttlCheckpoint.Status.Phase
			}, time.Minute*2, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has completion time")
			Expect(ttlCheckpoint.Status.CompletionTime).NotTo(BeNil())

			By("Verifying checkpoint still exists (TTL of 1d has not expired)")
			Consistently(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      ttlCheckpoint.Name,
					Namespace: ttlCheckpoint.Namespace,
				}, ttlCheckpoint)
				return err == nil
			}, time.Second*30, time.Second*5).Should(BeTrue(), "Checkpoint with 1d TTL should still exist after 30 seconds")
		})
	})

	Context("CP-E2E-012: FeatureGate disabled", func() {
		It("should leave Checkpoint CR unprocessed when SandboxCheckpoint feature gate is false", func() {
			By("Patching controller deployment to disable SandboxCheckpoint feature gate")
			deploy := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      "sandbox-controller-manager",
				Namespace: "sandbox-system",
			}, deploy)).To(Succeed())

			originalArgs := deploy.Spec.Template.Spec.Containers[0].Args
			newArgs := make([]string, len(originalArgs))
			copy(newArgs, originalArgs)
			for i, arg := range newArgs {
				if arg == "--feature-gates=SandboxCheckpoint=true" {
					newArgs[i] = "--feature-gates=SandboxCheckpoint=false"
					break
				}
			}

			deploy.Spec.Template.Spec.Containers[0].Args = newArgs
			Expect(k8sClient.Update(ctx, deploy)).To(Succeed())

			DeferCleanup(func() {
				By("Restoring controller deployment feature gate to true")
				restoreDeploy := &appsv1.Deployment{}
				Expect(k8sClient.Get(ctx, types.NamespacedName{
					Name:      "sandbox-controller-manager",
					Namespace: "sandbox-system",
				}, restoreDeploy)).To(Succeed())
				restoreArgs := restoreDeploy.Spec.Template.Spec.Containers[0].Args
				for i, arg := range restoreArgs {
					if arg == "--feature-gates=SandboxCheckpoint=false" {
						restoreArgs[i] = "--feature-gates=SandboxCheckpoint=true"
						break
					}
				}
				restoreDeploy.Spec.Template.Spec.Containers[0].Args = restoreArgs
				Expect(k8sClient.Update(ctx, restoreDeploy)).To(Succeed())
				Eventually(func() bool {
					d := &appsv1.Deployment{}
					_ = k8sClient.Get(ctx, types.NamespacedName{
						Name:      "sandbox-controller-manager",
						Namespace: "sandbox-system",
					}, d)
					return d.Status.ReadyReplicas == *d.Spec.Replicas
				}, time.Minute*2, time.Second*5).Should(BeTrue(), "Controller should be ready after restoring feature gate")
			})

			By("Waiting for controller rollout to complete with feature gate disabled")
			Eventually(func() bool {
				d := &appsv1.Deployment{}
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      "sandbox-controller-manager",
					Namespace: "sandbox-system",
				}, d)
				return d.Status.UpdatedReplicas == *d.Spec.Replicas &&
					d.Status.ReadyReplicas == *d.Spec.Replicas &&
					d.Status.ObservedGeneration >= d.Generation
			}, time.Minute*2, time.Second*5).Should(BeTrue(), "Controller rollout should complete with feature gate disabled")

			By("Verifying only new pods with disabled feature gate are running")
			Eventually(func() bool {
				pods := &corev1.PodList{}
				_ = k8sClient.List(ctx, pods,
					client.InNamespace("sandbox-system"),
					client.MatchingLabels{"control-plane": "sandbox-controller-manager"},
				)
				if len(pods.Items) != 1 {
					klog.InfoS("Waiting for single controller pod", "count", len(pods.Items))
					return false
				}
				pod := pods.Items[0]
				if pod.Status.Phase != corev1.PodRunning {
					return false
				}
				for _, arg := range pod.Spec.Containers[0].Args {
					if arg == "--feature-gates=SandboxCheckpoint=false" {
						return true
					}
				}
				klog.InfoS("Controller pod does not have feature gate disabled", "args", pod.Spec.Containers[0].Args)
				return false
			}, time.Minute*2, time.Second*5).Should(BeTrue(), "Single controller pod with feature gate disabled should be running")

			By("Waiting 5 seconds for controller to fully initialize")
			time.Sleep(time.Second * 5)

			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Waiting 30 seconds and verifying checkpoint is not processed")
			Consistently(func() string {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				return string(checkpoint.Status.Phase)
			}, time.Second*30, time.Second*5).Should(BeEmpty(), "Checkpoint phase should remain empty when feature gate is disabled")

			By("Verifying checkpoint has no finalizers")
			Expect(checkpoint.Finalizers).To(BeEmpty())

			By("Verifying checkpoint has no conditions")
			Expect(checkpoint.Status.Conditions).To(BeEmpty())

			By("Verifying checkpoint has no archive location")
			Expect(checkpoint.Status.ArchiveLocation).To(BeEmpty())
		})
	})

	Context("CP-E2E-013: Node-Agent unreachable retry", func() {
		It("should keep Checkpoint in Pending and retry when node-agent is unreachable, then succeed after recovery", func() {
			By("Patching node-agent DaemonSet to prevent pod recreation")
			ds := &appsv1.DaemonSet{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      "sandbox-node-agent",
				Namespace: "sandbox-system",
			}, ds)).To(Succeed())

			ds.Spec.Template.Spec.NodeSelector = map[string]string{"non-existing-node-label-ckpt-test": "true"}
			Expect(k8sClient.Update(ctx, ds)).To(Succeed())

			DeferCleanup(func() {
				By("Restoring node-agent DaemonSet nodeSelector")
				restoreDS := &appsv1.DaemonSet{}
				Expect(k8sClient.Get(ctx, types.NamespacedName{
					Name:      "sandbox-node-agent",
					Namespace: "sandbox-system",
				}, restoreDS)).To(Succeed())
				restoreDS.Spec.Template.Spec.NodeSelector = nil
				Expect(k8sClient.Update(ctx, restoreDS)).To(Succeed())
				Eventually(func() bool {
					d := &appsv1.DaemonSet{}
					_ = k8sClient.Get(ctx, types.NamespacedName{
						Name:      "sandbox-node-agent",
						Namespace: "sandbox-system",
					}, d)
					return d.Status.NumberReady == d.Status.DesiredNumberScheduled && d.Status.DesiredNumberScheduled > 0
				}, time.Minute*2, time.Second*5).Should(BeTrue(), "Node-agent should be ready after restoring DaemonSet")
			})

			By("Deleting existing node-agent pods")
			podList := &corev1.PodList{}
			Expect(k8sClient.List(ctx, podList,
				client.InNamespace("sandbox-system"),
				client.MatchingLabels{"app": "sandbox-node-agent"},
			)).To(Succeed())
			for i := range podList.Items {
				Expect(k8sClient.Delete(ctx, &podList.Items[i])).To(Succeed())
			}

			By("Waiting for all node-agent pods to be terminated")
			Eventually(func() int {
				pods := &corev1.PodList{}
				_ = k8sClient.List(ctx, pods,
					client.InNamespace("sandbox-system"),
					client.MatchingLabels{"app": "sandbox-node-agent"},
				)
				return len(pods.Items)
			}, time.Minute*1, time.Second*5).Should(BeZero(), "All node-agent pods should be terminated")

			By("Creating a sandbox")
			Expect(k8sClient.Create(ctx, sandbox)).To(Succeed())

			By("Waiting for sandbox to reach Running phase")
			Eventually(func() sandboxv1alpha1.SandboxPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      sandbox.Name,
					Namespace: sandbox.Namespace,
				}, sandbox)
				return sandbox.Status.Phase
			}, time.Second*60, time.Millisecond*500).Should(Equal(sandboxv1alpha1.SandboxRunning))

			By("Creating a checkpoint")
			Expect(k8sClient.Create(ctx, checkpoint)).To(Succeed())

			By("Verifying checkpoint stays in Pending (not Failed) for 30 seconds")
			Consistently(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				klog.InfoS("checkpoint status while node-agent down", "phase", checkpoint.Status.Phase)
				return checkpoint.Status.Phase
			}, time.Second*30, time.Second*5).Should(Or(
				Equal(sandboxv1alpha1.CheckpointPhase("")),
				Equal(sandboxv1alpha1.CheckpointPending),
			), "Checkpoint should stay Pending or empty, not Failed, when node-agent is unreachable")

			By("Restoring node-agent DaemonSet")
			restoreDS := &appsv1.DaemonSet{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      "sandbox-node-agent",
				Namespace: "sandbox-system",
			}, restoreDS)).To(Succeed())
			restoreDS.Spec.Template.Spec.NodeSelector = nil
			Expect(k8sClient.Update(ctx, restoreDS)).To(Succeed())

			By("Waiting for node-agent pod to become ready")
			Eventually(func() bool {
				d := &appsv1.DaemonSet{}
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      "sandbox-node-agent",
					Namespace: "sandbox-system",
				}, d)
				return d.Status.NumberReady == d.Status.DesiredNumberScheduled && d.Status.DesiredNumberScheduled > 0
			}, time.Minute*2, time.Second*5).Should(BeTrue(), "Node-agent should be ready after restoring DaemonSet")

			By("Verifying checkpoint automatically recovers and succeeds")
			Eventually(func() sandboxv1alpha1.CheckpointPhase {
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name:      checkpoint.Name,
					Namespace: checkpoint.Namespace,
				}, checkpoint)
				klog.InfoS("checkpoint status after node-agent recovery", "phase", checkpoint.Status.Phase, "message", checkpoint.Status.Message)
				return checkpoint.Status.Phase
			}, time.Minute*3, time.Second*5).Should(Equal(sandboxv1alpha1.CheckpointSucceeded))

			By("Verifying checkpoint has archive location and size")
			Expect(checkpoint.Status.ArchiveLocation).NotTo(BeEmpty())
			Expect(checkpoint.Status.ArchiveSize).To(BeNumerically(">", 0))
		})
	})
})

var _ = Describe("Snapshot File Persistence", Label("storage-persistence"), func() {
	var (
		spCtx   context.Context
		tempDir string
	)

	BeforeEach(func() {
		spCtx = context.Background()
		var err error
		tempDir, err = os.MkdirTemp("", "sp-e2e-*")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
	})

	createTestArchive := func(content string) string {
		GinkgoHelper()
		srcFile := filepath.Join(tempDir, "src-archive.tar.gz")
		err := os.WriteFile(srcFile, []byte(content), 0644)
		Expect(err).NotTo(HaveOccurred())
		return srcFile
	}

	computeSHA256 := func(filePath string) string {
		GinkgoHelper()
		f, err := os.Open(filePath)
		Expect(err).NotTo(HaveOccurred())
		defer f.Close()
		h := sha256.New()
		_, err = io.Copy(h, f)
		Expect(err).NotTo(HaveOccurred())
		return hex.EncodeToString(h.Sum(nil))
	}

	newTestMetadata := func(checkpointID, namespace string) *castorage.Metadata {
		return &castorage.Metadata{
			CheckpointID:       checkpointID,
			Namespace:          namespace,
			ContainerID:        "containerd://abc123def456",
			NodeName:           "test-node-1",
			RuntimeType:        "kata-qemu",
			CreationTime:       time.Now(),
			ArchiveSize:        104857600,
			ChecksumSHA256:     "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
			PersistentContents: []string{"memory", "filesystem"},
		}
	}

	Context("SP-E2E-001: local backend full lifecycle", func() {
		It("should complete upload, metadata, checksum, download, and delete", func() {
			baseDir := filepath.Join(tempDir, "local-backend")
			backend, err := castorage.NewLocalBackend(baseDir)
			Expect(err).NotTo(HaveOccurred())

			archiveContent := "local-backend-test-archive-content"
			srcFile := createTestArchive(archiveContent)
			expectedChecksum := computeSHA256(srcFile)

			By("Uploading archive to local backend")
			result, err := backend.Upload(spCtx, "default", "sp-cp-001", srcFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			Expect(result.Location).To(HavePrefix("local://"))
			Expect(result.Size).To(Equal(int64(len(archiveContent))))
			Expect(result.ChecksumSHA256).To(Equal(expectedChecksum))

			By("Writing metadata alongside the archive")
			metadata := newTestMetadata("sp-cp-001", "default")
			metadata.ChecksumSHA256 = expectedChecksum
			err = backend.WriteMetadata(spCtx, "default", "sp-cp-001", metadata)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying metadata.json exists on disk")
			archivePath := result.Location
			if len(archivePath) > 8 && archivePath[:8] == "local://" {
				archivePath = archivePath[8:]
			}
			metaPath := filepath.Join(filepath.Dir(archivePath), "metadata.json")
			_, err = os.Stat(metaPath)
			Expect(err).NotTo(HaveOccurred())

			By("Retrieving checksum from metadata")
			checksum, err := backend.GetChecksum(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
			Expect(checksum).To(Equal(expectedChecksum))

			By("Downloading archive and verifying content")
			destFile := filepath.Join(tempDir, "downloaded.tar.gz")
			err = backend.Download(spCtx, result.Location, destFile)
			Expect(err).NotTo(HaveOccurred())
			downloadedContent, err := os.ReadFile(destFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(downloadedContent)).To(Equal(archiveContent))

			By("Deleting archive and verifying cleanup")
			err = backend.Delete(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
			checkpointDir := filepath.Dir(archivePath)
			_, err = os.Stat(checkpointDir)
			Expect(os.IsNotExist(err)).To(BeTrue())
		})
	})

	Context("SP-E2E-002: CSI backend full lifecycle", func() {
		It("should complete upload, metadata, download, and delete on CSI volume", func() {
			volumePath := filepath.Join(tempDir, "csi-volume")
			Expect(os.MkdirAll(volumePath, 0755)).To(Succeed())

			backend, err := castorage.NewCSIBackend(volumePath)
			Expect(err).NotTo(HaveOccurred())

			archiveContent := "csi-backend-test-archive-content"
			srcFile := createTestArchive(archiveContent)
			expectedChecksum := computeSHA256(srcFile)

			By("Uploading archive to CSI backend")
			result, err := backend.Upload(spCtx, "default", "sp-cp-002", srcFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Location).To(HavePrefix("csi://"))
			Expect(result.Size).To(Equal(int64(len(archiveContent))))
			Expect(result.ChecksumSHA256).To(Equal(expectedChecksum))

			By("Writing metadata")
			metadata := newTestMetadata("sp-cp-002", "default")
			metadata.ChecksumSHA256 = expectedChecksum
			err = backend.WriteMetadata(spCtx, "default", "sp-cp-002", metadata)
			Expect(err).NotTo(HaveOccurred())

			By("Retrieving checksum from metadata")
			checksum, err := backend.GetChecksum(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
			Expect(checksum).To(Equal(expectedChecksum))

			By("Downloading and verifying content")
			destFile := filepath.Join(tempDir, "csi-downloaded.tar.gz")
			err = backend.Download(spCtx, result.Location, destFile)
			Expect(err).NotTo(HaveOccurred())
			downloadedContent, err := os.ReadFile(destFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(downloadedContent)).To(Equal(archiveContent))

			By("Deleting and verifying cleanup")
			err = backend.Delete(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("SP-E2E-003: NFS backend full lifecycle", func() {
		It("should complete upload, metadata, download, and delete on NFS mount", func() {
			mountPath := filepath.Join(tempDir, "nfs-mount")
			Expect(os.MkdirAll(mountPath, 0755)).To(Succeed())

			backend, err := castorage.NewNFSBackend(mountPath)
			Expect(err).NotTo(HaveOccurred())

			archiveContent := "nfs-backend-test-archive-content"
			srcFile := createTestArchive(archiveContent)
			expectedChecksum := computeSHA256(srcFile)

			By("Uploading archive to NFS backend")
			result, err := backend.Upload(spCtx, "default", "sp-cp-003", srcFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Location).To(HavePrefix("nfs://"))
			Expect(result.Size).To(Equal(int64(len(archiveContent))))
			Expect(result.ChecksumSHA256).To(Equal(expectedChecksum))

			By("Writing metadata")
			metadata := newTestMetadata("sp-cp-003", "default")
			metadata.ChecksumSHA256 = expectedChecksum
			err = backend.WriteMetadata(spCtx, "default", "sp-cp-003", metadata)
			Expect(err).NotTo(HaveOccurred())

			By("Retrieving checksum from metadata")
			checksum, err := backend.GetChecksum(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
			Expect(checksum).To(Equal(expectedChecksum))

			By("Downloading and verifying content")
			destFile := filepath.Join(tempDir, "nfs-downloaded.tar.gz")
			err = backend.Download(spCtx, result.Location, destFile)
			Expect(err).NotTo(HaveOccurred())
			downloadedContent, err := os.ReadFile(destFile)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(downloadedContent)).To(Equal(archiveContent))

			By("Deleting and verifying cleanup")
			err = backend.Delete(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("SP-E2E-004: registry multi-backend coexistence", func() {
		It("should support registering and retrieving multiple backends", func() {
			localDir := filepath.Join(tempDir, "registry-local")
			csiDir := filepath.Join(tempDir, "registry-csi")
			nfsDir := filepath.Join(tempDir, "registry-nfs")
			Expect(os.MkdirAll(csiDir, 0755)).To(Succeed())
			Expect(os.MkdirAll(nfsDir, 0755)).To(Succeed())

			registry := castorage.NewRegistry()

			localBackend, err := castorage.NewLocalBackend(localDir)
			Expect(err).NotTo(HaveOccurred())
			registry.Register(localBackend, castorage.BackendLocal)

			csiBackend, err := castorage.NewCSIBackend(csiDir)
			Expect(err).NotTo(HaveOccurred())
			registry.Register(csiBackend, castorage.BackendCSI)

			nfsBackend, err := castorage.NewNFSBackend(nfsDir)
			Expect(err).NotTo(HaveOccurred())
			registry.Register(nfsBackend, castorage.BackendNFS)

			By("Verifying all backends are registered")
			Expect(registry.IsSupported(castorage.BackendLocal)).To(BeTrue())
			Expect(registry.IsSupported(castorage.BackendCSI)).To(BeTrue())
			Expect(registry.IsSupported(castorage.BackendNFS)).To(BeTrue())
			Expect(registry.IsSupported(castorage.BackendS3)).To(BeFalse())

			By("Verifying ListSupported returns all registered types")
			supported := registry.ListSupported()
			Expect(supported).To(HaveLen(3))
			Expect(supported).To(ContainElements(castorage.BackendLocal, castorage.BackendCSI, castorage.BackendNFS))

			By("Retrieving each backend and performing upload")
			for _, backendType := range []castorage.BackendType{castorage.BackendLocal, castorage.BackendCSI, castorage.BackendNFS} {
				backend, err := registry.Get(backendType)
				Expect(err).NotTo(HaveOccurred())

				srcFile := createTestArchive("registry-test-" + string(backendType))
				result, err := backend.Upload(spCtx, "default", "sp-cp-004-"+string(backendType), srcFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(result.Location).NotTo(BeEmpty())
			}

			By("Verifying Get returns error for unregistered backend")
			_, err = registry.Get(castorage.BackendS3)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not registered"))
		})
	})

	Context("SP-E2E-005: config-based backend initialization", func() {
		It("should initialize backends from environment variables", func() {
			By("Initializing local backend via config")
			localDir := filepath.Join(tempDir, "config-local")
			os.Setenv("STORAGE_LOCAL_BASE_DIR", localDir)
			defer os.Unsetenv("STORAGE_LOCAL_BASE_DIR")

			registry := castorage.NewRegistry()
			err := castorage.InitRegistry(spCtx, registry, "local")
			Expect(err).NotTo(HaveOccurred())
			Expect(registry.IsSupported(castorage.BackendLocal)).To(BeTrue())

			By("Initializing CSI backend via config")
			csiDir := filepath.Join(tempDir, "config-csi")
			Expect(os.MkdirAll(csiDir, 0755)).To(Succeed())
			os.Setenv("STORAGE_CSI_VOLUME_PATH", csiDir)
			defer os.Unsetenv("STORAGE_CSI_VOLUME_PATH")

			registry2 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry2, "csi")
			Expect(err).NotTo(HaveOccurred())
			Expect(registry2.IsSupported(castorage.BackendCSI)).To(BeTrue())

			By("Initializing NFS backend via config")
			nfsDir := filepath.Join(tempDir, "config-nfs")
			Expect(os.MkdirAll(nfsDir, 0755)).To(Succeed())
			os.Setenv("STORAGE_NFS_MOUNT_PATH", nfsDir)
			defer os.Unsetenv("STORAGE_NFS_MOUNT_PATH")

			registry3 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry3, "nfs")
			Expect(err).NotTo(HaveOccurred())
			Expect(registry3.IsSupported(castorage.BackendNFS)).To(BeTrue())

			By("Verifying unknown backend type returns error")
			registry4 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry4, "unknown-type")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("unknown storage backend type"))

			By("Verifying case-insensitive backend type")
			localDir2 := filepath.Join(tempDir, "config-local-upper")
			os.Setenv("STORAGE_LOCAL_BASE_DIR", localDir2)
			registry5 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry5, "LOCAL")
			Expect(err).NotTo(HaveOccurred())
			Expect(registry5.IsSupported(castorage.BackendLocal)).To(BeTrue())
		})
	})

	Context("SP-E2E-006: checksum integrity verification", func() {
		It("should detect tampered archive via checksum mismatch", func() {
			baseDir := filepath.Join(tempDir, "checksum-backend")
			backend, err := castorage.NewLocalBackend(baseDir)
			Expect(err).NotTo(HaveOccurred())

			archiveContent := "checksum-verification-test-content"
			srcFile := createTestArchive(archiveContent)
			expectedChecksum := computeSHA256(srcFile)

			By("Uploading archive and metadata")
			result, err := backend.Upload(spCtx, "default", "sp-cp-006", srcFile)
			Expect(err).NotTo(HaveOccurred())

			metadata := newTestMetadata("sp-cp-006", "default")
			metadata.ChecksumSHA256 = expectedChecksum
			err = backend.WriteMetadata(spCtx, "default", "sp-cp-006", metadata)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying checksum matches for unmodified archive")
			checksum, err := backend.GetChecksum(spCtx, result.Location)
			Expect(err).NotTo(HaveOccurred())
			Expect(checksum).To(Equal(expectedChecksum))

			By("Downloading and verifying downloaded file checksum matches")
			destFile := filepath.Join(tempDir, "checksum-downloaded.tar.gz")
			err = backend.Download(spCtx, result.Location, destFile)
			Expect(err).NotTo(HaveOccurred())
			downloadedChecksum := computeSHA256(destFile)
			Expect(downloadedChecksum).To(Equal(expectedChecksum))

			By("Tampering with the source archive and verifying checksum differs")
			tamperedFile := filepath.Join(tempDir, "tampered.tar.gz")
			err = os.WriteFile(tamperedFile, []byte("tampered-content"), 0644)
			Expect(err).NotTo(HaveOccurred())
			tamperedChecksum := computeSHA256(tamperedFile)
			Expect(tamperedChecksum).NotTo(Equal(expectedChecksum))
		})
	})

	Context("SP-E2E-007: metadata round-trip with all fields", func() {
		It("should persist and retrieve all metadata fields correctly", func() {
			baseDir := filepath.Join(tempDir, "metadata-backend")
			backend, err := castorage.NewLocalBackend(baseDir)
			Expect(err).NotTo(HaveOccurred())

			creationTime := time.Now().Truncate(time.Second)
			metadata := &castorage.Metadata{
				CheckpointID:       "sp-cp-007",
				Namespace:          "test-namespace",
				ContainerID:        "containerd://xyz789",
				NodeName:           "worker-node-3",
				RuntimeType:        "kata-qemu",
				CreationTime:       creationTime,
				ArchiveSize:        524288000,
				ChecksumSHA256:     "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
				PersistentContents: []string{"memory", "filesystem"},
			}

			By("Writing metadata")
			err = backend.WriteMetadata(spCtx, "test-namespace", "sp-cp-007", metadata)
			Expect(err).NotTo(HaveOccurred())

			By("Reading metadata.json directly and verifying all fields")
			metaPath := filepath.Join(baseDir, "test-namespace", "sp-cp-007", "metadata.json")
			metaBytes, err := os.ReadFile(metaPath)
			Expect(err).NotTo(HaveOccurred())

			var readMetadata castorage.Metadata
			err = json.Unmarshal(metaBytes, &readMetadata)
			Expect(err).NotTo(HaveOccurred())

			Expect(readMetadata.CheckpointID).To(Equal("sp-cp-007"))
			Expect(readMetadata.Namespace).To(Equal("test-namespace"))
			Expect(readMetadata.ContainerID).To(Equal("containerd://xyz789"))
			Expect(readMetadata.NodeName).To(Equal("worker-node-3"))
			Expect(readMetadata.RuntimeType).To(Equal("kata-qemu"))
			Expect(readMetadata.ArchiveSize).To(Equal(int64(524288000)))
			Expect(readMetadata.ChecksumSHA256).To(Equal("abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"))
			Expect(readMetadata.PersistentContents).To(Equal([]string{"memory", "filesystem"}))
		})
	})

	Context("SP-E2E-008: delete cleans up archive and metadata", func() {
		It("should remove both checkpoint.tar.gz and metadata.json on delete", func() {
			for _, tc := range []struct {
				name    string
				backend castorage.Backend
				prefix  string
			}{
				{
					name: "local",
					backend: func() castorage.Backend {
						b, _ := castorage.NewLocalBackend(filepath.Join(tempDir, "del-local"))
						return b
					}(),
					prefix: "local://",
				},
				{
					name: "csi",
					backend: func() castorage.Backend {
						dir := filepath.Join(tempDir, "del-csi")
						os.MkdirAll(dir, 0755)
						b, _ := castorage.NewCSIBackend(dir)
						return b
					}(),
					prefix: "csi://",
				},
				{
					name: "nfs",
					backend: func() castorage.Backend {
						dir := filepath.Join(tempDir, "del-nfs")
						os.MkdirAll(dir, 0755)
						b, _ := castorage.NewNFSBackend(dir)
						return b
					}(),
					prefix: "nfs://",
				},
			} {
				By("Testing delete for " + tc.name + " backend")
				srcFile := createTestArchive("delete-test-" + tc.name)
				result, err := tc.backend.Upload(spCtx, "default", "sp-cp-008-"+tc.name, srcFile)
				Expect(err).NotTo(HaveOccurred())

				metadata := newTestMetadata("sp-cp-008-"+tc.name, "default")
				err = tc.backend.WriteMetadata(spCtx, "default", "sp-cp-008-"+tc.name, metadata)
				Expect(err).NotTo(HaveOccurred())

				err = tc.backend.Delete(spCtx, result.Location)
				Expect(err).NotTo(HaveOccurred())

				archivePath := result.Location
				if len(archivePath) > len(tc.prefix) && archivePath[:len(tc.prefix)] == tc.prefix {
					archivePath = archivePath[len(tc.prefix):]
				}
				checkpointDir := filepath.Dir(archivePath)
				_, statErr := os.Stat(checkpointDir)
				Expect(os.IsNotExist(statErr)).To(BeTrue(), "checkpoint directory should be removed after delete")
			}
		})
	})

	Context("SP-E2E-009: location format validation", func() {
		It("should reject invalid location formats for each backend", func() {
			By("CSI backend rejects non-csi:// location")
			csiDir := filepath.Join(tempDir, "loc-csi")
			Expect(os.MkdirAll(csiDir, 0755)).To(Succeed())
			csiBackend, err := castorage.NewCSIBackend(csiDir)
			Expect(err).NotTo(HaveOccurred())

			err = csiBackend.Download(spCtx, "s3://bucket/key", filepath.Join(tempDir, "out"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid CSI location"))

			By("NFS backend rejects non-nfs:// location")
			nfsDir := filepath.Join(tempDir, "loc-nfs")
			Expect(os.MkdirAll(nfsDir, 0755)).To(Succeed())
			nfsBackend, err := castorage.NewNFSBackend(nfsDir)
			Expect(err).NotTo(HaveOccurred())

			err = nfsBackend.Download(spCtx, "local://some/path", filepath.Join(tempDir, "out"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid NFS location"))

			By("CSI backend rejects invalid delete location")
			err = csiBackend.Delete(spCtx, "invalid-location")
			Expect(err).To(HaveOccurred())

			By("NFS backend rejects invalid delete location")
			err = nfsBackend.Delete(spCtx, "invalid-location")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("SP-E2E-010: backend constructor validation", func() {
		It("should reject invalid constructor arguments", func() {
			By("CSI backend rejects empty volume path")
			_, err := castorage.NewCSIBackend("")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("empty"))

			By("CSI backend rejects non-existent path")
			_, err = castorage.NewCSIBackend("/nonexistent/path/xyz")
			Expect(err).To(HaveOccurred())

			By("CSI backend rejects file path (not directory)")
			filePath := filepath.Join(tempDir, "not-a-dir")
			Expect(os.WriteFile(filePath, []byte("file"), 0644)).To(Succeed())
			_, err = castorage.NewCSIBackend(filePath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not a directory"))

			By("NFS backend rejects empty mount path")
			_, err = castorage.NewNFSBackend("")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("empty"))

			By("NFS backend rejects non-existent path")
			_, err = castorage.NewNFSBackend("/nonexistent/nfs/xyz")
			Expect(err).To(HaveOccurred())

			By("NFS backend rejects file path (not directory)")
			_, err = castorage.NewNFSBackend(filePath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not a directory"))
		})
	})

	Context("SP-E2E-011: config rejects missing required env vars", func() {
		It("should return error when required environment variables are missing", func() {
			By("S3 backend requires all env vars")
			os.Unsetenv("STORAGE_S3_ENDPOINT")
			os.Unsetenv("STORAGE_S3_ACCESS_KEY_ID")
			os.Unsetenv("STORAGE_S3_SECRET_ACCESS_KEY")
			os.Unsetenv("STORAGE_S3_BUCKET_NAME")
			registry := castorage.NewRegistry()
			err := castorage.InitRegistry(spCtx, registry, "s3")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("S3 backend require"))

			By("MinIO backend requires all env vars")
			os.Unsetenv("STORAGE_MINIO_ENDPOINT")
			os.Unsetenv("STORAGE_MINIO_ACCESS_KEY_ID")
			os.Unsetenv("STORAGE_MINIO_SECRET_ACCESS_KEY")
			os.Unsetenv("STORAGE_MINIO_BUCKET_NAME")
			registry2 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry2, "minio")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("MinIO backend require"))

			By("CSI backend requires STORAGE_CSI_VOLUME_PATH")
			os.Unsetenv("STORAGE_CSI_VOLUME_PATH")
			registry3 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry3, "csi")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("CSI backend require"))

			By("NFS backend requires STORAGE_NFS_MOUNT_PATH")
			os.Unsetenv("STORAGE_NFS_MOUNT_PATH")
			registry4 := castorage.NewRegistry()
			err = castorage.InitRegistry(spCtx, registry4, "nfs")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("NFS backend require"))
		})
	})

	Context("SP-E2E-012: atomic write verification", func() {
		It("should not leave partial .tmp files after successful upload", func() {
			baseDir := filepath.Join(tempDir, "atomic-backend")
			backend, err := castorage.NewLocalBackend(baseDir)
			Expect(err).NotTo(HaveOccurred())

			srcFile := createTestArchive("atomic-write-test-content")

			By("Uploading archive")
			result, err := backend.Upload(spCtx, "default", "sp-cp-012", srcFile)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying no .tmp files remain in checkpoint directory")
			archivePath := result.Location
			if len(archivePath) > 8 && archivePath[:8] == "local://" {
				archivePath = archivePath[8:]
			}
			checkpointDir := filepath.Dir(archivePath)
			entries, err := os.ReadDir(checkpointDir)
			Expect(err).NotTo(HaveOccurred())
			for _, entry := range entries {
				Expect(filepath.Ext(entry.Name())).NotTo(Equal(".tmp"),
					"no .tmp files should remain after successful upload")
			}

			By("Verifying archive file exists and is readable")
			_, err = os.Stat(archivePath)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("SP-E2E-013: multiple checkpoints in same namespace", func() {
		It("should isolate checkpoint archives by checkpointID", func() {
			baseDir := filepath.Join(tempDir, "multi-backend")
			backend, err := castorage.NewLocalBackend(baseDir)
			Expect(err).NotTo(HaveOccurred())

			checkpoints := []struct {
				id      string
				content string
			}{
				{"sp-cp-013-a", "archive-content-alpha"},
				{"sp-cp-013-b", "archive-content-beta"},
				{"sp-cp-013-c", "archive-content-gamma"},
			}

			var results []*castorage.BackendUploadResult

			By("Uploading multiple checkpoints to same namespace")
			for _, cp := range checkpoints {
				srcFile := createTestArchive(cp.content)
				result, err := backend.Upload(spCtx, "shared-ns", cp.id, srcFile)
				Expect(err).NotTo(HaveOccurred())
				results = append(results, result)
			}

			By("Verifying each checkpoint is independently downloadable")
			for i, cp := range checkpoints {
				destFile := filepath.Join(tempDir, "multi-download-"+cp.id)
				err := backend.Download(spCtx, results[i].Location, destFile)
				Expect(err).NotTo(HaveOccurred())
				content, err := os.ReadFile(destFile)
				Expect(err).NotTo(HaveOccurred())
				Expect(string(content)).To(Equal(cp.content))
			}

			By("Deleting one checkpoint does not affect others")
			err = backend.Delete(spCtx, results[1].Location)
			Expect(err).NotTo(HaveOccurred())

			for i, cp := range checkpoints {
				if i == 1 {
					continue
				}
				destFile := filepath.Join(tempDir, "multi-verify-"+cp.id)
				err := backend.Download(spCtx, results[i].Location, destFile)
				Expect(err).NotTo(HaveOccurred())
			}
		})
	})
})

func ptrBool(v bool) *bool {
	return &v
}

func ptrString(v string) *string {
	return &v
}

func createNamespace(ctx context.Context) string {
	rand.Seed(time.Now().UnixNano())
	chars := "abcdefghijklmnopqrstuvwxyz0123456789"
	suffix := make([]byte, 5)
	for i := range suffix {
		suffix[i] = chars[rand.Intn(len(chars))]
	}
	name := "checkpoint-e2e-" + string(suffix)
	obj := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	Expect(k8sClient.Create(ctx, obj)).To(Succeed())
	return name
}

// findCondition returns the condition with the given type from the condition list, or nil if not found.
func findCondition(conditions []metav1.Condition, conditionType string) *metav1.Condition {
	for i := range conditions {
		if conditions[i].Type == conditionType {
			return &conditions[i]
		}
	}
	return nil
}

