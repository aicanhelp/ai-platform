/*
Copyright 2024.

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

package controller

import (
	"context"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"os"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	clusterscomputingoperatorv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
)

var _ = Describe("TenantManagedCluster Controller", func() {
	Context("When reconciling a resource", func() {
		const tcResourceName = "tc-test-resource"
		const tmcResourceName = "tc-test-resource"
		const namespace = "default"
		const nodeList = "node[1-10000]"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      tcResourceName,
			Namespace: namespace,
		}
		tenantmanagedcluster := &clusterscomputingoperatorv1alpha1.TenantManagedCluster{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind TenantManagedCluster")
			err := k8sClient.Get(ctx, typeNamespacedName, tenantmanagedcluster)
			if err != nil && errors.IsNotFound(err) {
				// 创建 TenantCluster
				tcResource := &clusterscomputingoperatorv1alpha1.TenantCluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      tcResourceName,
						Namespace: namespace,
					},
					Spec: clusterscomputingoperatorv1alpha1.TenantClusterSpec{
						Mode: "managed",
						NodeList: []string{
							nodeList,
						},
						ManagedClusters: clusterscomputingoperatorv1alpha1.ManagedClusters{
							Installer: clusterscomputingoperatorv1alpha1.Installer{
								FuyaoInstaller: clusterscomputingoperatorv1alpha1.FuyaoInstaller{
									Command: []string{
										"/bin/sh",
										"-c",
										"/opt/fuyao/fuyao-latest-openEuler-22.03-amd64/fuyao create cluster -f /root/cluster/config/%s.yaml --offline --artifact-path /opt/fuyao/fuyao-offline-artifact-multi.tar.gz --debug && echo install success && scp root@%s:/etc/kubernetes/admin.conf managed.cluster.admin.conf && kubectl --kubeconfig=config/tenant.cluster.admin.config create secret generic %s-managed-kubeconfig --from-file=kubeconfig=managed.cluster.admin.conf --dry-run=client -o yaml | kubectl --kubeconfig=config/tenant.cluster.admin.config apply -f -",
									},
								},
								Image: "ubuntu-huawei-cco:latest",
							},
						},
					},
				}
				Expect(k8sClient.Create(ctx, tcResource)).To(Succeed())

				// 创建 TenantManagedCluster
				tmcResource := &clusterscomputingoperatorv1alpha1.TenantManagedCluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      tmcResourceName,
						Namespace: namespace,
					},
					Spec: clusterscomputingoperatorv1alpha1.TenantManagedClusterSpec{
						Mode:       "managed",
						MasterList: []string{"192.168.1.10"},
					},
				}
				controllerutil.SetControllerReference(tcResource, tmcResource, k8sClient.Scheme())
				Expect(k8sClient.Create(ctx, tmcResource)).To(Succeed())
			}
		})

		AfterEach(func() {
			// TODO(user): Cleanup logic after each test, like removing the resource instance.
			resource := &clusterscomputingoperatorv1alpha1.TenantManagedCluster{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance TenantManagedCluster")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})
		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			oldDir, _ := os.Getwd()
			defer os.Chdir(oldDir)
			os.Chdir("../../")
			controllerReconciler := &TenantManagedClusterReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Created managedClusterJob")
			managedClusterJobList := &batchv1.JobList{}
			err = k8sClient.List(ctx, managedClusterJobList, &client.ListOptions{})
			Expect(managedClusterJobList.Items).To(HaveLen(1))
		})
	})
})
