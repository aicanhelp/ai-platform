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
	karmadav1alpha1 "github.com/karmada-io/karmada/operator/pkg/apis/operator/v1alpha1"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"strconv"

	clusterscomputingoperatorv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
)

var _ = Describe("TenantCluster Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"
		const namespaceName = "default"
		const nodeList = "node[1-10000]"
		const numClusters = 2
		const karmadaName = "tenantcluster-karmada"
		const karmadaEtcdVersion = "3.5.16-0"

		ctx := context.Background()
		labels := map[string]string{
			ManagedLabel: clusterscomputingoperatorv1alpha1.GroupVersion.Group,
		}

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: namespaceName,
		}
		tenantcluster := &clusterscomputingoperatorv1alpha1.TenantCluster{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind TenantCluster")
			os.Setenv("KUBERNETES_SERVICE_HOST", "127.0.0.1")
			os.Setenv("KUBERNETES_SERVICE_PORT", "6443")
			err := k8sClient.Get(ctx, typeNamespacedName, tenantcluster)
			if err != nil && errors.IsNotFound(err) {
				resource := &clusterscomputingoperatorv1alpha1.TenantCluster{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: namespaceName,
					},
					Spec: clusterscomputingoperatorv1alpha1.TenantClusterSpec{
						Mode: "managed",
						NodeList: []string{
							nodeList,
						},
						ControlPlane: clusterscomputingoperatorv1alpha1.ControlPlane{
							SuperClusterd: clusterscomputingoperatorv1alpha1.ManagerDeployment{
								Image: "superclusterd",
							},
							Karmada: clusterscomputingoperatorv1alpha1.KarmadaDeployment{
								Name:           karmadaName,
								EtcdVersion:    karmadaEtcdVersion,
								KarmadaVersion: "v1.11.1",
								KubeVersion:    "v1.31.3",
								CrdUrl:         "http://localhost:8080/crds.tar.gz",
							},
						},
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {

			// TODO(user): Cleanup logic after each test, like removing the resource instance.
			resource := &clusterscomputingoperatorv1alpha1.TenantCluster{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance TenantCluster")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})

		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			scheme := k8sClient.Scheme()
			karmadav1alpha1.AddToScheme(scheme)
			oldDir, _ := os.Getwd()
			defer os.Chdir(oldDir)
			os.Chdir("../../")
			controllerReconciler := &TenantClusterReconciler{
				Client: k8sClient,
				Scheme: scheme,
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			var deploymentList appsv1.DeploymentList
			By("Created superclusterd Deployment")
			err = k8sClient.List(ctx, &deploymentList, client.InNamespace(namespaceName), client.MatchingLabels(labels))
			Expect(err).NotTo(HaveOccurred())

			Expect(len(deploymentList.Items)).To(Equal(1))
			deployment := deploymentList.Items[0]
			Expect(strings.HasPrefix(deployment.Name, resourceName+"-superclusterd")).To(Equal(true))
			Expect(deployment.OwnerReferences[0].Name).To(Equal(resourceName))

			By("Created karmada instance")
			karmada := &karmadav1alpha1.Karmada{}
			err = k8sClient.Get(ctx, client.ObjectKey{
				Name:      karmadaName,
				Namespace: namespaceName,
			}, karmada)
			Expect(err).NotTo(HaveOccurred())
			Expect(karmada.Spec.Components.Etcd.Local.ImageTag).To(Equal(karmadaEtcdVersion))

			By("Created member TenantManagedClusters")
			for i := 1; i <= numClusters; i++ {
				clusterName := resourceName + "-member-" + strconv.Itoa(i)
				mCluster := &clusterscomputingoperatorv1alpha1.TenantManagedCluster{}
				name := types.NamespacedName{
					Name:      clusterName,
					Namespace: namespaceName,
				}

				err = k8sClient.Get(ctx, name, mCluster)
				Expect(err).NotTo(HaveOccurred())
				Expect(mCluster.OwnerReferences[0].Name).To(Equal(resourceName))
			}
		})
	})
})
