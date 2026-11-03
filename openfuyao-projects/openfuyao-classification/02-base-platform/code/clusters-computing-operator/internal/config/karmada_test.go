package config_test

import (
	"fmt"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1alpha1 "openfuyao.com/clusters-computing-operator/api/v1alpha1"
	"openfuyao.com/clusters-computing-operator/internal/config"
	"os"
	"path/filepath"
)

var _ = Describe("LoadKarmadaConfig", func() {

	const (
		apiVersion     = "operator.karmada.io/v1alpha1"
		kind           = "Karmada"
		clusterName    = "test-cluster"
		namespace      = "test-namespace"
		karmadaName    = "tenantcluster-karmada"
		etcdRepository = "registry.k8s.io/etcd"
		etcdVersion    = "3.5.16-0"

		KarmadaAPIServerRepository           = "registry.k8s.io/kube-apiserver"
		karmadaAggregatedAPIServerRepository = "docker.io/karmada/karmada-aggregated-apiserver"
		karmadaVersion                       = "v1.11.1"
		kubeVersion                          = "v1.31.3"
		crdUrl                               = "https://example.com/crds.tar.gz"
	)
	var tenantCluster *clusterv1alpha1.TenantCluster

	BeforeEach(func() {
		tenantCluster = &clusterv1alpha1.TenantCluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      clusterName,
				Namespace: namespace,
			},
			Spec: clusterv1alpha1.TenantClusterSpec{
				ControlPlane: clusterv1alpha1.ControlPlane{
					Karmada: clusterv1alpha1.KarmadaDeployment{
						Name:           karmadaName,
						EtcdVersion:    etcdVersion,
						KarmadaVersion: karmadaVersion,
						KubeVersion:    kubeVersion,
						CrdUrl:         crdUrl,
					},
				},
			},
		}
	})
	Context("Use the correct YAML file", func() {
		It("Should correctly load the configuration", func() {
			cwd, _ := os.Getwd()
			fmt.Println("Current working directory:", cwd)
			karmada, err := config.LoadKarmadaConfig("../../config/app/cr/karmada.yaml", tenantCluster)
			Expect(err).NotTo(HaveOccurred())
			Expect(karmada).NotTo(BeNil())

			Expect(karmada.APIVersion).To(Equal(apiVersion))
			Expect(karmada.Kind).To(Equal(kind))
			Expect(karmada.Name).To(Equal(karmadaName))
			Expect(karmada.Namespace).To(Equal(namespace))

			// check crd url
			Expect(karmada.Spec.CRDTarball.HTTPSource.URL).To(Equal(crdUrl))

			// check some components
			Expect(karmada.Spec.Components.Etcd.Local.ImageRepository).To(Equal(etcdRepository))
			Expect(karmada.Spec.Components.Etcd.Local.ImageTag).To(Equal(etcdVersion))
			Expect(karmada.Spec.Components.KarmadaAPIServer.ImageRepository).To(Equal(KarmadaAPIServerRepository))
			Expect(karmada.Spec.Components.KarmadaAPIServer.ImageTag).To(Equal(kubeVersion))
			Expect(karmada.Spec.Components.KarmadaAggregatedAPIServer.ImageRepository).To(Equal(karmadaAggregatedAPIServerRepository))
			Expect(karmada.Spec.Components.KarmadaAggregatedAPIServer.ImageTag).To(Equal(karmadaVersion))
		})
	})

	Context("File does not exist", func() {
		It("Should return an error", func() {
			nonExistentPath := filepath.Join("config", "app", "cr", "nonexistent.yaml")
			cfg, err := config.LoadKarmadaConfig(nonExistentPath, tenantCluster)
			Expect(err).To(HaveOccurred())
			Expect(cfg).To(BeNil())
		})
	})
})
