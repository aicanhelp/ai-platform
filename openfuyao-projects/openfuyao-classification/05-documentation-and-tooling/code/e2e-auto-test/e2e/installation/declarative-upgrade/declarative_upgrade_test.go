package declarative_upgrade

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

// declarative-upgrade E2E skeleton (DT-01~06).
// Requires management cluster with declarative-upgrade feature gate and OCI artifacts.
// See docs/bke-declarative-upgrade-e2e-verification.md for manual steps.

var _ = Describe("Declarative Cluster Version Upgrade", Label("declarative-upgrade", "custom-repo"), func() {
	var (
		dynamicClient dynamic.Interface
		duHelper      *DeclarativeUpgradeHelper
		clusterName   string
		namespace     string
	)

	BeforeEach(func() {
		if strings.EqualFold(os.Getenv("ENABLE_DECLARATIVE_UPGRADE_E2E"), "true") {
			return
		}
		Skip("set ENABLE_DECLARATIVE_UPGRADE_E2E=true and provide OCI/mgmt cluster to run DT cases")
	})

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())

		duHelper = NewDeclarativeUpgradeHelper(dynamicClient)
		clusterName = os.Getenv("DECLARATIVE_TEST_CLUSTER")
		namespace = os.Getenv("DECLARATIVE_TEST_NAMESPACE")
		if namespace == "" {
			namespace = "default"
		}
		Expect(clusterName).NotTo(BeEmpty(), "DECLARATIVE_TEST_CLUSTER must be set")
	})

	Describe("DT-01 install auto ReleaseImage", func() {
		// Author: zqw1
		It("should have ClusterVersion and Valid ReleaseImage after install", func() {
			cv, err := duHelper.GetClusterVersion(namespace, clusterName)
			Expect(err).NotTo(HaveOccurred())
			desired, _, _ := unstructured.NestedString(cv.Object, "spec", "desiredVersion")
			Expect(desired).NotTo(BeEmpty())

			riList, err := duHelper.ListReleaseImages(namespace)
			Expect(err).NotTo(HaveOccurred())
			Expect(riList.Items).NotTo(BeEmpty())

			found := false
			for _, ri := range riList.Items {
				version, _, _ := unstructured.NestedString(ri.Object, "spec", "version")
				if version == desired {
					phase, _, _ := unstructured.NestedString(ri.Object, "status", "phase")
					Expect(phase).To(Equal("Valid"))
					found = true
					break
				}
			}
			Expect(found).To(BeTrue(), "no ReleaseImage matching desired version")
		})
	})

	Describe("DT-02 no ReleaseImage recreate after ClusterReady", func() {
		// Author: zqw1
		It("should not recreate ReleaseImage within 3 minutes after delete", func() {
			riList, err := duHelper.ListReleaseImages(namespace)
			Expect(err).NotTo(HaveOccurred())
			Expect(riList.Items).NotTo(BeEmpty())

			initialCount := len(riList.Items)
			deletedName := riList.Items[0].GetName()
			expectedCount := initialCount - 1

			err = dynamicClient.Resource(releaseImageGVR).Namespace(namespace).Delete(
				context.TODO(), deletedName, metav1.DeleteOptions{},
			)
			Expect(err).NotTo(HaveOccurred())

			// Wait for delete/finalizer to finish before the stability window.
			Eventually(func(g Gomega) {
				list, listErr := duHelper.ListReleaseImages(namespace)
				g.Expect(listErr).NotTo(HaveOccurred())
				g.Expect(list.Items).To(HaveLen(expectedCount))
				for _, ri := range list.Items {
					g.Expect(ri.GetName()).NotTo(Equal(deletedName))
				}
			}, 2*time.Minute, 5*time.Second).Should(Succeed())

			// [DT-02] Count stays at initial-1; deleted RI name must not reappear for 3 minutes.
			duHelper.AssertReleaseImageNotRecreated(namespace, deletedName, expectedCount, 3*time.Minute)
		})
	})

	Describe("DT-03 declarative upgrade precheck", func() {
		// Author: zqw1
		It("should mark BKECluster upgrade-ready after patching desiredVersion", func() {
			target := os.Getenv("DECLARATIVE_TARGET_VERSION")
			Expect(target).NotTo(BeEmpty(), "DECLARATIVE_TARGET_VERSION required")

			Expect(duHelper.PatchClusterVersionDesiredVersion(namespace, clusterName, target)).To(Succeed())

			// Wait by spec.version (not metadata.name); aligns with capbke ResolveReleaseImageForVersion.
			duHelper.WaitReleaseImageValidByVersion(namespace, target, 20*time.Minute)

			bc, err := duHelper.GetBKECluster(namespace, clusterName)
			Expect(err).NotTo(HaveOccurred())
			Expect(duHelper.BKEClusterHasUpgradeReadyAnnotation(bc, target)).To(BeTrue())
		})
	})

	Describe("DT-06 negative UpgradePath blocked", Label("declarative-upgrade-negative"), func() {
		var (
			targetVersion   string
			currentVersion  string
			upgradePathName string
			upBackup        *unstructured.Unstructured
		)

		BeforeEach(func() {
			targetVersion = os.Getenv("DECLARATIVE_TARGET_VERSION")
			Expect(targetVersion).NotTo(BeEmpty(), "DECLARATIVE_TARGET_VERSION required")

			upgradePathName = os.Getenv("DECLARATIVE_UPGRADE_PATH_NAME")
			if upgradePathName == "" {
				upgradePathName = "openfuyao-upgrade-paths"
			}

			duHelper.AssertUpgradePathActive(upgradePathName)

			var err error
			currentVersion, err = duHelper.ResolveClusterCurrentVersion(namespace, clusterName)
			Expect(err).NotTo(HaveOccurred())
			Expect(currentVersion).NotTo(Equal(targetVersion),
				"current and target must differ for blocked-path test")

			up, err := duHelper.GetUpgradePath(upgradePathName)
			Expect(err).NotTo(HaveOccurred())
			upBackup = up.DeepCopy()

			// Reset desiredVersion to current so precheck runs from a clean baseline.
			Expect(duHelper.PatchClusterVersionDesiredVersion(namespace, clusterName, currentVersion)).To(Succeed())
			duHelper.WaitBKEClusterUpgradeReadyCleared(namespace, clusterName, 3*time.Minute)

			DeferCleanup(func() {
				if upBackup != nil {
					_ = duHelper.RestoreUpgradePathSpec(upBackup)
				}
				_ = duHelper.PatchClusterVersionDesiredVersion(namespace, clusterName, currentVersion)
			})
		})

		// Author: zqw1
		It("should set ClusterVersion PreCheckFailed when path is blocked", func() {
			// [DT-06-1] Block the upgrade edge on cluster-scoped UpgradePath.
			Expect(duHelper.SetUpgradePathEdgeBlocked(
				upgradePathName, currentVersion, targetVersion, true,
			)).To(Succeed())

			// [DT-06-2] Trigger upgrade precheck (same user action as declarative upgrade).
			Expect(duHelper.PatchClusterVersionDesiredVersion(namespace, clusterName, targetVersion)).To(Succeed())

			// [DT-06-3] CV must fail precheck; BC must not receive upgrade-ready.
			duHelper.WaitClusterVersionPhase(namespace, clusterName, "PreCheckFailed", 5*time.Minute)

			bc, err := duHelper.GetBKECluster(namespace, clusterName)
			Expect(err).NotTo(HaveOccurred())
			duHelper.AssertBKEClusterNotUpgradeReady(bc, targetVersion)
		})
	})
})
