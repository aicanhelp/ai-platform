package declarative_upgrade

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

var (
	clusterVersionGVR = schema.GroupVersionResource{
		Group:    "config.openfuyao.com",
		Version:  "v1alpha1",
		Resource: "clusterversions",
	}
	releaseImageGVR = schema.GroupVersionResource{
		Group:    "config.openfuyao.com",
		Version:  "v1alpha1",
		Resource: "releaseimages",
	}
	upgradePathGVR = schema.GroupVersionResource{
		Group:    "config.openfuyao.com",
		Version:  "v1alpha1",
		Resource: "upgradepaths",
	}
	bkeClusterGVR = schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkeclusters",
	}
)

// DeclarativeUpgradeHelper wraps dynamic client queries for CV/RI/UP CRs.
type DeclarativeUpgradeHelper struct {
	client dynamic.Interface
}

func NewDeclarativeUpgradeHelper(client dynamic.Interface) *DeclarativeUpgradeHelper {
	return &DeclarativeUpgradeHelper{client: client}
}

func (h *DeclarativeUpgradeHelper) GetClusterVersion(namespace, name string) (*unstructured.Unstructured, error) {
	return h.client.Resource(clusterVersionGVR).Namespace(namespace).Get(
		context.TODO(), name, metav1.GetOptions{})
}

// GetBKECluster fetches BKECluster by exact metadata.name (no bke- prefix normalization).
func (h *DeclarativeUpgradeHelper) GetBKECluster(namespace, name string) (*unstructured.Unstructured, error) {
	return h.client.Resource(bkeClusterGVR).Namespace(namespace).Get(
		context.TODO(), name, metav1.GetOptions{})
}

func (h *DeclarativeUpgradeHelper) GetReleaseImage(namespace, name string) (*unstructured.Unstructured, error) {
	return h.client.Resource(releaseImageGVR).Namespace(namespace).Get(
		context.TODO(), name, metav1.GetOptions{})
}

func (h *DeclarativeUpgradeHelper) ListReleaseImages(namespace string) (*unstructured.UnstructuredList, error) {
	return h.client.Resource(releaseImageGVR).Namespace(namespace).List(
		context.TODO(), metav1.ListOptions{})
}

func (h *DeclarativeUpgradeHelper) WaitReleaseImagePhase(namespace, name, phase string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		ri, err := h.GetReleaseImage(namespace, name)
		g.Expect(err).NotTo(HaveOccurred())
		got, _, _ := unstructured.NestedString(ri.Object, "status", "phase")
		g.Expect(got).To(Equal(phase))
	}, timeout, 10*time.Second).Should(Succeed())
}

// WaitReleaseImageValidByVersion waits until some ReleaseImage with spec.version==version
// reaches status.phase=Valid. Matches capbke ResolveReleaseImageForVersion lookup (conventional
// name release-v* first, then any RI with matching spec.version).
func (h *DeclarativeUpgradeHelper) WaitReleaseImageValidByVersion(namespace, version string, timeout time.Duration) {
	version = strings.TrimSpace(version)
	Eventually(func(g Gomega) {
		list, err := h.ListReleaseImages(namespace)
		g.Expect(err).NotTo(HaveOccurred())
		found := false
		for _, ri := range list.Items {
			gotVersion, _, _ := unstructured.NestedString(ri.Object, "spec", "version")
			if strings.TrimSpace(gotVersion) != version {
				continue
			}
			phase, _, _ := unstructured.NestedString(ri.Object, "status", "phase")
			g.Expect(phase).To(Equal("Valid"))
			found = true
			break
		}
		g.Expect(found).To(BeTrue(), "no ReleaseImage with version %q in namespace %s", version, namespace)
	}, timeout, 10*time.Second).Should(Succeed())
}

// AssertReleaseImageNotRecreated verifies that after deleting one ReleaseImage on a Ready
// cluster, the remaining count stays at expectedCount and deletedName does not reappear.
func (h *DeclarativeUpgradeHelper) AssertReleaseImageNotRecreated(
	namespace, deletedName string, expectedCount int, duration time.Duration,
) {
	deletedName = strings.TrimSpace(deletedName)
	Consistently(func(g Gomega) {
		list, err := h.ListReleaseImages(namespace)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(list.Items).To(HaveLen(expectedCount),
			"ReleaseImage count must not increase after delete on a Ready cluster")
		for _, ri := range list.Items {
			g.Expect(ri.GetName()).NotTo(Equal(deletedName),
				"deleted ReleaseImage %q must not be recreated", deletedName)
		}
	}, duration, 15*time.Second).Should(Succeed())
}

func (h *DeclarativeUpgradeHelper) WaitClusterVersionPhase(namespace, name, phase string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		cv, err := h.GetClusterVersion(namespace, name)
		g.Expect(err).NotTo(HaveOccurred())
		got, _, _ := unstructured.NestedString(cv.Object, "status", "phase")
		g.Expect(got).To(Equal(phase))
	}, timeout, 10*time.Second).Should(Succeed())
}

func (h *DeclarativeUpgradeHelper) PatchClusterVersionDesiredVersion(namespace, name, version string) error {
	cv, err := h.GetClusterVersion(namespace, name)
	if err != nil {
		return err
	}
	if err := unstructured.SetNestedField(cv.Object, version, "spec", "desiredVersion"); err != nil {
		return err
	}
	_, err = h.client.Resource(clusterVersionGVR).Namespace(namespace).Update(
		context.TODO(), cv, metav1.UpdateOptions{})
	return err
}

func (h *DeclarativeUpgradeHelper) BKEClusterHasUpgradeReadyAnnotation(
	bc *unstructured.Unstructured, target string,
) bool {
	annotations := bc.GetAnnotations()
	if annotations == nil {
		return false
	}
	ready, ok := annotations["cvo.openfuyao.cn/upgrade-ready"]
	return ok && strings.TrimSpace(ready) == target
}

func (h *DeclarativeUpgradeHelper) AssertUpgradePathActive(name string) {
	list, err := h.client.Resource(upgradePathGVR).List(context.TODO(), metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	found := false
	for _, item := range list.Items {
		if item.GetName() == name {
			phase, _, _ := unstructured.NestedString(item.Object, "status", "phase")
			Expect(phase).To(Equal("Active"))
			found = true
			break
		}
	}
	Expect(found).To(BeTrue(), fmt.Sprintf("UpgradePath %q not found or not Active", name))
}

// GetUpgradePath fetches a cluster-scoped UpgradePath by name.
func (h *DeclarativeUpgradeHelper) GetUpgradePath(name string) (*unstructured.Unstructured, error) {
	return h.client.Resource(upgradePathGVR).Get(context.TODO(), name, metav1.GetOptions{})
}

// ResolveClusterCurrentVersion returns status.currentVersion, falling back to BKECluster openFuyaoVersion.
func (h *DeclarativeUpgradeHelper) ResolveClusterCurrentVersion(namespace, clusterName string) (string, error) {
	cv, err := h.GetClusterVersion(namespace, clusterName)
	if err != nil {
		return "", err
	}
	current, _, _ := unstructured.NestedString(cv.Object, "status", "currentVersion")
	current = strings.TrimSpace(current)
	if current != "" {
		return current, nil
	}
	bc, err := h.GetBKECluster(namespace, clusterName)
	if err != nil {
		return "", err
	}
	current, _, _ = unstructured.NestedString(bc.Object, "status", "openFuyaoVersion")
	current = strings.TrimSpace(current)
	if current == "" {
		return "", fmt.Errorf("current version empty on ClusterVersion and BKECluster %q", clusterName)
	}
	return current, nil
}

// SetUpgradePathEdgeBlocked sets spec.paths[from->to].blocked on a cluster-scoped UpgradePath.
func (h *DeclarativeUpgradeHelper) SetUpgradePathEdgeBlocked(
	upName, from, to string, blocked bool,
) error {
	upName = strings.TrimSpace(upName)
	from = strings.TrimSpace(from)
	to = strings.TrimSpace(to)

	up, err := h.GetUpgradePath(upName)
	if err != nil {
		return err
	}
	paths, found, err := unstructured.NestedSlice(up.Object, "spec", "paths")
	if err != nil {
		return err
	}
	if !found || len(paths) == 0 {
		return fmt.Errorf("upgrade path %q has no spec.paths", upName)
	}

	matched := false
	for i, item := range paths {
		rule, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		ruleFrom, _, _ := unstructured.NestedString(rule, "from")
		ruleTo, _, _ := unstructured.NestedString(rule, "to")
		if strings.TrimSpace(ruleFrom) != from || strings.TrimSpace(ruleTo) != to {
			continue
		}
		rule["blocked"] = blocked
		paths[i] = rule
		matched = true
		break
	}
	if !matched {
		return fmt.Errorf("no upgrade path edge %q -> %q in %q", from, to, upName)
	}
	if err := unstructured.SetNestedSlice(up.Object, paths, "spec", "paths"); err != nil {
		return err
	}
	_, err = h.client.Resource(upgradePathGVR).Update(context.TODO(), up, metav1.UpdateOptions{})
	return err
}

// RestoreUpgradePathSpec replaces spec on the live UpgradePath from backup.
func (h *DeclarativeUpgradeHelper) RestoreUpgradePathSpec(backup *unstructured.Unstructured) error {
	if backup == nil {
		return nil
	}
	name := backup.GetName()
	current, err := h.GetUpgradePath(name)
	if err != nil {
		return err
	}
	spec, found, err := unstructured.NestedMap(backup.Object, "spec")
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("backup UpgradePath %q has no spec", name)
	}
	if err := unstructured.SetNestedMap(current.Object, spec, "spec"); err != nil {
		return err
	}
	_, err = h.client.Resource(upgradePathGVR).Update(context.TODO(), current, metav1.UpdateOptions{})
	return err
}

// WaitBKEClusterUpgradeReadyCleared waits until upgrade-ready annotation is absent.
func (h *DeclarativeUpgradeHelper) WaitBKEClusterUpgradeReadyCleared(
	namespace, clusterName string, timeout time.Duration,
) {
	Eventually(func(g Gomega) {
		bc, err := h.GetBKECluster(namespace, clusterName)
		g.Expect(err).NotTo(HaveOccurred())
		annotations := bc.GetAnnotations()
		if annotations == nil {
			return
		}
		_, ok := annotations["cvo.openfuyao.cn/upgrade-ready"]
		g.Expect(ok).To(BeFalse(), "upgrade-ready annotation should be cleared")
	}, timeout, 5*time.Second).Should(Succeed())
}

// AssertBKEClusterNotUpgradeReady fails if BC carries upgrade-ready for targetVersion.
func (h *DeclarativeUpgradeHelper) AssertBKEClusterNotUpgradeReady(
	bc *unstructured.Unstructured, targetVersion string,
) {
	Expect(h.BKEClusterHasUpgradeReadyAnnotation(bc, targetVersion)).To(BeFalse(),
		"BKECluster must not have upgrade-ready=%q when upgrade path is blocked", targetVersion)
}
