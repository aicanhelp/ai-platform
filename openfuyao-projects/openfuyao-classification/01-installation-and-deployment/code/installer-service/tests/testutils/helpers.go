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

package testutils

import (
	"net/http/httptest"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"installer-service/pkg/api/clustermanage"
	"installer-service/pkg/installer"
)

// StartServerWithFakeClients starts a restful Container registered with installer
// handlers backed by the provided fake clients. It patches installer.NewInstallerOperation
// to return an installer.Operation implemented with the fake clients.
// Returns the test server, the fake clientset, the fake dynamic client, and a cleanup function.
func StartServerWithFakeClients() (*httptest.Server, *fake.Clientset, *dynamicfake.FakeDynamicClient, *gomonkey.Patches) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	// register list kinds required by installer (avoid dynamic fake panic on List)
	bkeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
	nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}
	upgradePathGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "upgradepaths"}
	clusterVersionGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "clusterversions"}
	nsGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}
	listKinds := map[schema.GroupVersionResource]string{
		bkeGVR:            "BKEClusterList",
		nodeGVR:           "BKENodeList",
		upgradePathGVR:    "UpgradePathList",
		clusterVersionGVR: "ClusterVersionList",
		nsGVR:             "NamespaceList",
	}

	dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds)
	cs := fake.NewSimpleClientset()

	// Register BKE API resources in fake discovery so restmapper works in CreateCluster
	cs.Fake.Resources = []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "namespaces", Kind: "Namespace", Namespaced: false},
				{Name: "configmaps", Kind: "ConfigMap", Namespaced: true},
			},
		},
		{
			GroupVersion: "bke.bocloud.com/v1beta1",
			APIResources: []metav1.APIResource{
				{Name: "bkeclusters", Kind: "BKECluster", Namespaced: true},
				{Name: "bkenodes", Kind: "BKENode", Namespaced: true},
			},
		},
	}

	// patch installer.NewInstallerOperation to return an installer backed by our fake clients
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(installer.NewInstallerOperation, func(_ *rest.Config) (installer.Operation, error) {
		return installer.NewInstallerOperationWithClients(cs, dyn)
	})

	container := restful.NewContainer()
	// call ConfigInstaller which will call our patched NewInstallerOperation
	_ = clustermanage.ConfigInstaller(container, nil)
	srv := httptest.NewServer(container)

	return srv, cs, dyn, patches
}

// MakeBKEClusterUnstructured returns a minimal, valid BKECluster unstructured object
// with controlPlaneEndpoint and clusterConfig populated so production code can read
// nested fields without nil checks failing.
func MakeBKEClusterUnstructured(name, namespace, host, openFuyaoVersion, kubernetesVersion string) *unstructured.Unstructured {
	if namespace == "" {
		namespace = name
	}
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKECluster",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"controlPlaneEndpoint": map[string]interface{}{"host": host},
			"clusterConfig": map[string]interface{}{
				"cluster": map[string]interface{}{
					"openFuyaoVersion":  openFuyaoVersion,
					"kubernetesVersion": kubernetesVersion,
					"containerRuntime":  map[string]interface{}{"CRI": "containerd"},
				},
			},
		},
	}}
}

// MakeBKENodeUnstructured returns a minimal BKENode unstructured object
func MakeBKENodeUnstructured(name, namespace, ip string, roles []string, labels map[string]string) *unstructured.Unstructured {
	roleIface := make([]interface{}, 0, len(roles))
	for _, r := range roles {
		roleIface = append(roleIface, r)
	}
	lbls := make(map[string]interface{})
	for k, v := range labels {
		lbls[k] = v
	}
	if namespace == "" {
		namespace = name
	}
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
			"labels":    lbls,
		},
		"spec": map[string]interface{}{
			"ip":   ip,
			"role": roleIface,
		},
	}}
}
