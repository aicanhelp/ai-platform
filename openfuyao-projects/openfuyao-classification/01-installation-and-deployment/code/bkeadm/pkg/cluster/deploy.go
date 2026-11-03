/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cluster

import (
	"fmt"
	"os"
	"path"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// applyImmutableOSAnnotation 当 --immutable-os=kubeos 时，在 BKECluster 上注入 annotation。
// 同时在 CustomExtra 中设置标志，传递到 agent 端 plugin。
func applyImmutableOSAnnotation(bkeCluster *configv1beta1.BKECluster, immutableOS string) {
	if immutableOS != utils.ImmutableOSAnnotationValueKubeOS {
		return
	}
	if bkeCluster.Annotations == nil {
		bkeCluster.Annotations = make(map[string]string)
	}
	bkeCluster.Annotations[utils.ImmutableOSAnnotation] = utils.ImmutableOSAnnotationValueKubeOS

	// 同时在 CustomExtra 中设置，使 agent 端 plugin 能感知不可变模式
	if bkeCluster.Spec.ClusterConfig != nil {
		if bkeCluster.Spec.ClusterConfig.CustomExtra == nil {
			bkeCluster.Spec.ClusterConfig.CustomExtra = make(map[string]string)
		}
		bkeCluster.Spec.ClusterConfig.CustomExtra["immutable-os"] = utils.ImmutableOSAnnotationValueKubeOS
	}
}

func (op *Options) Cluster() {
	if err := global.EnsureWorkspaceLayout(global.Workspace); err != nil {
		log.Warnf("Failed to ensure workspace layout, %v", err)
	}
	resources, err := NewClusterResourcesFromFiles(op.File, op.NodesFile)
	if err != nil {
		log.Errorf("Failed to load the configuration file. %v", err)
		return
	}

	bkeCluster := resources.BKECluster
	bkeNodes := resources.BKENodes
	if len(op.NtpServer) > 0 {
		bkeCluster.Spec.ClusterConfig.Cluster.NTPServer = op.NtpServer
	}
	applyImmutableOSAnnotation(bkeCluster, op.ImmutableOS)

	if err := alignBKENodesNamespace(bkeNodes, bkeCluster.Name, bkeCluster.Namespace); err != nil {
		log.Errorf("reject cluster create due to namespace mismatch: clusterFile=%s nodesFile=%s cluster=%s/%s err=%v",
			op.File, op.NodesFile, bkeCluster.Namespace, bkeCluster.Name, err)
		return
	}

	bkefile, err := marshalAndWriteClusterYAML(bkeCluster)
	if err != nil {
		return
	}
	nodeFiles, err := marshalAndWriteNodeYAMLs(bkeCluster.Name, bkeCluster.Namespace, bkeNodes)
	if err != nil {
		log.Errorf("Failed to write BKENode files: clusterFile=%s nodesFile=%s cluster=%s/%s err=%v",
			op.File, op.NodesFile, bkeCluster.Namespace, bkeCluster.Name, err)
		return
	}

	ensureClusterNamespace(bkeCluster.Namespace)
	if err := submitClusterCreateYAMLs(bkefile, nodeFiles, len(bkeNodes), bkeCluster.Namespace); err != nil {
		return
	}
}

// ensureClusterNamespace creates the target namespace when it is non-empty.
func ensureClusterNamespace(namespace string) {
	if namespace == "" {
		return
	}
	ns := corev1.Namespace{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Namespace"},
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}
	if err := global.K8s.CreateNamespace(&ns); err != nil {
		log.Warnf("create namespace %s failed: %v", namespace, err)
	}
}

// submitClusterCreateYAMLs installs BKENodes first, then BKECluster, and watches events.
func submitClusterCreateYAMLs(bkefile string, nodeFiles []string, nodeCount int, namespace string) error {
	log.Info("Submit cluster-api yaml to the cluster")
	for _, nodeFile := range nodeFiles {
		if err := global.K8s.InstallYaml(nodeFile, map[string]string{}, ""); err != nil {
			log.Errorf("Failed to install bke-node, %v", err)
			return err
		}
	}
	if err := global.K8s.InstallYaml(bkefile, map[string]string{}, ""); err != nil {
		log.Errorf("Failed to install bke-cluster, %v", err)
		return err
	}
	log.Infof("Submit the configuration to the cluster (1 BKECluster + %d BKENodes)", nodeCount)
	log.Info("Waiting for the cluster to start...")
	global.K8s.WatchEventByAnnotation(namespace)
	return nil
}

// alignBKENodesNamespace ensures every BKENode shares the BKECluster namespace.
// Empty node namespace is filled from the cluster; a non-empty mismatch fails.
func alignBKENodesNamespace(nodes []configv1beta1.BKENode, clusterName, namespace string) error {
	for i := range nodes {
		node := &nodes[i]
		if node.Namespace == "" {
			node.Namespace = namespace
			continue
		}
		if node.Namespace != namespace {
			return fmt.Errorf(
				"BKENode %q (namespace=%q) does not match BKECluster %q (namespace=%q); "+
					"set the same metadata.namespace in bc.yaml and bn.yaml, then retry",
				node.Name, node.Namespace, clusterName, namespace)
		}
	}
	return nil
}

// marshalAndWriteNodeYAMLs writes BKENode resources to YAML files
func marshalAndWriteNodeYAMLs(clusterName, namespace string, nodes []configv1beta1.BKENode) ([]string, error) {
	var files []string

	if err := os.MkdirAll(path.Join(global.Workspace, "cluster"), utils.DefaultDirPermission); err != nil {
		log.Warnf("Failed to create cluster directory: %v", err)
	}

	if err := alignBKENodesNamespace(nodes, clusterName, namespace); err != nil {
		return nil, err
	}

	for i := range nodes {
		node := &nodes[i]

		by, err := yaml2.Marshal(node)
		if err != nil {
			return nil, err
		}

		nsPart := namespace
		if nsPart == "" {
			nsPart = "default"
		}
		nodeFile := path.Join(global.Workspace, "cluster", fmt.Sprintf("%s-%s-node.yaml", nsPart, node.Name))
		if err := os.WriteFile(nodeFile, by, utils.DefaultFilePermission); err != nil {
			return nil, err
		}
		files = append(files, nodeFile)
	}

	return files, nil
}
