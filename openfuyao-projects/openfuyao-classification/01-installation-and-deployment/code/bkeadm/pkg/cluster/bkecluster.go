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
	"errors"
	"fmt"
	"os"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/securityinit"
)

// ClusterResources contains BKECluster and associated BKENode resources
type ClusterResources struct {
	BKECluster *configv1beta1.BKECluster
	BKENodes   []configv1beta1.BKENode
}

// NewClusterResourcesFromFiles loads BKECluster and BKENodes from separate files
func NewClusterResourcesFromFiles(clusterFile, nodesFile string) (*ClusterResources, error) {
	resources := &ClusterResources{}

	bkeCluster, err := loadBKEClusterFromFile(clusterFile)
	if err != nil {
		return nil, err
	}
	resources.BKECluster = bkeCluster

	bkeNodes, err := loadBKENodesFromFile(nodesFile)
	if err != nil {
		return nil, err
	}
	resources.BKENodes = bkeNodes

	if err := processClusterResources(resources); err != nil {
		return nil, err
	}

	return resources, nil
}

// loadBKEClusterFromFile loads a BKECluster from a YAML file
func loadBKEClusterFromFile(file string) (*configv1beta1.BKECluster, error) {
	conf := &configv1beta1.BKECluster{}
	yamlFile, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	if err := yaml2.Unmarshal(yamlFile, conf); err != nil {
		return nil, err
	}
	if conf.Spec.ClusterConfig == nil {
		return nil, errors.New("the cluster configuration cannot be empty")
	}
	return conf, nil
}

// loadBKENodesFromFile loads BKENodes from a YAML file (supports multi-document)
func loadBKENodesFromFile(file string) ([]configv1beta1.BKENode, error) {
	yamlFile, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	var nodes []configv1beta1.BKENode

	nodeList := &configv1beta1.BKENodeList{}
	if err := yaml2.Unmarshal(yamlFile, nodeList); err == nil && len(nodeList.Items) > 0 {
		return nodeList.Items, nil
	}

	// Check if this is a multi-document YAML (contains "---" separator)
	// Must try multi-document parsing BEFORE single node parsing,
	if strings.Contains(string(yamlFile), "\n---") {
		docs := splitYAMLDocuments(yamlFile)
		for _, doc := range docs {
			if len(doc) == 0 {
				continue
			}
			n := configv1beta1.BKENode{}
			if err := yaml2.Unmarshal(doc, &n); err == nil && n.Spec.IP != "" {
				nodes = append(nodes, n)
			}
		}
		if len(nodes) > 0 {
			return nodes, nil
		}
	}

	node := &configv1beta1.BKENode{}
	if err := yaml2.Unmarshal(yamlFile, node); err == nil && node.Spec.IP != "" {
		nodes = append(nodes, *node)
		return nodes, nil
	}

	if len(nodes) == 0 {
		return nil, errors.New("no valid BKENode resources found in the file")
	}

	return nodes, nil
}

// splitYAMLDocuments splits a multi-document YAML into separate documents
func splitYAMLDocuments(data []byte) [][]byte {
	var docs [][]byte
	var currentDoc []byte

	lines := splitLines(data)
	for _, line := range lines {
		if string(line) == "---" {
			if len(currentDoc) > 0 {
				docs = append(docs, currentDoc)
				currentDoc = nil
			}
			continue
		}
		currentDoc = append(currentDoc, line...)
		currentDoc = append(currentDoc, '\n')
	}
	if len(currentDoc) > 0 {
		docs = append(docs, currentDoc)
	}

	return docs
}

// splitLines splits data into lines
func splitLines(data []byte) [][]byte {
	var lines [][]byte
	var line []byte
	for _, b := range data {
		if b == '\n' {
			lines = append(lines, line)
			line = nil
			continue
		}
		if b != '\r' {
			line = append(line, b)
		}
	}
	if len(line) > 0 {
		lines = append(lines, line)
	}
	return lines
}

// processClusterResources validates and processes the cluster resources
func processClusterResources(resources *ClusterResources) error {
	bc, err := configinit.NewBkeConfigFromClusterConfig(resources.BKECluster.Spec.ClusterConfig)
	if err != nil {
		return err
	}
	if err := bc.Validate(); err != nil {
		return err
	}

	if len(resources.BKENodes) == 0 {
		return errors.New("at least one BKENode resource is required")
	}
	if err := validation.ValidateBKENodes(resources.BKENodes); err != nil {
		return err
	}

	// 在加密 BKENode 密码前，初始化 AES 密钥来源（从管理集群 Secret 直读）。
	// 必须先于下方 security.AesDecrypt/AesEncrypt 调用。
	if err := securityinit.Init(); err != nil {
		return fmt.Errorf("ensure password encryption key: %w", err)
	}

	for i := range resources.BKENodes {
		node := &resources.BKENodes[i]
		_, err := security.AesDecrypt(node.Spec.Password)
		if err != nil {

			p, err := security.AesEncrypt(node.Spec.Password)
			if err != nil {
				return err
			}
			node.Spec.Password = p
		}

		if node.Labels == nil {
			node.Labels = make(map[string]string)
		}
		node.Labels["cluster.x-k8s.io/cluster-name"] = resources.BKECluster.Name
		if node.Namespace == "" {
			node.Namespace = resources.BKECluster.Namespace
		}
	}

	resources.BKECluster.Spec.ClusterConfig, _ = configinit.ConvertBkEConfig(bc)
	if resources.BKECluster.Spec.ClusterConfig.CustomExtra == nil {
		resources.BKECluster.Spec.ClusterConfig.CustomExtra = map[string]string{
			bkecommon.BKEAgentListenerAnnotationKey: bkecommon.BKEAgentListenerCurrent,
		}
	} else {
		if len(resources.BKECluster.Spec.ClusterConfig.CustomExtra[bkecommon.BKEAgentListenerAnnotationKey]) == 0 {
			resources.BKECluster.Spec.ClusterConfig.CustomExtra[bkecommon.BKEAgentListenerAnnotationKey] = bkecommon.BKEAgentListenerCurrent
		}
	}

	return nil
}

// NewBKEClusterFromFile loads BKECluster from file (legacy function, for backward compatibility during transition)
// Deprecated: Use NewClusterResourcesFromFiles instead
func NewBKEClusterFromFile(file string) (*configv1beta1.BKECluster, error) {
	return loadBKEClusterFromFile(file)
}
