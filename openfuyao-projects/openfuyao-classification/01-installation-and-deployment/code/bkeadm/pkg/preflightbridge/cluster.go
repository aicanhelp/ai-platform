/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/preflight/api"
)

const (
	maxClusterInputBytes = 16 << 20
	minimumPortNumber    = 1
	maximumPortNumber    = 65535
)

// ClusterResources is the pure, parsed view of BKECluster + BKENodes without
// any submission side effects (no Secret read, no Setenv, no encryption, no
// label mutation). preflight consumes this; the existing cluster create path
// continues to use processClusterResources for submission preparation.
type ClusterResources struct {
	BKECluster *configv1beta1.BKECluster
	BKENodes   []configv1beta1.BKENode
}

// bkeNodeListDocument accepts the metadata fields used by the node-list files
// generated for bkeadm. The provider's BKENodeList embeds metav1.ListMeta,
// while the file format also carries name/namespace metadata, so decoding
// directly into BKENodeList rejects an otherwise valid input before it reaches
// the items.
type bkeNodeListDocument struct {
	APIVersion string                  `json:"apiVersion,omitempty"`
	Kind       string                  `json:"kind,omitempty"`
	Metadata   bkeNodeListMetadata     `json:"metadata,omitempty"`
	Items      []configv1beta1.BKENode `json:"items"`
}

type bkeNodeListMetadata struct {
	Name               string            `json:"name,omitempty"`
	GenerateName       string            `json:"generateName,omitempty"`
	Namespace          string            `json:"namespace,omitempty"`
	UID                string            `json:"uid,omitempty"`
	ResourceVersion    string            `json:"resourceVersion,omitempty"`
	Continue           string            `json:"continue,omitempty"`
	RemainingItemCount *int64            `json:"remainingItemCount,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
	Annotations        map[string]string `json:"annotations,omitempty"`
}

// ParseClusterResources loads BKECluster and BKENodes from separate files
// using strict YAML decode, preserving input errors. It performs NO Secret
// read, NO os.Setenv, NO password encryption and NO label mutation — those
// stay in the host's PrepareClusterResourcesForSubmission.
//
// This is the pure first layer; preflight uses it + ResolveClusterResources.
// The existing NewClusterResourcesFromFiles / processClusterResources stays
// unchanged for the cluster create execution path.
func ParseClusterResources(clusterFile, nodesFile string) (*ClusterResources, []api.RunIssue, error) {
	var issues []api.RunIssue
	resources := &ClusterResources{}
	if strings.TrimSpace(clusterFile) == "" || strings.TrimSpace(nodesFile) == "" {
		field := "cluster"
		cause := "cluster file is required"
		if strings.TrimSpace(clusterFile) != "" {
			field, cause = "nodes", "nodes file is required"
		}
		return nil, []api.RunIssue{{Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Field: field,
			Cause: cause, Action: "provide both --file and --nodes, then run preflight again"}}, errors.New(cause)
	}

	bkeCluster, err := loadBKEClusterStrict(clusterFile)
	if err != nil {
		return nil, append(issues, api.RunIssue{
			Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Field: inputErrorField("cluster", err), Cause: err.Error(),
			Action: "correct the reported field in the cluster file, then run preflight again",
		}), err
	}
	resources.BKECluster = bkeCluster

	bkeNodes, err := loadBKENodesStrict(nodesFile)
	if err != nil {
		return nil, append(issues, api.RunIssue{
			Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Field: inputErrorField("nodes", err), Cause: err.Error(),
			Action: "correct the reported field in the nodes file, then run preflight again",
		}), err
	}
	resources.BKENodes = bkeNodes

	return resources, issues, nil
}

// ResolveClusterResources validates only the structural prerequisites needed
// to form a plan. Business validation is deliberately retained as Report
// findings by the Provider resolver. It does NOT read
// Secrets, set env vars, encrypt passwords, or mutate labels. Run issues carry
// validation problems; BLOCK_START issues map to INPUT_ERROR.
func ResolveClusterResources(r *ClusterResources) []api.RunIssue {
	var issues []api.RunIssue
	if r == nil || r.BKECluster == nil {
		return []api.RunIssue{{
			Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Field: "cluster", Cause: "cluster configuration is empty",
			Action: "provide a cluster configuration, then run preflight again",
		}}
	}
	if r.BKECluster.Spec.ClusterConfig == nil {
		return []api.RunIssue{{
			Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Field: "clusterConfig", Cause: "the cluster configuration cannot be empty",
			Action: "set spec.clusterConfig, then run preflight again",
		}}
	}
	if len(r.BKENodes) == 0 {
		return append(issues, api.RunIssue{
			Code: "CLUSTER-INPUT-REQUIRED", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart,
			Field: "nodes", Cause: "at least one BKENode resource is required",
			Action: "provide at least one BKENode, then run preflight again",
		})
	}
	return issues
}

// loadBKEClusterStrict loads a BKECluster with strict YAML (unknown fields
// rejected).
func loadBKEClusterStrict(file string) (*configv1beta1.BKECluster, error) {
	yamlFile, err := readClusterInput(file)
	if err != nil {
		return nil, fmt.Errorf("read cluster file %s: %w", file, err)
	}
	conf := &configv1beta1.BKECluster{}
	if err := decodeYAMLStrict(yamlFile, conf, "cluster"); err != nil {
		return nil, fmt.Errorf("parse cluster file %s: %w", file, err)
	}
	if conf.Spec.ClusterConfig == nil {
		return nil, errors.New("the cluster configuration cannot be empty")
	}
	return conf, nil
}

// loadBKENodesStrict loads BKENodes with strict YAML, supporting both
// BKENodeList and multi-document forms.
func loadBKENodesStrict(file string) ([]configv1beta1.BKENode, error) {
	yamlFile, err := readClusterInput(file)
	if err != nil {
		return nil, fmt.Errorf("read nodes file %s: %w", file, err)
	}
	nodes, err := decodeBKENodeDocuments(yamlFile)
	if err != nil {
		return nil, fmt.Errorf("parse nodes file %s: %w", file, err)
	}
	return nodes, nil
}

func readClusterInput(file string) ([]byte, error) {
	info, err := os.Stat(file)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("input must be a regular file")
	}
	if info.Size() > maxClusterInputBytes {
		return nil, fmt.Errorf("input exceeds %d byte limit", maxClusterInputBytes)
	}
	return os.ReadFile(file)
}

// ClusterCreateOptions is the resolved cluster-create input for preflight.
type ClusterCreateOptions struct {
	File  string
	Nodes string
}

// ResolveClusterCreate loads and validates cluster/nodes purely, returning
// run issues. preflight uses this to build its AdapterInput without any
// submission side effects.
func ResolveClusterCreate(file, nodes string) (*ClusterResources, []api.RunIssue, error) {
	res, issues, err := ParseClusterResources(file, nodes)
	if err != nil {
		return nil, issues, err
	}
	issues = append(issues, ResolveClusterResources(res)...)
	if api.HasStartBlocker(issues) {
		return nil, issues, errors.New("cluster-create input has errors")
	}
	return res, issues, nil
}

// ToAdapterInput converts resolved cluster resources to the SDK's AdapterInput.
func (r ClusterResources) ToAdapterInput() (AdapterInput, error) {
	// The adapter receives the raw YAML bytes; the SDK adapter does the
	// BKECluster→OperationPlan mapping. We re-marshal here to keep the bridge
	// stateless and avoid passing BKE Go types into the SDK.
	clusterYAML, err := marshalSanitized(r.BKECluster)
	if err != nil {
		return AdapterInput{}, fmt.Errorf("marshal cluster resource: %w", err)
	}
	nodesYAML, err := marshalSanitized(r.BKENodes)
	if err != nil {
		return AdapterInput{}, fmt.Errorf("marshal node resources: %w", err)
	}
	return AdapterInput{
		Operation: api.OperationClusterCreate,
		ClusterCreate: &ClusterCreateInput{
			ClusterYAML: clusterYAML,
			NodesYAML:   nodesYAML,
		},
	}, nil
}

// ApplyProviderManagementDefaults applies the plan-relevant subset of the
// Provider webhook's bke-config ConfigMap defaults. The input resources are an
// in-memory parse owned by preflight; no Kubernetes object or input file is
// mutated. Random/persistent defaults (for example VRIDs) are intentionally
// excluded because they do not affect the preflight plan.
func ApplyProviderManagementDefaults(resources *ClusterResources, defaults map[string]string) error {
	if resources == nil || resources.BKECluster == nil || resources.BKECluster.Spec.ClusterConfig == nil ||
		len(defaults) == 0 {
		return nil
	}
	config := resources.BKECluster.Spec.ClusterConfig
	cluster := &config.Cluster
	applyPort := func(current *string, key string) error {
		if *current != "" || defaults[key] == "" {
			return nil
		}
		port, err := strconv.Atoi(defaults[key])
		if err != nil || port < minimumPortNumber || port > maximumPortNumber {
			return fmt.Errorf("management default %s is not a valid port", key)
		}
		*current = defaults[key]
		return nil
	}
	if err := applyPort(&cluster.HTTPRepo.Port, "yumRepoPort"); err != nil {
		return err
	}
	if err := applyPort(&cluster.ImageRepo.Port, "imageRepoPort"); err != nil {
		return err
	}
	if cluster.ImageRepo.Prefix == "" {
		cluster.ImageRepo.Prefix = defaults["imageRepoPrefix"]
	}
	if cluster.NTPServer == "" {
		cluster.NTPServer = defaults["ntpServer"]
	}
	applyProviderHostDefaults(config, cluster, defaults)
	applyProviderRuntimeDefault(config, defaults)
	return nil
}

func applyProviderHostDefaults(
	config *configv1beta1.BKEConfig,
	cluster *configv1beta1.Cluster,
	defaults map[string]string,
) {
	host := strings.TrimSpace(defaults["host"])
	if host == "" {
		return
	}
	if cluster.HTTPRepo.Ip == "" {
		cluster.HTTPRepo.Ip = host
	}
	if cluster.ImageRepo.Ip == "" {
		cluster.ImageRepo.Ip = host
	}
	if config.CustomExtra == nil {
		config.CustomExtra = map[string]string{}
	}
	for key, value := range map[string]string{"host": host, "nfsServer": host, "nfsRootDir": "/", "nfsVersion": "4.1"} {
		if _, exists := config.CustomExtra[key]; !exists {
			config.CustomExtra[key] = value
		}
	}
}

func applyProviderRuntimeDefault(config *configv1beta1.BKEConfig, defaults map[string]string) {
	containerd := strings.TrimSpace(defaults["containerd"])
	if containerd == "" {
		return
	}
	if config.CustomExtra == nil {
		config.CustomExtra = map[string]string{}
	}
	if _, exists := config.CustomExtra["containerd"]; !exists {
		containerd = strings.ReplaceAll(containerd, "amd64", "{.arch}")
		containerd = strings.ReplaceAll(containerd, "arm64", "{.arch}")
		config.CustomExtra["containerd"] = containerd
	}
}

func marshalSanitized(value any) ([]byte, error) {
	raw, err := yaml2.Marshal(value)
	if err != nil {
		return nil, err
	}
	var document any
	if err := yaml2.Unmarshal(raw, &document); err != nil {
		return nil, err
	}
	sanitizeSensitiveLeaves(document)
	return yaml2.Marshal(document)
}

func sanitizeSensitiveLeaves(value any) {
	switch typed := value.(type) {
	case map[string]any:
		sanitizeSensitiveMap(typed)
	case []any:
		for _, child := range typed {
			sanitizeSensitiveLeaves(child)
		}
	default:
		return
	}
}

func sanitizeSensitiveMap(document map[string]any) {
	for key, child := range document {
		if isSensitiveField(key) {
			delete(document, key)
		} else {
			sanitizeSensitiveLeaves(child)
		}
	}
}

func isSensitiveField(key string) bool {
	switch strings.ToLower(key) {
	case "password", "passwd", "token", "identitytoken", "registrytoken", "privatekey", "clientkey", "auth":
		return true
	default:
		return false
	}
}
