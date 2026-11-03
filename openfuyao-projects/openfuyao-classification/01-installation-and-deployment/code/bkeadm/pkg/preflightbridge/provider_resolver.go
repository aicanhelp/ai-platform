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
	"context"
	"errors"
	"fmt"
	"net"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	providernode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
)

// ProviderResolver is a versioned interface that performs the pure,
// side-effect-free defaulting and validation that the Provider webhook does,
// but WITHOUT reading ConfigMaps/BKENodes from the API, generating random
// VRID, mutating objects, or creating resources. The host injects the
// concrete resolver; tests inject a fake.
//
// This is a temporary compatibility layer in bkeadm. When the Provider repo
// can be modified, this moves upstream as a webhook+Adapter shared Resolver.
type ProviderResolver interface {
	ContractVersion() string
	// Resolve applies pure defaulting and validation to cluster/nodes WITHOUT
	// side effects. It does not read Secrets, set env vars, encrypt passwords,
	// generate VRID, or mutate the submitted objects. Fields that are random
	// or runtime-only are marked runtime-unresolved and do NOT enter PlanHash.
	Resolve(
		ctx context.Context,
		cluster *configv1beta1.BKECluster,
		nodes []configv1beta1.BKENode,
	) (ResolveResult, error)
}

// ResolveResult carries the resolved topology + defaults computed by the
// Provider resolver. Only plan-relevant facts; no secrets.
type ResolveResult struct {
	// BootstrapNodeID is empty unless a host has independently proved that the
	// bke bootstrap machine is one of the declared nodes.
	BootstrapNodeID string
	// MasterCount, EtcdCount, WorkerCount from the final node roles.
	MasterCount, EtcdCount, WorkerCount int
	// ControlPlaneEndpoint explicit? (false => defaulted to first master).
	EndpointExplicit bool
	// APIBindPort (default 6443 from Provider defaults.go).
	APIBindPort int
	// Runtime / data-root / kubelet root defaults.
	Runtime     string
	DataRoot    string
	KubeletRoot string
	EtcdDataDir string
	// Node roles by node ID for plan instantiation.
	NodeRoles map[string][]string
	// Node addresses by node ID.
	NodeAddresses map[string]string
	NodeMetadata  map[string]map[string]string
	Sources       []api.Source
	// Pod/Service CIDR.
	PodSubnet        string
	ServiceSubnet    string
	AddonValid       bool
	AddonSourceValid bool
	VersionSupported bool
	ClusterValid     bool
	NodesValid       bool
	ValidationFields map[string]ValidationFinding
	// Issues prevent the Provider plan from being completed; the host maps them
	// to BLOCK_START run issues.
	Issues []string
}

// ValidationFinding identifies the input field behind one failed local
// Provider validation without coupling the generic SDK to BKE Go types.
type ValidationFinding struct {
	Field string
	Cause string
}

// ProviderResolverV122 is the v1.2.2 (and the post-merge pseudo-version)
// resolver. It reuses the Provider's published pure constants and
// ValidateBKENodes, and applies only the minimal defaulting that does not
// require API reads, randomness or persistence.
type ProviderResolverV122 struct {
	// Policy is optional for compatibility with callers that only need the
	// Provider parity resolver. When present, config defaults are used only
	// when the submitted object did not provide an explicit value.
	Policy *preflightconfig.Config
}

// ContractVersion is the resolver contract version.
func (r *ProviderResolverV122) ContractVersion() string { return "preflight-resolver/v1" }

// Resolve applies pure defaulting + validation.
func (r *ProviderResolverV122) Resolve(
	ctx context.Context,
	cluster *configv1beta1.BKECluster,
	nodes []configv1beta1.BKENode,
) (ResolveResult, error) {
	res := newProviderResolveResult()
	if cluster == nil {
		res.Issues = append(res.Issues, "cluster is nil")
		return res, errors.New("cluster is nil")
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if cluster.Spec.ClusterConfig == nil {
		res.Issues = append(res.Issues, "clusterConfig is required")
		return res, nil
	}
	defaulted, err := configinit.NewBkeConfigFromClusterConfig(cluster.Spec.ClusterConfig.DeepCopy())
	if err != nil {
		return res, fmt.Errorf("apply Provider defaults: %w", err)
	}
	r.applyResolvedDefaults(&res, cluster, defaulted)
	projectProviderNodes(&res, nodes, defaulted)
	res.EndpointExplicit = cluster.Spec.ControlPlaneEndpoint.Host != ""
	return res, nil
}

func newProviderResolveResult() ResolveResult {
	return ResolveResult{
		APIBindPort: configinit.DefaultAPIBindPort, Runtime: configinit.DefaultRuntime,
		DataRoot: configinit.DefaultCRIContainerdDataRootDir, KubeletRoot: configinit.DefaultKubeletRootDir,
		EtcdDataDir: configinit.DefaultEtcdDataDir, PodSubnet: configinit.DefaultPodSubnet,
		ServiceSubnet: configinit.DefaultServicesSubnet, NodeRoles: make(map[string][]string),
		NodeAddresses: make(map[string]string), NodeMetadata: make(map[string]map[string]string),
		ValidationFields: make(map[string]ValidationFinding),
	}
}

func (r *ProviderResolverV122) applyResolvedDefaults(
	res *ResolveResult,
	cluster *configv1beta1.BKECluster,
	defaulted *configinit.BkeConfig,
) {
	res.APIBindPort = int(defaulted.Cluster.APIServer.Port)
	if r.Policy != nil && !explicitAPIPort(cluster) {
		if configured := policyPort(*r.Policy, api.OperationClusterCreate, "apiServer"); configured > 0 {
			res.APIBindPort = configured
		}
	}
	res.Runtime = defaulted.Cluster.ContainerRuntime.CRI
	res.DataRoot = defaulted.Cluster.ContainerRuntime.Param["data-root"]
	res.KubeletRoot = configinit.DefaultKubeletRootDir
	res.EtcdDataDir = defaulted.Cluster.Etcd.DataDir
	res.PodSubnet = defaulted.Cluster.Networking.PodSubnet
	res.ServiceSubnet = defaulted.Cluster.Networking.ServiceSubnet
	res.Sources = providerSources(defaulted.Cluster)
	addonErr := validation.ValidateAddons(defaulted.Addons)
	res.AddonValid = addonErr == nil
	res.recordValidation("cluster.addon.closure", "cluster.spec.clusterConfig.addons", addonErr)
	res.AddonSourceValid = addonSourcesValid(defaulted)
	if !res.AddonSourceValid {
		res.ValidationFields["cluster.addon.source"] = ValidationFinding{
			Field: "cluster.spec.clusterConfig.cluster.chartRepo",
			Cause: "a chart addon requires a configured chart repository",
		}
	}
	versionErr := validation.ValidateK8sVersion(defaulted.Cluster.KubernetesVersion)
	res.VersionSupported = versionErr == nil
	res.recordValidation(
		"cluster.ref.versionSupported",
		"cluster.spec.clusterConfig.cluster.kubernetesVersion",
		versionErr,
	)
	clusterField, clusterErr := validateClusterPure(defaulted)
	res.ClusterValid = clusterErr == nil
	res.recordValidation("local.cluster.valid", clusterField, clusterErr)
}

func projectProviderNodes(res *ResolveResult, nodes []configv1beta1.BKENode, defaulted *configinit.BkeConfig) {
	// Credential presence/authentication is evaluated by the SSH collector;
	// the sanitized AdapterInput intentionally carries no password.
	nodeErr := validation.ValidateBKENodesNonStandard(nodes)
	res.NodesValid = nodeErr == nil
	res.recordValidation("local.node.valid", nodeValidationField(nodes, nodeErr), nodeErr)
	providerNodes := providernode.SetDefaultsForNodes(providernode.ConvertBKENodesToNodes(nodes))
	for i, n := range nodes {
		roles := n.Spec.Role
		res.NodeRoles[n.Name] = logicalRoles(roles)
		res.NodeAddresses[n.Name] = n.Spec.IP
		if i < len(providerNodes) {
			res.NodeMetadata[n.Name] = map[string]string{
				"sshPort": providerNodes[i].Port, "sshUser": providerNodes[i].Username,
				"hostname": n.Spec.Hostname, "runtimeDataRoot": res.DataRoot,
				"kubeletRoot": res.KubeletRoot, "etcdDataDir": res.EtcdDataDir,
				"certificatesDir": defaulted.Cluster.CertificatesDir,
			}
		}
		for _, role := range roles {
			switch role {
			case providernode.MasterNodeRole:
				res.MasterCount++
			case providernode.MasterWorkerNodeRole:
				res.MasterCount++
				res.WorkerCount++
			case providernode.EtcdNodeRole:
				res.EtcdCount++
			case providernode.WorkerNodeRole:
				res.WorkerCount++
			default:
				continue
			}
		}
	}
}

func (r *ResolveResult) recordValidation(key, field string, err error) {
	if err == nil {
		return
	}
	if r.ValidationFields == nil {
		r.ValidationFields = make(map[string]ValidationFinding)
	}
	r.ValidationFields[key] = ValidationFinding{Field: field, Cause: err.Error()}
}

func explicitAPIPort(cluster *configv1beta1.BKECluster) bool {
	return cluster != nil && cluster.Spec.ClusterConfig != nil &&
		cluster.Spec.ClusterConfig.Cluster.APIServer != nil && cluster.Spec.ClusterConfig.Cluster.APIServer.Port != 0
}

func policyPort(config preflightconfig.Config, operation api.Operation, key string) int {
	for _, port := range config.Operation(operation).Ports {
		if port.Key == key {
			return port.DefaultPort
		}
	}
	return 0
}

func validateClusterPure(config *configinit.BkeConfig) (string, error) {
	cluster := config.Cluster
	if cluster.CertificatesDir == "" {
		return "cluster.spec.clusterConfig.cluster.certificatesDir", errors.New("certificatesDir is required")
	}
	if err := validation.ValidateK8sVersion(cluster.KubernetesVersion); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.kubernetesVersion", err)
	}
	if err := validation.ValidateKubeletComponent(cluster.Kubelet); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.kubelet", err)
	}
	if err := validation.ValidateControlPlaneComponents(cluster.ControlPlane); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.controlPlane", err)
	}
	if err := validation.ValidateNetworking(cluster.Networking); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.networking", err)
	}
	if err := validation.ValidateContainerRuntime(cluster.ContainerRuntime); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.containerRuntime", err)
	}
	if err := validation.ValidateRepo(cluster.HTTPRepo); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.httpRepo", err)
	}
	if err := validation.ValidateRepo(cluster.ImageRepo); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.imageRepo", err)
	}
	if err := validation.ValidateAddons(config.Addons); err != nil {
		return invalidClusterField("cluster.spec.clusterConfig.addons", err)
	}
	chartRepoErr := validation.ValidateRepo(cluster.ChartRepo)
	if validation.IsContainChartAddon(config.Addons) && chartRepoErr != nil {
		return invalidClusterField("cluster.spec.clusterConfig.cluster.chartRepo", chartRepoErr)
	}
	return "", nil
}

func invalidClusterField(field string, err error) (string, error) {
	return refineClusterValidationField(field, err), err
}

func refineClusterValidationField(base string, err error) string {
	message := strings.ToLower(err.Error())
	for _, leaf := range []string{
		"kubernetesVersion", "manifestsDir", "dataDir", "podSubnet", "serviceSubnet",
		"dnsDomain", "cri", "runtime", "port", "domain", "ip",
	} {
		if strings.Contains(message, strings.ToLower(leaf)) && !strings.HasSuffix(base, leaf) {
			return base + "." + leaf
		}
	}
	return base
}

func nodeValidationField(nodes []configv1beta1.BKENode, err error) string {
	if err == nil {
		return ""
	}
	message := strings.ToLower(err.Error())
	var leaf string
	if strings.Contains(message, "hostname") {
		leaf = "hostname"
	} else if strings.Contains(message, "ip") {
		leaf = "ip"
	} else {
		leaf = "role"
	}
	for index, node := range nodes {
		if (node.Name != "" && strings.Contains(message, strings.ToLower(node.Name))) ||
			(node.Spec.IP != "" && strings.Contains(message, strings.ToLower(node.Spec.IP))) {
			return fmt.Sprintf("nodes[%d].spec.%s", index, leaf)
		}
	}
	return "nodes[*].spec." + leaf
}

func addonSourcesValid(config *configinit.BkeConfig) bool {
	for _, addon := range config.Addons {
		if addon.Name == "" {
			return false
		}
		if addon.Type == "chart" && config.Cluster.ChartRepo.Domain == "" && config.Cluster.ChartRepo.Ip == "" {
			return false
		}
	}
	return true
}

func logicalRoles(roles []string) []string {
	out := make([]string, 0, len(roles)+1)
	for _, role := range roles {
		switch role {
		case providernode.MasterWorkerNodeRole:
			out = append(out, "master", "worker")
		case providernode.WorkerNodeRole:
			out = append(out, "worker")
		default:
			out = append(out, role)
		}
	}
	return out
}

func providerSources(cluster configv1beta1.Cluster) []api.Source {
	candidates := []api.Source{
		providerSource("image-repo", "image", cluster.ImageRepo),
		providerSource("chart-repo", "chart", cluster.ChartRepo),
		providerSource("http-repo", "package", cluster.HTTPRepo),
	}
	sources := make([]api.Source, 0, len(candidates))
	seenEndpoints := make(map[string]struct{}, len(candidates))
	for _, source := range candidates {
		endpoint := strings.TrimSpace(source.Endpoint)
		if endpoint == "" {
			continue
		}
		if _, exists := seenEndpoints[endpoint]; exists {
			continue
		}
		seenEndpoints[endpoint] = struct{}{}
		sources = append(sources, source)
	}
	return sources
}

func providerSource(id, kind string, repo configv1beta1.Repo) api.Source {
	domain := strings.TrimSpace(repo.Domain)
	ip := strings.TrimSpace(repo.Ip)
	host := domain
	if host == "" {
		host = ip
	}
	endpoint := host
	if host == "" {
		endpoint = ""
	} else if repo.Port != "" {
		endpoint = net.JoinHostPort(host, repo.Port)
	} else if parsed := net.ParseIP(strings.Trim(host, "[]")); parsed != nil && strings.Contains(host, ":") {
		endpoint = "[" + strings.Trim(host, "[]") + "]"
	} else {
		endpoint = host
	}
	// Both a domain and an explicit IP configured: keep the domain as the
	// operator-facing endpoint and record the IP so node checks connect to
	// the configured address while preserving the domain as Host/SNI.
	address := ""
	if domain != "" && ip != "" {
		port := repo.Port
		if port == "" {
			port = "443"
		}
		address = net.JoinHostPort(strings.Trim(ip, "[]"), port)
	}
	scheme := "http"
	if repo.Port == "443" || repo.Port == "40443" || repo.Port == "" && kind == "image" {
		scheme = "https"
	}
	authRef := ""
	if repo.AuthSecretRef != nil {
		authRef = repo.AuthSecretRef.Namespace + "/" + repo.AuthSecretRef.Name
	}
	return api.Source{
		ID:         id,
		Kind:       kind,
		Endpoint:   endpoint,
		Address:    address,
		Scheme:     scheme,
		ObjectPath: strings.TrimPrefix(repo.Prefix, "/"),
		AuthRef:    authRef,
	}
}

// TopologySummary returns the resolved topology as a string for logging.
func (r ResolveResult) TopologySummary() string {
	return fmt.Sprintf(
		"masters=%d etcd=%d workers=%d endpointExplicit=%v",
		r.MasterCount,
		r.EtcdCount,
		r.WorkerCount,
		r.EndpointExplicit,
	)
}
