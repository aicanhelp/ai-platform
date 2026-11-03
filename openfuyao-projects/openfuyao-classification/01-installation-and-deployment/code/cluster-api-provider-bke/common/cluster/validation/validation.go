/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package validation

import (
	"fmt"
	"net"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/blang/semver"
	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	MinSupportedK8sVersion, _       = semver.ParseTolerant("v1.21.0")
	DockerMinSupportedK8sVersion, _ = semver.ParseTolerant("v1.24.0")
	MaxSupportedK8sVersion, _       = semver.ParseTolerant("v1.28.0")
	masterNodeEvenDivisor           = 2
	maxFieldLength                  = 255
)

// ValidateBKENodes validates a list of BKENode resources
// This is the entry point for node validation after BKECluster splitting
func ValidateBKENodes(bkeNodes []v1beta1.BKENode) error {
	if len(bkeNodes) == 0 {
		return errors.New("at least one BKENode must be defined")
	}

	// Convert BKENodes to internal Node slice for validation
	nodes := node.ConvertBKENodesToNodes(bkeNodes)

	// Reuse existing validation logic
	return ValidateNodesFields(nodes)
}

// ValidateBKENodesNonStandard validates BKENodes with non-standard rules
func ValidateBKENodesNonStandard(bkeNodes []v1beta1.BKENode) error {
	if len(bkeNodes) == 0 {
		return nil
	}

	// Convert BKENodes to internal Node slice for validation
	nodes := node.ConvertBKENodesToNodes(bkeNodes)

	// Reuse existing validation logic
	return ValidateNonStandardNodesFields(nodes)
}

func ValidateBKEConfig(bkeConfig v1beta1.BKEConfig) error {
	if err := ValidateCluster(bkeConfig); err != nil {
		return err
	}
	if err := ValidateLoadBalanceType(bkeConfig.Cluster.LoadBalanceType); err != nil {
		return err
	}
	//todo remove this
	if err := ValidateCustomExtra(bkeConfig.CustomExtra); err != nil {
		return err
	}
	if err := ValidateAddons(bkeConfig.Addons); err != nil {
		return err
	}
	return nil
}

func ValidateNonStandardBKEConfig(bkeConfig v1beta1.BKEConfig) error {
	if bkeConfig.Cluster.KubernetesVersion != "" {
		if err := ValidateK8sVersion(bkeConfig.Cluster.KubernetesVersion); err != nil {
			return err
		}
	}
	if err := ValidateKubeletComponent(bkeConfig.Cluster.Kubelet); err != nil {
		return err
	}

	if err := ValidateNetworking(bkeConfig.Cluster.Networking); err != nil {
		return err
	}

	if err := ValidateContainerRuntime(bkeConfig.Cluster.ContainerRuntime); err != nil {
		return err
	}

	if err := ValidateRepo(bkeConfig.Cluster.HTTPRepo); err != nil {
		return errors.Errorf("Cluster httpRepo %s", err.Error())
	}
	if err := ValidateRepo(bkeConfig.Cluster.ImageRepo); err != nil {
		return errors.Errorf("Cluster imageRepo %s", err.Error())
	}
	return nil
}

func ValidateNodesFields(nodes node.Nodes) error {
	if err := validateRequiredNodes(nodes); err != nil {
		return err
	}
	workerNodes := nodes.Worker()
	masterWorkerNodes := nodes.MasterWorker()
	if workerNodes.Length() == 0 && masterWorkerNodes.Length() == 0 {
		return NoWorkerNodeError()
	}

	// 单节点字段校验已在 BKENode 创建时通过 webhook ValidateSingleNode 完成，此处不再重复校验
	for _, n := range nodes {
		// check field is unique
		if n.Hostname != "" {
			if nodes.Filter(node.FilterOptions{"Hostname": n.Hostname}).Length() > 1 {
				return errors.Errorf("node field is not valid, hostname is not unique %q", n.Hostname)
			}
		}
		if nodes.Filter(node.FilterOptions{"IP": n.IP}).Length() > 1 {
			return errors.Errorf("node field is not valid, IP is not unique %q", n.IP)
		}
	}

	return nil
}

func ValidateNonStandardNodesFields(nodes node.Nodes) error {
	if nodes == nil || nodes.Length() == 0 {
		return nil
	}
	if err := ValidateNodesRole(nodes); err != nil {
		return err
	}
	if err := ValidateNodesHostnameUnique(nodes); err != nil {
		return err
	}
	if err := ValidateNodesIPUnique(nodes); err != nil {
		return err
	}
	return nil
}

func ValidateNodesHostnameUnique(nodes node.Nodes) error {
	for _, n := range nodes {
		if n.Hostname != "" {
			if nodes.Filter(node.FilterOptions{"Hostname": n.Hostname}).Length() > 1 {
				return errors.Errorf("nodes field is not valid, hostname is not unique %q", n.Hostname)
			}
		}
	}
	return nil
}

func ValidateNodesIPUnique(nodes node.Nodes) error {
	for _, n := range nodes {
		if n.IP == "" {
			return errors.Errorf("node %s field is not valid, IP is required", node.NodeInfo(n))
		}
		if nodes.Filter(node.FilterOptions{"IP": n.IP}).Length() > 1 {
			return errors.Errorf("node field is not valid, IP is not unique %q", n.IP)
		}
	}
	return nil
}

func ValidateNodesRole(nodes node.Nodes) error {
	if err := validateRequiredNodes(nodes); err != nil {
		return err
	}
	for _, n := range nodes {
		roles := n.Role
		if utils.SliceContainsSlice(roles, []string{node.MasterNodeRole, node.WorkerNodeRole}) {
			return errors.Errorf("node %s cannot set %q and %q roles at the same time, please set %q instead", node.NodeInfo(v1beta1.Node(n)), node.MasterNodeRole, node.WorkerNodeRole, node.MasterWorkerNodeRole)
		}
		if utils.SliceContainsSlice(roles, []string{node.MasterNodeRole, node.EtcdNodeRole, node.MasterWorkerNodeRole}) {
			return errors.Errorf("node %s cannot set %q 、%q and %q roles at the same time", node.NodeInfo(v1beta1.Node(n)), node.MasterNodeRole, node.EtcdNodeRole, node.MasterWorkerNodeRole)
		}
		if utils.SliceContainsSlice(roles, []string{node.WorkerNodeRole, node.EtcdNodeRole}) {
			return errors.Errorf("node %s with %q role cannot set %q and %q roles at the same time", node.NodeInfo(v1beta1.Node(n)), node.WorkerNodeRole, node.WorkerNodeRole, node.EtcdNodeRole)
		}
	}

	return nil
}

func validateRequiredNodes(nodes node.Nodes) error {
	if nodes == nil || nodes.Length() == 0 {
		return errors.New("nodes is required")
	}

	masterNodes := nodes.Master()
	if masterNodes.Length() == 0 {
		return NoMasterNodeError()
	}
	if masterNodes.Length()%masterNodeEvenDivisor == 0 {
		return MasterNodeOddError()
	}
	etcdNodes := nodes.Etcd()
	if etcdNodes.Length() == 0 {
		return NoEtcdNodeError()
	}

	return nil
}

func ValidateSingleNode(n node.Node) error {
	val := reflect.ValueOf(&n).Elem()
	typ := reflect.TypeOf(&n).Elem()
	for i := 0; i < typ.NumField(); i++ {
		// validate Role field
		if typ.Field(i).Name == "Role" {
			refval := reflect.ValueOf(n)
			rolesValue, ok := val.Field(i).Interface().([]string)
			if !ok {
				return errors.New("node role field type assertion failed")
			}
			roles := rolesValue
			if utils.SliceContainsSlice(roles, []string{node.MasterNodeRole, node.WorkerNodeRole}) {
				return errors.Errorf("node %s cannot set %q and %q roles at the same time, please set %q instead", node.NodeInfo(v1beta1.Node(n)), node.MasterNodeRole, node.WorkerNodeRole, node.MasterWorkerNodeRole)
			}
			if utils.SliceContainsSlice(roles, []string{node.MasterNodeRole, node.EtcdNodeRole, node.MasterWorkerNodeRole}) {
				return errors.Errorf("node %s cannot set %q 、%q and %q roles at the same time", node.NodeInfo(v1beta1.Node(n)), node.MasterNodeRole, node.EtcdNodeRole, node.MasterWorkerNodeRole)
			}
			if utils.SliceContainsSlice(roles, []string{node.WorkerNodeRole, node.EtcdNodeRole}) {
				return errors.Errorf("node %s with %q role cannot set %q and %q roles at the same time", node.NodeInfo(v1beta1.Node(n)), node.WorkerNodeRole, node.WorkerNodeRole, node.EtcdNodeRole)
			}
			if utils.SliceContainsString(roles, node.WorkerNodeRole) && !refval.FieldByName("ControlPlane").IsZero() {
				n.ControlPlane = v1beta1.ControlPlane{}
				return errors.Errorf("node %s with %q role cannot configure control plane components", node.NodeInfo(v1beta1.Node(n)), node.WorkerNodeRole)
			}
		}

		// exclude validation of non-string fields (v1beta1.ControlPlane ,v1beta1.Kubelet)
		if typ.Field(i).Type != reflect.TypeOf("") {
			// todo add validation for non-string fields
			continue
		}
		// validate string fields value != ""
		if val.Field(i).IsZero() {
			// not validate Hostname field, allow empty
			if typ.Field(i).Name != "Hostname" {
				return errors.Errorf("node %s field %q is required", node.NodeInfo(v1beta1.Node(n)), typ.Field(i).Name)
			}
		} else {
			if val.Field(i).Len() > maxFieldLength {
				return errors.Errorf("nodes %s field %q cannot be longer than %d characters", node.NodeInfo(v1beta1.Node(n)), typ.Field(i).Name, maxFieldLength)
			}
		}
		// validate IP field value != ""
		if typ.Field(i).Name == "IP" && !bkenet.ValidIP(val.Field(i).String()) {
			return errors.Errorf("node %s field IP %q is not valid", node.NodeInfo(v1beta1.Node(n)), val.Field(i).String())
		}
	}
	return nil
}

// ValidateCluster validates the cluster config
func ValidateCluster(bkeConfig v1beta1.BKEConfig) error {
	obj := bkeConfig.Cluster
	if obj.CertificatesDir == "" {
		return errors.New("The certificatesDir is required. ")
	}

	if err := ValidateK8sVersion(obj.KubernetesVersion); err != nil {
		return err
	}

	if err := ValidateKubeletComponent(obj.Kubelet); err != nil {
		return err
	}

	if err := ValidateControlPlaneComponents(obj.ControlPlane); err != nil {
		return err
	}

	if err := ValidateNetworking(obj.Networking); err != nil {
		return err
	}

	if err := ValidateContainerRuntime(obj.ContainerRuntime); err != nil {
		return err
	}

	if err := ValidateRepo(obj.HTTPRepo); err != nil {
		return errors.Errorf("Cluster httpRepo %s", err.Error())
	}
	if err := ValidateRepo(obj.ImageRepo); err != nil {
		return errors.Errorf("Cluster imageRepo %s", err.Error())
	}
	if err := ValidateChartRepo(obj.ChartRepo, bkeConfig.Addons); err != nil {
		return errors.Errorf("Cluster chartRepo %s", err.Error())
	}
	return nil
}

// IsContainChartAddon is check addons is contain chart addon.
func IsContainChartAddon(addons addon.Addons) bool {
	if len(addons) == 0 {
		return false
	}

	for _, item := range addons {
		if item.Type == addon.ChartAddon {
			return true
		}
	}
	return false
}

// ValidateChartRepo is validate chart repo.
func ValidateChartRepo(chartRepo v1beta1.Repo, addons addon.Addons) error {
	hasChart := IsContainChartAddon(addons)
	if !hasChart {
		return nil
	}

	if chartRepo == (v1beta1.Repo{}) {
		return fmt.Errorf("not set chart repo, please set it in the cluster config file")
	}

	if err := ValidateRepo(chartRepo); err != nil {
		return err
	}
	if _, err := ResolveReachableRepoAddress(chartRepo); err != nil {
		return err
	}
	return nil
}

// BuildRepoURL builds the repo URL.
func BuildRepoURL(host, port, prefix string) string {
	addr := host
	if port != "" {
		addr = net.JoinHostPort(host, port)
	}
	return fmt.Sprintf("%s/%s", addr, prefix)
}

// 校验连通性
func checkReachable(addr string) bool {
	host, port := parseHostPort(addr)
	if host == "" {
		return false
	}

	if checkHTTPReachable(host, port) {
		return true
	}

	conn, err := net.DialTimeout("tcp", buildDialAddress(host, port), 5*time.Second)
	if err != nil {
		return false
	}
	defer func() {
		if err := conn.Close(); err != nil {
			log.Warnf("failed to close conn: %v, address: %s", err, addr)
		}
	}()
	return true
}

func parseHostPort(addr string) (string, string) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "", ""
	}
	host, port, err := net.SplitHostPort(addr)
	if err == nil {
		return strings.TrimSpace(host), strings.TrimSpace(port)
	}
	// fallback for inputs without explicit port
	return strings.TrimSpace(addr), ""
}

func buildDialAddress(host, port string) string {
	if port == "" {
		return net.JoinHostPort(host, "80")
	}
	return net.JoinHostPort(host, port)
}

func buildHTTPProbeURL(host, port string) string {
	if port == "" {
		return fmt.Sprintf("http://%s", host)
	}
	return fmt.Sprintf("http://%s", net.JoinHostPort(host, port))
}

func checkHTTPReachable(host, port string) bool {
	client := &http.Client{Timeout: 5 * time.Second}
	url := buildHTTPProbeURL(host, port)

	// HEAD is cheap; some servers don't allow it, so fallback to GET.
	for _, method := range []string{http.MethodHead, http.MethodGet} {
		req, err := http.NewRequest(method, url, nil)
		if err != nil {
			continue
		}
		if method == http.MethodGet {
			req.Header.Set("Range", "bytes=0-0")
		}
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Warnf("failed to close HTTP probe response body: %v, url: %s", closeErr, url)
		}
		// Any non-5xx status means endpoint is reachable.
		if resp.StatusCode < 500 {
			return true
		}
	}
	return false
}

// ResolveReachableRepoAddress returns the reachable repo address.
func ResolveReachableRepoAddress(repo v1beta1.Repo) (string, error) {
	if repo.Domain != "" {
		url := BuildRepoURL(repo.Domain, repo.Port, repo.Prefix)
		if checkReachable(net.JoinHostPort(repo.Domain, repo.Port)) {
			return url, nil
		}
	}
	if repo.Ip != "" {
		url := BuildRepoURL(repo.Ip, repo.Port, repo.Prefix)
		if checkReachable(net.JoinHostPort(repo.Ip, repo.Port)) {
			return url, nil
		}
	}
	return "", fmt.Errorf("repo not reachable via domain or ip")
}

// BuildHTTPRepoBaseURL builds http://host:port for HTTP binary downloads.
func BuildHTTPRepoBaseURL(host, port string) string {
	if host == "" {
		if port == "" {
			return "http://"
		}
		return fmt.Sprintf("http://:%s", port)
	}
	if port == "" {
		return fmt.Sprintf("http://%s", host)
	}
	return fmt.Sprintf("http://%s", net.JoinHostPort(host, port))
}

// ResolveReachableHTTPRepoHost picks HTTP repo host: prefer domain when reachable, else IP.
func ResolveReachableHTTPRepoHost(repo v1beta1.Repo) string {
	if repo.Domain != "" && repo.Ip != "" {
		if checkReachable(net.JoinHostPort(repo.Domain, repo.Port)) {
			return repo.Domain
		}
		log.Warnf("HTTP repo domain %q unreachable, fallback to IP %q", repo.Domain, repo.Ip)
		return repo.Ip
	}
	if repo.Domain != "" && checkReachable(net.JoinHostPort(repo.Domain, repo.Port)) {
		return repo.Domain
	}
	if repo.Ip != "" {
		return repo.Ip
	}
	return repo.Domain
}

// ResolveReachableHTTPRepoBaseURL returns http://host:port for HTTP binary downloads.
// When domain is not reachable (e.g. offline env without DNS), fallback to configured IP.
func ResolveReachableHTTPRepoBaseURL(repo v1beta1.Repo) string {
	return BuildHTTPRepoBaseURL(ResolveReachableHTTPRepoHost(repo), repo.Port)
}

func ValidateK8sVersion(version string) error {
	if version == "" {
		return errors.New("The kubernetesVersion is required. ")
	}

	v, err := semver.ParseTolerant(version)
	if err != nil {
		return err
	}
	// 不验证最高支持版本
	if v.LT(MinSupportedK8sVersion) {
		return errors.Errorf("The kubernetesVersion %s is not supported. The minimum supported version of BKE is %s", version, MinSupportedK8sVersion)
	}
	return nil
}

func ValidateControlPlaneComponents(obj v1beta1.ControlPlane) error {
	if obj.Etcd == nil {
		return errors.New("The etcd is required. ")
	}
	if obj.Etcd.DataDir == "" {
		return errors.New("Cluster etcd dataDir is required. ")
	}
	return nil
}

func ValidateKubeletComponent(obj *v1beta1.Kubelet) error {
	if obj == nil {
		return nil
	}
	if obj.ManifestsDir == "" {
		return errors.New("Cluster kubelet manifestsDir is required. ")
	}
	return nil
}

func ValidateNetworking(obj v1beta1.Networking) error {
	if obj.PodSubnet == "" {
		return errors.New("The podSubnet is required. ")
	}
	if err := ValidateIPNet(obj.PodSubnet); err != nil {
		return errors.Errorf("Cluster podSubnet %s", err.Error())
	}
	if obj.ServiceSubnet == "" {
		return errors.New("The serviceSubnet is required. ")
	}
	if err := ValidateIPNet(obj.ServiceSubnet); err != nil {
		return errors.Errorf("Cluster serviceSubnet %s", err.Error())
	}
	if obj.DNSDomain == "" {
		return errors.New("The dnsDomain is required. ")
	}
	if err := bkenet.IsDNS1123Subdomain(obj.DNSDomain); err != nil {
		return err
	}
	return nil
}

func ValidateContainerRuntime(obj v1beta1.ContainerRuntime) error {
	if obj.CRI != "" {
		if obj.CRI != "docker" && obj.CRI != "containerd" {
			return errors.New("The cri only support docker or containerd. ")
		}
	}
	if obj.Runtime != "" {
		if obj.Runtime != "kata" && obj.Runtime != "runc" && obj.Runtime != "richrunc" {
			return errors.New("The runtime only support kata, runc or richrunc. ")
		}
	}
	return nil
}

func ValidateIPNet(cidr string) error {
	_, _, err := net.ParseCIDR(cidr)
	if err != nil {
		return errors.Errorf("%q is invalid, couldn't parse subnet", cidr)
	}
	return err
}

func ValidateRepo(repo v1beta1.Repo) error {
	if repo.Domain != "" {
		if err := bkenet.IsDNS1123Subdomain(repo.Domain); err != nil {
			return err
		}
	}
	if repo.Ip != "" {
		if !bkenet.ValidIP(repo.Ip) {
			return errors.New("repo ip is not a valid IP")
		}
	}
	if repo.Port != "" {
		if v, err := strconv.Atoi(repo.Port); err != nil {
			return errors.Errorf("The repo port %s is invalid, %s", repo.Port, err.Error())
		} else if v <= 0 {
			return errors.Errorf("The repo port %s is invalid, port can't less than or equal to zero", repo.Port)
		}
	}

	if repo.Domain == "" && repo.Ip == "" {
		return errors.New("The domain or ip are required when repo is public.")
	}

	return nil
}

func GetImageRepoAddress(repo v1beta1.Repo) string {
	address := ""

	// 域名 端口不为空，ip为空 域名+端口
	if repo.Domain != "" && repo.Port != "" && repo.Ip == "" {
		address = fmt.Sprintf("%s:%s", repo.Domain, repo.Port)
		if repo.Port == "443" {
			address = repo.Domain
		}
	}
	// ip 端口不为空，域名为空 ip+端口
	if repo.Domain == "" && repo.Port != "" && repo.Ip != "" {
		address = fmt.Sprintf("%s:%s", repo.Ip, repo.Port)
		if repo.Port == "443" {
			address = repo.Ip
		}
	}
	// 域名 ip 端口都不为空 域名+端口
	if repo.Domain != "" && repo.Port != "" && repo.Ip != "" {
		address = fmt.Sprintf("%s:%s", repo.Domain, repo.Port)
		if repo.Port == "443" {
			address = repo.Domain
		}
	}
	// 域名 不为空 端口 ip为空 域名
	if repo.Domain != "" && repo.Port == "" && repo.Ip == "" {
		address = repo.Domain
	}
	// ip 不为空 端口 域名为空 ip
	if repo.Domain == "" && repo.Port == "" && repo.Ip != "" {
		address = repo.Ip
	}
	return address
}

func ValidateCustomExtra(extra map[string]string) error {
	if len(extra) == 0 {
		return errors.New("must contain the required parameter containerd")
	}
	if _, ok := extra["containerd"]; !ok {
		return errors.New("must contain the required parameter containerd")
	}
	match, err := regexp.MatchString("^containerd-([a-z0-9\\.]+)-([a-z0-9]+)-([a-z0-9\\.\\{\\}]+).tar.gz$", extra["containerd"])
	if err != nil {
		return err
	}
	if !match {
		return errors.New("The value of containerd must meet the regular expression ^containerd-([a-z0-9\\.]+)-([a-z0-9]+)-([a-z0-9\\.\\{\\}]+).tar.gz$")
	}
	return nil
}

// ValidateLoadBalanceType validates the loadBalanceType enum value; empty is treated as the default "ha".
func ValidateLoadBalanceType(lbType string) error {
	switch lbType {
	case "", "ha", "nginx":
		return nil
	default:
		return errors.Errorf(`loadBalanceType must be "ha" or "nginx", got %q`, lbType)
	}
}

func ValidateAddons(addons addon.Addons) error {
	if len(addons) == 0 {
		return nil
	}
	for _, ad := range addons {
		if addons.Filter(addon.FilterOptions{"Name": ad.Name, "Version": ad.Version}).Length() != 1 {
			return errors.Errorf("The addon name: %q version: %q must be unique", ad.Name, ad.Version)
		}
		if ad.Type != addon.YamlAddon && ad.Type != addon.ChartAddon && ad.Type != "" {
			return fmt.Errorf("the addon %s type error", ad.Name)
		}
	}
	return nil
}
