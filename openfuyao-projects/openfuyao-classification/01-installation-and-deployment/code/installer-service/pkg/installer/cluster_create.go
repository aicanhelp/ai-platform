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

package installer

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const defaultClusterYaml = `apiVersion: bke.bocloud.com/v1beta1
kind: BKECluster
metadata:
  creationTimestamp: null
  name: bke-cluster
  namespace: bke-cluster
spec:
  clusterConfig:
    addons:
    - name: kubeproxy
      param:
        clusterNetworkMode: calico
      version: v1.28.8
    - block: true
      name: calico
      param:
        calicoMode: vxlan
        ipAutoDetectionMethod: skip-interface=nerdctl*
      version: v3.27.3
    - name: coredns
      version: v1.10.1
    - name: openfuyao-system-controller
      param:
        helmRepo: https://helm.openfuyao.cn/_core
        tagVersion: latest
      version: latest
    cluster:
      agentHealthPort: "58080"
      apiServer:
        extraArgs:
          max-mutating-requests-inflight: "3000"
          max-requests-inflight: "1000"
          watch-cache-sizes: node#1000,pod#5000
      certificatesDir: /etc/kubernetes/pki
      containerRuntime:
        cri: containerd
        param:
          data-root: /var/lib/containerd
        runtime: runc
      controllerManager:
        extraArgs:
          kube-api-burst: "1000"
          kube-api-qps: "1000"
      etcd:
        dataDir: /var/lib/etcd
      httpRepo:
        domain: http.bocloud.k8s
        ip: 192.168.100.20
        port: "40080"
        prefix: ""
      imageRepo:
        domain: default
        ip: ""
        port: "443"
        prefix: ""
      kubelet:
        extraArgs:
          kube-api-burst: "2000"
          kube-api-qps: "1000"
        extraVolumes:
        - hostPath: /var/lib/kubelet
          name: kubelet-root-dir
      kubernetesVersion: v1.28.8
      openFuyaoVersion: v00.00
      containerdVersion: v0.0.0
      networking:
        dnsDomain: cluster.local
        podSubnet: 10.250.0.0/16
        serviceSubnet: 10.96.0.0/16
      ntpServer: cn.pool.ntp.org:123
      scheduler:
        extraArgs:
          kube-api-qps: "1000"
    customExtra:
      chart_password: chart
      chart_username: chart
      chartRepoPort: "38080"
      clusterapi: latest
      containerd: containerd-v2.1.1-linux-{.arch}.tar.gz
      domain: deploy.bocloud.k8s
      host: 192.168.201.85
      imageRepoPort: "40443"
      nfsserverpath: /
      otherRepo: cr.openfuyao.cn/openfuyao/
      otherRepoIp: 119.3.216.97
      otherSource: ""
      yumRepoPort: "40080"
  controlPlaneEndpoint: {}
  pause: false
status:
  agentStatus: {}
  kubernetesVersion: ""
  ready: false
`

const (
	defaultControlPlanePort int64 = 36443
	openFuyaoAddonAlias           = "openFuyao-cores"
	openFuyaoAddonName            = "openfuyao-system-controller"
)

// BuildCreateClusterYaml builds a multi-doc YAML for BKECluster and BKENode resources.
func BuildCreateClusterYaml(req CreateClusterRequest, defaults DefaultResp) (string, error) {
	if strings.TrimSpace(req.Cluster.Name) == "" {
		return "", fmt.Errorf("cluster name is required")
	}
	clusterObj := map[string]any{}
	if err := yaml.Unmarshal([]byte(defaultClusterYaml), &clusterObj); err != nil {
		return "", fmt.Errorf("unmarshal default cluster yaml failed: %w", err)
	}
	setIfNotEmpty(&clusterObj, req.Cluster.Name, "metadata", "name")
	setIfNotEmpty(&clusterObj, req.Cluster.Name, "metadata", "namespace")
	setIfNotEmpty(&clusterObj, req.Cluster.OpenFuyaoVersion, "spec", "clusterConfig", "cluster", "openFuyaoVersion")
	if defaults.KubernetesVersion != "" {
		setIfNotEmpty(&clusterObj, defaults.KubernetesVersion, "spec", "clusterConfig", "cluster", "kubernetesVersion")
	}
	if defaults.ContainerRuntime != "" {
		setIfNotEmpty(&clusterObj, defaults.ContainerRuntime, "spec", "clusterConfig", "cluster", "containerRuntime", "cri")
	}
	if defaults.AgentHealthPort != "" {
		setIfNotEmpty(&clusterObj, defaults.AgentHealthPort, "spec", "clusterConfig", "cluster", "agentHealthPort")
	}
	if defaults.Ip != "" {
		setIfNotEmpty(&clusterObj, defaults.Ip, "spec", "clusterConfig", "cluster", "httpRepo", "ip")
		setIfNotEmpty(&clusterObj, defaults.Ip, "spec", "clusterConfig", "customExtra", "host")
	}
	imageRepoDomain := strings.TrimSpace(req.Cluster.ImageRepo.Url)
	imageRepoIP := strings.TrimSpace(req.Cluster.ImageRepo.Ip)
	if imageRepoIP != "" {
		setIfNotEmpty(&clusterObj, imageRepoIP, "spec", "clusterConfig", "cluster", "imageRepo", "ip")
	}
	if imageRepoDomain != "" && !strings.Contains(imageRepoDomain, "cr.openfuyao.cn") {
		setIfNotEmpty(&clusterObj, imageRepoDomain, "spec", "clusterConfig", "cluster", "imageRepo", "domain")
		setIfNotEmpty(&clusterObj, "", "spec", "clusterConfig", "customExtra", "otherRepo")
		setIfNotEmpty(&clusterObj, "", "spec", "clusterConfig", "customExtra", "otherRepoIp")
		setIfNotEmpty(&clusterObj, "40443", "spec", "clusterConfig", "cluster", "imageRepo", "port")
		setIfNotEmpty(&clusterObj, "kubernetes", "spec", "clusterConfig", "cluster", "imageRepo", "prefix")
	}
	if strings.TrimSpace(req.ControlPlaneEndpoint) != "" {
		_ = unstructured.SetNestedMap(clusterObj, map[string]any{"host": req.ControlPlaneEndpoint, "port": defaultControlPlanePort}, "spec", "controlPlaneEndpoint")
	}
	addonsRaw, _, _ := unstructured.NestedSlice(clusterObj, "spec", "clusterConfig", "addons")
	_ = unstructured.SetNestedSlice(clusterObj, mergeAddons(addonsRaw, req.Addons), "spec", "clusterConfig", "addons")
	builder := strings.Builder{}
	for _, node := range req.Nodes {
		nodeYaml, err := buildNodeYaml(req.Cluster.Name, node)
		if err != nil {
			return "", err
		}
		builder.Write(nodeYaml)
		builder.WriteString("\n---\n")
	}
	clusterYaml, err := yaml.Marshal(clusterObj)
	if err != nil {
		return "", fmt.Errorf("marshal cluster yaml failed: %w", err)
	}
	builder.Write(clusterYaml)
	return builder.String(), nil
}

func setIfNotEmpty(obj *map[string]any, value string, fields ...string) {
	if value == "" {
		return
	}
	_ = unstructured.SetNestedField(*obj, value, fields...)
}

func buildNodeYaml(clusterName string, node CreateClusterNode) ([]byte, error) {
	port := strings.TrimSpace(node.Port)
	if port == "" || port == "0" {
		port = "22"
	}
	role := normalizeNodeRoles(node.Role)
	labels := map[string]any{
		"cluster.x-k8s.io/cluster-name": clusterName,
	}
	nodeObj := map[string]any{
		"apiVersion": "bke.bocloud.com/v1beta1",
		"kind":       "BKENode",
		"metadata": map[string]any{
			"name":      node.Hostname,
			"namespace": clusterName,
			"labels":    labels,
		},
		"spec": map[string]any{
			"hostname": node.Hostname,
			"ip":       node.Ip,
			"port":     port,
			"username": node.Username,
			"password": node.Password,
			"role":     role,
		},
	}
	nodeYaml, err := yaml.Marshal(nodeObj)
	if err != nil {
		return nil, fmt.Errorf("marshal node yaml failed: %w", err)
	}
	return nodeYaml, nil
}

func normalizeNodeRoles(roles []string) []string {
	result := make([]string, 0, len(roles))
	for _, role := range roles {
		switch role {
		case "master":
			result = append(result, "master/node")
		case "worker":
			result = append(result, "node")
		default:
			result = append(result, role)
		}
	}
	return result
}

func mergeAddons(defaultAddons []any, reqAddons []CreateClusterAddon) []any {
	if len(defaultAddons) == 0 && len(reqAddons) == 0 {
		return defaultAddons
	}
	reqMap := map[string]CreateClusterAddon{}
	for _, addon := range reqAddons {
		name := normalizeAddonName(addon.Name)
		addon.Name = name
		reqMap[name] = addon
	}

	result := make([]any, 0, len(defaultAddons))
	for _, raw := range defaultAddons {
		addonMap, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		name, ok := addonMap["name"].(string)
		if !ok || name == "" {
			continue
		}
		if name == openFuyaoAddonName {
			if _, exists := reqMap[name]; !exists {
				continue
			}
		}
		if reqAddon, exists := reqMap[name]; exists {
			if isAddonDisabled(reqAddon) {
				continue
			}
			currentParam := getOrCreateParamMap(addonMap)
			for k, v := range reqAddon.Params {
				currentParam[k] = v
			}
			addonMap["param"] = currentParam
		}
		result = append(result, addonMap)
		delete(reqMap, name)
	}

	for _, addon := range reqMap {
		if isAddonDisabled(addon) {
			continue
		}
		addonMap := map[string]any{
			"name":  addon.Name,
			"param": map[string]any{},
		}
		if len(addon.Params) > 0 {
			param := make(map[string]any, len(addon.Params))
			for k, v := range addon.Params {
				param[k] = v
			}
			addonMap["param"] = param
		}
		result = append(result, addonMap)
	}

	return result
}

func normalizeAddonName(name string) string {
	if name == openFuyaoAddonAlias {
		return openFuyaoAddonName
	}
	return name
}

func isAddonDisabled(addon CreateClusterAddon) bool {
	if addon.Params == nil {
		return false
	}
	enabled, ok := addon.Params["enabled"]
	if !ok {
		return false
	}
	value := strings.TrimSpace(strings.ToLower(enabled))
	return value == "false" || value == "0" || value == "no"
}
