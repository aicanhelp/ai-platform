/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package prometheus

import (
	"fmt"
	"regexp"
	"strings"

	"openfuyao.com/monitoring-service/pkg/client"
)

func makeExpr(metric string, opts client.QueryOptions) string {
	tmpl := promQLTemplatesResources[metric]
	switch opts.Level {
	case client.LevelCluster:
		return tmpl
	case client.LevelNode:
		return makeNodeMetricExpr(tmpl, opts)
	case client.LevelDeployment:
		return makeDeploymentMetricExpr(tmpl, opts)
	case client.LevelStatefulset:
		return makeStatefulsetMetricExpr(tmpl, opts)
	case client.LevelDaemonset:
		return makeDaemonsetMetricExpr(tmpl, opts)
	case client.LevelPod:
		return makePodMetricExpr(tmpl, opts)
	case client.LevelContainer:
		return makeContainerMetricExpr(tmpl, opts)
	case client.LevelControlPlaneComponents:
		tmpl = promQLTemplatesComponents[metric]
		return makeComponentsExpr(tmpl, opts)
	case client.LevelNodeComponents:
		tmpl = promQLTemplatesComponents[metric]
		return makeComponentsExpr(tmpl, opts)
	case client.LevelAddonsComponents:
		tmpl = promQLTemplatesComponents[metric]
		return makeComponentsExpr(tmpl, opts)
	default:
		return tmpl
	}
}

func makeNodeMetricExpr(tmpl string, o client.QueryOptions) string {
	var instanceSelector, nodeSelector string

	instanceSelector = fmt.Sprintf(`instance=~"%s"`, o.NodeName)

	patternMachine := `\b(machine_|cluster:|node_namespace_)[\w:]+`
	matchesMachine := findMatches(tmpl, patternMachine)
	nodeSelector = fmt.Sprintf(`node=~"%s"`, o.NodeName)
	for _, match := range matchesMachine {
		tmpl = strings.Replace(tmpl, match, match+"{"+nodeSelector+"}", -1)
	}

	patternNode := `\bnode_\w*(\{[^}]*\})?`
	matchesNode := findMatches(tmpl, patternNode)
	for _, match := range matchesNode {
		if strings.HasPrefix(match, "node_namespace") {
			continue
		}

		length := len(match)
		if match[length-1] == '}' {
			tmpl = strings.Replace(tmpl, match, match[0:(length-1)]+","+instanceSelector+"}", -1)
		} else {
			tmpl = strings.Replace(tmpl, match, match+"{"+instanceSelector+"}", -1)
		}
	}
	return tmpl
}

func makeDeploymentMetricExpr(tmpl string, o client.QueryOptions) string {
	var deploymentSelector string

	deploymentSelector = makeWorkloadSelector("ReplicaSet", o.DeploymentName)
	return makeWorkloadMetricExpr(deploymentSelector, tmpl, o)
}

func makeStatefulsetMetricExpr(tmpl string, o client.QueryOptions) string {
	var statefulsetSelector string

	statefulsetSelector = makeWorkloadSelector("StatefulSet", o.StatefulsetName)
	return makeWorkloadMetricExpr(statefulsetSelector, tmpl, o)
}

func makeDaemonsetMetricExpr(tmpl string, o client.QueryOptions) string {
	var daemonsetSelector string

	daemonsetSelector = makeWorkloadSelector("DaemonSet", o.DaemonsetName)
	return makeWorkloadMetricExpr(daemonsetSelector, tmpl, o)
}

func makeWorkloadMetricExpr(workloadSelector, tmpl string, o client.QueryOptions) string {
	var podSelector string

	podSelector = fmt.Sprintf(`pod=~".*(%s).*", namespace=~"%s"`, o.ResourceFilter, o.NamespaceName)

	matchesPodOwner := findMatches(tmpl, "kube_pod_owner{")
	matchesPodInfo := findMatches(tmpl, "kube_pod_info{")

	if len(matchesPodOwner) > 0 {
		tmpl = strings.Replace(tmpl, matchesPodOwner[0], matchesPodOwner[0]+workloadSelector+", ", -1)
	}

	if len(matchesPodInfo) > 0 {
		tmpl = strings.Replace(tmpl, matchesPodInfo[0], matchesPodInfo[0]+podSelector+", ", -1)
	}

	return tmpl
}

func makePodMetricExpr(tmpl string, o client.QueryOptions) string {
	var podSelector string

	podSelector = fmt.Sprintf(`pod=~"%s", namespace=~"%s"`, o.PodName, o.NamespaceName)

	patternContainer := `\bcontainer[^{}]*\{`
	matchesContainer := findMatches(tmpl, patternContainer)

	for _, match := range matchesContainer {
		tmpl = strings.Replace(tmpl, match, match+podSelector+", ", -1)

	}

	return tmpl
}

func makeContainerMetricExpr(tmpl string, o client.QueryOptions) string {
	var containerSelector string

	containerSelector = fmt.Sprintf(`namespace=~"%s", pod=~"%s", container=~"%s"`, o.NamespaceName, o.PodName,
		o.ContainerName)

	patternContainer := `\bcontainer[^{}]*\{`
	matchesContainer := findMatches(tmpl, patternContainer)

	for _, match := range matchesContainer {
		tmpl = strings.Replace(tmpl, match, match+containerSelector+", ", -1)
	}

	return tmpl
}

func makeComponentsExpr(tmpl string, o client.QueryOptions) string {
	replacements := make(map[string]string)
	if o.Component == client.KubeAPIServer {
		replacements["verb"] = o.Verb
		replacements["resource"] = o.Resource
	} else {
		replacements["instance"] = o.Instance
	}
	return replaceTemplatePlaceholders(tmpl, replacements)
}

func replaceTemplatePlaceholders(template string, replacements map[string]string) string {
	for placeholder, value := range replacements {
		placeholderPattern := `=~"$` + placeholder + `"`
		template = strings.Replace(template, placeholderPattern, `=~"`+value+`"`, -1)
	}
	return template
}

func makeWorkloadSelector(workloadKind, workloadName string) string {
	switch workloadKind {
	case "ReplicaSet":
		return fmt.Sprintf(`owner_kind="ReplicaSet", owner_name=~"^(%s)-[^-]{1,10}$"`, workloadName)
	case "StatefulSet":
		return fmt.Sprintf(`owner_kind="StatefulSet", owner_name=~"%s"`, workloadName)
	case "DaemonSet":
		return fmt.Sprintf(`owner_kind="DaemonSet", owner_name=~"%s"`, workloadName)
	default:
		return fmt.Sprintf(`owner_kind="Unknown", owner_name=~"%s"`, workloadName)
	}
}

func removeDuplicates(input []string) []string {
	seen := make(map[string]struct{})
	var result []string

	for _, value := range input {
		if _, ok := seen[value]; !ok {
			result = append(result, value)
			seen[value] = struct{}{}
		}
	}

	return result
}

func findMatches(tmpl string, pattern string) []string {
	re := regexp.MustCompile(pattern)
	matches := re.FindAllString(tmpl, -1)
	matches = removeDuplicates(matches)
	return matches
}
