/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package topology

// Default upgrade component names aligned with pkg/upgrade/catalog.go (used in tests).
const (
	defaultPreUpgradeResources = "pre-upgrade-resources"
	defaultProvider            = "provider"
	defaultBKEAgentUpgrade     = "bkeagent"
	defaultKubeProxy           = "kube-proxy"
	defaultCoreDNS             = "coredns"
	defaultEtcd                = "etcd"
	defaultContainerd          = "containerd"
	defaultK8sMaster           = "kubernetes-master"
	defaultK8sWorker           = "kubernetes-worker"
)

// DefaultDependenciesFor returns no fallback prerequisites.
// Upgrade DAG edges come from ComponentVersion.spec.dependencies in the release bundle only.
func DefaultDependenciesFor(_ string) []string {
	return nil
}

// DefaultDependencyResolver wraps DefaultDependenciesFor as a DependencyResolver.
func DefaultDependencyResolver() DependencyResolver {
	return func(name, _ string) ([]string, error) {
		return DefaultDependenciesFor(name), nil
	}
}
