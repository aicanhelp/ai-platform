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
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/collectors/artifact"
	"gopkg.openfuyao.cn/preflight/collectors/endpoint"
	"gopkg.openfuyao.cn/preflight/collectors/local"
	"gopkg.openfuyao.cn/preflight/collectors/router"
	sshcollector "gopkg.openfuyao.cn/preflight/collectors/ssh"
)

const (
	endpointProbeTimeout = 5 * time.Second
	sshProbeTimeout      = 20 * time.Second
)

// VerifiedPlatforms converts the human-authored platform policy into the
// collector's neutral compatibility rows. It deliberately keeps only public
// platform facts; evidence references are not part of the policy file.
func VerifiedPlatforms(config preflightconfig.Config) []api.VerifiedPlatform {
	seen := map[string]struct{}{}
	var out []api.VerifiedPlatform
	for _, policy := range []preflightconfig.OperationPolicy{config.Operations.Init, config.Operations.ClusterCreate} {
		for _, distribution := range policy.Platform.Distributions {
			if len(distribution.KernelPrefixes) == 0 {
				continue
			}
			key := distribution.Name + "\x00" + distribution.Version
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, api.VerifiedPlatform{
				Distribution:   distribution.Name,
				Version:        distribution.Version,
				Architectures:  append([]string(nil), distribution.Architectures...),
				KernelPrefixes: append([]string(nil), distribution.KernelPrefixes...),
			})
		}
	}
	return out
}

// NewRuntimeCollector composes the production read-only collectors used by
// the bke CLI. Remote SSH and management-cluster routes can be prepended by
// the cluster-create host after it constructs their scoped credentials.
func NewRuntimeCollector(extra ...router.Route) api.Collector {
	return newRuntimeCollector(endpoint.New(endpointProbeTimeout), nil, extra...)
}

// NewInitRuntimeCollector keeps repository credentials in the bkeadm host
// layer and passes them only to the short-lived endpoint collector.
func NewInitRuntimeCollector(options InitOptions, extra ...router.Route) api.Collector {
	return NewInitRuntimeCollectorWithPlatforms(options, nil, extra...)
}

// NewInitRuntimeCollectorWithPlatforms creates the init collector with release-verified platform metadata.
func NewInitRuntimeCollectorWithPlatforms(
	options InitOptions,
	platforms []api.VerifiedPlatform,
	extra ...router.Route,
) api.Collector {
	credentials := map[string]endpoint.Credential{}
	if options.ImageRepoUsername != "" && options.ImageRepoPassword != "" && options.OtherRepo != "" {
		if host := sourceHost(options.OtherRepo); host != "" {
			credentials[host] = endpoint.Credential{
				Username: options.ImageRepoUsername,
				Password: options.ImageRepoPassword,
			}
		}
	}
	return newRuntimeCollector(endpoint.NewWithCredentials(endpointProbeTimeout, credentials), platforms, extra...)
}

// NewInitRuntimeCollectorWithCustom adds validated init custom checks.
func NewInitRuntimeCollectorWithCustom(
	options InitOptions,
	platforms []api.VerifiedPlatform,
	policy preflightconfig.OperationPolicy,
	extra ...router.Route,
) (api.Collector, error) {
	route, err := customRoute(api.OperationInit, policy, nil)
	if err != nil {
		return nil, fmt.Errorf("create init custom collector: %w", err)
	}
	routes := append([]router.Route{route}, extra...)
	return NewInitRuntimeCollectorWithPlatforms(options, platforms, routes...), nil
}

func newRuntimeCollector(
	endpointCollector api.Collector,
	platforms []api.VerifiedPlatform,
	extra ...router.Route,
) api.Collector {
	artifactCollector := artifact.New()
	routes := append([]router.Route(nil), extra...)
	routes = append(routes,
		router.Route{Match: func(_ api.Subject, key string) bool {
			return strings.HasPrefix(key, "artifact./")
		}, Collector: artifactCollector},
		router.Route{Match: func(_ api.Subject, key string) bool {
			return strings.HasPrefix(key, "endpoint.")
		}, Collector: endpointCollector},
		router.Route{Match: func(subject api.Subject, _ string) bool {
			return subject.ID == "bootstrap" || subject.ID == "cluster"
		}, Collector: local.NewWithVerifiedPlatforms(local.DefaultProbe(), platforms)},
	)
	return router.New(routes...)
}

func sourceHost(raw string) string {
	value := strings.TrimSpace(raw)
	if parsed, err := url.Parse(value); err == nil && parsed.Host != "" {
		value = parsed.Host
	} else {
		value = strings.SplitN(value, "/", 2)[0]
	}
	if _, _, err := net.SplitHostPort(value); err == nil {
		return value
	}
	return net.JoinHostPort(strings.Trim(value, "[]"), "443")
}

// NewClusterRuntimeCollector adds the real SSH-node route while retaining the
// local and endpoint collectors. SSH host-key trust is intentionally not a
// preflight gate; this matches the existing OpenFuyao SSH installation path.
func NewClusterRuntimeCollector(nodes []configv1beta1.BKENode, extra ...router.Route) api.Collector {
	return NewClusterRuntimeCollectorWithPlatforms(nodes, nil, extra...)
}

// NewClusterRuntimeCollectorWithPlatforms creates the cluster collector with release-verified platform metadata.
func NewClusterRuntimeCollectorWithPlatforms(
	nodes []configv1beta1.BKENode,
	platforms []api.VerifiedPlatform,
	extra ...router.Route,
) api.Collector {
	nodeCollector := sshcollector.NewWithVerifiedPlatforms(newNodeSSHExecutor(nodes), nil, sshProbeTimeout, platforms)
	routes := append([]router.Route{{Match: func(subject api.Subject, _ string) bool {
		return subject.ID != "" && subject.ID != "cluster" && subject.ID != "bootstrap"
	}, Collector: nodeCollector}}, extra...)
	return newRuntimeCollector(endpoint.New(endpointProbeTimeout), platforms, routes...)
}

// NewClusterRuntimeCollectorWithCustom adds validated custom checks on each
// selected remote node.
func NewClusterRuntimeCollectorWithCustom(
	nodes []configv1beta1.BKENode,
	platforms []api.VerifiedPlatform,
	policy preflightconfig.OperationPolicy,
	extra ...router.Route,
) (api.Collector, error) {
	executor := newNodeSSHExecutor(nodes)
	nodeCollector := sshcollector.NewWithVerifiedPlatforms(executor, nil, sshProbeTimeout, platforms)
	custom, err := customRoute(api.OperationClusterCreate, policy, executor)
	if err != nil {
		return nil, fmt.Errorf("create cluster custom collector: %w", err)
	}
	routes := []router.Route{
		custom,
		{Match: func(subject api.Subject, _ string) bool {
			return subject.ID != "" && subject.ID != "cluster" && subject.ID != "bootstrap"
		}, Collector: nodeCollector},
	}
	routes = append(routes, extra...)
	return newRuntimeCollector(endpoint.New(endpointProbeTimeout), platforms, routes...), nil
}
