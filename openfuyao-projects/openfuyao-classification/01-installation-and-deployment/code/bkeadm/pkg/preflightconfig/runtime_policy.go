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

package preflightconfig

import "fmt"

// ValidateRuntimeCompleteness prevents an external policy from removing
// release-defined check families. Values and presentation may change, and
// customChecks may be added, but every fixed policy key remains explicit.
func ValidateRuntimeCompleteness(release, runtime Config) error {
	if err := validateOperationCompleteness(
		"init", release.Operations.Init, runtime.Operations.Init,
	); err != nil {
		return err
	}
	return validateOperationCompleteness(
		"clusterCreate", release.Operations.ClusterCreate, runtime.Operations.ClusterCreate,
	)
}

func validateOperationCompleteness(name string, release, runtime OperationPolicy) error {
	if len(release.Platform.Distributions) != 0 && len(runtime.Platform.Distributions) == 0 {
		return fmt.Errorf("operations.%s.platform.distributions must not be empty", name)
	}
	checks := []error{
		validateRequiredKeys("operations."+name+".ports", release.Ports, runtime.Ports,
			func(item PortCheck) string { return item.Key }),
		validateRequiredKeys("operations."+name+".resources", release.Resources, runtime.Resources,
			func(item ResourceCheck) string { return item.Key }),
		validateRequiredKeys("operations."+name+".paths", release.Paths, runtime.Paths,
			func(item PathCheck) string { return item.Key }),
		validateRequiredKeys("operations."+name+".tools", release.Tools, runtime.Tools,
			func(item ToolCheck) string { return item.Key }),
	}
	for _, err := range checks {
		if err != nil {
			return err
		}
	}
	if err := validatePathRequirements(name, release.Paths, runtime.Paths); err != nil {
		return err
	}
	if err := validateTopologyCompleteness(name, release.Topology, runtime.Topology); err != nil {
		return err
	}
	return validateManagementCompleteness(name, release.ManagementCluster, runtime.ManagementCluster)
}

func validateRequiredKeys[T any](path string, release, runtime []T, key func(T) string) error {
	actual := make(map[string]struct{}, len(runtime))
	for _, item := range runtime {
		actual[key(item)] = struct{}{}
	}
	for _, item := range release {
		required := key(item)
		if _, exists := actual[required]; !exists {
			return fmt.Errorf("%s is missing release key %q", path, required)
		}
	}
	return nil
}

func validatePathRequirements(operation string, release, runtime []PathCheck) error {
	actual := make(map[string][]string, len(runtime))
	for _, item := range runtime {
		actual[item.Key] = item.Require
	}
	for _, item := range release {
		if missing := firstMissingValue(item.Require, actual[item.Key]); missing != "" {
			return fmt.Errorf(
				"operations.%s.paths[%s].require is missing release value %q",
				operation,
				item.Key,
				missing,
			)
		}
	}
	return nil
}

func validateTopologyCompleteness(operation string, release, runtime TopologyPolicy) error {
	if missing := firstMissingValue(release.Unique, runtime.Unique); missing != "" {
		return fmt.Errorf("operations.%s.topology.unique is missing release value %q", operation, missing)
	}
	if release.Masters.Odd && !runtime.Masters.Odd {
		return fmt.Errorf("operations.%s.topology.masters.odd must remain enabled", operation)
	}
	return nil
}

func validateManagementCompleteness(
	operation string,
	release, runtime ManagementClusterPolicy,
) error {
	if release.API && !runtime.API {
		return fmt.Errorf("operations.%s.managementCluster.api must remain enabled", operation)
	}
	if release.Discovery && !runtime.Discovery {
		return fmt.Errorf("operations.%s.managementCluster.discovery must remain enabled", operation)
	}
	if missing := firstMissingValue(release.CRDs, runtime.CRDs); missing != "" {
		return fmt.Errorf("operations.%s.managementCluster.crds is missing release value %q", operation, missing)
	}
	if release.RequiredKubeconfig != "" && runtime.RequiredKubeconfig == "" {
		return fmt.Errorf("operations.%s.managementCluster.requiredKubeconfig is required", operation)
	}
	return nil
}

func firstMissingValue(required, actual []string) string {
	values := make(map[string]struct{}, len(actual))
	for _, value := range actual {
		values[value] = struct{}{}
	}
	for _, value := range required {
		if _, exists := values[value]; !exists {
			return value
		}
	}
	return ""
}
