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

package healthcheck

import (
	"fmt"
	"time"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

// FromCRD converts nested CRD HealthCheckSpec to the flat runtime Spec.
// Custom checks become `/bin/sh -c <command>` per API contract.
func FromCRD(in *cvv1alpha1.HealthCheckSpec) (Spec, error) {
	if in == nil {
		return Spec{}, nil
	}
	out := Spec{Enabled: in.Enabled}
	if !in.Enabled {
		return out, nil
	}

	if in.Timeout != "" {
		d, err := time.ParseDuration(in.Timeout)
		if err != nil {
			return Spec{}, fmt.Errorf("parse healthCheck.timeout: %w", err)
		}
		out.Timeout = d
	}
	if in.Interval != "" {
		d, err := time.ParseDuration(in.Interval)
		if err != nil {
			return Spec{}, fmt.Errorf("parse healthCheck.interval: %w", err)
		}
		out.Interval = d
	}

	for i, item := range in.Checks {
		check, err := fromCRDCheck(item)
		if err != nil {
			return Spec{}, fmt.Errorf("healthCheck.checks[%d]: %w", i, err)
		}
		out.Checks = append(out.Checks, check)
	}
	return out, nil
}

func fromCRDCheck(item cvv1alpha1.HealthCheckItemSpec) (Check, error) {
	switch item.Type {
	case string(CheckTypePodReady):
		if item.PodReady == nil {
			return Check{}, fmt.Errorf("podReady config is required")
		}
		return Check{
			Type:          CheckTypePodReady,
			Namespace:     item.PodReady.Namespace,
			LabelSelector: item.PodReady.LabelSelector,
			MinReady:      int(item.PodReady.MinReady),
		}, nil
	case string(CheckTypeEndpointReady):
		if item.EndpointReady == nil {
			return Check{}, fmt.Errorf("endpointReady config is required")
		}
		return Check{
			Type:        CheckTypeEndpointReady,
			Namespace:   item.EndpointReady.Namespace,
			ServiceName: item.EndpointReady.ServiceName,
			Port:        int(item.EndpointReady.Port),
		}, nil
	case string(CheckTypeCustom):
		if item.Custom == nil {
			return Check{}, fmt.Errorf("custom config is required")
		}
		if item.Custom.Command == "" {
			return Check{}, fmt.Errorf("custom.command is required")
		}
		return Check{
			Type:    CheckTypeCustom,
			Command: []string{"/bin/sh", "-c", item.Custom.Command},
		}, nil
	default:
		return Check{}, fmt.Errorf("unsupported health check type %q", item.Type)
	}
}
