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

package v1

import (
	"fmt"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/pkg/errors"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/utils"
)

const (
	timeLayout  = "2006-01-02 15:04:05"
	defaultStep = 15 * time.Second
)

const (
	errTimeParamConflict = "'time' and the combination of 'start' and 'end' are mutually exclusive"
	errTimeParamMissing  = "missing 'start' or 'end'"
	errInvalidStartEnd   = "'start' must be before 'end'"
	errUnnecessaryParam  = "'step' parameter is unnecessary for instant queries"
	errTimeParamParsing  = "fail to parse time"
)

const (
	errKubeApiServerParamMissing               = "missing 'verb' or 'resource' for the `kube-apiserver` component"
	errKubeApiServerParamConflict              = "`instance` is not required for `kube-apiserver` component"
	errRestControlPlaneComponentsParamConflict = "`verb` or `resource` is not required"
	errRestControlPlaneComponentsParamMissing  = "Missing `instance`"
)

const (
	defaultFilter = ".*"
)

type reqParams struct {
	time            string
	start           string
	end             string
	step            string
	expression      string
	metricFilter    string
	resourceFilter  string
	nodeName        string
	namespaceName   string
	deploymentName  string
	statefulsetName string
	daemonsetName   string
	podName         string
	containerName   string
	component       string
	verb            string
	resource        string
	instance        string
}

type queryOptions struct {
	start        time.Time
	end          time.Time
	time         time.Time
	step         time.Duration
	metricFilter string
	namedMetrics []string
	option       client.QueryOption
}

func parseRequestParamsResources(req *restful.Request) reqParams {
	var r reqParams
	r.time = utils.EscapeSpecialChars(req.QueryParameter("time"))
	r.start = utils.EscapeSpecialChars(req.QueryParameter("start"))
	r.end = utils.EscapeSpecialChars(req.QueryParameter("end"))
	r.step = utils.EscapeSpecialChars(req.QueryParameter("step"))
	r.expression = utils.EscapeSpecialChars(req.QueryParameter("expr"))
	r.metricFilter = utils.EscapeSpecialChars(getDefaultParam(req, "metrics_filter", defaultFilter))
	r.resourceFilter = utils.EscapeSpecialChars(getDefaultParam(req, "resources_filter", defaultFilter))
	r.nodeName = utils.EscapeSpecialChars(req.PathParameter("node"))
	r.namespaceName = utils.EscapeSpecialChars(req.PathParameter("namespace"))
	r.deploymentName = utils.EscapeSpecialChars(req.PathParameter("deployment"))
	r.statefulsetName = utils.EscapeSpecialChars(req.PathParameter("statefulset"))
	r.daemonsetName = utils.EscapeSpecialChars(req.PathParameter("daemonset"))
	r.podName = utils.EscapeSpecialChars(req.PathParameter("pod"))
	r.containerName = utils.EscapeSpecialChars(req.PathParameter("container"))

	return r
}

func parseRequestParamsComponents(req *restful.Request) reqParams {
	var r reqParams
	r.time = utils.EscapeSpecialChars(req.QueryParameter("time"))
	r.start = utils.EscapeSpecialChars(req.QueryParameter("start"))
	r.end = utils.EscapeSpecialChars(req.QueryParameter("end"))
	r.step = utils.EscapeSpecialChars(req.QueryParameter("step"))
	r.metricFilter = utils.EscapeSpecialChars(getDefaultParam(req, "metrics_filter", defaultFilter))
	r.component = utils.EscapeSpecialChars(req.PathParameter("component"))
	r.verb = utils.EscapeSpecialChars(req.QueryParameter("verb"))
	r.resource = utils.EscapeSpecialChars(req.QueryParameter("resource"))
	r.instance = utils.EscapeSpecialChars(req.QueryParameter("instance"))
	return r
}

func transformQueryOptionsResources(reqParams reqParams, lvl client.Level) (queryOpts queryOptions,
	err error) {
	queryOpts.metricFilter = reqParams.metricFilter

	switch lvl {
	case client.LevelQuery:
		break

	case client.LevelCluster:
		queryOpts.option = client.WithClusterLevel()
		queryOpts.namedMetrics = clusterMetrics

	case client.LevelNode:
		queryOpts.option = client.WithNodeName(reqParams.nodeName)
		queryOpts.namedMetrics = nodeMetrics

	case client.LevelDeployment:
		queryOpts.option = client.WithDeploymentDetails(reqParams.resourceFilter, reqParams.namespaceName,
			reqParams.deploymentName)
		queryOpts.namedMetrics = workloadMetrics

	case client.LevelStatefulset:
		queryOpts.option = client.WithStatefulsetDetails(reqParams.resourceFilter, reqParams.namespaceName,
			reqParams.statefulsetName)
		queryOpts.namedMetrics = workloadMetrics

	case client.LevelDaemonset:
		queryOpts.option = client.WithDaemonsetDetails(reqParams.resourceFilter, reqParams.namespaceName,
			reqParams.daemonsetName)
		queryOpts.namedMetrics = workloadMetrics

	case client.LevelPod:
		queryOpts.option = client.WithPodDetails(reqParams.namespaceName, reqParams.podName)
		queryOpts.namedMetrics = podMetrics

	case client.LevelContainer:
		queryOpts.option = client.WithContainerDetails(reqParams.namespaceName, reqParams.podName,
			reqParams.containerName)
		queryOpts.namedMetrics = containerMetrics

	default:
		return queryOpts, fmt.Errorf("unhandled level: %v", lvl)
	}

	queryOpts, err = transformTimeBasedQueryOptions(reqParams, queryOpts)
	if err != nil {
		return queryOpts, err
	}
	return queryOpts, nil
}

func transformQueryOptionsComponents(reqParams reqParams, lvl client.Level) (queryOpts queryOptions,
	err error) {
	queryOpts.metricFilter = reqParams.metricFilter

	switch lvl {
	case client.LevelControlPlaneComponents:
		queryOpts, err = setControlPlaneComponentOptions(reqParams, queryOpts)

	case client.LevelNodeComponents:
		queryOpts, err = setNodeComponentOptions(reqParams, queryOpts)

	case client.LevelAddonsComponents:
		queryOpts, err = setAddonsComponentOptions(reqParams, queryOpts)

	default:
		err = fmt.Errorf("unhandled kubernetes components: %v", lvl)
	}

	if err != nil {
		return queryOpts, err
	}

	queryOpts, err = transformVerbResourceInstanceQueryOptions(reqParams, queryOpts)
	if err != nil {
		return queryOpts, err
	}

	queryOpts, err = transformTimeBasedQueryOptions(reqParams, queryOpts)
	if err != nil {
		return queryOpts, err
	}
	return queryOpts, nil
}

func setControlPlaneComponentOptions(reqParams reqParams, queryOpts queryOptions) (queryOptions, error) {
	switch reqParams.component {
	case client.KubeAPIServer:
		queryOpts.option = client.WithControlPlaneComponentsKubeApiServerDetails(reqParams.component,
			reqParams.verb, reqParams.resource)
		queryOpts.namedMetrics = controlPlaneComponentsKubeApiServerMetrics

	case client.ETCD:
		queryOpts.option = client.WithControlPlaneComponentsETCD(reqParams.component, reqParams.instance)
		queryOpts.namedMetrics = controlPlaneComponentsETCDMetrics

	case client.KubeScheduler:
		queryOpts.option = client.WithControlPlaneComponentsKubeScheduler(reqParams.component,
			reqParams.instance)
		queryOpts.namedMetrics = controlPlaneComponentsKubeSchedulerMetrics

	case client.KubeControllerManager:
		queryOpts.option = client.WithControlPlaneComponentsKubeControllerManager(reqParams.component,
			reqParams.instance)
		queryOpts.namedMetrics = controlPlaneComponentsKubeControllerManagerMetrics

	default:
		return queryOpts, fmt.Errorf("unhandled control plane component: %s", reqParams.component)
	}
	return queryOpts, nil
}

func setNodeComponentOptions(reqParams reqParams, queryOpts queryOptions) (queryOptions, error) {
	switch reqParams.component {
	case client.Kubelet:
		queryOpts.option = client.WithNodeComponentKubelet(reqParams.component, reqParams.instance)
		queryOpts.namedMetrics = nodeComponentsKubeletMetrics

	case client.KubeProxy:
		queryOpts.option = client.WithNodeComponentKubeProxy(reqParams.component, reqParams.instance)
		queryOpts.namedMetrics = nodeComponentsKubeProxyMetrics

	default:
		return queryOpts, fmt.Errorf("unhandled node component: %s", reqParams.component)
	}
	return queryOpts, nil
}

func setAddonsComponentOptions(reqParams reqParams, queryOpts queryOptions) (queryOptions, error) {
	switch reqParams.component {
	case client.CoreDNS:
		queryOpts.option = client.WithAddonsComponentCoredns(reqParams.component, reqParams.instance)
		queryOpts.namedMetrics = addonsComponentsCorednsMetrics
	default:
		return queryOpts, fmt.Errorf("unhandled addons component: %s", reqParams.component)
	}
	return queryOpts, nil
}

func transformVerbResourceInstanceQueryOptions(reqParams reqParams, queryOpts queryOptions) (queryOptions, error) {
	switch reqParams.component {
	case client.KubeAPIServer:
		if reqParams.verb == "" || reqParams.resource == "" {
			return queryOpts, errors.New(errKubeApiServerParamMissing)
		} else if reqParams.verb != "" && reqParams.resource != "" && reqParams.instance != "" {
			return queryOpts, errors.New(errKubeApiServerParamConflict)
		}

	default:
		if reqParams.verb != "" || reqParams.resource != "" {
			return queryOpts, errors.New(errRestControlPlaneComponentsParamConflict)
		} else if reqParams.instance == "" {
			return queryOpts, errors.New(errRestControlPlaneComponentsParamMissing)
		}
	}

	return queryOpts, nil
}

func transformTimeBasedQueryOptions(reqParams reqParams, queryOpts queryOptions) (queryOptions, error) {
	var err error

	if reqParams.start != "" && reqParams.end != "" && reqParams.time == "" {
		queryOpts.start, err = time.ParseInLocation(timeLayout, reqParams.start, time.Local)
		if err != nil {
			return queryOpts, errors.New(errTimeParamParsing)
		}
		queryOpts.end, err = time.ParseInLocation(timeLayout, reqParams.end, time.Local)
		if err != nil {
			return queryOpts, errors.New(errTimeParamParsing)
		}
		if !queryOpts.start.Before(queryOpts.end) {
			return queryOpts, errors.New(errInvalidStartEnd)
		}
		queryOpts.step = defaultStep
		if reqParams.step != "" {
			queryOpts.step, err = time.ParseDuration(reqParams.step)
			if err != nil {
				return queryOpts, errors.New(errTimeParamParsing)
			}
		}
		return queryOpts, nil
	} else if reqParams.start == "" && reqParams.end == "" {
		queryOpts.time = time.Now()
		if reqParams.time != "" {
			queryOpts.time, err = time.ParseInLocation(timeLayout, reqParams.time, time.Local)
			if err != nil {
				return queryOpts, errors.New(errTimeParamParsing)
			}
		}
		if reqParams.step != "" {
			return queryOpts, errors.New(errUnnecessaryParam)
		}
		return queryOpts, nil
	} else if reqParams.start != "" && reqParams.end != "" && reqParams.time != "" {
		return queryOpts, errors.New(errTimeParamConflict)
	} else {
		return queryOpts, errors.New(errTimeParamMissing)
	}
}

func getDefaultParam(req *restful.Request, paramName, defaultValue string) string {
	value := req.QueryParameter(paramName)
	if value == "" {
		return defaultValue
	}
	return value
}

func (q queryOptions) isRangeQuery() bool {
	return q.time.IsZero()
}
