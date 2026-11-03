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

// Package helm
package helm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/kube"
	"helm.sh/helm/v3/pkg/release"
	"helm.sh/helm/v3/pkg/releaseutil"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/param"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/utils/util"
	"application-management-service/pkg/zlog"
)

const (
	defaultHistoryNums       = 255
	defaultRollbackHisNums   = 10
	maxReleaseNameLength     = 53
	uninstallReleaseNotFound = "uninstall: Release not loaded: %s: release: not found"
	releaseNotFound          = "release: not found"
)

// workloads which contains pod log
var resourceLogMap = map[string]bool{
	"Deployment":  true,
	"Job":         true,
	"StatefulSet": true,
	"DaemonSet":   true,
	"CronJob":     true,
	"Pod":         true,
}

func (c *helmClient) GetReleases(releaseParameter param.ReleaseParameter) (*httputil.ResponseJson, int) {

	actionConfig, err := getActionConfig(releaseParameter.Namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	// validate the length of release name
	if utf8.RuneCountInString(releaseParameter.Release) >= maxReleaseNameLength {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "release name is too long",
			Data: nil,
		}, http.StatusBadRequest
	}

	releases, err := listReleases(actionConfig, releaseParameter).Run()
	if err != nil {
		zlog.Errorf("failed to list releases: %s", err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "failed to list releases: ",
				Data: releases},
			http.StatusInternalServerError
	}
	// extension apps are managed separately
	filteredReleases := deleteReleasesValue(filterReleaseByName(filterExtensions(releaseParameter.Extension, releases),
		releaseParameter.Release))

	return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  "success",
			Data: convertResponseResult(filteredReleases)},
		http.StatusOK
}

func (c *helmClient) Install(deploy helm.DeployParam) (*httputil.ResponseJson, int) {
	if response := validateDeployParam(deploy); response != nil {
		return response, http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(deploy.Namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	if utf8.RuneCountInString(deploy.Release) >= maxReleaseNameLength {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "release name is too long",
			Data: nil,
		}, http.StatusBadRequest
	}
	// 检查是否安装过该release
	exists, err := c.ReleaseExists(actionConfig, deploy.Release, deploy.Namespace)
	// [TO DO] errors.isAlready exist?
	if err != nil && err.Error() != releaseNotFound {
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  err.Error(),
				Data: nil,
			},
			http.StatusInternalServerError
	}

	if exists {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "release already exist",
		}, http.StatusBadRequest
	}

	install, err := c.createInstall(actionConfig, deploy)
	if err != nil {
		zlog.Errorf("failed to create install action: %s", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	chartInfo, values, err := c.getChartInfo(deploy)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	_, err = install.Run(chartInfo, values)
	if err != nil {
		zlog.Errorf("install failed : %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  err.Error(),
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "helm chart installing",
	}, http.StatusAccepted
}

func (c *helmClient) Upgrade(deploy helm.DeployParam) (*httputil.ResponseJson, int) {
	actionConfig, err := getActionConfig(deploy.Namespace, c.kubeConfig)
	if err != nil {
		zlog.Errorf("failed to initialize helm action config: %s", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	upgrade, err := c.createUpgrade(actionConfig, deploy)
	if err != nil {
		zlog.Errorf("failed to create upgrade action: %s", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	chartInfo, values, err := c.getChartInfo(deploy)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	_, err = upgrade.Run(deploy.Release, chartInfo, values)
	if err != nil {
		zlog.Errorf("Upgrade failed : %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "An exception occurred during the upgrading",
		}, http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "helm chart upgrading",
	}, http.StatusAccepted
}

func (c *helmClient) Uninstall(namespace, releaseName string) (*httputil.ResponseJson, int) {
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	uninstall := action.NewUninstall(actionConfig)
	_, err = uninstall.Run(releaseName)
	if err != nil {
		if fmt.Sprintf(uninstallReleaseNotFound, releaseName) == err.Error() {
			return &httputil.ResponseJson{
				Code: constant.ResourceNotFound,
				Msg:  "release not fount: ",
				Data: nil,
			}, http.StatusBadRequest
		}
		zlog.Errorf("run command failed, error: %v", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to uninstall release: ",
			Data: nil,
		}, http.StatusInternalServerError
	}
	zlog.Infof("namespace: %s, name: %s, run command success", namespace, releaseName)
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}

func getHelmRelease(actionConfig *action.Configuration, releaseName, namespace string) (*release.Release, error) {
	helmStatus := action.NewStatus(actionConfig)
	helmRelease, err := helmStatus.Run(releaseName)
	if err != nil {
		// release not found is expected behavior and should not report as error
		if err.Error() != releaseNotFound {
			zlog.Errorf("namespace: %s, name: %s, run command failed, error: %v", namespace, releaseName, err)
		}
		return nil, err
	}
	zlog.Infof("namespace: %s, name: %s, run command success", namespace, releaseName)
	return helmRelease, nil
}

func (c *helmClient) GetReleaseInfo(namespace, releaseName string) (*httputil.ResponseJson, int) {
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	helmRelease, err := getHelmRelease(actionConfig, releaseName, namespace)
	if err != nil {
		zlog.Errorf("failed to get release info : %s", err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "failed to get release info: ",
				Data: helmRelease},
			http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: ConvertToRelease(actionConfig, helmRelease, true),
	}, http.StatusOK
}

// GetReleaseHistory 获取应用的历史记录信息
func (c *helmClient) GetReleaseHistory(namespace, releaseName string, max int) (*httputil.ResponseJson, int) {
	zlog.Debugf("GetReleaseHistory release namespace is %s, release name is %s", namespace,
		releaseName)
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}

	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	releases, err := getReleaseHistory(actionConfig, releaseName)
	if err != nil {
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "failed to initialize NewHistory action:  ",
				Data: releases},
			http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: convertToReleaseHis(getMaxHistories(max, releases)),
	}, http.StatusOK
}

// RollbackRelease 回滚应用到指定的版本，如果不指定则回滚到上一个版本
func (c *helmClient) RollbackRelease(namespace,
	releaseName, chartName string,
	version, maxHistory int) (*httputil.ResponseJson, int) {
	zlog.Debugf("GetReleaseInfo  release namespace is %s, release name is %s", namespace,
		releaseName)
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	err = c.getRollBack(actionConfig, version, maxHistory, chartName).Run(releaseName)
	if err != nil {
		zlog.Errorf("failed to run rollback action: %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to run rollback action: ",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}

func (c *helmClient) getRollBack(actionConfig *action.Configuration, version int, maxHistory int,
	chartName string) *action.Rollback {
	rollback := action.NewRollback(actionConfig)
	rollback.Version = version

	// 用户没有传此值的话，默认10个，防止存放太多，和更新保留数量保持一致
	if maxHistory > 0 {
		rollback.MaxHistory = maxHistory
	} else {
		rollback.MaxHistory = defaultRollbackHisNums
	}

	// add timeout and wait param
	amsConfig, err := util.GetApplicationManagementServiceConfig(c.clientset)
	if err != nil {
		return rollback
	}
	if util.Contains(amsConfig.HelmWaitCharts, chartName) {
		timeout, err := time.ParseDuration(amsConfig.HelmTimeout)
		if err != nil {
			return rollback
		}
		rollback.Timeout = timeout
		rollback.Wait = true
		rollback.WaitForJobs = true
	}

	return rollback
}

func (c *helmClient) GetManifest(namespace, releaseName string) (*httputil.ResponseJson, int) {
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	getManifest := action.NewGet(actionConfig)

	result, err := getManifest.Run(releaseName)
	if err != nil {
		zlog.Errorf("failed to get manifest: %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to get manifest: ",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: result.Manifest,
	}, http.StatusOK
}

func (c *helmClient) GetValues(namespace, releaseName string) (*httputil.ResponseJson, int) {
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	getValues := action.NewGetValues(actionConfig)
	getValues.AllValues = true
	result, err := getValues.Run(releaseName)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to get values: ",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: result,
	}, http.StatusOK
}

func (c *helmClient) GetReleasePods(namespace, releaseName string) (*httputil.ResponseJson, int) {
	if err := validateParams(releaseName, namespace); err != nil {
		return httputil.GetParamsEmptyErrorResponseJson(), http.StatusBadRequest
	}
	actionConfig, err := getActionConfig(namespace, c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	getManifest := action.NewGet(actionConfig)

	release, err := getManifest.Run(releaseName)
	if err != nil {
		zlog.Errorf("failed to get manifest while getting pods info: %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to get manifest while getting pods info: ",
			Data: nil,
		}, http.StatusInternalServerError
	}

	resources, err := actionConfig.KubeClient.Build(bytes.NewBuffer([]byte(release.Manifest)), false)
	if err != nil {
		zlog.Errorf("failed to get resources list %s", err)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "failed to get resources list: ",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: getPodsContainersInfo(resources, c),
	}, http.StatusOK
}

func getPodsContainersInfo(resources kube.ResourceList, c *helmClient) map[string][]string {
	result := make(map[string][]string)
	for _, resourceInfo := range resources {
		gvk := resourceInfo.Object.GetObjectKind().GroupVersionKind()
		namespace := resourceInfo.Namespace
		if _, ok := resourceLogMap[gvk.Kind]; ok {
			labelSelector := getLabelSelector(gvk, c, resourceInfo)
			getInfoFromSelector(namespace, labelSelector, c, resourceInfo, result)
		}
	}
	return result
}

func getInfoFromSelector(namespace string, labelSelector *v1.LabelSelector, c *helmClient,
	resourceInfo *resource.Info, result map[string][]string) {
	if labelSelector != nil {
		pods, err := c.clientset.CoreV1().Pods(namespace).List(context.TODO(), v1.ListOptions{
			LabelSelector: v1.FormatLabelSelector(labelSelector),
		})
		if err != nil {
			zlog.Errorf("failed to list pod info %v", err)
			return
		}
		for _, value := range pods.Items {
			var containerList []string
			for _, c := range value.Spec.Containers {
				containerList = append(containerList, c.Name)
			}
			for _, c := range value.Spec.InitContainers {
				containerList = append(containerList, c.Name)
			}
			if result != nil {
				result[value.Name] = containerList
			}
		}
	} else {
		pods, err := c.clientset.CoreV1().Pods(resourceInfo.Namespace).Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to list pod info %v", err)
			return
		}
		var containerList []string
		for _, c := range pods.Spec.Containers {
			containerList = append(containerList, c.Name)
		}
		for _, c := range pods.Spec.InitContainers {
			containerList = append(containerList, c.Name)
		}
		if result != nil {
			result[pods.Name] = containerList
		}
	}
}

func getLabelSelector(gvk schema.GroupVersionKind, c *helmClient, resourceInfo *resource.Info) *v1.LabelSelector {
	switch gvk.Kind {
	case "StatefulSet":
		statefulSet, err := c.clientset.AppsV1().
			StatefulSets(resourceInfo.Namespace).
			Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to get statefulset:  %s . %s", resourceInfo.Name, err)
			return nil
		}
		return statefulSet.Spec.Selector
	case "DaemonSet":
		daemonSet, err := c.clientset.AppsV1().
			DaemonSets(resourceInfo.Namespace).
			Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to get daemonset:  %s . %s", resourceInfo.Name, err)
			return nil
		}
		return daemonSet.Spec.Selector
	case "Deployment":
		deployment, err := c.clientset.AppsV1().
			Deployments(resourceInfo.Namespace).
			Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to get deployment:  %s . %s", resourceInfo.Name, err)
			return nil
		}
		return deployment.Spec.Selector
	case "Job":
		job, err := c.clientset.BatchV1().
			Jobs(resourceInfo.Namespace).
			Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to get job:  %s . %s", resourceInfo.Name, err)
			return nil
		}
		return job.Spec.Selector
	case "CronJob":
		cronJob, err := c.clientset.BatchV1beta1().
			CronJobs(resourceInfo.Namespace).
			Get(context.TODO(), resourceInfo.Name, v1.GetOptions{})
		if err != nil {
			zlog.Errorf("failed to get cronjob:  %s . %s", resourceInfo.Name, err)
			return nil
		}
		return cronJob.Spec.JobTemplate.Spec.Selector
	default:
		return nil
	}
}

func getActionConfig(namespace string, kubeConfig *rest.Config) (*action.Configuration, error) {
	restClientGetter := NewRESTClientConfig(namespace, kubeConfig)
	actionConfig := new(action.Configuration)

	if err := actionConfig.Init(restClientGetter, namespace, "", zlog.Infof); err != nil {
		zlog.Errorf("failed to initialize helm action config: %s", err)
		return nil, err
	}
	return actionConfig, nil
}

func (c *helmClient) ReleaseExists(actionConfig *action.Configuration, releaseName, namespace string) (bool, error) {
	// err == nil helm release exist
	// err != nil && error == "release: not found" helm release not exist
	// err != nil && error != "release: not found" something else goes wrong
	_, err := getHelmRelease(actionConfig, releaseName, namespace)
	if err != nil {
		return false, err
	} else {
		return true, nil
	}
}

func (c *helmClient) createInstall(actionConfig *action.Configuration,
	deploy helm.DeployParam) (*action.Install, error) {
	install := action.NewInstall(actionConfig)
	if install == nil {
		return install, fmt.Errorf("failed to prepare install action")
	}
	install.Namespace = deploy.Namespace
	install.ReleaseName = deploy.Release
	install.Version = deploy.Version
	install.Labels = map[string]string{
		"openfuyao.io.repo": deploy.RepoName,
	}
	// add extra label for multi-cluster-service chart
	if deploy.ChartName == constant.McsChartName {
		install.Labels["openfuyao.io/chart-name"] = constant.McsChartName
	}
	// add timeout and wait param
	amsConfig, err := util.GetApplicationManagementServiceConfig(c.clientset)
	if err != nil {
		return install, nil
	}
	if util.Contains(amsConfig.HelmWaitCharts, deploy.ChartName) {
		timeout, err := time.ParseDuration(amsConfig.HelmTimeout)
		if err != nil {
			return install, nil
		}
		install.Timeout = timeout
		install.Wait = true
		install.WaitForJobs = true
	}
	return install, nil
}

func (c *helmClient) createUpgrade(actionConfig *action.Configuration,
	deploy helm.DeployParam) (*action.Upgrade, error) {
	upgrade := action.NewUpgrade(actionConfig)
	if upgrade == nil {
		return upgrade, fmt.Errorf("failed to prepare upgrade action")
	}
	upgrade.Namespace = deploy.Namespace
	upgrade.Version = deploy.Version
	upgrade.Labels = map[string]string{
		"openfuyao.io.repo": deploy.RepoName,
	}
	// add extra label for multi-cluster-service chart
	if deploy.ChartName == constant.McsChartName {
		upgrade.Labels["openfuyao.io/chart-name"] = constant.McsChartName
	}
	// add timeout and wait param
	amsConfig, err := util.GetApplicationManagementServiceConfig(c.clientset)
	if err != nil {
		return upgrade, nil
	}
	if util.Contains(amsConfig.HelmWaitCharts, deploy.ChartName) {
		timeout, err := time.ParseDuration(amsConfig.HelmTimeout)
		if err != nil {
			return upgrade, nil
		}
		upgrade.Timeout = timeout
		upgrade.Wait = true
		upgrade.WaitForJobs = true
	}
	return upgrade, nil
}

func listReleases(actionConfig *action.Configuration, releaseParameter param.ReleaseParameter) *action.List {
	list := action.NewList(actionConfig)
	return list
}

func filterReleaseByName(releases []*release.Release, releaseName string) []*release.Release {
	if releaseName == "" {
		return releases
	}
	filtered := make([]*release.Release, 0)
	for _, r := range releases {
		if strings.Contains(r.Name, releaseName) {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func deleteReleasesValue(releases []*release.Release) []*release.Release {
	for _, r := range releases {
		deleteReleaseValue(r)
	}
	return releases
}

func deleteReleaseValue(r *release.Release) {
	r.Chart.Values = nil
	r.Config = nil
}

func convertResponseResult(releases []*release.Release) *helm.ListResponse {
	result := make([]interface{}, 0, len(releases))
	for _, r := range releases {
		result = append(result, ConvertToRelease(nil, r, false))
	}
	return &helm.ListResponse{
		Items: result,
	}
}

func filterExtensions(extension string, releases []*release.Release) []*release.Release {
	var filteredReleases []*release.Release
	isExtensionEnabled := extension == "true"
	for _, rel := range releases {
		hasFuyaoExtension := false
		hasFuyaoComponent := false
		// iterate the keyword set of chart to check whether the app is an extension app
		for _, keyword := range rel.Chart.Metadata.Keywords {
			if keyword == constant.FuyaoComponent {
				hasFuyaoComponent = true
				break
			}
			if keyword == constant.FuyaoExtensionKeyword {
				hasFuyaoExtension = true
				break
			}
		}
		// divide the apps into extension apps and business apps according to the flag
		isApplication := !isExtensionEnabled && !hasFuyaoComponent && !hasFuyaoExtension
		if (isExtensionEnabled && hasFuyaoExtension) || isApplication {
			filteredReleases = append(filteredReleases, rel)
		}
	}
	return filteredReleases
}

func convertToReleaseHis(releases []*release.Release) []*helm.ReleaseHistoryResponse {
	// 数据转换
	result := make([]*helm.ReleaseHistoryResponse, len(releases))
	for i, r := range releases {
		result[i] = ConvertToReleaseHistory(r)
	}
	return result
}

func getMaxHistories(max int, releases []*release.Release) []*release.Release {
	// maximum number of revision to include in history (default 256)
	// SDK 中没有提供 maxHistory 功能，参考 helm cli 逻辑补充
	maxHistory := defaultHistoryNums
	if max > 0 {
		maxHistory = max
	}
	limit := GetMinInt(len(releases), maxHistory)
	rels := releases[:limit]
	return rels
}

func getReleaseHistory(actionConfig *action.Configuration, releaseName string) ([]*release.Release, error) {
	releases, err := action.NewHistory(actionConfig).Run(releaseName)
	if err != nil {
		zlog.Errorf("failed to initialize NewHistory action: %s ", err)
		return nil, err
	}

	releaseutil.Reverse(releases, releaseutil.SortByRevision)
	return releases, nil
}

func (c *helmClient) getChartInfo(deploy helm.DeployParam) (*chart.Chart, chartutil.Values, error) {
	chartInfo, err := c.chartGetter.Get(deploy.RepoName, deploy.ChartName, deploy.Version)
	if err != nil {
		zlog.Errorf("get version err %s", err)
		return nil, nil, err
	}
	values, err := chartutil.ReadValues([]byte(deploy.Values))
	if err != nil {
		zlog.Errorf("read values err: %w", err)
		return nil, nil, err
	}
	return chartInfo, values, nil
}
func (c *helmClient) CheckReleaseStatus(chartName string) (*httputil.ResponseJson, int) {

	actionConfig, err := getActionConfig("", c.kubeConfig)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	list := action.NewList(actionConfig)
	releases, err := list.Run()
	if err != nil {
		zlog.Errorf("failed to list releases: %s", err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "failed to list releases: ",
				Data: releases},
			http.StatusInternalServerError
	}
	checkReleaseRuning(releases, chartName)
	return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  "success",
			Data: checkReleaseRuning(releases, chartName)},
		http.StatusOK
}
func checkReleaseRuning(releases []*release.Release, chartName string) string {
	for _, rel := range releases {
		if strings.Contains(rel.Chart.Name(), chartName) && rel.Info.Status == "deployed" {
			zlog.Infof("%s is deployed", chartName)
			return chartName + " running"
		}
	}
	zlog.Infof("%s is not ready", chartName)
	return chartName + " not ready"
}

func validateParams(name, namespace string) error {
	if namespace == "" || name == "" {
		err := errors.New("namespace or name is empty")
		return err
	}
	return nil
}

func validateDeployParam(deploy helm.DeployParam) *httputil.ResponseJson {
	if deploy.Namespace == "" || deploy.Release == "" {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide proper release name and namespace",
		}
	}
	if deploy.ChartName == "" || deploy.RepoName == "" || deploy.Version == "" {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide proper chart info",
		}
	}
	return nil
}
