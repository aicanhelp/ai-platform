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

package helm

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/kube"
	"helm.sh/helm/v3/pkg/release"
	"helm.sh/helm/v3/pkg/repo"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/cli-utils/pkg/kstatus/status"

	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/zlog"
)

// ConvertToRelease convert release info to restful response
func ConvertToRelease(cfg *action.Configuration, release *release.Release, showResource bool) *helm.ReleaseResponse {
	var (
		info = &helm.Info{}
		err  error
	)

	if release.Info != nil {
		info = &helm.Info{
			Description:   release.Info.Description,
			Status:        string(release.Info.Status),
			Notes:         release.Info.Notes,
			FirstDeployed: release.Info.FirstDeployed,
			LastDeployed:  release.Info.LastDeployed,
			Deleted:       release.Info.Deleted,
		}
	}

	chart := ConvertChart(release, info, showResource)
	releaseModel := &helm.ReleaseResponse{
		Name:      release.Name,
		Info:      info,
		Chart:     chart,
		Values:    release.Config,
		Resources: nil,
		Version:   release.Version,
		Namespace: release.Namespace,
		Labels:    release.Labels,
	}
	if showResource {
		releaseModel.Resources, err = GetResourcesFromManifest(cfg, release.Manifest)
		if err != nil {
			zlog.Warnf("get resourcesFromManifest err %s", err)
		}
	}

	return releaseModel
}

// GetResourcesFromManifest get resource from manifest
func GetResourcesFromManifest(cfg *action.Configuration, manifest string) ([]helm.ReleaseResource, error) {
	if err := cfg.KubeClient.IsReachable(); err != nil {
		return nil, err
	}
	kubeClient, ok := cfg.KubeClient.(kube.InterfaceResources)
	if !ok {
		return nil, fmt.Errorf("unable to get kubeClient with interface InterfaceResources")
	}
	resources, err := cfg.KubeClient.Build(bytes.NewBufferString(manifest), false)
	if err != nil {
		return nil, err
	}
	resp, err := kubeClient.Get(resources, false)
	if err != nil {
		return nil, err
	}
	result := make([]helm.ReleaseResource, 0)
	for _, objs := range resp {
		for _, obj := range objs {
			accessor, err := meta.Accessor(obj)
			if err != nil {
				zlog.Warnf("meta.Accessor err %s", err)
				continue
			}
			result = append(result, getResourceInfo(accessor, obj))
		}
	}
	return result, nil
}

func getResourceInfo(accessor v1.Object, obj runtime.Object) helm.ReleaseResource {
	r := helm.ReleaseResource{
		Name:      accessor.GetName(),
		Namespace: accessor.GetNamespace(),
		Uid:       string(accessor.GetUID()),
	}
	gvk := obj.GetObjectKind().GroupVersionKind()
	r.APIVersion, r.Kind = gvk.ToAPIVersionAndKind()
	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		zlog.Warnf("getResourcesFromManifest Converter To Unstructured err is %s", err)
	} else {
		r.Status, err = status.Compute(&unstructured.Unstructured{Object: unstructuredObj})
		if err != nil {
			zlog.Warnf("getResourcesFromManifest Compute err is %s", err)
		}
	}
	return r
}

// ConvertChart convert release to chart
func ConvertChart(release *release.Release, info *helm.Info, showResource bool) *helm.Chart {
	var chart = &helm.Chart{}
	if release.Chart == nil {
		return chart
	}
	chart = &helm.Chart{
		Values: release.Chart.Values,
	}
	if release.Chart.Metadata != nil {
		chart.Metadata = release.Chart.Metadata
	}
	if showResource {
		for _, f := range release.Chart.Files {
			if f == nil {
				continue
			}
			if readmes[strings.ToLower(f.Name)] {
				info.Readme = string(f.Data)
			}
		}
	}
	return chart
}

// ConvertToListInterface convert any type of slice to slice of any type
// i.e. []double -> []interface{}
func ConvertToListInterface(data interface{}) []interface{} {
	listValue := reflect.ValueOf(data)
	if listValue.Kind() != reflect.Slice {
		return nil
	}

	length := listValue.Len()
	interfaceList := make([]interface{}, length)
	for i := 0; i < length; i++ {
		interfaceList[i] = listValue.Index(i).Interface()
	}
	return interfaceList
}

// ConvertToReleaseHistory convert release to release history
func ConvertToReleaseHistory(release *release.Release) *helm.ReleaseHistoryResponse {
	releaseHistory := &helm.ReleaseHistoryResponse{
		Name:      release.Name,
		Namespace: release.Namespace,
		Version:   release.Version,
		Labels:    release.Labels,
	}

	if release.Info != nil {
		releaseHistory.FirstDeployed = release.Info.FirstDeployed
		releaseHistory.LastDeployed = release.Info.LastDeployed
		releaseHistory.Status = string(release.Info.Status)
	}

	if release.Chart != nil && release.Chart.Metadata != nil {
		releaseHistory.ChartName = release.Chart.Metadata.Name
		releaseHistory.ChartVersion = release.Chart.Metadata.Version
		releaseHistory.AppVersion = release.Chart.Metadata.AppVersion
		releaseHistory.APIVersion = release.Chart.Metadata.APIVersion
	}

	return releaseHistory
}

// GetMinInt get minimum integer
func GetMinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Paging  does paging action
func Paging(temp []*release.Release, pagination *param.Pagination) ([]*release.Release, int, int) {
	total := len(temp)
	if pagination == nil {
		pagination = param.GetNoPagination()
	}
	start, end, totalPages := pagination.GetPaginationResult(total)
	temp = temp[start:end]
	return temp, totalPages, total
}

// ParseQueryInt parse string to integer
func ParseQueryInt(param string) int {
	val, err := strconv.Atoi(param)
	if err != nil {
		zlog.Warn("Failed to strconv Atoi: %s", err)
		return 0
	}
	return val
}

// ExtractUniqueKeywords put all chartversion's keywords into one latest version
func ExtractUniqueKeywords(chartVersions repo.ChartVersions) []string {
	unique := make(map[string]bool)

	for _, chartVersion := range chartVersions {
		for _, keyword := range chartVersion.Keywords {
			unique[keyword] = true
		}
	}

	allKeywords := make([]string, 0, len(unique))
	for key := range unique {
		allKeywords = append(allKeywords, key)
	}

	return allKeywords
}
