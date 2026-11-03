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

package request

import (
	"context"
	"net/http"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes"

	"console-service/pkg/constant"
)

const (
	resourcePathMinLength = 3
	clusterNameIndex      = 1
	clusterPathOffset     = 2
)

const requestInfoKey requestInfoKeyType = iota

// RequestInfoResolver returns new RequestInfo
type RequestInfoResolver interface {
	NewRequestInfo(req *http.Request, k8sClient kubernetes.Interface) (*RequestInfo, error)
}

var k8sAPIPrefixes = sets.New("api", "apis")

// RequestInfo contains detailed information extracted from an http.Request.
// It extends the functionality provied in k8s.io/apiserver/pkg/endpoints/request/requestinfo.go
// by including additional fields and methods specific to our application's requirements.
type RequestInfo struct {
	*k8srequest.RequestInfo

	IsK8sRequest  bool
	ResourceScope string

	ClusterName        string
	ClusterProxyScheme string
	ClusterProxyHost   string
	ClusterProxyURL    string
}

// SetSingleCluster set requestInfo to single cluster
func (r *RequestInfo) SetSingleCluster() {
	r.ClusterName = ""
	r.ClusterProxyHost = ""
	r.ClusterProxyURL = ""
	r.ClusterProxyScheme = ""
}

// RequestInfoFactory request info factory
type RequestInfoFactory struct {
	*k8srequest.RequestInfoFactory

	GlobalResources []schema.GroupResource
}

// NewRequestInfo returns the information from http request
func (r *RequestInfoFactory) NewRequestInfo(req *http.Request, k8sClient kubernetes.Interface) (*RequestInfo, error) {
	// 到这一步 /clusters/{cluster}/api/kubernetes 或者 /clusters/{cluster}/rest 两种前缀
	requestInfo := RequestInfo{
		RequestInfo: &k8srequest.RequestInfo{
			Path: req.URL.Path,
			Verb: req.Method,
		},
		IsK8sRequest: false,
		ClusterName:  "",
	}

	k8sFactory := k8srequest.RequestInfoFactory{
		APIPrefixes:          r.APIPrefixes,
		GrouplessAPIPrefixes: r.GrouplessAPIPrefixes,
	}

	defer setIsK8sRequest(&requestInfo)

	// the pathname where the /clusters/{cluster} part is extracted
	clusterName, pathname, ok := r.extractCluster(req.URL.Path)
	if ok {
		// /cluster/{cluster} part exists, then update Cluster field and path
		requestInfo.ClusterName = clusterName
		req.URL.Path = pathname
		r.setClusterProxyURL(&requestInfo, k8sClient)
	}
	if !r.checkK8sRequest(req) {
		return &requestInfo, nil
	}

	k8sInfo, err := k8sFactory.NewRequestInfo(req)
	requestInfo.RequestInfo = k8sInfo
	requestInfo.ResourceScope = r.resolveResourceScope(requestInfo)

	return &requestInfo, err
}

func setIsK8sRequest(ri *RequestInfo) {
	prefix := ri.APIPrefix
	if prefix == "" {
		currentParts := splitPath(ri.Path)
		if len(currentParts) > 0 {
			prefix = currentParts[0]
		}
	}
	if k8sAPIPrefixes.Has(prefix) {
		ri.IsK8sRequest = true
	}
}

type requestInfoKeyType int

// RequestInfoFrom returns the value of the RequestInfo key on the ctx
func RequestInfoFrom(ctx context.Context) (*RequestInfo, bool) {
	info, exist := ctx.Value(requestInfoKey).(*RequestInfo)
	return info, exist
}

// WithRequestInfo returns a copy of parent in which the request info value is set,
func WithRequestInfo(parent context.Context, info *RequestInfo) context.Context {
	return k8srequest.WithValue(parent, requestInfoKey, info)
}

// splitPath segments the url Path
func splitPath(uriPath string) []string {
	uriPath = strings.Trim(uriPath, "/")
	if uriPath == "" {
		return []string{}
	}
	return strings.Split(uriPath, "/")
}

func (r *RequestInfoFactory) extractCluster(urlPath string) (string, string, bool) {
	clusterName := ""
	currentParts := splitPath(urlPath)

	// [TO DO]: what is actually the MinLength
	if len(currentParts) < resourcePathMinLength {
		// the request is non-resource
		return clusterName, urlPath, false
	}

	if currentParts[0] != "clusters" {
		// the request doesn't contain cluster
		return "", urlPath, false
	}

	if len(currentParts) > resourcePathMinLength-clusterPathOffset {
		clusterName = currentParts[clusterNameIndex]
	}
	if len(currentParts) > resourcePathMinLength-clusterNameIndex {
		currentParts = currentParts[clusterPathOffset:]
	}
	dispatchedPathname := "/" + strings.Join(currentParts, "/")
	return clusterName, dispatchedPathname, true
}

func (r *RequestInfoFactory) setClusterProxyURL(info *RequestInfo, k8sClient kubernetes.Interface) {
	// check whether karmad-apiserver service exists
	info.ClusterProxyHost = ""
	info.ClusterProxyURL = ""
	info.ClusterProxyScheme = ""

	if k8sClient == nil {
		return
	}

	_, err := k8sClient.CoreV1().Services(constant.KaramdaNamespace).Get(context.TODO(),
		constant.KaramdaAPIServer, metav1.GetOptions{})
	if err != nil {
		return
	}

	info.ClusterProxyHost = constant.MultiClusterProxyHost
	info.ClusterProxyURL = strings.Replace(constant.MultiClusterProxyURL, "{cluster}", info.ClusterName, 1)
	info.ClusterProxyScheme = constant.MultiClusterProxyScheme

	return
}

func (r *RequestInfoFactory) checkK8sRequest(req *http.Request) bool {
	k8sPrefix := "/api/kubernetes"

	if strings.HasPrefix(req.URL.Path, k8sPrefix) {
		req.URL.Path = strings.TrimPrefix(req.URL.Path, k8sPrefix)
		return true
	}

	return false
}

const (
	globalScope    = "Global"
	clusterScope   = "Cluster"
	namespaceScope = "Namespace"
)

func (r *RequestInfoFactory) resolveResourceScope(info RequestInfo) string {
	for _, gResource := range r.GlobalResources {
		if gResource.Group == info.APIGroup && gResource.Resource == info.Resource {
			return globalScope
		}
	}

	if info.Namespace != "" {
		return namespaceScope
	}

	return clusterScope
}
