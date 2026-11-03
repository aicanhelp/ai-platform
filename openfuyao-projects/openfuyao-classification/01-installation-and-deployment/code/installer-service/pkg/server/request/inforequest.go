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

// Package request build request info
package request

import (
	"context"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
)

const (
	resourcePathMinLength = 3
	clusterNameIndex      = 1
	clusterPathOffset     = 2
)

// RequestInfoResolver returns new RequestInfo
type RequestInfoResolver interface {
	NewRequestInfo(req *http.Request) (*RequestInfo, error)
}

var k8sAPIPrefixes = sets.New("api", "apis")

// RequestInfo RequestInfo 包含从 http.Request 中提取的详细信息
type RequestInfo struct {
	*k8srequest.RequestInfo

	IsK8sRequest bool

	Cluster string

	ResourceScope string
}

// RequestInfoFactory request info factory
type RequestInfoFactory struct {
	*k8srequest.RequestInfoFactory

	GlobalResources []schema.GroupResource
}

// NewRequestInfo 返回来自 HTTP 请求的信息，返回的 RequestInfo
func (r *RequestInfoFactory) NewRequestInfo(req *http.Request) (*RequestInfo, error) {
	requestInfo := RequestInfo{
		IsK8sRequest: false,
		RequestInfo: &k8srequest.RequestInfo{
			Path: req.URL.Path,
			Verb: req.Method,
		},
		Cluster: "",
	}

	defer setIsK8sRequest(&requestInfo)

	k8sFactory := k8srequest.RequestInfoFactory{
		APIPrefixes:          r.APIPrefixes,
		GrouplessAPIPrefixes: r.GrouplessAPIPrefixes,
	}

	oriPathname := req.URL.Path

	// the pathname where the /cluster/{cluster} part is extracted
	clusterName, pathname, ok := r.extractCluster(k8sFactory, req.URL.Path)
	if ok {
		// /cluster/{cluster} part exists, then update Cluster field and path
		requestInfo.Cluster = clusterName
		req.URL.Path = pathname
	}

	k8sInfo, err := k8sFactory.NewRequestInfo(req)
	requestInfo.RequestInfo = k8sInfo
	requestInfo.ResourceScope = r.resolveResourceScope(requestInfo)

	req.URL.Path = oriPathname

	return &requestInfo, err
}

func setIsK8sRequest(requestInfo *RequestInfo) {
	prefix := requestInfo.APIPrefix
	if prefix == "" {
		currentParts := splitPath(requestInfo.Path)
		if len(currentParts) > 0 {
			prefix = currentParts[0]
		}
	}
	if k8sAPIPrefixes.Has(prefix) {
		requestInfo.IsK8sRequest = true
	}
}

type requestInfoKeyType int

const requestInfoKey requestInfoKeyType = iota

// WithRequestInfo 返回父对象的一个副本，其中设置了请求信息的值
func WithRequestInfo(parent context.Context, info *RequestInfo) context.Context {
	return k8srequest.WithValue(parent, requestInfoKey, info)
}

// RequestInfoFrom  返回上下文 ctx 中 RequestInfo 键的值
func RequestInfoFrom(ctx context.Context) (*RequestInfo, bool) {
	info, exist := ctx.Value(requestInfoKey).(*RequestInfo)
	return info, exist
}

func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

func (r *RequestInfoFactory) extractCluster(
	factory k8srequest.RequestInfoFactory, pathname string,
) (string, string, bool) {
	clusterName := ""
	currentParts := splitPath(pathname)

	if len(currentParts) < resourcePathMinLength || !factory.APIPrefixes.Has(currentParts[0]) {
		return clusterName, pathname, false
	}

	APIPrefix := currentParts[0]
	currentParts = currentParts[1:]

	if currentParts[0] != "clusters" {
		// the request doesn't contain cluster
		return "", pathname, false
	}

	if len(currentParts) > resourcePathMinLength-clusterPathOffset {
		clusterName = currentParts[clusterNameIndex]
	}
	if len(currentParts) > resourcePathMinLength-clusterNameIndex {
		currentParts = currentParts[clusterPathOffset:]
	}
	dispatchedPathname := strings.Join([]string{APIPrefix, strings.Join(currentParts, "/")}, "/")
	return clusterName, dispatchedPathname, true
}

const (
	globalScope    = "Global"
	clusterScope   = "Cluster"
	namespaceScope = "Namespace"
)

func (r *RequestInfoFactory) resolveResourceScope(info RequestInfo) string {
	for _, groupResource := range r.GlobalResources {
		if groupResource.Group == info.APIGroup && groupResource.Resource == info.Resource {
			return globalScope
		}
	}

	if info.Namespace != "" {
		return namespaceScope
	}

	return clusterScope
}
