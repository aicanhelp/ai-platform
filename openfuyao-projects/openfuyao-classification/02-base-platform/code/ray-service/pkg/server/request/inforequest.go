/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

// RequestInfo contains detailed information extracted from an http.Request.
// It extends the functionality provided in k8s.io/apiserver/pkg/endpoints/request/requestinfo.go
// by including additional fields and methods specific to our application's requirements.
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

// NewRequestInfo extracts request information from the http request.
// It extends the default Kubernetes RequestInfo by adding support for multi-cluster URLs.
func (r *RequestInfoFactory) NewRequestInfo(req *http.Request) (*RequestInfo, error) {
	base := &k8srequest.RequestInfo{
		Path: req.URL.Path,
		Verb: req.Method,
	}

	requestInfo := RequestInfo{
		IsK8sRequest: false,
		RequestInfo:  base,
		Cluster:      "",
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

	// recover the pathname with original /cluster/{cluster} part
	req.URL.Path = oriPathname

	return &requestInfo, err
}

func setIsK8sRequest(requestInfo *RequestInfo) {
	apiPrefix := requestInfo.APIPrefix

	switch {
	case apiPrefix == "":
		segments := splitPath(requestInfo.Path)
		if len(segments) > 0 {
			apiPrefix = segments[0]
		}
	default:
		// do nothing
	}

	if k8sAPIPrefixes.Has(apiPrefix) {
		requestInfo.IsK8sRequest = true
	}
}

type requestInfoKeyType int

const requestInfoKey requestInfoKeyType = iota

// WithRequestInfo returns a copy of parent in which the request info value is set.
func WithRequestInfo(ctx context.Context, info *RequestInfo) context.Context {
	return k8srequest.WithValue(ctx, requestInfoKey, info)
}

// RequestInfoFrom returns the value of the RequestInfo key on the ctx.
func RequestInfoFrom(ctx context.Context) (*RequestInfo, bool) {
	info, exist := ctx.Value(requestInfoKey).(*RequestInfo)
	return info, exist
}

func splitPath(path string) []string {
	newPath := strings.Trim(path, "/")
	if len(path) > 0 {
		return strings.Split(newPath, "/")
	}
	return []string{}
}

func (r *RequestInfoFactory) extractCluster(
	factory k8srequest.RequestInfoFactory, pathname string,
) (string, string, bool) {
	clusterName := ""
	currentParts := splitPath(pathname)

	if len(currentParts) < resourcePathMinLength || !factory.APIPrefixes.Has(currentParts[0]) {
		// the request is non-resource
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
	globalScopeVal    = "Global"
	clusterScopeVal   = "Cluster"
	namespaceScopeVal = "Namespace"
)

func (r *RequestInfoFactory) resolveResourceScope(info RequestInfo) string {
	index := 0
	for index < len(r.GlobalResources) {
		groupResource := r.GlobalResources[index]
		if groupResource.Group == info.APIGroup && groupResource.Resource == info.Resource {
			return globalScopeVal
		}
		index++
	}

	if info.Namespace != "" {
		return namespaceScopeVal
	}

	return clusterScopeVal
}
