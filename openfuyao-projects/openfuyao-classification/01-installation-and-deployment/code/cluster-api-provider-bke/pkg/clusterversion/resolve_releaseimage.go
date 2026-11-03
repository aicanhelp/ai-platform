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

package clusterversion

import (
	"context"
	"fmt"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

// ResolveReleaseImageForVersion locates a ReleaseImage for the given openFuyao version string.
// Lookup uses conventional metadata.name first, then spec.version.
func ResolveReleaseImageForVersion(
	ctx context.Context,
	c client.Client,
	namespace, version string,
) (*cvv1alpha1.ReleaseImage, error) {
	if c == nil {
		return nil, fmt.Errorf("client is nil")
	}
	version = strings.TrimSpace(version)
	if version == "" {
		return nil, fmt.Errorf("release image version is empty")
	}
	namespace = strings.TrimSpace(namespace)
	if namespace == "" {
		return nil, fmt.Errorf("namespace is empty")
	}

	if ri, err := getReleaseImageByConventionalName(ctx, c, namespace, version); err != nil {
		return nil, err
	} else if ri != nil {
		return ri, nil
	}

	return findReleaseImageByVersion(ctx, c, namespace, version)
}

// ResolveReleaseImageForDesiredVersion locates the ReleaseImage for cv.spec.desiredVersion.
func ResolveReleaseImageForDesiredVersion(
	ctx context.Context,
	c client.Client,
	cv *cvv1alpha1.ClusterVersion,
) (*cvv1alpha1.ReleaseImage, error) {
	if cv == nil {
		return nil, fmt.Errorf("cluster version is nil")
	}
	version := strings.TrimSpace(cv.Spec.DesiredVersion)
	if version == "" {
		return nil, fmt.Errorf("cluster version %s/%s has empty spec.desiredVersion", cv.Namespace, cv.Name)
	}
	return ResolveReleaseImageForVersion(ctx, c, cv.Namespace, version)
}

func getReleaseImageByConventionalName(
	ctx context.Context,
	c client.Client,
	namespace, version string,
) (*cvv1alpha1.ReleaseImage, error) {
	ri := &cvv1alpha1.ReleaseImage{}
	err := c.Get(ctx, types.NamespacedName{
		Namespace: namespace,
		Name:      ReleaseImageRefForVersion(version),
	}, ri)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(ri.Spec.Version) != version {
		return nil, nil
	}
	return ri, nil
}

func findReleaseImageByVersion(
	ctx context.Context,
	c client.Client,
	namespace, version string,
) (*cvv1alpha1.ReleaseImage, error) {
	list := &cvv1alpha1.ReleaseImageList{}
	if err := c.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil, err
	}

	var fallback *cvv1alpha1.ReleaseImage
	for i := range list.Items {
		ri := &list.Items[i]
		if strings.TrimSpace(ri.Spec.Version) != version {
			continue
		}
		if ri.Status.Phase == cvv1alpha1.ReleaseImagePhaseValid {
			return ri, nil
		}
		if fallback == nil {
			fallback = ri
		}
	}
	if fallback != nil {
		return fallback, nil
	}
	return nil, fmt.Errorf("ReleaseImage for version %q not found in namespace %s", version, namespace)
}
