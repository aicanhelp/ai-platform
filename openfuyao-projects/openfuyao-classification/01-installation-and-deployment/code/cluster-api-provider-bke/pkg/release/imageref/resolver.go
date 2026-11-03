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

package imageref

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

const (
	AnnotationBKEClusterName = annotation.ReleaseImageBKEClusterNameAnnotationKey

	releaseImageName = "release-image"
)

// ReleaseImageRefs returns candidate OCI refs for a release image (domain first, then IP).
func ReleaseImageRefs(ctx context.Context, c client.Client, namespace, preferredClusterName, version string) ([]string, error) {
	if strings.TrimSpace(version) == "" {
		return nil, fmt.Errorf("release version is empty")
	}
	return ResolveImageOCIRefs(ctx, c, ResolveOptions{
		Namespace:            namespace,
		PreferredClusterName: preferredClusterName,
		ImageName:            releaseImageName,
		Tag:                  version,
	})
}

// ResolveOptions identifies the cluster and image whose OCI references are required.
type ResolveOptions struct {
	// Namespace limits cluster lookup to one namespace.
	Namespace string
	// PreferredClusterName selects a cluster when several are available.
	PreferredClusterName string
	// ImageName is the repository name appended to the registry address.
	ImageName string
	// Tag is the requested image version.
	Tag string
}

// ResolveImageOCIRefs builds candidate OCI refs from BKECluster imageRepo: domain first, then IP.
func ResolveImageOCIRefs(ctx context.Context, c client.Client, options ResolveOptions) ([]string, error) {
	bc, err := selectBKECluster(ctx, c, options.Namespace, options.PreferredClusterName)
	if err != nil {
		return nil, err
	}
	refs, err := upgradepath.ResolveImageOCIRefsFromCluster(bc, options.ImageName, options.Tag)
	if err != nil {
		if bc.Spec.ClusterConfig == nil {
			return nil, err
		}
		// Matches bkeadm prepareImageRepoConfig when only --onlineImage is set (empty prefix).
		if strings.Trim(bc.Spec.ClusterConfig.Cluster.ImageRepo.Prefix, "/") == "" {
			return []string{upgradepath.DefaultReleaseImageOCIRef(options.Tag)}, nil
		}
		return nil, fmt.Errorf("failed to resolve upgrade path OCI refs from bke-config: %w", err)
	}
	return refs, nil
}

func selectBKECluster(ctx context.Context, c client.Client, namespace, preferredName string) (*bkev1beta1.BKECluster, error) {
	if preferredName != "" && namespace != "" {
		bc := &bkev1beta1.BKECluster{}
		if err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: preferredName}, bc); err != nil {
			return nil, err
		}
		return bc, nil
	}

	list := &bkev1beta1.BKEClusterList{}
	var opts []client.ListOption
	if namespace != "" {
		opts = append(opts, client.InNamespace(namespace))
	}
	if err := c.List(ctx, list, opts...); err != nil {
		return nil, err
	}
	if preferredName != "" {
		matches := make([]bkev1beta1.BKECluster, 0, 1)
		for _, item := range list.Items {
			if item.Name == preferredName {
				matches = append(matches, item)
			}
		}
		if len(matches) == 1 {
			return &matches[0], nil
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("BKECluster %q not found", preferredName)
		}
		return nil, fmt.Errorf("multiple BKEClusters named %q found, use a namespaced ReleaseImage or UpgradePath", preferredName)
	}
	if len(list.Items) == 0 {
		return nil, fmt.Errorf("BKECluster not found")
	}

	sort.Slice(list.Items, func(i, j int) bool {
		return client.ObjectKeyFromObject(&list.Items[i]).String() < client.ObjectKeyFromObject(&list.Items[j]).String()
	})

	return &list.Items[0], nil
}
