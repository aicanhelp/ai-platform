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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

// ProvisionOptions controls initial ClusterVersion spec/status when created from a BKECluster.
type ProvisionOptions struct {
	Phase          cvv1alpha1.ClusterVersionPhase
	CurrentVersion string
}

// InstallProvision returns the options used when BKECluster starts installation.
// ClusterVersion.status is populated by BKECluster Reconciler when install completes.
func InstallProvision() ProvisionOptions {
	return ProvisionOptions{}
}

// OpenFuyaoVersionForBKECluster returns status version, else spec.cluster.openFuyaoVersion.
func OpenFuyaoVersionForBKECluster(bc *bkev1beta1.BKECluster) string {
	if bc == nil {
		return ""
	}
	if bc.Status.OpenFuyaoVersion != "" {
		return strings.TrimSpace(bc.Status.OpenFuyaoVersion)
	}
	if bc.Spec.ClusterConfig != nil {
		return strings.TrimSpace(bc.Spec.ClusterConfig.Cluster.OpenFuyaoVersion)
	}
	return ""
}

// ReleaseImageRefForVersion derives the default ReleaseImage metadata.name for a version string.
func ReleaseImageRefForVersion(version string) string {
	version = strings.ToLower(strings.TrimSpace(version))
	version = strings.TrimPrefix(version, "v")
	version = strings.NewReplacer("/", "-", "_", "-", "@", "-", ":", "-").Replace(version)
	return "release-v" + strings.Trim(version, "-")
}

// NewClusterVersionFromBKECluster builds a ClusterVersion owned by the BKECluster (not persisted).
func NewClusterVersionFromBKECluster(
	bc *bkev1beta1.BKECluster,
	opts ProvisionOptions,
) (*cvv1alpha1.ClusterVersion, error) {
	if bc == nil {
		return nil, fmt.Errorf("bke cluster is nil")
	}
	desired := OpenFuyaoVersionForBKECluster(bc)
	if desired == "" {
		return nil, fmt.Errorf("openFuyaoVersion is empty on BKECluster %s/%s", bc.Namespace, bc.Name)
	}

	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name:      bc.Name,
			Namespace: bc.Namespace,
		},
		Spec: cvv1alpha1.ClusterVersionSpec{
			DesiredVersion: desired,
		},
	}
	return cv, nil
}

// EnsureClusterVersionForBKECluster creates a ClusterVersion when none is associated with bc.
// Install path uses Phase=Installing; BKECluster reconciler patches status on install completion.
func EnsureClusterVersionForBKECluster(
	ctx context.Context,
	c client.Client,
	scheme *runtime.Scheme,
	bc *bkev1beta1.BKECluster,
) (created bool, err error) {
	if bc == nil {
		return false, fmt.Errorf("bke cluster is nil")
	}
	if !bc.DeletionTimestamp.IsZero() {
		return false, nil
	}

	_, err = GetClusterVersionForBKECluster(ctx, c, bc)
	if err == nil {
		return false, nil
	}
	if !apierrors.IsNotFound(err) {
		return false, err
	}

	cv, err := NewClusterVersionFromBKECluster(bc, InstallProvision())
	if err != nil {
		return false, err
	}
	if scheme != nil {
		if err := controllerutil.SetControllerReference(bc, cv, scheme); err != nil {
			return false, err
		}
	}
	if err := c.Create(ctx, cv); err != nil {
		return false, client.IgnoreAlreadyExists(err)
	}
	return true, nil
}
