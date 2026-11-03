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
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/imageref"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const releaseImageRequeueInterval = 15 * time.Second

// ReleaseImageEnsurer pulls ri OCI artifacts and ensures ReleaseImage CRs exist.
type ReleaseImageEnsurer struct {
	Client    client.Client
	Scheme    *runtime.Scheme
	OCIClient *oci.Client
}

// Ensure resolves or creates a ReleaseImage for version.
func (e *ReleaseImageEnsurer) Ensure(
	ctx context.Context,
	bc *bkev1beta1.BKECluster,
	cv *cvv1alpha1.ClusterVersion,
	version string,
) (*cvv1alpha1.ReleaseImage, ctrl.Result, error) {
	if e == nil || e.Client == nil {
		return nil, ctrl.Result{}, fmt.Errorf("release image ensurer is not configured")
	}
	version = strings.TrimSpace(version)
	if version == "" {
		return nil, ctrl.Result{}, fmt.Errorf("release version is empty")
	}

	clusterName := cv.Name

	ri, err := e.findReleaseImageByVersion(ctx, cv.Namespace, version)
	if err != nil {
		return nil, ctrl.Result{}, err
	}
	if ri == nil {
		ri, err = e.createReleaseImageFromOCI(ctx, bc, cv, clusterName, version)
		if err != nil {
			return nil, ctrl.Result{}, err
		}
	}

	switch ri.Status.Phase {
	case cvv1alpha1.ReleaseImagePhaseValid:
		return ri, ctrl.Result{}, nil
	case cvv1alpha1.ReleaseImagePhaseInvalid,
		cvv1alpha1.ReleaseImagePhaseManifestMissing,
		cvv1alpha1.ReleaseImagePhaseCompatibilityFailed:
		return ri, ctrl.Result{}, fmt.Errorf("release image %s/%s phase %s: %s",
			ri.Namespace, ri.Name, ri.Status.Phase, ri.Status.Message)
	default:
		return ri, ctrl.Result{RequeueAfter: releaseImageRequeueInterval}, nil
	}
}

func (e *ReleaseImageEnsurer) findReleaseImageByVersion(
	ctx context.Context,
	namespace, version string,
) (*cvv1alpha1.ReleaseImage, error) {
	list := &cvv1alpha1.ReleaseImageList{}
	if err := e.Client.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil, err
	}
	for i := range list.Items {
		if list.Items[i].Spec.Version == version {
			return list.Items[i].DeepCopy(), nil
		}
	}
	return nil, nil
}

func (e *ReleaseImageEnsurer) createReleaseImageFromOCI(
	ctx context.Context,
	bc *bkev1beta1.BKECluster,
	cv *cvv1alpha1.ClusterVersion,
	clusterName, version string,
) (*cvv1alpha1.ReleaseImage, error) {
	ociRefs, err := imageref.ReleaseImageRefs(ctx, e.Client, cv.Namespace, clusterName, version)
	if err != nil {
		return nil, fmt.Errorf("resolve release OCI ref: %w", err)
	}

	var (
		parsed  *cvv1alpha1.ReleaseImage
		lastErr error
	)
	for _, ref := range ociRefs {
		log.Info("pulling release image", "ociRef", ref, "clusterVersion", cv.Namespace+"/"+cv.Name, "version", version)
		parsed, err = pullReleaseImageSpec(ctx, e.OCIClient, ref)
		if err == nil {
			log.Info("release image pulled", "ociRef", ref, "clusterVersion", cv.Namespace+"/"+cv.Name, "version", version)
			break
		}
		log.Error("release image pull failed, trying next OCI ref", "ociRef", ref, "error", err.Error())
		lastErr = fmt.Errorf("pull release manifest from %s: %w", ref, err)
	}
	if parsed == nil {
		if lastErr != nil {
			return nil, lastErr
		}
		return nil, fmt.Errorf("no OCI ref resolved for release image")
	}

	ri := releaseImageCRFromParsed(parsed, cv.Namespace, clusterName)
	if e.Scheme != nil && bc != nil {
		if err := controllerutil.SetControllerReference(bc, ri, e.Scheme); err != nil {
			return nil, err
		}
	}

	if err := e.Client.Create(ctx, ri); err != nil {
		if apierrors.IsAlreadyExists(err) {
			existing := &cvv1alpha1.ReleaseImage{}
			if getErr := e.Client.Get(ctx, types.NamespacedName{Namespace: ri.Namespace, Name: ri.Name}, existing); getErr != nil {
				return nil, getErr
			}
			return existing, nil
		}
		return nil, err
	}
	return ri, nil
}

func pullReleaseImageSpec(ctx context.Context, ociClient *oci.Client, ociRef string) (*cvv1alpha1.ReleaseImage, error) {
	if ociClient == nil {
		return nil, fmt.Errorf("OCI client is not configured")
	}
	img, err := ociClient.Pull(ctx, ociRef)
	if err != nil {
		return nil, err
	}
	layer, err := img.GetLayerByPath("release.yaml")
	if err != nil {
		return nil, fmt.Errorf("release.yaml not found in %s: %w", ociRef, err)
	}
	parsed := &cvv1alpha1.ReleaseImage{}
	if err := layer.UnmarshalYAML(parsed); err != nil {
		return nil, fmt.Errorf("unmarshal release.yaml: %w", err)
	}
	if strings.TrimSpace(parsed.Spec.Version) == "" {
		return nil, fmt.Errorf("release.yaml missing spec.version")
	}
	return parsed, nil
}

func releaseImageCRFromParsed(
	parsed *cvv1alpha1.ReleaseImage,
	namespace, clusterName string,
) *cvv1alpha1.ReleaseImage {
	ri := parsed.DeepCopy()
	ri.TypeMeta = metav1.TypeMeta{
		APIVersion: cvv1alpha1.GroupVersion.String(),
		Kind:       "ReleaseImage",
	}
	ri.Namespace = namespace
	if strings.TrimSpace(ri.Name) == "" {
		ri.Name = ReleaseImageRefForVersion(parsed.Spec.Version)
	}
	if ri.Annotations == nil {
		ri.Annotations = map[string]string{}
	}
	ri.Annotations[imageref.AnnotationBKEClusterName] = clusterName
	ri.ResourceVersion = ""
	ri.UID = ""
	ri.CreationTimestamp = metav1.Time{}
	ri.Status = cvv1alpha1.ReleaseImageStatus{}
	return ri
}
