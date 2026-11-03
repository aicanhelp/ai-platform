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

package phaseutil

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// SyncReferencedKubeletAndContainerdConfig 将 BKECluster ref 引用的 KCT/CCT 从控制器集群同步到目标集群
func SyncReferencedKubeletAndContainerdConfig(
	ctx context.Context,
	localClient client.Client,
	remoteClientSet kubernetes.Interface,
	remoteDynamic dynamic.Interface,
	bkeCluster *bkev1beta1.BKECluster,
) error {
	if bkeCluster == nil {
		return nil
	}
	if remoteClientSet == nil || remoteDynamic == nil {
		return errors.New("remote client is required to sync referenced config CRs")
	}

	var errs []error
	if ref := bkeCluster.Spec.KubeletConfigRef; ref != nil && ref.Name != "" {
		namespace := resolveKubeletConfigNamespace(ref, bkeCluster.Namespace)
		if err := syncConfigCR(ctx, localClient, remoteClientSet, remoteDynamic, configCRSyncTarget{
			gvr:       confv1beta1.KubeletConfigGVR(),
			gvk:       confv1beta1.KubeletConfigGVK(),
			namespace: namespace,
			name:      ref.Name,
		}); err != nil {
			errs = append(errs, errors.Wrapf(err, "sync KubeletConfig %q", client.ObjectKey{Namespace: namespace, Name: ref.Name}))
		}
	}

	if bkeCluster.Spec.ClusterConfig != nil {
		if ref := bkeCluster.Spec.ClusterConfig.Cluster.ContainerdConfigRef; ref != nil && ref.Name != "" {
			namespace := resolveContainerdConfigNamespace(ref, bkeCluster.Namespace)
			if err := syncConfigCR(ctx, localClient, remoteClientSet, remoteDynamic, configCRSyncTarget{
				gvr:       confv1beta1.ContainerdConfigGVR(),
				gvk:       confv1beta1.ContainerdConfigGVK(),
				namespace: namespace,
				name:      ref.Name,
			}); err != nil {
				errs = append(errs, errors.Wrapf(err, "sync ContainerdConfig %q", client.ObjectKey{Namespace: namespace, Name: ref.Name}))
			}
		}
	}

	if len(errs) > 0 {
		return errors.Errorf("sync referenced config CRs failed: %v", errs)
	}
	return nil
}

func resolveKubeletConfigNamespace(ref *confv1beta1.KubeletConfigRef, clusterNamespace string) string {
	if ref == nil || ref.Namespace == "" {
		return clusterNamespace
	}
	return ref.Namespace
}

func resolveContainerdConfigNamespace(ref *confv1beta1.ContainerdConfigRef, clusterNamespace string) string {
	if ref == nil || ref.Namespace == "" {
		return clusterNamespace
	}
	return ref.Namespace
}

type configCRSyncTarget struct {
	gvr       schema.GroupVersionResource
	gvk       schema.GroupVersionKind
	namespace string
	name      string
}

func syncConfigCR(
	ctx context.Context,
	localClient client.Client,
	remoteClientSet kubernetes.Interface,
	remoteDynamic dynamic.Interface,
	target configCRSyncTarget,
) error {
	source := &unstructured.Unstructured{}
	source.SetGroupVersionKind(target.gvk)
	if err := localClient.Get(ctx, client.ObjectKey{Namespace: target.namespace, Name: target.name}, source); err != nil {
		return errors.Wrapf(err, "get %s from controller cluster", target.gvk.Kind)
	}

	if err := ensureRemoteNamespace(ctx, remoteClientSet, target.namespace); err != nil {
		return err
	}

	remoteObj := source.DeepCopy()
	prepareConfigCRForApply(remoteObj)

	remoteResource := remoteDynamic.Resource(target.gvr).Namespace(target.namespace)
	_, err := remoteResource.Create(ctx, remoteObj, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsAlreadyExists(err) {
		return errors.Wrapf(err, "create %s on target cluster", target.gvk.Kind)
	}

	return RetryOnConflict(func() error {
		existing, getErr := remoteResource.Get(ctx, target.name, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}
		remoteObj.SetResourceVersion(existing.GetResourceVersion())
		_, updateErr := remoteResource.Update(ctx, remoteObj, metav1.UpdateOptions{})
		return updateErr
	})
}

func prepareConfigCRForApply(obj *unstructured.Unstructured) {
	obj.SetResourceVersion("")
	obj.SetUID("")
	obj.SetGeneration(0)
	obj.SetSelfLink("")
	obj.SetCreationTimestamp(metav1.Time{})
	obj.SetManagedFields(nil)
	obj.SetOwnerReferences(nil)
	unstructured.RemoveNestedField(obj.Object, "status")
}

func ensureRemoteNamespace(ctx context.Context, remoteClientSet kubernetes.Interface, namespace string) error {
	if namespace == "" {
		return fmt.Errorf("namespace is required")
	}
	_, err := remoteClientSet.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return errors.Wrapf(err, "get remote namespace %q", namespace)
	}
	_, err = remoteClientSet.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrapf(err, "create remote namespace %q", namespace)
	}
	return nil
}
