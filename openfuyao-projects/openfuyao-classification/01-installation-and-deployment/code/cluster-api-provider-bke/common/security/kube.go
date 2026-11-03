/*
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package security

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// bkenodeGVR is used by dynamic (scheme-less) clients to list BKENodes.
var bkenodeGVR = schema.GroupVersionResource{
	Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes",
}

// translateGetSecretErr maps a Secret get result to the SecretKeyProvider
// convention: NotFound becomes (nil, nil) so the provider can distinguish
// "missing" from real API errors.
func translateGetSecretErr(s *corev1.Secret, err error) (*corev1.Secret, error) {
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return s, nil
}

// translateCreateSecretErr maps a Secret create result to the
// SecretKeyProvider convention: AlreadyExists becomes (false, nil) so the
// provider re-reads the winning replica's Secret.
func translateCreateSecretErr(err error) (bool, error) {
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// CtrlRuntimeSecretIO adapts controller-runtime clients for the
// SecretKeyProvider. The reader should be mgr.GetAPIReader() (direct API
// reads, bypassing the informer cache, so a Secret created by another replica
// is immediately visible); the writer should be mgr.GetClient() (writes go
// straight to the API even before the cache starts).
func CtrlRuntimeSecretIO(reader client.Reader, writer client.Writer,
	namespace, name string) (get func(ctx context.Context) (*corev1.Secret, error),
	create func(ctx context.Context, s *corev1.Secret) (bool, error)) {

	get = func(ctx context.Context) (*corev1.Secret, error) {
		s := &corev1.Secret{}
		err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, s)
		return translateGetSecretErr(s, err)
	}
	create = func(ctx context.Context, s *corev1.Secret) (bool, error) {
		return translateCreateSecretErr(writer.Create(ctx, s))
	}
	return get, create
}

// ClientGoSecretIO adapts a client-go clientset (bkeadm host process).
func ClientGoSecretIO(clientset kubernetes.Interface,
	namespace, name string) (get func(ctx context.Context) (*corev1.Secret, error),
	create func(ctx context.Context, s *corev1.Secret) (bool, error)) {

	get = func(ctx context.Context) (*corev1.Secret, error) {
		s, err := clientset.CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
		return translateGetSecretErr(s, err)
	}
	create = func(ctx context.Context, s *corev1.Secret) (bool, error) {
		_, err := clientset.CoreV1().Secrets(namespace).Create(ctx, s, metav1.CreateOptions{})
		return translateCreateSecretErr(err)
	}
	return get, create
}

// isBKENodeResourceMissing reports whether err means the BKENode resource
// type does not exist on the cluster (CRD not installed yet). When the CRD
// is absent no BKENode can hold new-format ciphertext, so there is no
// key-loss risk and detection can safely be skipped (first-time install via
// bke init deploys 002-cluster-api-bke.yaml, which carries the CRD, only
// after the encryption key is ensured).
func isBKENodeResourceMissing(err error) bool {
	return apierrors.IsNotFound(err)
}

// deferListErr wraps a BKENode list failure: the generation attempt is
// deferred because warnings must never be silently skipped while the cluster
// state cannot be verified.
func deferListErr(err error) error {
	return fmt.Errorf("deferring new password encryption key generation: "+
		"cannot list BKENodes to detect key loss: %w", err)
}

// CtrlRuntimeBKENodeGuard is the key-loss detection for the controller
// process. It does NOT block generation (see design doc decision D1):
//   - any BKENode whose password is new-format ciphertext gets a Warning
//     Event (reason PasswordKeyLost) plus an ERROR log, then the guard
//     returns nil (generation proceeds);
//   - when the BKENode resource type does not exist (CRD missing), the
//     detection is skipped and generation proceeds;
//   - any other listing failure defers this generation attempt (retried on
//     the next call / pod restart) so warnings are never silently skipped;
//   - a nil recorder disables Event emission (logs only).
//
// The reader should be mgr.GetAPIReader() to avoid stale informer reads.
func CtrlRuntimeBKENodeGuard(reader client.Reader, recorder record.EventRecorder) func(ctx context.Context) error {
	return func(ctx context.Context) error {
		list := &confv1beta1.BKENodeList{}
		if err := reader.List(ctx, list); err != nil {
			if isBKENodeResourceMissing(err) {
				log.Info("BKENode resource not found (CRD not installed yet?), skipping key-loss detection")
				return nil
			}
			return deferListErr(err)
		}
		for i := range list.Items {
			n := &list.Items[i]
			if !IsNewFormatCiphertext(n.Spec.Password) {
				continue
			}
			log.Errorf("DATA LOSS WARNING: secret %s/%s is missing and a new key is being "+
				"generated; BKENode %s/%s has a password encrypted with the lost key, its "+
				"SSH will fail until spec.password is updated with the correct plaintext "+
				"(the webhook will re-encrypt it with the new key)",
				DefaultSecretNamespace, DefaultSecretName, n.Namespace, n.Name)
			if recorder != nil {
				recorder.Eventf(n, corev1.EventTypeWarning, "PasswordKeyLost",
					"The password encryption key Secret %s/%s was missing and a new key "+
						"is being generated; the stored password was encrypted with the "+
						"lost key and SSH will fail until spec.password is updated with "+
						"the correct plaintext (it will be re-encrypted automatically)",
					DefaultSecretNamespace, DefaultSecretName)
			}
		}
		return nil
	}
}

// DynamicBKENodeGuard is the bkeadm-side guard (scheme-less dynamic client).
// Same policy as CtrlRuntimeBKENodeGuard: detected key loss is reported as a
// console WARN with the affected node list and recovery guidance, then the
// guard returns nil; a missing BKENode resource type (fresh install before
// the CRD is deployed) skips detection; other listing failures defer the
// generation attempt.
func DynamicBKENodeGuard(dyn dynamic.Interface) func(ctx context.Context) error {
	return func(ctx context.Context) error {
		list, err := dyn.Resource(bkenodeGVR).List(ctx, metav1.ListOptions{})
		if err != nil {
			if isBKENodeResourceMissing(err) {
				log.Info("BKENode resource not found (CRD not installed yet?), skipping key-loss detection")
				return nil
			}
			return deferListErr(err)
		}
		var affected []string
		for _, item := range list.Items {
			password, found, _ := unstructured.NestedString(item.Object, "spec", "password")
			if found && IsNewFormatCiphertext(password) {
				affected = append(affected, item.GetNamespace()+"/"+item.GetName())
			}
		}
		if len(affected) > 0 {
			log.Warnf("DATA LOSS WARNING: secret %s/%s is missing and a new key is being "+
				"generated. The following BKENodes have passwords encrypted with the lost "+
				"key; their SSH will fail until spec.password is updated with the correct "+
				"plaintext (the webhook will re-encrypt them): %s",
				DefaultSecretNamespace, DefaultSecretName, strings.Join(affected, ", "))
		}
		return nil
	}
}

// SecretWatchRunnable is a controller-runtime manager Runnable that watches
// the encryption key Secret and keeps the SecretKeyProvider cache in sync
// with external changes, removing the need to restart the controller:
//   - Added/Modified (external rotation): the cached key is refreshed from
//     the Secret content; invalid data keeps the current cache;
//   - Deleted (accidental deletion): the Secret is recreated from the cached
//     in-memory key, so the deletion self-heals with zero key loss.
type SecretWatchRunnable struct {
	provider  *SecretKeyProvider
	clientset kubernetes.Interface
}

// NewSecretWatchRunnable builds the watcher. The clientset must point to the
// management cluster (kubernetes.NewForConfig(mgr.GetConfig())).
func NewSecretWatchRunnable(provider *SecretKeyProvider, clientset kubernetes.Interface) *SecretWatchRunnable {
	return &SecretWatchRunnable{provider: provider, clientset: clientset}
}

// isKeySecret reports whether the object is the watched encryption key
// Secret. The informer field selector is a server-side optimization only;
// client-side filtering keeps fake-clientset (no server filtering) and
// custom deployments safe.
func (r *SecretWatchRunnable) isKeySecret(obj interface{}) (*corev1.Secret, bool) {
	sec, ok := obj.(*corev1.Secret)
	if !ok {
		return nil, false
	}
	return sec, sec.Namespace == DefaultSecretNamespace && sec.Name == DefaultSecretName
}

// Start implements manager.Runnable.
func (r *SecretWatchRunnable) Start(ctx context.Context) error {
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = "metadata.name=" + DefaultSecretName
			return r.clientset.CoreV1().Secrets(DefaultSecretNamespace).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = "metadata.name=" + DefaultSecretName
			return r.clientset.CoreV1().Secrets(DefaultSecretNamespace).Watch(ctx, options)
		},
	}
	_, ctrl := cache.NewInformer(lw, &corev1.Secret{}, 0, r.eventHandlers(ctx))
	go ctrl.Run(ctx.Done())
	log.Infof("watching encryption key secret %s/%s for external changes", DefaultSecretNamespace, DefaultSecretName)
	return nil
}

// eventHandlers builds the informer handlers driving the provider cache.
func (r *SecretWatchRunnable) eventHandlers(ctx context.Context) cache.ResourceEventHandlerFuncs {
	return cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if sec, ok := r.isKeySecret(obj); ok {
				r.provider.UpdateFromSecret(sec)
			}
		},
		UpdateFunc: func(_, newObj interface{}) {
			if sec, ok := r.isKeySecret(newObj); ok {
				r.provider.UpdateFromSecret(sec)
			}
		},
		DeleteFunc: func(obj interface{}) {
			if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = d.Obj
			}
			if _, ok := r.isKeySecret(obj); ok {
				r.provider.RestoreSecret(ctx)
			}
		},
	}
}
