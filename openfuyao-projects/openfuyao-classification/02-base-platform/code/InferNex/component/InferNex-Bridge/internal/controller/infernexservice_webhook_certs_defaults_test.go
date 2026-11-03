package controller

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestGatewayDefaultsAndResolvedName(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	r := &InferNexServiceReconciler{}
	if got := r.resolvedGatewayNamespacedName(owner, nil); got != (types.NamespacedName{Namespace: "ns-a", Name: "demo-" + managedGatewaySuffix}) {
		t.Fatalf("unexpected default gateway namespaced name: %v", got)
	}
	igr := &infernexv1alpha1.IntelligentGatewayRoutingSpec{
		Gateway: &infernexv1alpha1.GatewayRefSpec{Ref: &infernexv1alpha1.NamedRef{Name: "custom-gw"}},
	}
	if got := r.resolvedGatewayNamespacedName(owner, igr); got.Name != "custom-gw" {
		t.Fatalf("expected custom gateway name, got %v", got)
	}
}

func TestWebhookCertHelpers(t *testing.T) {
	t.Parallel()
	if !componentNeedsWebhookCert(elasticScalerComponent) || !componentNeedsWebhookCert(rsgComponent) {
		t.Fatal("expected elastic-scaler and rsg need webhook cert")
	}
	if componentNeedsWebhookCert(cacheIndexerComponent) {
		t.Fatal("cache-indexer should not need webhook cert")
	}
	if name := componentWebhookCertSecretName("demo", "rsg"); name == "" {
		t.Fatal("secret name should not be empty")
	}
}

func TestEnsureWebhookCertVolumeAndMount_Idempotent(t *testing.T) {
	t.Parallel()
	tpl := &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "c1"}, {Name: "c2"}},
		},
	}
	ensureWebhookCertVolumeAndMount(tpl, "secret-a")
	ensureWebhookCertVolumeAndMount(tpl, "secret-b")
	if len(tpl.Spec.Volumes) != 1 {
		t.Fatalf("expected single cert volume, got %d", len(tpl.Spec.Volumes))
	}
	if tpl.Spec.Volumes[0].Secret == nil || tpl.Spec.Volumes[0].Secret.SecretName != "secret-b" {
		t.Fatalf("expected updated secret name, got %#v", tpl.Spec.Volumes[0].Secret)
	}
	for _, c := range tpl.Spec.Containers {
		if len(c.VolumeMounts) != 1 || c.VolumeMounts[0].MountPath != webhookCertMountPath || !c.VolumeMounts[0].ReadOnly {
			t.Fatalf("unexpected volume mounts for %s: %#v", c.Name, c.VolumeMounts)
		}
	}
}

func TestEnsureAndDeleteComponentWebhookCertSecret(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := gatewayTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	name, err := r.ensureComponentWebhookCertSecret(ctx, owner, elasticScalerComponent)
	if err != nil {
		t.Fatalf("ensureComponentWebhookCertSecret error: %v", err)
	}
	secret := &corev1.Secret{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, secret); err != nil {
		t.Fatalf("expected secret created: %v", err)
	}
	if secret.Type != corev1.SecretTypeTLS {
		t.Fatalf("expected tls secret type, got %q", secret.Type)
	}
	cert := secret.Data[corev1.TLSCertKey]
	if len(cert) == 0 || len(secret.Data[corev1.TLSPrivateKeyKey]) == 0 {
		t.Fatal("expected tls cert and key data present")
	}
	pb, _ := pem.Decode(cert)
	if pb == nil {
		t.Fatal("failed to decode certificate pem")
	}
	if _, err := x509.ParseCertificate(pb.Bytes); err != nil {
		t.Fatalf("failed to parse generated cert: %v", err)
	}

	if err := r.deleteComponentWebhookCertSecret(ctx, owner, elasticScalerComponent); err != nil {
		t.Fatalf("deleteComponentWebhookCertSecret error: %v", err)
	}
	err = cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: name}, &corev1.Secret{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected cert secret deleted, got err=%v", err)
	}
}

