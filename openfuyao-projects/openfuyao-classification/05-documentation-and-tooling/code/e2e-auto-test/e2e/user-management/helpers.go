package usermanagement

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"

	"golang.org/x/crypto/pbkdf2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

func discoverUserGVR(ctx context.Context, client *k8s.K8SClient) (schema.GroupVersionResource, bool, error) {
	_, lists, err := client.Clientset.Discovery().ServerGroupsAndResources()
	if err != nil && lists == nil {
		return schema.GroupVersionResource{}, false, err
	}
	for _, list := range lists {
		gv, err := schema.ParseGroupVersion(list.GroupVersion)
		if err != nil {
			continue
		}
		for _, r := range list.APIResources {
			if r.Kind != "User" || r.Name != "users" || strings.Contains(r.Name, "/") {
				continue
			}
			return schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: r.Name}, r.Namespaced, nil
		}
	}
	return schema.GroupVersionResource{}, false, fmt.Errorf("cluster has no User/users CRD")
}

func getUserEncryptedPassword(ctx context.Context, client *k8s.K8SClient, name string) (string, error) {
	gvr, namespaced, err := discoverUserGVR(ctx, client)
	if err != nil {
		return "", err
	}
	var obj *unstructured.Unstructured
	if namespaced {
		obj, err = client.DynamicClient.Resource(gvr).Namespace("default").Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			obj, err = client.DynamicClient.Resource(gvr).Namespace("").Get(ctx, name, metav1.GetOptions{})
		}
	} else {
		obj, err = client.DynamicClient.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
	}
	if err != nil {
		return "", err
	}
	for _, key := range []string{"EncryptedPassword", "encryptedPassword"} {
		v, found, nerr := unstructured.NestedString(obj.Object, "spec", key)
		if nerr != nil {
			return "", nerr
		}
		if found && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v), nil
		}
	}
	return "", fmt.Errorf("spec.EncryptedPassword empty on User/%s", name)
}

func verifyPBKDF2Password(stored, plaintext string) error {
	first, err := base64.StdEncoding.DecodeString(stored)
	if err != nil {
		return fmt.Errorf("first base64 decode: %w", err)
	}
	decoded, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(first)))
	if err != nil {
		return fmt.Errorf("second base64 decode: %w", err)
	}
	const saltLen = 16
	if len(decoded) <= saltLen {
		return fmt.Errorf("encrypted blob too short: %d", len(decoded))
	}
	salt := decoded[:saltLen]
	want := decoded[saltLen:]
	got := pbkdf2.Key([]byte(plaintext), salt, 100000, 64, sha256.New)
	if !bytes.Equal(got, want) {
		// encrypted data are BY NO MEANS sensitive contents and SHOULD be allowed to be printed
		return fmt.Errorf("PBKDF2 password mismatch, want: %s, got: %s", want, got)
	}
	return nil
}

func listOAuthServerPods(ctx context.Context, client *k8s.K8SClient, ns string) ([]corev1.Pod, error) {
	list, err := client.ListPods(ctx, ns, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var out []corev1.Pod
	for i := range list.Items {
		p := list.Items[i]
		if strings.Contains(p.Name, "oauth-server") {
			out = append(out, p)
		}
	}
	return out, nil
}

func runningOAuthPodNames(pods []corev1.Pod) []string {
	var names []string
	for _, p := range pods {
		if p.DeletionTimestamp != nil {
			continue
		}
		if p.Status.Phase == corev1.PodRunning {
			names = append(names, p.Name)
		}
	}
	return names
}

func oauthPodReady(p corev1.Pod) bool {
	if p.DeletionTimestamp != nil || p.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, c := range p.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
