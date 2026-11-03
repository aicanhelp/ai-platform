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

// Package utils contains a collection of utility functions and helpers that provide
// common functionalities useful across the entire engineering project.
package utils

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

const (
	openFuyaoPlatformRole = "platformrole"
	openFuyaoClusterRole  = "clusterrole"

	operationsJoin   = "join:cluster"
	operationsUnjoin = "unjoin:cluster"
	operationsTag    = "tag:cluster"
	operationsList   = "list:cluster"
)

const (
	watchKarmadaServiceTimeout  = 2 * time.Minute
	watchKarmadaServiceInterval = 5 * time.Second
)

var roleActions = map[string][]string{
	openFuyaoPlatformRole: []string{operationsJoin, operationsUnjoin, operationsTag, operationsList},
	openFuyaoClusterRole:  []string{},
}

// IsAccessPass checks if the current user has permission for cluster opeartions.
func IsAccessPass(c kubernetes.Interface, name string, operation string) (bool, error) {
	memList, err := CheckUserAccessList(c, name)
	if err != nil {
		return false, err
	}

	var userRole string

	if len(memList) == 0 {
		return false, nil
	}

	if len(memList) == 1 && memList[0] == "*" {
		userRole = openFuyaoPlatformRole
	} else {
		userRole = openFuyaoClusterRole
	}

	if hasString(roleActions[userRole], operation) {
		zlog.LogInfof("PASS: operations %s for user %s \n", operation, name)
		return true, nil
	}

	zlog.Warnf("Access Deined: operations %s for user %s \n", operation, name)
	return false, nil
}

// HasKarmadaServiceOrMCService checks if karmada service exists in joined=cluster
func HasKarmadaServiceOrMCService(c kubernetes.Interface) (bool, bool) {
	name := "karmada-system"
	_, err := c.CoreV1().Namespaces().Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		zlog.LogInfof("No %s namespace exists", name)
		return false, false
	}

	existKarmada := true
	deployKarmadaAPI := "karmada-apiserver"
	_, err = c.AppsV1().Deployments(name).Get(context.TODO(), deployKarmadaAPI, metav1.GetOptions{})
	if err != nil {
		zlog.LogInfof("No %s deployment exists", deployKarmadaAPI)
		existKarmada = false
	}

	existMcs := true
	deployMcs := "multicluster-service"
	_, err = c.AppsV1().Deployments(name).Get(context.TODO(), deployMcs, metav1.GetOptions{})
	if err != nil {
		zlog.LogInfof("No %s deployment exists", deployMcs)
		existMcs = false
	}

	return existKarmada, existMcs
}

// IsKarmadaServiceReady checks if the karmada service is ready for cluster opeartions.
func IsKarmadaServiceReady(karmadaclient *versioned.Clientset) error {
	// confirm apiservcie is ok
	apiServiceURL := fmt.Sprintf("%s%s", KarmadaEndPoints, "/readyz")
	err := isKarmadaAPIServiceReady(apiServiceURL)
	if err != nil {
		zlog.LogErrorf("Error Communication to karmada API server : %v", err)
		return err
	}
	zlog.LogInfof("Karmada api service is now ready")

	// confirm aggreapiservcie is ok
	apiAggreServiceURL := fmt.Sprintf("%s%s", KarmadaAggreEndPoints, "/readyz")
	err = isKarmadaAPIServiceReady(apiAggreServiceURL)
	if err != nil {
		zlog.LogErrorf("Error Communication to karmada AggreAPI server : %v", err)
		return err
	}
	zlog.LogInfof("Karmada aggre api service is now ready")

	// configm karmadaclient list cluster is ok
	err = wait.PollUntilContextTimeout(context.TODO(), WatchKarmadaListInterval, WatchKarmadaListTimeout, false,
		func(context.Context) (done bool, err error) {
			_, err = karmadaclient.ClusterV1alpha1().Clusters().List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				return false, err
			}

			zlog.LogInfof("Waiting for the karmada cluster list ok.")
			return true, nil
		})
	if err != nil {
		zlog.LogErrorf("Error Requesting karmada cluster list, timeout : %v", err)
		return err
	}

	zlog.LogInfof("Karmada service requests: retrieving karmada cluster list ok.")
	return nil

}

// IsClusterDeleted polls to check if a specified cluster, identified by `tag`, has been deleted.
// It continues to poll until the cluster is confirmed deleted or until a timeout is reached.
func IsClusterDeleted(karmadaclient *versioned.Clientset, tag string) error {
	var err error
	err = wait.PollUntilContextTimeout(context.TODO(), WatchClusterInterval, WatchClusterTimeout, false,
		func(context.Context) (done bool, err error) {
			_, err = karmadaclient.ClusterV1alpha1().Clusters().Get(context.TODO(), tag, metav1.GetOptions{})
			if err != nil {
				if errors.IsNotFound(err) {
					return true, nil
				}
				return false, err
			}

			zlog.LogInfof("Waiting for the cluster object %s to be deleted.", tag)
			return false, nil
		})
	if err != nil {
		zlog.LogErrorf("Error Deleting cluster object %s, timeout : %v", tag, err)
		return err
	}

	zlog.LogInfof("Cluster object %s has been delted from karmada controlplane.", tag)
	return nil
}

func isNamespaceDeleted(c kubernetes.Interface, tag string) error {
	var err error
	err = wait.PollUntilContextTimeout(context.TODO(), WatchInterval, WatchTimeout, false,
		func(context.Context) (done bool, err error) {
			ns, err := c.CoreV1().Namespaces().Get(context.Background(), tag, metav1.GetOptions{})
			if err != nil {
				if errors.IsNotFound(err) {
					return true, nil
				}

				return false, nil
			}

			if len(ns.Spec.Finalizers) > 0 {
				zlog.LogInfof("get ns finalizer %s to be deleted.", ns.Spec.Finalizers[0])
			}

			if ns.Status.Phase == "Terminating" {
				ns.SetFinalizers(nil)
				newns, err := c.CoreV1().Namespaces().Update(context.Background(), ns, metav1.UpdateOptions{})
				if err != nil {
					zlog.LogErrorf("update ns %s finalizer failed : %v", tag, err)
					return false, err
				}
				zlog.LogInfof("update newns ns finalizer %d successfully", len(newns.Spec.Finalizers))
				return false, nil
			}

			return false, nil
		})
	if err != nil {
		zlog.LogErrorf("Error Deleting memcluster namespace %s : %v", tag, err)
		return err
	}

	return nil
}

// IsClusterUIDValid checks if a cluster identified by `tag` has a valid UID in the karmada control plane.
func IsClusterUIDValid(karmadaclient *versioned.Clientset, tag string) (bool, error) {
	clusters, err := karmadaclient.ClusterV1alpha1().Clusters().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada cluster list : %v", err)
		return false, err
	}

	for _, cluster := range clusters.Items {
		if cluster.Name == tag {
			zlog.LogInfof("cluster %s is valid", tag)
			return true, nil
		}
	}

	return false, nil
}

// IsClusterUIDUnique checks the uniqueness of a cluster UID using kubernetes client to fetch the UID,
// and verifies against the karmada control plane to ensure no duplicate cluster object exists.
func IsClusterUIDUnique(clusterclient kubernetes.Interface, karmadaclient *versioned.Clientset) (string, error) {
	ns, err := clusterclient.CoreV1().Namespaces().Get(context.TODO(), metav1.NamespaceSystem, metav1.GetOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving cluster default namespace: %v", err)
		return "", err
	}
	uid := string(ns.UID)

	clusters, err := karmadaclient.ClusterV1alpha1().Clusters().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada cluster list : %v", err)
		return "", err
	}

	for _, cluster := range clusters.Items {
		if cluster.Spec.ID == uid {
			zlog.Warnf("Warning Registering the cluster %s already exists ", cluster.GetName())
			return "", nil
		}
	}

	return uid, nil
}

// IsClusterNameRepeated check the uniqueness of the registering cluster object's name.
func IsClusterNameRepeated(name string, karmadaclient *versioned.Clientset) (bool, error) {
	clusters, err := karmadaclient.ClusterV1alpha1().Clusters().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada cluster list : %v", err)
		return false, err
	}

	for _, cluster := range clusters.Items {
		if cluster.Name == name {
			zlog.Warnf("Warning Registering the cluster %s name repeated ", cluster.GetName())
			return true, nil
		}
	}

	return false, nil
}

// IsNameStringValid checks if the provided string meets the metadata naming conventions.
func IsNameStringValid(name string) bool {
	return isMetaNameStringValid(name)
}

// IsLabelsValid checks if the provided string meets the metadata naming conventions.
func IsLabelsValid(labels map[string]string) bool {
	for key, values := range labels {
		if !isMetaLabelStringValid(key) || !isMetaLabelStringValid(values) {
			return false
		}
	}

	return true
}

func isMetaLabelStringValid(str string) bool {
	if len(str) > MaxLabelLength {
		return false
	}
	pattern := MetadataLabelPattern
	matched, err := regexp.MatchString(pattern, str)
	if err != nil {
		zlog.LogErrorf("Error matching label against pattern: %v", err)
		return false
	}
	return matched
}

func isMetaNameStringValid(str string) bool {
	if len(str) > MaxNameLength {
		return false
	}
	pattern := MetadataNamePattern
	matched, err := regexp.MatchString(pattern, str)
	if err != nil {
		zlog.LogErrorf("Error matching name against pattern: %v", err)
		return false
	}
	return matched
}

func isKarmadaAPIServiceReady(url string) error {
	ctx, cancel := context.WithTimeout(context.Background(), watchKarmadaServiceTimeout)
	defer cancel()

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}
	c := &http.Client{Transport: tr}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			resp, err := c.Get(url)
			if err != nil {
				zlog.Warnf("Karmada apiserver need to ready %v: wait for service to become ready", err)
				time.Sleep(watchKarmadaServiceInterval)
				continue
			}
			defer resp.Body.Close()

			if resp.StatusCode == http.StatusOK {
				return nil
			}

			zlog.Warnf("Karmada service is not ready, status code:", resp.StatusCode)
			time.Sleep(watchKarmadaServiceInterval)
		}
	}

}
