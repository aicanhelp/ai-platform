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
	"fmt"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

func createClusterServiceAcc(c kubernetes.Interface, labels map[string]string,
	tag string) (*corev1.ServiceAccount, string, error) {
	var err error
	serviceAccountName := DefaultSAPrefix + "-" + tag
	clusterSAObj := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: DefaultKarmadaNamespace,
			Name:      serviceAccountName,
			Labels:    labels,
		},
	}
	if clusterSAObj, err = checkServiceAccount(c, clusterSAObj); err != nil {
		zlog.LogErrorf("Error checking cluster serviceaccount : %v", err)
		return clusterSAObj, serviceAccountName, err
	}
	return clusterSAObj, serviceAccountName, nil
}

func checkCreateClusterRole(c kubernetes.Interface, serviceAccountName string,
	labels map[string]string) (*rbacv1.ClusterRole, error) {
	var err error
	clusterRoleName := fmt.Sprint(DefaultClusterRolePrefix + ":" + serviceAccountName)
	clusterRoleObj := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:   clusterRoleName,
			Labels: labels,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"*"},
				Resources: []string{"*"},
				Verbs:     []string{"*"},
			},
			{
				NonResourceURLs: []string{"*"},
				Verbs:           []string{"*"},
			},
		},
	}

	if clusterRoleObj, err = checkClusterRole(c, clusterRoleObj); err != nil {
		zlog.LogErrorf("Error checking cluster clusterrole : %v", err)
		return clusterRoleObj, err
	}
	return clusterRoleObj, nil
}

func checkCreateClusterRoleBinding(c kubernetes.Interface, clusterRoleObj *rbacv1.ClusterRole,
	labels map[string]string, clusterSAObj *corev1.ServiceAccount) (*rbacv1.ClusterRoleBinding, error) {
	var err error
	clusterRolebindingObj := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:   clusterRoleObj.Name,
			Labels: labels,
		},
		Subjects: buildRolebindingSubjects("", clusterSAObj.Name, clusterSAObj.Namespace),
		RoleRef:  builderRolebindingReference(clusterRoleObj.Name),
	}
	if clusterRolebindingObj, err = checkCluterRoleBinding(c, clusterRolebindingObj); err != nil {
		zlog.LogErrorf("Error checking cluster clusterrolebinding : %v", err)
		return clusterRolebindingObj, err
	}
	return clusterRolebindingObj, nil
}

func checkCreateImpersonator(c kubernetes.Interface, labels map[string]string) (*corev1.ServiceAccount, error) {
	var err error
	imServiceAccountName := DefaultSAPrefix + "-" + DefaultImpersonatorPrefix
	impersonatorSAObj := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: DefaultKarmadaNamespace,
			Name:      imServiceAccountName,
			Labels:    labels,
		},
	}

	if impersonatorSAObj, err = checkServiceAccount(c, impersonatorSAObj); err != nil {
		zlog.LogErrorf("Error checking impersonator serviceaccount : %v", err)
		return impersonatorSAObj, err
	}
	return impersonatorSAObj, nil
}

// GenerateClusterSecret creates Kubernetes secrets for secvice accounts and impersonator SA
// in a member cluster identified by `tag`.
// It binds the cluster SA with a full-permission clusterrole, and returns the secrets containing
// tokens for both service accounts.
func GenerateClusterSecret(c kubernetes.Interface, tag string) (*corev1.Secret, *corev1.Secret, error) {
	var err error
	var serviceAccountName string
	var clusterSAObj *corev1.ServiceAccount
	var clusterRoleObj *rbacv1.ClusterRole
	var impersonatorSAObj *corev1.ServiceAccount
	labels := map[string]string{
		KarmadaSystemLabel:      KaramadaSystemLabelValue,
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
	}

	// create namespace
	if _, err := checkNamespace(c, labels); err != nil {
		zlog.LogErrorf("Error checking namespace : %v", err)
		return nil, nil, err
	}

	if clusterSAObj, serviceAccountName, err = createClusterServiceAcc(c, labels, tag); err != nil {
		return nil, nil, err
	}

	if clusterRoleObj, err = checkCreateClusterRole(c, serviceAccountName, labels); err != nil {
		return nil, nil, err
	}

	if _, err = checkCreateClusterRoleBinding(c, clusterRoleObj, labels, clusterSAObj); err != nil {
		return nil, nil, err
	}

	// create impersonator serviceaccount
	if impersonatorSAObj, err = checkCreateImpersonator(c, labels); err != nil {
		return nil, nil, err
	}

	// create cluster sa secret
	var clusterSASecret *corev1.Secret
	var impersonatorSASecret *corev1.Secret
	clusterSASecret, err = checkSecret(c, clusterSAObj)
	if err != nil {
		zlog.LogErrorf("Error checking cluster secret : %v", err)
		return nil, nil, err
	}

	impersonatorSASecret, err = checkSecret(c, impersonatorSAObj)
	if err != nil {
		zlog.LogErrorf("Error checking impersonator secret : %v", err)
		return nil, nil, err
	}

	return clusterSASecret, impersonatorSASecret, nil
}

// GeneratePlatformAuthPolicy syncs the authorization policy for the specified user based on operations.
func GeneratePlatformAuthPolicy(c kubernetes.Interface, username string, operation string, isAdmin bool) error {
	var err error

	clusterroleName := "platform-admin"
	if !isAdmin {
		clusterroleName = "platform-regular"
	}

	crbName := fmt.Sprintf("%s-%s", username, clusterroleName)
	openFuyaoClusterRole := fmt.Sprintf("%s-%s", "openfuyao", clusterroleName)

	if operation == "unjoin" {
		err := c.RbacV1().ClusterRoleBindings().Delete(context.TODO(), crbName, metav1.DeleteOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				zlog.LogErrorf("Error Deleting clusterrolebinding : %v", err)
			}
			zlog.Warnf("Member Cluster has no clusterrolebinding %s.", crbName)
		}
		return err
	}

	if operation != "join" {
		return fmt.Errorf("unSupported Operation %s : Access Denied", operation)
	}

	// join: creat new clusterrolebinding
	labels := map[string]string{
		CreatedByopenFuyaoLabel:      CreatedByopenFuyaoLabelValue,
		DefaultopenFuyaoRoleRefLabel: clusterroleName,
		DefaultopenFuyaoUserRefLabel: username,
		DefaultopenFuyaoRoleType:     "platform-role",
	}

	clusterRolebindingObj := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:   crbName,
			Labels: labels,
		},
		Subjects: buildRolebindingSubjects(username, "", ""),
		RoleRef:  builderRolebindingReference(openFuyaoClusterRole),
	}
	if clusterRolebindingObj, err = checkCluterRoleBinding(c, clusterRolebindingObj); err != nil {
		zlog.LogErrorf("Error checking cluster clusterrolebinding : %v", err)
		return err
	}

	return nil
}

// DeleteClusterResources removes all resources created within a member cluster identified by `tag`.
// It uses the provided Kubernetes client to perform the deletions and returns and error if any occur
// during the process.
func DeleteClusterResources(c kubernetes.Interface, tag string) error {
	var err error
	// delete clusterrolebinding
	err = deleteClusterRoleBinding(c, tag)
	if err != nil {
		return err
	}
	err = deleteClusterRoleBinding(c, "")
	if err != nil {
		return err
	}

	// delete clusterrole
	err = deleteClusterRole(c, tag)
	if err != nil {
		return err
	}
	err = deleteClusterRole(c, "")
	if err != nil {
		return err
	}

	// delete serviceaccount
	err = deleteServiceAccount(c, tag)
	if err != nil {
		return err
	}

	return nil
}

func checkSecret(c kubernetes.Interface, saObj *corev1.ServiceAccount) (*corev1.Secret, error) {
	var secretObj *corev1.Secret
	err := wait.PollUntilContextTimeout(context.TODO(), WatchInterval, WatchTimeout, false,
		func(context.Context) (bool, error) {
			exist, err := isServiceAccountExist(c, saObj.Name, saObj.Namespace)
			if !exist {
				return false, err
			}
			secretObj, exist, err = isSecretExist(c, saObj.Name, saObj.Namespace)
			if !exist && err != nil {
				return false, err
			}

			if !exist {
				secretObj, err = createSecret(c, saObj.Name, saObj.Namespace)
				if err != nil {
					return false, err
				}
			}

			if secretObj == nil {
				return false, fmt.Errorf("secret found/created failed")
			}

			if _, ok := secretObj.Data["token"]; !ok {
				return false, nil
			}

			return true, nil
		})
	if err != nil {
		return nil, err
	}

	if secretObj == nil {
		return nil, fmt.Errorf("secret found/created failed")
	}

	zlog.LogInfof("Successfully Creating Secret %s.", secretObj.Name)
	return secretObj, nil
}

func checkCluterRoleBinding(c kubernetes.Interface, crbinding *rbacv1.ClusterRoleBinding) (
	*rbacv1.ClusterRoleBinding, error) {
	// check if clusterrolebinding exists
	ok, err := isClusterRoleBindingExist(c, crbinding.Name)
	if err != nil {
		zlog.LogErrorf("Error checking clusterrolebinding if exists: %v", err)
		return nil, err
	}
	if ok {
		return crbinding, nil
	}

	// create clusterrolebinding
	crbindingObj, err := creareClusterRoleBinding(c, crbinding)
	if err != nil {
		return nil, err
	}

	zlog.LogInfof("Successfully Creating ClusterRoleBinding %s.", crbindingObj.Name)
	return crbindingObj, nil
}

func checkClusterRole(c kubernetes.Interface, clusterrole *rbacv1.ClusterRole) (*rbacv1.ClusterRole, error) {
	// check if clusterrole exists
	ok, err := isClusterRoleExist(c, clusterrole.Name)
	if err != nil {
		zlog.LogErrorf("Error checking clusterrole if exists: %v", err)
		return nil, err
	}
	if ok {
		return clusterrole, nil
	}

	// create clusterrole
	crObj, err := createClusterRole(c, clusterrole)
	if err != nil {
		return nil, err
	}

	zlog.LogInfof("Successfully Creating ClusterRole %s.", crObj.Name)
	return crObj, nil
}

func checkServiceAccount(c kubernetes.Interface, sa *corev1.ServiceAccount) (*corev1.ServiceAccount, error) {
	// check if serviceaccount exists
	ok, err := isServiceAccountExist(c, sa.Name, sa.Namespace)
	if err != nil {
		zlog.LogErrorf("Error checking serviceaccount if exists: %v", err)
		return nil, err
	}
	if ok {
		return sa, nil
	}

	// create servceiaccount
	saObj, err := createServiceAccount(c, sa)
	if err != nil {
		return nil, err
	}

	zlog.LogInfof("Successfully Creating ServiceAccount %s.", saObj.Name)
	return saObj, nil
}

func checkNamespace(c kubernetes.Interface, labels map[string]string) (*corev1.Namespace, error) {
	// check if namespace exists
	ok, err := isNamespaceExist(c, DefaultKarmadaNamespace)
	if err != nil {
		zlog.LogErrorf("Error checking namespace if exists: %v", err)
		return nil, err
	}
	if ok {
		return nil, nil
	}

	// create namespace
	nsObj := &corev1.Namespace{}
	nsObj.Name = DefaultKarmadaNamespace
	nsObj.Labels = labels
	if err = createNamespace(c, nsObj); err != nil {
		return nil, err
	}

	zlog.LogInfof("Successfully Creating Namespace %s.", DefaultKarmadaNamespace)
	return nsObj, nil

}

func isClusterRoleBindingExist(c kubernetes.Interface, crbindingName string) (bool, error) {
	_, err := c.RbacV1().ClusterRoleBindings().Get(context.TODO(), crbindingName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.LogInfof("NotFound ClusterRolebinding")
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func isClusterRoleExist(c kubernetes.Interface, clusterroleName string) (bool, error) {
	_, err := c.RbacV1().ClusterRoles().Get(context.TODO(), clusterroleName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.LogInfof("NotFound ClusterRole")
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func isNamespaceExist(c kubernetes.Interface, namespaceName string) (bool, error) {
	_, err := c.CoreV1().Namespaces().Get(context.Background(), namespaceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.LogInfof("NotFound Namespace")
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func isSecretExist(c kubernetes.Interface, saName string, saNamespace string) (*corev1.Secret, bool, error) {
	secretObj, err := c.CoreV1().Secrets(saNamespace).Get(context.Background(), saName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.LogInfof("NotFound Secret")
			return nil, false, nil
		}
		return nil, false, err
	}

	return secretObj, true, nil
}

func isServiceAccountExist(c kubernetes.Interface, saName string, saNamespace string) (bool, error) {
	_, err := c.CoreV1().ServiceAccounts(saNamespace).Get(context.Background(), saName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.LogInfof("NotFound ServiceAccount")
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func creareClusterRoleBinding(c kubernetes.Interface, crbindingObj *rbacv1.ClusterRoleBinding) (
	*rbacv1.ClusterRoleBinding, error) {
	createdObj, err := c.RbacV1().ClusterRoleBindings().Create(context.TODO(), crbindingObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating clusterrolebinding %s", crbindingObj.Name, ", already exist !")
			return crbindingObj, nil
		}

		zlog.LogErrorf("Error Creating clusterrolebinding: %v", err)
		return nil, err
	}

	return createdObj, nil

}

func createClusterRole(c kubernetes.Interface, crObj *rbacv1.ClusterRole) (*rbacv1.ClusterRole, error) {
	clusterRoleObj, err := c.RbacV1().ClusterRoles().Create(context.TODO(), crObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating clusterrole %s", crObj.Name, ", already exist !")
			return clusterRoleObj, nil
		}

		zlog.LogErrorf("Error Creating clusterrole: %v", err)
		return nil, err
	}

	return clusterRoleObj, nil
}

func createNamespace(c kubernetes.Interface, nsObj *corev1.Namespace) error {
	_, err := c.CoreV1().Namespaces().Create(context.TODO(), nsObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating namespace %s", nsObj.Name, ", already exist !")
			return nil
		}

		zlog.LogErrorf("Error Creating namespace: %v", err)
		return err
	}

	return nil
}

func createSecret(c kubernetes.Interface, saName string, saNamespace string) (*corev1.Secret, error) {
	secretObj := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: saNamespace,
			Annotations: map[string]string{
				corev1.ServiceAccountNameKey: saName,
			},
		},
		Type: corev1.SecretTypeServiceAccountToken,
	}

	st, err := c.CoreV1().Secrets(secretObj.Namespace).Create(context.TODO(), secretObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating secret %s", secretObj.Name, ", already exist !")
			return c.CoreV1().Secrets(secretObj.Namespace).Get(context.TODO(), secretObj.Name, metav1.GetOptions{})
		}

		zlog.LogErrorf("Error Creating secret: %v", err)
		return nil, err
	}

	return st, nil

}

func createServiceAccount(c kubernetes.Interface, saObj *corev1.ServiceAccount) (*corev1.ServiceAccount, error) {
	_, err := c.CoreV1().ServiceAccounts(saObj.Namespace).Create(context.TODO(), saObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating serviceaccount %s", saObj.Name, ", already exist !")
			return saObj, nil
		}

		zlog.LogErrorf("Error Creating serviceaccount: %v", err)
		return nil, err
	}

	return saObj, nil
}

func deleteNamespace(c kubernetes.Interface) error {
	nsName := DefaultKarmadaNamespace
	err := c.CoreV1().Namespaces().Delete(context.Background(), nsName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting namespace : %v", err)
			return err
		}
	}

	err = isNamespaceDeleted(c, nsName)
	if err != nil {
		if err.Error() == "context deadline exceeded" {
			zlog.Warnf("Warning deleting namespace : %v", err)
			return nil
		}
		zlog.LogErrorf("Error Deleting namespace : %v", err)
		return err
	}

	return nil
}

func deleteClusterRoleBinding(c kubernetes.Interface, tag string) error {
	var clusterRoleName string
	if tag != "" {
		serviceAccountName := DefaultSAPrefix + "-" + tag
		clusterRoleName = DefaultClusterRolePrefix + ":" + serviceAccountName
	} else {
		clusterRoleName = DefaultSAPrefix + "-" + DefaultImpersonatorPrefix
	}
	err := c.RbacV1().ClusterRoleBindings().Delete(context.TODO(), clusterRoleName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting clusterrolebinding : %v", err)
			return err
		}
	}
	return nil
}

func deleteClusterRole(c kubernetes.Interface, tag string) error {
	var clusterRoleBindingName string
	if tag != "" {
		serviceAccountName := DefaultSAPrefix + "-" + tag
		clusterRoleBindingName = fmt.Sprint(DefaultClusterRolePrefix + ":" + serviceAccountName)
	} else {
		clusterRoleBindingName = DefaultSAPrefix + "-" + DefaultImpersonatorPrefix
	}
	err := c.RbacV1().ClusterRoles().Delete(context.TODO(), clusterRoleBindingName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting clusterrole : %v", err)
			return err
		}
	}
	return nil
}

func deleteServiceAccount(c kubernetes.Interface, tag string) error {
	serviceAccountName := DefaultSAPrefix + "-" + tag
	err := c.CoreV1().ServiceAccounts(DefaultKarmadaNamespace).
		Delete(context.Background(), serviceAccountName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting serviceaccount : %v", err)
			return err
		}
	}
	return nil
}
