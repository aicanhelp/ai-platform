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
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// GenerateKarmadaSecret retrieves tokens and CAs from member cluster secrets specified by `tag`
// and stores them in the Karmada control plane.
// Returns the cluster and impersonator secret created by karamada control plane.
func GenerateKarmadaSecret(c kubernetes.Interface, clusterSt *corev1.Secret, impersonatorSt *corev1.Secret,
	tag string) (*corev1.Secret, *corev1.Secret, error) {
	var err error
	labels := map[string]string{
		KarmadaSystemLabel:      KaramadaSystemLabelValue,
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
	}

	// create namespace
	if _, err := checkNamespace(c, labels); err != nil {
		zlog.LogErrorf("Error checking namespace : %v", err)
		return nil, nil, err
	}

	clusterSecretObjName := tag
	clusterSecretObj := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      clusterSecretObjName,
			Namespace: DefaultKarmadaNamespace,
			Labels:    labels,
		},
		Data: map[string][]byte{
			TLSCAData:    clusterSt.Data["ca.crt"],
			TLSTokenData: clusterSt.Data[TLSTokenData],
		},
	}

	clusterSecretObj, err = createKarmadaSecret(c, clusterSecretObj)
	if err != nil {
		return nil, nil, err
	}

	impersonatorSecretObjName := tag + "-" + DefaultImpersonatorPrefix
	impersonatorSecretObj := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      impersonatorSecretObjName,
			Namespace: DefaultKarmadaNamespace,
			Labels:    labels,
		},
		Data: map[string][]byte{
			TLSTokenData: impersonatorSt.Data[TLSTokenData],
		},
	}

	impersonatorSecretObj, err = createKarmadaSecret(c, impersonatorSecretObj)
	if err != nil {
		return nil, nil, err
	}

	return clusterSecretObj, impersonatorSecretObj, nil
}

// DeleteKarmadaResources removes secrets created within cluster object identified by `tag`.
func DeleteKarmadaResources(c kubernetes.Interface, tag string) error {
	var err error
	clusterSecretObjName := tag
	err = deleteKarmadaSecret(c, clusterSecretObjName)
	if err != nil {
		return err
	}

	impersonatorSecretObjName := tag + "-" + DefaultImpersonatorPrefix
	err = deleteKarmadaSecret(c, impersonatorSecretObjName)
	if err != nil {
		return err
	}

	err = updateKarmadaClusterrole(c, tag)
	if err != nil {
		return err
	}

	return nil
}

func deleteKarmadaSecret(c kubernetes.Interface, tag string) error {
	err := c.CoreV1().Secrets(DefaultKarmadaNamespace).Delete(context.Background(), tag, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting Karmada Secret : %v", err)
			return err
		}
	}
	return nil
}

func createKarmadaSecret(c kubernetes.Interface, stObj *corev1.Secret) (*corev1.Secret, error) {
	st, err := c.CoreV1().Secrets(stObj.Namespace).Create(context.TODO(), stObj, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating karmada cluster secret %s already exist !", stObj.Name)
			return c.CoreV1().Secrets(stObj.Namespace).Get(context.TODO(), stObj.Name, metav1.GetOptions{})
		}

		zlog.LogErrorf("Error Creating karmada cluster secret: %v", err)
		return nil, err
	}

	return st, nil
}

// CheckUserAccessList checks for user-specific access rights and returns a slice of strings
// representing the cluste resources the user can access.
func CheckUserAccessList(c kubernetes.Interface, name string) ([]string, error) {
	defaultLabels := fmt.Sprintf("%s,%s", CreatedByopenFuyaoLabel, DefaultopenFuyaoLabel)

	listOptions := metav1.ListOptions{
		LabelSelector: defaultLabels,
	}

	cr, err := c.RbacV1().ClusterRoles().List(context.TODO(), listOptions)
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada clusterrole list : %v", err)
		return nil, err
	}

	var memList []string
	for _, crObj := range cr.Items {
		arrayName := strings.Split(crObj.Name, ":")
		username := arrayName[len(arrayName)-1]
		if username != name {
			continue
		}
		if crObj.Labels[DefaultopenFuyaoLabel] == DefaultPlatformRoleLabel {
			memList = append(memList, "*")
			return memList, nil
		}

		for _, ruleObj := range crObj.Rules {
			if hasString(ruleObj.Resources, "clusters/proxy") ||
				hasString(ruleObj.Resources, "cluster") {
				memList = ruleObj.ResourceNames
				return memList, nil
			}
		}
	}

	return nil, nil
}

// GetClusterRoleList returns clusterRole objects with specific platform and cluster labels.
func GetClusterRoleList(c kubernetes.Interface) (map[string]*rbacv1.ClusterRoleList, error) {
	defaultLabels := CreatedByopenFuyaoLabel
	listOptions := metav1.ListOptions{
		LabelSelector: defaultLabels,
	}

	cr, err := c.RbacV1().ClusterRoles().List(context.TODO(), listOptions)
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada clusterrole list : %v", err)
		return nil, err
	}

	crList := make(map[string]*rbacv1.ClusterRoleList)
	crList["platform"] = &rbacv1.ClusterRoleList{}
	crList["cluster"] = &rbacv1.ClusterRoleList{}

	for _, crObj := range cr.Items {
		resourceType, err := getResourceType(crObj)
		if err != nil {
			continue
		}
		crList[resourceType].Items = append(crList[resourceType].Items, crObj)
	}

	return crList, nil
}

func getResourceType(crObj rbacv1.ClusterRole) (string, error) {
	if len(crObj.Labels) > 1 {
		if crObj.Labels[DefaultopenFuyaoLabel] == DefaultPlatformRoleLabel {
			return "platform", nil
		}
		if crObj.Labels[DefaultopenFuyaoLabel] == DefaultMemCusterRoleLabel {
			return "cluster", nil
		}
	}

	return "", fmt.Errorf("not Found resource cluster or clusters/proxy")
}

func hasString(slice []string, str string) bool {
	for _, elem := range slice {
		if elem == str {
			return true
		}
	}
	return false
}

func removeString(slice []string, str string) []string {
	var res []string
	for _, elem := range slice {
		if elem != str {
			res = append(res, elem)
		}
	}
	return res
}

func compareSlice(slice1, slice2 []string) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	sort.Strings(slice1)
	sort.Strings(slice2)
	for index := range slice1 {
		if slice1[index] != slice2[index] {
			return false
		}
	}
	return true
}

// UpdateClusterRole updates the role assignments based on user list.
func UpdateClusterRole(c kubernetes.Interface, crList *rbacv1.ClusterRoleList, userInfo map[string][]string) error {
	var err error
	var userList []string
	for _, crObj := range crList.Items {
		arrayName := strings.Split(crObj.Name, ":")
		username := arrayName[len(arrayName)-1]
		if memList, ok := userInfo[username]; ok {
			if err = checkMemberCluster(c, crObj, memList, ""); err != nil {
				return err
			}
			userList = append(userList, username)
		} else {
			// delete clusterrole
			if err = deleteClusterRoleResource(c, crObj.Name); err != nil {
				return err
			}
		}
	}

	for key, values := range userInfo {
		if hasString(userList, key) {
			continue
		}
		if values == nil {
			continue
		}
		if err = createClusterProxyResource(c, key, values); err != nil {
			return err
		}
	}

	return nil
}

func updateKarmadaClusterrole(c kubernetes.Interface, clustertag string) error {
	labels := map[string]string{
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
		DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
	}
	labelsSelector := metav1.FormatLabelSelector(&metav1.LabelSelector{MatchLabels: labels})
	listOptions := metav1.ListOptions{
		LabelSelector: labelsSelector,
	}

	cr, err := c.RbacV1().ClusterRoles().List(context.TODO(), listOptions)
	if err != nil {
		zlog.LogErrorf("Error Retrieving karmada clusterrole list : %v", err)
		return err
	}

	for _, crObj := range cr.Items {
		if err := checkMemberCluster(c, crObj, nil, clustertag); err != nil {
			zlog.LogErrorf("Error Updating karmada clusterrole %s : %v", crObj.Name, err)
			return err
		}
	}

	return nil
}

func checkMemberCluster(c kubernetes.Interface, crObj rbacv1.ClusterRole, memList []string, clusterTag string) error {
	var err error
	resourceClusterProxyType := "clusters/proxy"
	var policyRule rbacv1.PolicyRule
	var index int
	for idx, ruleObj := range crObj.Rules {
		if hasString(ruleObj.Resources, resourceClusterProxyType) {
			policyRule = ruleObj
			index = idx
			break
		}
	}

	if len(policyRule.Resources) == 0 {
		return fmt.Errorf("error Retrieving ClusterRole Resource Type")
	}

	if len(memList) == 0 && clusterTag == "" {
		return deleteClusterRoleResource(c, crObj.Name)
	}

	if len(memList) != 0 && clusterTag != "" {
		return fmt.Errorf("error updating clusterrole resource clusters/proxy member cluster list")
	}

	if clusterTag != "" {
		if !hasString(policyRule.ResourceNames, clusterTag) {
			return nil
		}
		if len(policyRule.ResourceNames) == 1 && policyRule.ResourceNames[0] == clusterTag {
			return deleteClusterRoleResource(c, crObj.Name)
		}
		newMemList := removeString(policyRule.ResourceNames, clusterTag)
		crObj.Rules[index].ResourceNames = newMemList
	}

	if len(memList) != 0 {
		if compareSlice(policyRule.ResourceNames, memList) {
			return nil
		}
		crObj.Rules[index].ResourceNames = memList
	}

	_, err = c.RbacV1().ClusterRoles().Update(context.TODO(), &crObj, metav1.UpdateOptions{})
	if err != nil {
		zlog.LogErrorf("Error Updating clusterrole resourcename member list: %v", err)
		return err
	}

	zlog.LogInfof("Updating member list from %v to %v", policyRule.ResourceNames, crObj.Rules[index].ResourceNames)

	return nil
}

// UpdatePlatformRole updates the role assignments based on user list.
func UpdatePlatformRole(c kubernetes.Interface, crList *rbacv1.ClusterRoleList, userList []string) error {
	var err error
	var crUser []string
	for _, crObj := range crList.Items {
		arrayName := strings.Split(crObj.Name, ":")
		username := arrayName[len(arrayName)-1]
		if hasString(userList, username) {
			crUser = append(crUser, username)
			continue
		}
		if err = deleteClusterRoleResource(c, crObj.Name); err != nil {
			return err
		}
	}

	for _, item := range userList {
		if hasString(crUser, item) {
			continue
		}
		if err := creatClusterRoleResource(c, item); err != nil {
			return err
		}
	}

	return err
}

func createClusterProxyResource(c kubernetes.Interface, username string, memList []string) error {
	var err error
	clusterRoleName := DefaultUserClusterRolePrefix + username
	labels := map[string]string{
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
		DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
	}
	clusterRoleObj := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:   clusterRoleName,
			Labels: labels,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups:     []string{DefaultKarmdaClusterAPIGroup},
				Resources:     []string{"clusters/proxy"},
				ResourceNames: memList,
				Verbs:         []string{"*"},
			},
			{
				APIGroups:     []string{DefaultKarmdaClusterAPIGroup},
				Resources:     []string{"cluster"},
				ResourceNames: memList,
				Verbs:         []string{"get", "list"},
			},
		},
	}

	if _, err = checkClusterRole(c, clusterRoleObj); err != nil {
		zlog.LogErrorf("Error checking cluster clusterrole : %v", err)
		return err
	}

	// create clusterrolebinding
	if err = buildRolebindingResource(c, labels, clusterRoleName, username); err != nil {
		return err
	}

	zlog.LogInfof("Successfully Creating clusters/proxy : clusterrole and clusterrolebinding.")
	return nil
}

func creatClusterRoleResource(c kubernetes.Interface, username string) error {
	var err error
	var createError error
	clusterRoleName := DefaultUserPlatformRolePrefix + username
	labels := map[string]string{
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
		DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
	}
	clusterRoleObj := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:   clusterRoleName,
			Labels: labels,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{DefaultKarmdaClusterAPIGroup},
				Resources: []string{"cluster"},
				Verbs:     []string{"*"},
			},
			{
				APIGroups: []string{DefaultKarmdaClusterAPIGroup},
				Resources: []string{"clusters/proxy"},
				Verbs:     []string{"*"},
			},
		},
	}

	if _, createError = checkClusterRole(c, clusterRoleObj); createError != nil {
		zlog.LogErrorf("Error checking cluster clusterrole : %v", createError)
		return createError
	}

	// create clusterrolebinding
	if err = buildRolebindingResource(c, labels, clusterRoleName, username); err != nil {
		return err
	}

	zlog.LogInfof("Successfully Creating clusterrole: clusterrole and clusterrolebinding.")
	return nil
}

func buildRolebindingResource(c kubernetes.Interface, labels map[string]string, name string, username string) error {
	clusterRolebindingObj := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Subjects: buildRolebindingSubjects(username, "", ""),
		RoleRef:  builderRolebindingReference(name),
	}
	if _, err := checkCluterRoleBinding(c, clusterRolebindingObj); err != nil {
		zlog.LogErrorf("Error checking cluster clusterrolebinding : %v", err)
		return err
	}
	return nil
}

func buildRolebindingSubjects(username string, saName string, saNamespace string) []rbacv1.Subject {
	if username != "" {
		return []rbacv1.Subject{
			{
				Kind:     rbacv1.UserKind,
				Name:     username,
				APIGroup: rbacv1.GroupName,
			},
		}
	}

	return []rbacv1.Subject{
		{
			Kind:      rbacv1.ServiceAccountKind,
			Name:      saName,
			Namespace: saNamespace,
		},
	}
}

func builderRolebindingReference(roleName string) rbacv1.RoleRef {
	return rbacv1.RoleRef{
		APIGroup: rbacv1.GroupName,
		Kind:     "ClusterRole",
		Name:     roleName,
	}
}

func deleteClusterRoleResource(c kubernetes.Interface, objName string) error {
	var err error
	err = c.RbacV1().ClusterRoles().Delete(context.TODO(), objName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting clusterrole : %v", err)
			return err
		}
	}

	err = c.RbacV1().ClusterRoleBindings().Delete(context.TODO(), objName, metav1.DeleteOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			zlog.LogErrorf("Error Deleting clusterrolebinding : %v", err)
			return err
		}
	}
	return nil
}
