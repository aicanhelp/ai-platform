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

package filters

import (
	"context"
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"github.com/pkg/errors"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/utils/util"
	"application-management-service/pkg/zlog"
)

var platformRoles = []string{"platform-admin", "platform-regular"}
var clusterRoles = []string{"cluster-admin", "cluster-viewer", "cluster-editor"}

// AuthorizeByAdmittedRoles authorizes the user by its corresponding clusterrole,
// only check when chart name is multi-cluster-service
func AuthorizeByAdmittedRoles(restfulRequest *restful.Request,
	admittedRoles []string, k8sClient kubernetes.Interface) bool {
	req := restfulRequest.Request
	zlog.Infof("authorize request: %s", req.RequestURI)

	// fetch userinfo
	userinfo, ok := req.Context().Value(constant.UserKey).(user.Info)
	if !ok {
		zlog.Error("cannot get userinfo from req.context")
		return false
	}

	// fetch clusterrolebinding
	username := userinfo.GetName()
	crbs, err := getOpenFuyaoClusterRoleBindings(username, k8sClient, req)
	if err != nil {
		zlog.Errorf("cannot get crb for %s", userinfo.GetName())
		return false
	}

	// check its platform admin
	for _, crb := range crbs {
		zlog.Infof("crb: %s with label: %v", crb.Name, crb.Labels)
		role := getOpenFuyaoClusterRoleType(crb)
		zlog.Infof("get matched clusterrole/platformrole type from CRB: %s", role)
		if openFuyaoRoleContains(admittedRoles, role) {
			zlog.Infof("get openfuyao cluterrole/platformrole type from %s", crb.Name)
			return true
		}
	}

	zlog.Errorf("cannot get openfuyao clusterrole or platformrole")
	return false
}

func getOpenFuyaoClusterRoleBindings(userName string, k8sClient kubernetes.Interface,
	req *http.Request) ([]*rbacv1.ClusterRoleBinding, error) {
	var crbList rbacv1.ClusterRoleBindingList

	crbListPointer, err := k8sClient.RbacV1().ClusterRoleBindings().List(context.Background(),
		v1.ListOptions{LabelSelector: constant.OpenFuyaoRoleLabel})
	crbList = *crbListPointer
	if err != nil {
		zlog.Errorf("cannot list clusterrolebinding, err: %v", err)
		return nil, err

	}

	crbs := make([]*rbacv1.ClusterRoleBinding, 0)
	for _, crb := range crbList.Items {
		for _, subj := range crb.Subjects {
			if subj.Name == userName {
				zlog.Infof("find clusterrolebinding %s", crb.Name)
				crbs = append(crbs, crb.DeepCopy())
			}
		}
	}

	if len(crbs) == 0 {
		errString := fmt.Sprintf("no matched clusterrolebinding for user %s", userName)
		zlog.Error(errString)
		return nil, errors.New(errString)
	}

	return crbs, nil
}

func getOpenFuyaoClusterRoleType(crb *rbacv1.ClusterRoleBinding) string {
	// get role-type value
	roleType := crb.Labels[constant.OpenFuyaoRoleLabel]
	roleName := crb.RoleRef.Name
	if roleType == "platform-role" {
		if openFuyaoRoleContains(platformRoles, roleName) {
			return roleName
		}
	} else if roleType == "cluster-role" {
		if openFuyaoRoleContains(clusterRoles, roleName) {
			return roleName
		}
	} else {
		return ""
	}
	return ""
}

func openFuyaoRoleContains(list []string, query string) bool {
	for _, item := range list {
		if util.TrimOpenFuyaoRolePrefix(query) == item {
			return true
		}
	}
	return false
}
