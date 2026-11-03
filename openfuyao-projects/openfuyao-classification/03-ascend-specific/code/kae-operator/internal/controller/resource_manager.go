/**
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
**/

package controller

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/client-go/kubernetes/scheme"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/constants"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

type assetsFromFile []byte

// Resources indicates resources managed by KAE operator
type Resources struct {
	ConfigMaps         []corev1.ConfigMap
	ClusterRoleBinding rbacv1.ClusterRoleBinding
	ClusterRole        rbacv1.ClusterRole
	DaemonSet          appsv1.DaemonSet
	Deployment         appsv1.Deployment
	RoleBinding        rbacv1.RoleBinding
	Role               rbacv1.Role
	ServiceAccount     corev1.ServiceAccount
	Service            corev1.Service
}

// filePathWalkDir 遍历根目录下所有文件，返回文件列表
func filePathWalkDir(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			zlog.Warnf("error in filepath.Walk on %s: %v", root, err)
			return nil
		}
		if !d.IsDir() {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func getAssetsFrom(path string) ([]assetsFromFile, error) {
	var manifests []assetsFromFile
	files, err := filePathWalkDir(path)
	if err != nil {
		return manifests, err
	}

	sort.Strings(files)
	for _, file := range files {
		buffer, err := os.ReadFile(file)
		if err != nil {
			return manifests, err
		}
		manifests = append(manifests, buffer)
	}
	return manifests, nil
}

func getResourcesAndHandlers(path string) (Resources, resourceHandler, error) {
	zlog.Infof("Getting assets from path: %s", path)
	res := Resources{}
	ctrl := resourceHandler{}
	manifests, err := getAssetsFrom(path)
	if err != nil {
		return res, ctrl, err
	}
	s := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme.Scheme, scheme.Scheme,
		json.SerializerOptions{Yaml: true, Pretty: false, Strict: false})
	for _, m := range manifests {
		kind := extractKindFromManifest(m)
		err := handleResource(kind, m, &res, &ctrl, s)
		if err != nil {
			return res, ctrl, err
		}
	}
	return res, ctrl, nil
}

func extractKindFromManifest(manifest []byte) string {
	reg := regexp.MustCompile(`\b(\w*kind:\w*)\B.*\b`)
	kind := reg.FindString(string(manifest))
	slice := strings.Split(kind, ":")
	if len(slice) > 1 {
		return strings.TrimSpace(slice[1])
	}
	zlog.Warnf("clould not extract Kind From Manifest: %s, KindInfo: %s", manifest, kind)
	return kind
}

func handleResource(kind string, manifest []byte, res *Resources, ctrl *resourceHandler, s *json.Serializer) error {
	zlog.Infof("Looking for Kind: %s", kind)
	switch kind {
	case constants.ServiceAccount:
		return decodeAndAppendHandler(manifest, &res.ServiceAccount, s, ctrl, HandlerServiceAccount)
	case constants.Role:
		return decodeAndAppendHandler(manifest, &res.Role, s, ctrl, HandlerRole)
	case constants.RoleBinding:
		return decodeAndAppendHandler(manifest, &res.RoleBinding, s, ctrl, HandlerRoleBinding)
	case constants.ClusterRole:
		return decodeAndAppendHandler(manifest, &res.ClusterRole, s, ctrl, HandlerClusterRole)
	case constants.ClusterRoleBinding:
		return decodeAndAppendHandler(manifest, &res.ClusterRoleBinding, s, ctrl, HandlerClusterRoleBinding)
	case constants.ConfigMap:
		cm := corev1.ConfigMap{}
		if err := decodeAndAppendHandler(manifest, &cm, s, ctrl, HandlerConfigMaps); err != nil {
			return err
		}
		res.ConfigMaps = append(res.ConfigMaps, cm)
		return nil
	case constants.DaemonSet:
		return decodeAndAppendHandler(manifest, &res.DaemonSet, s, ctrl, HandlerDaemonSet)
	case constants.Deployment:
		return decodeAndAppendHandler(manifest, &res.Deployment, s, ctrl, HandlerDeployment)
	case constants.Service:
		return decodeAndAppendHandler(manifest, &res.Service, s, ctrl, HandlerService)
	default:
		zlog.Warnf("Unknown Resource, Manifest: %s, Kind: %s", manifest, kind)
		return nil
	}
}

func decodeAndAppendHandler(manifest []byte, obj runtime.Object, s *json.Serializer, ctrl *resourceHandler,
	handler func(manager KAEPolicyManager) (kaev1.State, error)) error {
	_, _, err := s.Decode(manifest, nil, obj)
	if err != nil {
		return err
	}
	*ctrl = append(*ctrl, handler)
	return nil
}
