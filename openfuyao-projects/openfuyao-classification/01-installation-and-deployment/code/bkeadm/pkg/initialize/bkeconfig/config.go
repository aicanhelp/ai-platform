/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package bkeconfig

import (
	"bytes"
	"context"
	"fmt"
	"os"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func ensureNsExists(namespace string) error {
	if global.K8s == nil {
		var err error
		global.K8s, err = k8s.NewKubernetesClient("")
		if err != nil {
			return err
		}
	}
	client := global.K8s.GetClient()

	_, err := client.CoreV1().Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{})
	if err == nil {
		return nil
	}

	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("check namespace %s failed: %v", namespace, err)
	}

	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	}

	_, err = client.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}

	return fmt.Errorf("create namespace %s failed: %v", namespace, err)
}

func SetKubernetesConfig(data map[string]string, name, ns string) error {
	var err error
	if global.K8s == nil {
		global.K8s, err = k8s.NewKubernetesClient("")
		if err != nil {
			return err
		}
	}
	client := global.K8s.GetClient()

	// k8s configmap
	cm, err := client.CoreV1().ConfigMaps(ns).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			cm = &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: ns,
				},
				Data: data,
			}
			_, err = client.CoreV1().ConfigMaps(ns).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				return err
			}
		} else {
			return err
		}
	}
	return nil
}

func SetPatchConfig(version, yamlFilePath, cfgName string) error {
	yamlData, err := os.ReadFile(yamlFilePath)
	if err != nil {
		return err
	}

	cleanData := bytes.ReplaceAll(yamlData, []byte("\r\n"), []byte("\n"))
	cleanData = bytes.ReplaceAll(cleanData, []byte("\r"), []byte("\n"))

	data := map[string]string{
		version: string(cleanData),
	}

	if err = ensureNsExists(utils.PatchNameSpace); err != nil {
		return fmt.Errorf("failed to ensure ns %s exists: %w", utils.PatchNameSpace, err)
	}

	return SetKubernetesConfig(data, cfgName, utils.PatchNameSpace)
}
