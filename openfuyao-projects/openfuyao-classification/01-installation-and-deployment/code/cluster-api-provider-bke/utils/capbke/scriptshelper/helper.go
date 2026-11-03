/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package scriptshelper

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func isScriptFile(filename string) bool {
	return strings.HasSuffix(filename, ".sh") || strings.HasSuffix(filename, ".py")
}

// CollectScriptFiles defines recursively scan directories for script files
func CollectScriptFiles(scriptsDir string) (map[string]string, error) {
	scripts := make(map[string]string)

	err := filepath.Walk(scriptsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		if isScriptFile(info.Name()) {
			scripts[info.Name()] = path
		}

		return nil
	})

	if err != nil {
		return nil, errors.Wrapf(err, "scan directories failed: %s", scriptsDir)
	}

	return scripts, nil
}

// CreateScriptsConfigMaps create or update ConfigMaps
func CreateScriptsConfigMaps(c client.Client) error {
	scripts, err := CollectScriptFiles(constant.K8sScriptsDir)
	if err != nil {
		return err
	}

	var errs []error

	for name, path := range scripts {
		if err = createOrUpdateScriptConfigMap(c, name, path); err != nil {
			errs = append(errs, err)
		}
	}

	return kerrors.NewAggregate(errs)
}

func createOrUpdateScriptConfigMap(c client.Client, name, filePath string) error {
	content, err := readAndNormalizeScript(filePath)
	if err != nil {
		return errors.Wrapf(err, "get scripts file failed: %s", filePath)
	}

	cm := buildScriptConfigMap(name, content)

	if err = createOrUpdateConfigMap(c, cm); err != nil {
		return errors.Wrapf(err, "deal config map failed: %s", name)
	}

	log.Infof("load env script file %q to configmap %q", name, utils.ClientObjNS(cm))
	return nil
}

func readAndNormalizeScript(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	// docToUnix
	normalized := strings.ReplaceAll(string(content), "\r\n", "\n")
	return normalized, nil
}

func buildScriptConfigMap(name, content string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "cluster-system",
			Labels:    map[string]string{labelhelper.ScriptsLabelKey: ""},
		},
		Data: map[string]string{
			name: content,
		},
	}
}

func createOrUpdateConfigMap(c client.Client, cm *corev1.ConfigMap) error {
	if err := c.Create(context.Background(), cm); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return err
		}

		log.Infof("script file exist, update env init script file %q to configmap %q",
			cm.Name, utils.ClientObjNS(cm))

		key := client.ObjectKey{Name: cm.Name, Namespace: cm.Namespace}
		return phaseutil.RetryOnConflict(func() error {
			latest := &corev1.ConfigMap{}
			if err := c.Get(context.Background(), key, latest); err != nil {
				return err
			}
			latest.Data = cm.Data
			latest.Labels = cm.Labels
			return c.Update(context.Background(), latest)
		})
	}

	return nil
}

func ListScriptsConfigMaps(c client.Client) ([]string, error) {
	cmList := &corev1.ConfigMapList{}
	err := c.List(context.Background(), cmList, &client.ListOptions{
		Namespace: "cluster-system",
		LabelSelector: labels.SelectorFromSet(
			map[string]string{labelhelper.ScriptsLabelKey: ""},
		),
	})
	if err != nil {
		return nil, err
	}

	var names []string
	for _, cm := range cmList.Items {
		names = append(names, cm.Name)
	}
	return names, nil
}
