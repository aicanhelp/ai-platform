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

package kube

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"text/template"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	apierrors2 "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/restmapper"

	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	templateutil "gopkg.openfuyao.cn/cluster-api-provider-bke/common/template"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/addonutil"
)

const (
	// DefaultFilePermission represents the default file permission for temporary files
	DefaultFilePermission = 0644
	// DefaultYamlDecoderBufferSize represents the default buffer size for YAML decoder
	DefaultYamlDecoderBufferSize = 4096
)

var trueVar = true

// ApplyYaml applies yaml file to kubernetes cluster
func (c *Client) ApplyYaml(task *Task) error {
	c.Log.Infof("*****start %s yaml file %s*****", task.Operate, task.FilePath)
	defer c.Log.Infof("*****end %s yaml file %s*****", task.Operate, task.FilePath)
	defer func() {
		if err := recover(); err != nil {
			c.Log.Errorf("%s yaml file %s panic: %v", task.Operate, task.FilePath, err)
		}
	}()

	decoder, err := RenderYamlToDecoder(task)
	if err != nil {
		c.Log.Errorf("failed to render yaml %q: %v", task.FilePath, err)
		return errors.Wrapf(err, "failed to render yaml %q", task.FilePath)
	}

	restMapper, err := c.getRestMapper()
	if err != nil {
		return err
	}

	unstructList, err := GetUnStructListFromDecoder(decoder)
	if err != nil {
		c.Log.Errorf("failed to get unstruct list from file %s: %v", task.FilePath, err)
		return fmt.Errorf("failed to get unstruct list from decoder: %w", err)
	}

	finalUnstructuredList := c.processUnstructuredList(unstructList)
	finalUnstructuredList = c.sortUnstructuredList(finalUnstructuredList, task)

	return c.applyUnstructuredList(finalUnstructuredList, restMapper, task)
}

func (c *Client) getRestMapper() (meta.RESTMapper, error) {
	if c.RESTMapper == nil {
		return nil, errors.New("RESTMapper is not initialized")
	}
	return c.RESTMapper, nil
}

func (c *Client) rebuildRESTMapperFromDiscovery() (meta.RESTMapper, error) {
	if c.ClientSet == nil {
		return nil, errors.New("ClientSet is not initialized")
	}
	cachedDiscovery := memory.NewMemCacheClient(c.ClientSet.Discovery())
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	c.RESTMapper = mapper
	return mapper, nil
}

func (c *Client) processUnstructuredList(unstructList []*unstructured.Unstructured) []unstructured.Unstructured {
	var finalUnstructuredList []unstructured.Unstructured
	for _, unstruct := range unstructList {
		if addonutil.IsListKind(unstruct.GetKind()) {
			list, err := addonutil.UnwrapList(*unstruct)
			if err != nil {
				// 这里应该抛出错误，但由于是格式调整，保持原逻辑
				continue
			}
			finalUnstructuredList = append(finalUnstructuredList, list...)
			continue
		}
		finalUnstructuredList = append(finalUnstructuredList, *unstruct)
	}
	return finalUnstructuredList
}

func (c *Client) sortUnstructuredList(list []unstructured.Unstructured, task *Task) []unstructured.Unstructured {
	if task.Operate == bkeaddon.CreateAddon ||
		task.Operate == bkeaddon.UpdateAddon ||
		task.Operate == bkeaddon.UpgradeAddon {
		return addonutil.SortInstallUnstructuredByKind(list)
	}
	return addonutil.SortUninstallUnstructuredByKind(list)
}

func (c *Client) applyUnstructuredList(
	list []unstructured.Unstructured,
	restMapper meta.RESTMapper,
	task *Task) error {

	for _, unstruct := range list {
		var err error
		restMapper, err = c.getRestMapper()
		if err != nil {
			return err
		}

		gvk := unstruct.GroupVersionKind()
		c.Log.Debugf("gvk: %v", gvk.String())

		mapping, err := c.getMapping(restMapper, gvk, unstruct, task)
		if err != nil {
			return err
		}

		if mapping == nil {
			continue
		}

		dr, err := c.getResourceInterface(mapping, unstruct)
		if err != nil {
			return err
		}

		obj, err := c.handleOperation(dr, unstruct, task, gvk)
		if err != nil {
			return err
		}

		err = c.handleWaitAndLogging(obj, unstruct, task)
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) getMapping(
	restMapper meta.RESTMapper,
	gvk schema.GroupVersionKind,
	unstruct unstructured.Unstructured,
	task *Task) (*meta.RESTMapping, error) {

	mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		// if no resource match, try refresh restMapper
		if apierrors2.IsNoMatchError(err) && c.mapperCache != nil {
			c.Log.Infof("RESTMapping not found for %s/%s, invalidating RESTMapper cache and retrying", gvk.GroupKind().String(), gvk.Version)
			c.mapperCache.Invalidate()
		}
		// try again
		mapping, err = restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			freshMapper, freshErr := c.rebuildRESTMapperFromDiscovery()
			if freshErr == nil {
				c.Log.Infof("RESTMapping failed for %s/%s, rebuilding RESTMapper from current discovery client and retrying", gvk.GroupKind().String(), gvk.Version)
				mapping, err = freshMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
			} else {
				c.Log.Warnf("failed to rebuild RESTMapper from current discovery client: %v", freshErr)
			}
		}
		if err != nil {
			if apierrors2.IsNoMatchError(err) && gvk.Kind == "ServiceMonitor" {
				c.Log.Infof("addon obj Kind: %s, Name %s, APIVersion %s, not support %s in target cluster skip",
					unstruct.GetKind(), unstruct.GetName(), unstruct.GetAPIVersion(), task.Operate)
				return nil, nil
			}
			return nil, err
		}
	}
	return mapping, nil
}

func (c *Client) getResourceInterface(
	mapping *meta.RESTMapping,
	unstruct unstructured.Unstructured) (dynamic.ResourceInterface, error) {

	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		return c.DynamicClient.Resource(mapping.Resource).Namespace(unstruct.GetNamespace()), nil
	}
	return c.DynamicClient.Resource(mapping.Resource), nil
}

func (c *Client) handleOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task,
	gvk schema.GroupVersionKind) (*unstructured.Unstructured, error) {

	var obj *unstructured.Unstructured
	var err error

	switch task.Operate {
	case bkeaddon.CreateAddon:
		obj, err = c.handleCreateOperation(dr, unstruct, task)
	case bkeaddon.UpdateAddon:
		obj, err = c.handleUpdateOperation(dr, unstruct, task, gvk)
	case bkeaddon.UpgradeAddon:
		if task.ApplyStrategy == ApplyStrategyCreateOnly {
			obj, err = c.handleCreateOnlyOperation(dr, unstruct, task)
		} else {
			obj, err = c.handleUpgradeOperation(dr, unstruct, task)
		}
	case bkeaddon.RemoveAddon:
		err = c.handleRemoveOperation(dr, unstruct, task)
	default:
		c.Log.Warnf("Unknown operation type: %s", task.Operate)
		err = errors.Errorf("Unknown operation type: %s", task.Operate)
	}

	return obj, err
}

func (c *Client) handleCreateOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task) (*unstructured.Unstructured, error) {

	obj, err := dr.Apply(c.Ctx, unstruct.GetName(), &unstruct,
		metav1.ApplyOptions{Force: true, FieldManager: "bke"})
	if err != nil {
		c.Log.Errorf("failed to apply %v, error: %v", unstruct.GroupVersionKind(), err)
		return nil, err
	}
	if task.recorder != nil {
		task.recorder.Record(obj)
	}
	return obj, nil
}

func (c *Client) handleUpdateOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task,
	gvk schema.GroupVersionKind) (*unstructured.Unstructured, error) {

	if gvk.Kind == "CustomResourceDefinition" {
		c.Log.Infof("skip update CustomResourceDefinition %s", unstruct.GetName())
		return nil, nil
	}

	json, err := unstruct.MarshalJSON()
	if err != nil {
		return nil, err
	}

	obj, err := dr.Patch(c.Ctx, unstruct.GetName(), types.ApplyPatchType, json,
		metav1.PatchOptions{Force: &trueVar, FieldManager: "bke"})
	if err != nil {
		if apierrors.IsNotFound(err) {
			c.Log.Warnf("addon obj Kind: %s, Name %s not found, skip update",
				unstruct.GetKind(), unstruct.GetName())
			return nil, nil
		}
		c.Log.Errorf("failed to update %v, error: %v", gvk, err)
		return nil, err
	}

	if task.recorder != nil {
		task.recorder.Record(obj)
	}
	return obj, nil
}

func (c *Client) handleUpgradeOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task) (*unstructured.Unstructured, error) {

	obj, err := dr.Apply(c.Ctx, unstruct.GetName(), &unstruct,
		metav1.ApplyOptions{Force: true, FieldManager: "bke"})
	if err != nil {
		c.Log.Errorf("faild upgrade %v", unstruct.GroupVersionKind())
		return nil, err
	}

	if task.recorder != nil {
		task.recorder.Record(obj)
	}
	return obj, nil
}

// handleCreateOnlyOperation creates the object only when it does not already exist.
func (c *Client) handleCreateOnlyOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task) (*unstructured.Unstructured, error) {

	existing, err := dr.Get(c.Ctx, unstruct.GetName(), metav1.GetOptions{})
	if err == nil {
		c.Log.Infof("addon obj Kind: %s, Name %s already exists, skip create-only apply",
			unstruct.GetKind(), unstruct.GetName())
		if task.recorder != nil {
			task.recorder.Record(existing)
		}
		return existing, nil
	}
	if !apierrors.IsNotFound(err) {
		c.Log.Errorf("failed to get %v before create-only apply, error: %v", unstruct.GroupVersionKind(), err)
		return nil, errors.Wrapf(err, "get %s before create-only apply", unstruct.GroupVersionKind())
	}

	obj, err := dr.Create(c.Ctx, &unstruct, metav1.CreateOptions{})
	if err != nil {
		c.Log.Errorf("failed to create %v, error: %v", unstruct.GroupVersionKind(), err)
		return nil, errors.Wrapf(err, "create-only create %s", unstruct.GroupVersionKind())
	}
	if task.recorder != nil {
		task.recorder.Record(obj)
	}
	return obj, nil
}

func (c *Client) handleRemoveOperation(
	dr dynamic.ResourceInterface,
	unstruct unstructured.Unstructured,
	task *Task) error {

	propagation := metav1.DeletePropagationBackground
	err := dr.Delete(c.Ctx, unstruct.GetName(), metav1.DeleteOptions{
		PropagationPolicy: &propagation,
	})
	if err != nil {
		if apierrors.IsNotFound(err) {
			c.Log.Warnf("addon obj Kind: %s, Name %s not found, skip delete",
				unstruct.GetKind(), unstruct.GetName())
			return nil
		}
		c.Log.Errorf("faild delete %v", unstruct.GroupVersionKind())
		return err
	}
	return nil
}

func (c *Client) handleWaitAndLogging(
	obj *unstructured.Unstructured,
	unstruct unstructured.Unstructured,
	task *Task) error {

	// wait for obj when update or create or upgrade
	if task.Block && task.Operate != bkeaddon.RemoveAddon {
		// Check if obj is nil before using it
		if obj == nil {
			obj = &unstruct
		}
		c.Log.Infof("wait for addon obj Kind: %s, Name %s", obj.GetKind(), obj.GetName())
		if err := c.Wait(obj, task); err != nil {
			c.Log.Errorf("addon obj Kind: %s, Name %s  wait failed, err: %v",
				obj.GetKind(), obj.GetName(), err)
			return err
		}
	}

	if obj == nil {
		obj = &unstruct
	}
	c.Log.Infof("addon obj Kind: %s, Name %s, APIVersion %s, %s success",
		obj.GetKind(), obj.GetName(), obj.GetAPIVersion(), task.Operate)
	return nil
}

// RenderYamlToDecoder render yaml and return decoder
func RenderYamlToDecoder(task *Task) (*yamlutil.YAMLOrJSONDecoder, error) {
	rendered, err := renderTaskYAML(task)
	if err != nil {
		return nil, err
	}
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader(rendered), DefaultYamlDecoderBufferSize)
	return decoder, nil
}

// RenderManifest renders manifest bytes using the same template path as ApplyYaml.
func RenderManifest(name string, content []byte, params map[string]interface{}) ([]byte, error) {
	if len(content) == 0 {
		return nil, nil
	}
	task := NewTask(name, "", params)
	task.ManifestContent = content
	return renderTaskYAML(task)
}

func renderTaskYAML(task *Task) ([]byte, error) {
	if task == nil {
		return nil, errors.New("task is nil")
	}

	var source []byte
	var err error
	switch {
	case len(task.ManifestContent) > 0:
		source = append([]byte(nil), task.ManifestContent...)
	case task.FilePath == "":
		return nil, errors.New("task has no manifest content or file path")
	case strings.Contains(task.FilePath, "noexecute"):
		source, err = os.ReadFile(task.FilePath)
		if err != nil {
			return nil, err
		}
	default:
		source, err = os.ReadFile(task.FilePath)
		if err != nil {
			return nil, err
		}
	}

	if task.Param == nil || strings.Contains(task.FilePath, "noexecute") {
		return source, nil
	}

	tmpl, err := template.New(task.Name).Funcs(templateutil.CommonFuncMap()).Parse(string(source))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse manifest template %s", task.Name)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, task.Param); err != nil {
		return nil, errors.Wrapf(err, "failed to render manifest template %s", task.Name)
	}
	return buf.Bytes(), nil
}

func GetUnStructListFromDecoder(decoder *yamlutil.YAMLOrJSONDecoder) ([]*unstructured.Unstructured, error) {
	var unstructList []*unstructured.Unstructured
	latestUnstructName := ""
	for {
		unstruct, _, err := UnStructYaml(decoder)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("latest success resource %s, err: %w", latestUnstructName, err)
		}
		latestUnstructName = fmt.Sprintf("kind: %s name: %s", unstruct.GetKind(), unstruct.GetName())
		unstructList = append(unstructList, unstruct)
	}
	return unstructList, nil
}

// UnStructYaml get unstruct form yaml decoder
func UnStructYaml(decoder *yamlutil.YAMLOrJSONDecoder) (*unstructured.Unstructured, *schema.GroupVersionKind, error) {
	var rawObj runtime.RawExtension
	if err := decoder.Decode(&rawObj); err != nil {
		return nil, nil, err
	}
	obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	// runtime.Object convert to unstructured
	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, nil, err
	}
	unstruct := &unstructured.Unstructured{Object: unstructuredObj}
	return unstruct, gvk, nil
}
