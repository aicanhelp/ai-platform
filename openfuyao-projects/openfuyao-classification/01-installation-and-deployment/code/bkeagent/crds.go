package main

import (
	"context"
	"embed"
	_ "embed"
	"io"
	"strings"

	log "github.com/sirupsen/logrus"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/restmapper"
	ctrl "sigs.k8s.io/controller-runtime"

	"gopkg.openfuyao.cn/bkeagent/api/v1beta1"
)

//go:embed config/crd/bases
var crds embed.FS

func enableCrdHasInstalled() error {

	clientSet, err := kubernetes.NewForConfig(ctrl.GetConfigOrDie())
	if err != nil {
		return err
	}
	_, apiResourceList, err := clientSet.Discovery().ServerGroupsAndResources()
	if len(apiResourceList) == 0 {
		return err
	}

	needCreate := true
	for _, list := range apiResourceList {
		if list.GroupVersion != v1beta1.GroupVersion.String() {
			continue
		}
		for _, resource := range list.APIResources {
			// skip subresources such as "/status", "/scale" and etc because these are not real APIResources that we are caring about.
			if strings.Contains(resource.Name, "/") {
				continue
			}
			if resource.Kind == "Command" {
				needCreate = false
				break
			}
		}
	}

	if !needCreate {
		return nil
	}
	dynamicClient, err := dynamic.NewForConfig(ctrl.GetConfigOrDie())
	if err != nil {
		return err
	}

	if err = installCrd(clientSet, dynamicClient); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return err
	}

	return nil
}

func installCrd(clientSet *kubernetes.Clientset, dynamicClient dynamic.Interface) error {

	f, err := crds.Open("config/crd/bases/bkeagent.bocloud.com_commands.yaml")
	if err != nil {
		return err
	}
	defer f.Close()

	decoder := yamlutil.NewYAMLOrJSONDecoder(f, 4096)
	dc := clientSet.Discovery()
	restMapperRes, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return err
	}
	restMapper := restmapper.NewDiscoveryRESTMapper(restMapperRes)

	for {
		var rawObj runtime.RawExtension
		if err = decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			} else {
				return err
			}
		}
		// runtime.Object
		obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
		if err != nil {
			return err
		}
		mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			return err
		}
		// runtime.Object convert to unstructured
		unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			return err
		}
		unstruct := &unstructured.Unstructured{Object: unstructuredObj}

		var obj2 *unstructured.Unstructured

		obj2, err = dynamicClient.Resource(mapping.Resource).Create(context.Background(), unstruct, metav1.CreateOptions{})
		if err != nil {
			return err
		}
		log.Infof("The target resource %s/%s is created in the cluster", obj2.GetKind(), obj2.GetName())
	}
	return nil
}
