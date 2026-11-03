package weight_dispatcher

import (
	"context"

	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

var (
	clientset           *kubernetes.Clientset
	dynamicClient       dynamic.Interface
	apiExtensionsClient *apiextensionsclientset.Clientset
	restConfig          *rest.Config
	ctx                 context.Context
	cancel              context.CancelFunc
)
