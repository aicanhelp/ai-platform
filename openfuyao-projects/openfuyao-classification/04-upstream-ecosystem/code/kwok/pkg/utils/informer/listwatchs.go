package informer

import (
	"context"
	"crypto"
	"crypto/sha512"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	certificatesv1 "k8s.io/api/certificates/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	nodev1 "k8s.io/api/node/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/client-go/util/keyutil"

	"sigs.k8s.io/kwok/pkg/log"
)

var caFile = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

type ListWatchsManager interface {
	RunListWatchsForNode(ctx context.Context, nodeName string) error
	StopListWatchsForNode(ctx context.Context, nodeName string)
}

var _ ListWatchsManager = &listWatchsManager{}

// AdditionalListWatchsManager manages additional List-watch clients for simulating other K8s components
type AdditionalListWatchsManager interface {
	StartAdditionalListWatchsForNode(ctx context.Context, nodeName string) error
	StopAdditionalListWatchsForNode(ctx context.Context, nodeName string)
}

var _ AdditionalListWatchsManager = &additionalListWatchsManager{}

type listWatchsManager struct {
	keyData       []byte
	caData        []byte
	apiServerHost string

	client kubernetes.Interface
	stopFn map[string]context.CancelFunc
}

type additionalListWatchsManager struct {
	client        kubernetes.Interface
	resourceTypes []string
	numClients    int
	stopFn        map[string]context.CancelFunc
	clients       map[string][]kubernetes.Interface // nodeName -> clients
	mu            sync.Mutex
}

func InitListWatchsManager(client kubernetes.Interface, config *rest.Config) (ListWatchsManager, error) {
	// Generate private key
	keyData, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return nil, err
	}
	caData, err := os.ReadFile(caFile)
	if err != nil {
		return nil, err
	}

	return &listWatchsManager{
		keyData:       keyData,
		caData:        caData,
		apiServerHost: config.Host,
		client:        client,
		stopFn:        make(map[string]context.CancelFunc),
	}, nil

}

// KubeletListWatchs is used to simulate Kubelet's list-watch functionality against kube-apiserver
// It includes Pod, Node, Service, CSIDriver, RuntimeClass list-watch
// Note: Real kubelet also list-watch Secrets and ConfigMaps referenced by pods on the node,
// but considering kwok needs to simulate a large number of nodes, dynamic list-watch of
// Secrets and ConfigMaps is not implemented here to avoid excessive resource consumption
type KubeletListWatchs struct {
	lws map[string]*cache.ListWatch
}

func (lwm *listWatchsManager) RunListWatchsForNode(ctx context.Context, nodeName string) error {
	certData, err := requestNodeCertificate(ctx, lwm.client, lwm.keyData, types.NodeName(nodeName))
	if err != nil {
		return err
	}
	client, err := lwm.buildClientFromCert(certData)
	if err != nil {
		return err
	}
	ctx, fn := context.WithCancel(ctx)
	klw := initKubeletListWatchs(ctx, client, nodeName)
	lwm.stopFn[nodeName] = fn
	klw.Run(ctx, nodeName)
	return nil
}

func (lwm *listWatchsManager) StopListWatchsForNode(ctx context.Context, nodeName string) {
	logger := log.FromContext(ctx)
	if fn, ok := lwm.stopFn[nodeName]; ok {
		fn()
		delete(lwm.stopFn, nodeName)
	} else {
		logger.Warn("kubelet list-watch for node is not running", "nodeName", nodeName)
	}
}

func (lws *KubeletListWatchs) Run(ctx context.Context, nodeName string) {
	logger := log.FromContext(ctx)
	for resourceType, lw := range lws.lws {
		logger.Info("Starting list-watch for resource", "resourceType", resourceType, "node", nodeName)
		// Reflector is a low-level wrapper in client-go, simpler than SharedInformer
		reflector := cache.NewReflectorWithOptions(
			lw,
			getObjectForResource(resourceType),
			&cache.FakeCustomStore{}, // Do not store or process events
			cache.ReflectorOptions{
				Name: fmt.Sprintf("kubelet-listwatch-%s", resourceType),
			},
		)
		go reflector.Run(ctx.Done())
	}
}

func initKubeletListWatchs(ctx context.Context, client kubernetes.Interface, nodeName string) *KubeletListWatchs {
	klw := &KubeletListWatchs{
		lws: make(map[string]*cache.ListWatch),
	}

	// List-watch pods on this node
	klw.lws["pods"] = &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fields.OneTermEqualSelector("spec.nodeName", nodeName).String()
			return client.CoreV1().Pods("").List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fields.OneTermEqualSelector("spec.nodeName", nodeName).String()
			return client.CoreV1().Pods("").Watch(ctx, options)
		},
	}

	// List-watch this node
	klw.lws["nodes"] = &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fields.Set{metav1.ObjectNameField: nodeName}.String()
			return client.CoreV1().Nodes().List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fields.Set{metav1.ObjectNameField: nodeName}.String()
			return client.CoreV1().Nodes().Watch(ctx, options)
		},
	}

	// List-watch all services
	klw.lws["services"] = &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return client.CoreV1().Services("").List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return client.CoreV1().Services("").Watch(ctx, options)
		},
	}

	// List-watch all CSIDrivers
	klw.lws["csidrivers"] = &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return client.StorageV1().CSIDrivers().List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return client.StorageV1().CSIDrivers().Watch(ctx, options)
		},
	}

	// List-watch all RuntimeClasses
	klw.lws["runtimeclasses"] = &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return client.NodeV1().RuntimeClasses().List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return client.NodeV1().RuntimeClasses().Watch(ctx, options)
		},
	}

	return klw
}

// requestNodeCertificate will create a certificate signing request for a node
// (Organization and CommonName for the CSR will be set as expected for node
// certificates) and send it to API server, then it will watch the object's
// status, once approved by API server, it will return the API server's issued
// certificate (pem-encoded). If there is any errors, or the watch timeouts, it
// will return an error. This is intended for use on nodes (kubelet and
// kubeadm).
func requestNodeCertificate(ctx context.Context, client kubernetes.Interface, privateKeyData []byte, nodeName types.NodeName) (certData []byte, err error) {
	logger := log.FromContext(ctx)
	subject := &pkix.Name{
		Organization: []string{"system:nodes"},
		CommonName:   "system:node:" + string(nodeName),
	}

	privateKey, err := keyutil.ParsePrivateKeyPEM(privateKeyData)
	if err != nil {
		return nil, fmt.Errorf("invalid private key for certificate request: %v", err)
	}
	csrData, err := certutil.MakeCSR(privateKey, subject, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("unable to generate certificate request: %v", err)
	}

	usages := []certificatesv1.KeyUsage{
		certificatesv1.UsageDigitalSignature,
		certificatesv1.UsageKeyEncipherment,
		certificatesv1.UsageClientAuth,
	}

	// The Signer interface contains the Public() method to get the public key.
	signer, ok := privateKey.(crypto.Signer)
	if !ok {
		return nil, fmt.Errorf("private key does not implement crypto.Signer")
	}

	name, err := digestedName(signer.Public(), subject, usages)
	if err != nil {
		return nil, err
	}

	reqName, reqUID, err := csr.RequestCertificate(client, csrData, name, certificatesv1.KubeAPIServerClientKubeletSignerName, nil, usages, privateKey)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, 3600*time.Second)
	defer cancel()

	logger.Info("Waiting for client certificate to be issued")
	return csr.WaitForCertificate(ctx, client, reqName, reqUID)
}

// This digest should include all the relevant pieces of the CSR we care about.
// We can't directly hash the serialized CSR because of random padding that we
// regenerate every loop and we include usages which are not contained in the
// CSR. This needs to be kept up to date as we add new fields to the node
// certificates and with ensureCompatible.
func digestedName(publicKey interface{}, subject *pkix.Name, usages []certificatesv1.KeyUsage) (string, error) {
	hash := sha512.New512_256()

	// Here we make sure two different inputs can't write the same stream
	// to the hash. This delimiter is not in the base64.URLEncoding
	// alphabet so there is no way to have spill over collisions. Without
	// it 'CN:foo,ORG:bar' hashes to the same value as 'CN:foob,ORG:ar'
	const delimiter = '|'
	encode := base64.RawURLEncoding.EncodeToString

	write := func(data []byte) {
		hash.Write([]byte(encode(data)))
		hash.Write([]byte{delimiter})
	}

	publicKeyData, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return "", err
	}
	write(publicKeyData)

	write([]byte(subject.CommonName))
	for _, v := range subject.Organization {
		write([]byte(v))
	}
	for _, v := range usages {
		write([]byte(v))
	}

	return fmt.Sprintf("node-csr-%s", encode(hash.Sum(nil))), nil
}

func (lwm *listWatchsManager) buildClientFromCert(certData []byte) (*kubernetes.Clientset, error) {
	// Build rest.Config
	config := &rest.Config{
		Host: lwm.apiServerHost,
		TLSClientConfig: rest.TLSClientConfig{
			CertData: certData,
			KeyData:  lwm.keyData,
			CAData:   lwm.caData,
		},
	}
	return kubernetes.NewForConfig(config)
}

// InitAdditionalListWatchsManager creates a new additional List-watch manager
func InitAdditionalListWatchsManager(client kubernetes.Interface, numClients int, resourceTypes []string) AdditionalListWatchsManager {
	return &additionalListWatchsManager{
		client:        client,
		numClients:    numClients,
		resourceTypes: resourceTypes,
		stopFn:        make(map[string]context.CancelFunc),
		clients:       make(map[string][]kubernetes.Interface),
	}
}

func (alm *additionalListWatchsManager) StartAdditionalListWatchsForNode(ctx context.Context, nodeName string) error {
	logger := log.FromContext(ctx)
	alm.mu.Lock()
	defer alm.mu.Unlock()

	if alm.numClients <= 0 || len(alm.resourceTypes) == 0 {
		logger.Info("No additional List-watch clients configured for node", "node", nodeName)
		return nil
	}

	// Check if already started for this node
	if _, exists := alm.stopFn[nodeName]; exists {
		logger.Info("Additional List-watch already started for node", "node", nodeName)
		return nil
	}

	// Create InClusterConfig for additional clients
	config, err := rest.InClusterConfig()
	if err != nil {
		return fmt.Errorf("failed to create in-cluster config for node %s: %v", nodeName, err)
	}

	// Create node-specific context and clients
	ctx, cancel := context.WithCancel(ctx)

	nodeClients := make([]kubernetes.Interface, 0, alm.numClients)
	for i := 0; i < alm.numClients; i++ {
		configCopy := rest.CopyConfig(config)

		// Create a different Dial function instance for each client to bypass TLS cache
		configCopy.Dial = func(ctx context.Context, network, address string) (net.Conn, error) {
			dialer := &net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}
			return dialer.DialContext(ctx, network, address)
		}

		// Set different User-Agent for each client
		configCopy.UserAgent = fmt.Sprintf("kwok-addon-%d-%s", i, nodeName)

		client, err := kubernetes.NewForConfig(configCopy)
		if err != nil {
			cancel() // Clean up context
			return fmt.Errorf("failed to create kubernetes client %d for node %s: %v", i, nodeName, err)
		}
		nodeClients = append(nodeClients, client)
	}

	// Only update internal state after all clients are successfully created
	alm.stopFn[nodeName] = cancel
	alm.clients[nodeName] = nodeClients

	logger.Info("Starting additional List-watch clients for node", "node", nodeName, "numClients", alm.numClients, "resources", alm.resourceTypes)

	// Start List-watch for each client and resource
	for i, client := range nodeClients {
		for _, resourceType := range alm.resourceTypes {
			alm.runListWatchForResource(ctx, client, i, resourceType, nodeName)
		}
	}

	return nil
}

func (alm *additionalListWatchsManager) StopAdditionalListWatchsForNode(ctx context.Context, nodeName string) {
	logger := log.FromContext(ctx)
	alm.mu.Lock()
	defer alm.mu.Unlock()

	if cancel, exists := alm.stopFn[nodeName]; exists {
		cancel()
		delete(alm.stopFn, nodeName)
		delete(alm.clients, nodeName)
		logger.Info("Stopped additional List-watch for node", "node", nodeName)
	} else {
		logger.Info("Additional List-watch for node is not running", "node", nodeName)
	}
}

func (alm *additionalListWatchsManager) runListWatchForResource(ctx context.Context, client kubernetes.Interface, clientIdx int, resourceType string, nodeName string) {
	logger := log.FromContext(ctx)
	listWatch := alm.createListWatchForResource(ctx, client, resourceType)
	if listWatch == nil {
		logger.Warn("Unknown resource type %s for client %d, node %s", resourceType, clientIdx, nodeName)
		return
	}

	// Create a reflector to watch the resource
	reflector := cache.NewReflectorWithOptions(
		listWatch,
		getObjectForResource(resourceType),
		&cache.FakeCustomStore{},
		cache.ReflectorOptions{
			Name: fmt.Sprintf("additional-listwatch-%s-client%d-node%s", resourceType, clientIdx, nodeName),
		},
	)

	// Run the reflector until stopped
	go reflector.Run(ctx.Done())
}

func (alm *additionalListWatchsManager) createListWatchForResource(ctx context.Context, client kubernetes.Interface, resourceType string) *cache.ListWatch {
	switch resourceType {
	case "pods":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().Pods("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().Pods("").Watch(ctx, options)
			},
		}
	case "services":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().Services("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().Services("").Watch(ctx, options)
			},
		}
	case "nodes":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().Nodes().List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().Nodes().Watch(ctx, options)
			},
		}
	case "configmaps":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().ConfigMaps("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().ConfigMaps("").Watch(ctx, options)
			},
		}
	case "secrets":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().Secrets("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().Secrets("").Watch(ctx, options)
			},
		}
	case "endpointSlices":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.DiscoveryV1().EndpointSlices("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.DiscoveryV1().EndpointSlices("").Watch(ctx, options)
			},
		}
	case "namespaces":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.CoreV1().Namespaces().List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().Namespaces().Watch(ctx, options)
			},
		}
	case "deployments":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.AppsV1().Deployments("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.AppsV1().Deployments("").Watch(ctx, options)
			},
		}
	case "replicasets":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.AppsV1().ReplicaSets("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.AppsV1().ReplicaSets("").Watch(ctx, options)
			},
		}
	case "daemonsets":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.AppsV1().DaemonSets("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.AppsV1().DaemonSets("").Watch(ctx, options)
			},
		}
	case "statefulsets":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.AppsV1().StatefulSets("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.AppsV1().StatefulSets("").Watch(ctx, options)
			},
		}
	case "jobs":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.BatchV1().Jobs("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.BatchV1().Jobs("").Watch(ctx, options)
			},
		}
	case "cronjobs":
		return &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.BatchV1().CronJobs("").List(ctx, options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.BatchV1().CronJobs("").Watch(ctx, options)
			},
		}
	default:
		return nil
	}
}

func getObjectForResource(resourceType string) runtime.Object {
	switch resourceType {
	case "pods":
		return &corev1.Pod{}
	case "services":
		return &corev1.Service{}
	case "nodes":
		return &corev1.Node{}
	case "csidrivers":
		return &storagev1.CSIDriver{}
	case "runtimeclasses":
		return &nodev1.RuntimeClass{}
	case "configmaps":
		return &corev1.ConfigMap{}
	case "secrets":
		return &corev1.Secret{}
	case "endpointSlices":
		return &discoveryv1.EndpointSlice{}
	case "namespaces":
		return &corev1.Namespace{}
	case "deployments":
		return &appsv1.Deployment{}
	case "replicasets":
		return &appsv1.ReplicaSet{}
	case "daemonsets":
		return &appsv1.DaemonSet{}
	case "statefulsets":
		return &appsv1.StatefulSet{}
	case "jobs":
		return &batchv1.Job{}
	case "cronjobs":
		return &batchv1.CronJob{}
	default:
		return nil
	}
}
