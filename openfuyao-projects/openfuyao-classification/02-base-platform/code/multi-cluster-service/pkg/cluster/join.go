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

// Package cluster provides a suite og tools adn utilities for managing cluster objects.
package cluster

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"

	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/homedir"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
	"openfuyao.com/multi-cluster-service/pkg/utils"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// Join intergrates a new cluster into the karmada management system by performing necessary operations required
// for registrations.
//
// 1. Adding ServiceAccounts to the member cluster and karmada controlplane.
// 2. Creating the new cluster object to karmada controlplane and multicluster service.
func (c *OperationClient) Join(ctx context.Context, opt JoinOptions) error {
	var err error
	clustertag := opt.ClusterName
	if clustertag == "" {
		clustertag = "host"
	}

	// create kubernete client and api.config based on join-options.
	clusterKClient, apiKubeconfig, err := establishClientAndApiConfig(opt)
	if err != nil {
		return err
	}

	// check cluster id and name if valid
	uid, err := c.checkClusterIDAndName(clustertag, clusterKClient)
	if err != nil {
		return err
	}

	// check target cluster if karmada exists
	if clustertag != "host" {
		if existsKarmada, existsMcs := utils.HasKarmadaServiceOrMCService(clusterKClient); existsKarmada || existsMcs {
			zlog.Warnf("Cluster %s already has karmada system or multicluster service !", clustertag)
			return fmt.Errorf("target Cluster ID contains karmada system or multicluster service")
		}
	}

	// create credentials for registering cluster objects.
	kClusterSecret, kImpersonatorSecret, err := c.createCredentials(clusterKClient, clustertag)
	if err != nil {
		return err
	}

	buildOption := generateClusterOptions(clustertag, uid, apiKubeconfig, opt)
	buildOption.ClusterSecret = *kClusterSecret
	buildOption.ImpersonatorSecret = *kImpersonatorSecret

	if opt.KubeconfigPath != "" {
		buildOption.ClusterKubeconfigPath = opt.KubeconfigPath
	}

	kubeData, err := utils.ConfToUnstructure(apiKubeconfig)
	if err != nil {
		zlog.LogErrorf("Error Generating kubeconfig data : %v", err)
		return err
	}
	buildOption.ClusterKubeconfigData = kubeData
	buildOption.OutClusterConfig = opt.Kubeconfig
	if clustertag == "host" {
		configdata, err := utils.HostConfToJson()
		if err != nil {
			return err
		}
		buildOption.OutClusterConfig = configdata
	}
	registerObj := buildCluster(buildOption)
	if err := c.createClusterObject(buildOption, registerObj); err != nil {
		return err
	}
	return nil
}

// JoinHost initiates the registration of the host cluster, logging the start of this process.
func (c *OperationClient) JoinHost(ctx context.Context) {
	zlog.LogInfof("Starting Registering HostCluster !")

	if err := ctx.Err(); err != nil {
		zlog.LogErrorf("Error Receiving context : %v", err)
		return
	}

	cluster := InfoOfCluster{}
	cluster.ClusterLabels = make(map[string]string)
	cluster.ClusterLabels[utils.DefaultopenFuyaoLabel] = "host"

	homePath := homedir.HomeDir()
	kubeConfig := path.Join(homePath, ".kube/config")
	if _, err := os.Stat(kubeConfig); err == nil {
		cluster = InfoOfCluster{
			KubeconfigPath: kubeConfig,
		}
	}

	// before joinhost, check karmada api service if ok
	err := utils.IsKarmadaServiceReady(c.KarmadaVersionedClient)
	if err != nil {
		zlog.LogErrorf("Error Waiting karmada service ok: %v", err)
		return
	}

	joinOpts := Transfer2JoinOpts(cluster)
	err = c.Join(ctx, joinOpts)
	if err != nil {
		if err.Error() == "cluster ID already exists" {
			return
		}
		zlog.LogErrorf("Error Registering Hostserver : %v", err)
	}
	return
}

func testClient(mgrClient client.Client, tag string) {
	clusterObj := v1beta1.Cluster{}
	if err := mgrClient.Get(context.TODO(), client.ObjectKey{Name: tag}, &clusterObj); err != nil {
		zlog.LogErrorf("Error Retrieving cluster object: %v", err)
		return
	}

	kubeData := clusterObj.Spec.KubeConfig.Raw
	var kubeconfig api.Config
	err := json.Unmarshal(kubeData, &kubeconfig)
	if err != nil {
		zlog.LogErrorf("Errorf kubeconfig data unmarshal failed.")
		return
	}

	_, err = utils.CreateClusterClient(&kubeconfig)
	if err != nil {
		zlog.LogErrorf("Error Creating cluster client from json: %v", err)
		return
	}

	zlog.LogInfof("Creating cluster success!!!!!!!!!!")
	return
}

func createAPICluster(mgrClient client.Client, buildOption OptionsOfCluster) error {
	// buildOptions to api/v1beta1 cluster
	labels := map[string]string{
		utils.CreatedByopenFuyaoLabel: utils.CreatedByopenFuyaoLabelValue,
	}

	if len(buildOption.ClusterLabels) != 0 {
		for key, values := range buildOption.ClusterLabels {
			labels[key] = values
		}
	}

	kubeconfigRaw := runtime.RawExtension{
		Raw: buildOption.ClusterKubeconfigData,
	}

	outclusterconfig := runtime.RawExtension{
		Raw: buildOption.OutClusterConfig,
	}

	clusterSpec := &v1beta1.ClusterSpec{
		UID:              buildOption.ClusterID,
		APIEndpoint:      buildOption.ClusterAPIEndpoint,
		KubeConfig:       kubeconfigRaw,
		Kubeconfigpath:   buildOption.ClusterKubeconfigPath,
		OutClusterConfig: outclusterconfig,
	}

	v1beta1Cluster := &v1beta1.Cluster{
		ObjectMeta: v1.ObjectMeta{
			Name:   buildOption.ClusterName,
			Labels: labels,
		},
		Spec: *clusterSpec,
	}

	if err := mgrClient.Create(context.TODO(), v1beta1Cluster); err != nil {
		if errors.IsAlreadyExists(err) {
			zlog.LogInfof("Creating v1beta1 cluster object %s already exist !", v1beta1Cluster.Name)
			return nil
		}

		zlog.LogErrorf("Error creating v1beta1 cluster object : %v", err)
		return err
	}

	zlog.LogInfof("Cluster Object %s created.", v1beta1Cluster.Name)
	return nil
}

func registerHostCluster(kclient *versioned.Clientset, cluster *v1alpha1.Cluster) error {
	newCluster, err := kclient.ClusterV1alpha1().Clusters().Create(context.TODO(), cluster, v1.CreateOptions{})
	if err != nil {
		zlog.LogErrorf("Error Posting cluster object : %v", err)
		return err
	}

	zlog.LogInfof("Karmada Cluster Object %s registered.", newCluster.Name)
	return nil
}

func registerCluster(cluster v1alpha1.Cluster) error {
	clusterdata, err := json.Marshal(cluster)
	if err != nil {
		zlog.LogErrorf("Error Marshaling cluster data : %v", err)
		return err
	}

	url := utils.KarmadaEndPoints + utils.KarmadaClusterAPIPath
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(clusterdata))
	if err != nil {
		zlog.LogErrorf("Error Creating register cluster request : %v", err)
		return err
	}
	var token string
	token = ""
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}

	c := &http.Client{Transport: tr}
	resp, err := c.Do(req)
	if err != nil {
		zlog.LogErrorf("Error Sending cluster post request : %v", err)
		return err
	}

	defer resp.Body.Close()

	_, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		zlog.LogErrorf("Error Reading cluster post response : %v", err)
		return err
	}

	fmt.Println("Response Status:", resp.Status)
	return nil
}

func buildCluster(opt OptionsOfCluster) *v1alpha1.Cluster {
	clusterObj := &v1alpha1.Cluster{
		ObjectMeta: v1.ObjectMeta{
			Name: opt.ClusterName,
		},
	}

	if len(opt.ClusterLabels) != 0 {
		clusterObj.Labels = make(map[string]string)
		for key, values := range opt.ClusterLabels {
			clusterObj.Labels[key] = values
		}
	}

	clusterSpec := &v1alpha1.ClusterSpec{
		APIEndpoint:                 opt.ClusterAPIEndpoint,
		SyncMode:                    v1alpha1.Push,
		InsecureSkipTLSVerification: opt.ClusterTLSVerify,
		ID:                          opt.ClusterID,
	}

	secretRef := &v1alpha1.LocalSecretReference{
		Name:      opt.ClusterSecret.Name,
		Namespace: opt.ClusterSecret.Namespace,
	}
	clusterSpec.SecretRef = secretRef

	impersonatorSecretRef := &v1alpha1.LocalSecretReference{
		Name:      opt.ImpersonatorSecret.Name,
		Namespace: opt.ImpersonatorSecret.Namespace,
	}
	clusterSpec.ImpersonatorSecretRef = impersonatorSecretRef

	clusterObj.Spec = *clusterSpec

	return clusterObj
}

func establishClientAndApiConfig(opt JoinOptions) (*kubernetes.Clientset, *api.Config, error) {
	var err error

	clusterKClient := &kubernetes.Clientset{}
	apiKubeconfig := api.NewConfig()
	if opt.KubeconfigPath != "" {
		clusterKClient, apiKubeconfig, err = utils.CreateClient(opt.KubeconfigPath)
		if err != nil {
			zlog.LogErrorf("Error Creating cluster client from path: %v", err)
			return nil, nil, err
		}
	} else if opt.Kubeconfig != nil {
		err := utils.LoadKubeconfigRawData(opt.Kubeconfig, apiKubeconfig)
		if err != nil {
			zlog.LogErrorf("Error Loading Kubeconfig RawData: %v", err)
			return nil, nil, err
		}

		clusterKClient, err = utils.CreateClusterClient(apiKubeconfig)
		if err != nil {
			zlog.LogErrorf("Error Creating cluster client from config json data: %v", err)
			return nil, nil, err
		}
	} else {
		clusterKClient, apiKubeconfig, err = utils.CreateHostClient()
		if err != nil {
			zlog.LogErrorf("Error Creating Hostcluster client : %v", err)
			return nil, nil, err
		}
	}

	return clusterKClient, apiKubeconfig, nil
}

func (c *OperationClient) createCredentials(k kubernetes.Interface, t string) (*corev1.Secret, *corev1.Secret, error) {
	// create serviceaccount
	clusterSecret, impersonatorSecret, err := utils.GenerateClusterSecret(k, t)
	if err != nil {
		zlog.LogErrorf("Error Genarating cluster Secrets : %v", err)
		return nil, nil, err
	}

	// create serviceaccount
	kClusterSecret, kImpersonatorSecret, err := utils.GenerateKarmadaSecret(c.KarmadaClient, clusterSecret,
		impersonatorSecret, t)
	if err != nil {
		zlog.LogErrorf("Error Genarating karmada Secrets : %v", err)
		return nil, nil, err
	}

	return kClusterSecret, kImpersonatorSecret, nil
}

func (c *OperationClient) createClusterObject(buildOption OptionsOfCluster, registerObj *v1alpha1.Cluster) error {
	if err := registerHostCluster(c.KarmadaVersionedClient, registerObj); err != nil {
		zlog.LogErrorf("Error Registering Karmada Cluster : %v", err)
		return err
	}

	if err := createAPICluster(c.MgrClient, buildOption); err != nil {
		zlog.LogErrorf("Error Creating v1beta1 Cluster Object : %v", err)
		return err
	}
	return nil
}

func generateClusterOptions(name string, uid string, apiConfig *api.Config, opt JoinOptions) OptionsOfCluster {
	contextName := apiConfig.CurrentContext
	apiCtx, _ := apiConfig.Contexts[contextName]
	clusterMessage, _ := apiConfig.Clusters[apiCtx.Cluster]

	return OptionsOfCluster{
		ClusterName:        name,
		ClusterID:          uid,
		ClusterLabels:      opt.ClusterLabels,
		ClusterAPIEndpoint: clusterMessage.Server,
		ClusterTLSVerify:   clusterMessage.InsecureSkipTLSVerify,
		ClusterKubeconfig:  apiConfig,
	}
}

func (c *OperationClient) checkClusterIDAndName(name string, client *kubernetes.Clientset) (string, error) {
	// check name if repeated
	isRepeated, err := utils.IsClusterNameRepeated(name, c.KarmadaVersionedClient)
	if err != nil {
		return "", err
	}
	if isRepeated {
		return "", fmt.Errorf("cluster ID already exists")
	}

	// check uid if unique
	uid, err := utils.IsClusterUIDUnique(client, c.KarmadaVersionedClient)
	if err != nil {
		return "", err
	}
	if uid == "" {
		return "", fmt.Errorf("cluster ID already exists")
	}
	return uid, nil
}
