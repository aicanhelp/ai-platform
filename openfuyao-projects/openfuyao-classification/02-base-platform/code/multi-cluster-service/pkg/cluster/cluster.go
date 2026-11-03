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
	"context"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// OperationsInterface contains methods to work with Cluster resources.
type OperationsInterface interface {
	JoinHost(ctx context.Context)
	Join(ctx context.Context, opts JoinOptions) error
	Unjoin(ctx context.Context, opts UnjoinOptions) error
	Edit(ctx context.Context, opts EditOptions) error
	Inspect(ctx context.Context, opts InspectOptions) (*ListOfCluster, error)
	Authorize(ctx context.Context, opts AuthorizeOptions) error
}

// OperationClient encapsulates various clients necessary for managing cluster resources.
type OperationClient struct {
	// MgrClient is the controller client to interact with CRD, not specific to any kubernetes version.
	MgrClient client.Client

	// KarmadaClient is a client for interacting with the karmada API - kubernetes native resources.
	KarmadaClient *kubernetes.Clientset

	// KarmadaVersionedClient provides a versioned client for karmada the can be used to perform API
	// operations with a specific resources like karmada `Cluster`.
	KarmadaVersionedClient *versioned.Clientset
}

// NewClusteClient defines a new clusterclient structure.
func NewClusteClient(mgr client.Client, kClient *kubernetes.Clientset,
	kVClient *versioned.Clientset) OperationsInterface {
	return &OperationClient{
		MgrClient:              mgr,
		KarmadaClient:          kClient,
		KarmadaVersionedClient: kVClient,
	}
}

// Transfer2JoinOpts constructs JoinOptions based on the provided structure.
func Transfer2JoinOpts(clusterinfo InfoOfCluster) JoinOptions {
	labels := deepcopyLabels(clusterinfo.ClusterLabels)
	return JoinOptions{
		ClusterName:    clusterinfo.Name,
		Kubeconfig:     clusterinfo.Kubeconfig,
		KubeconfigPath: clusterinfo.KubeconfigPath,
		ClusterLabels:  labels,
	}
}

// Transfer2UnjoinOpts constructs UnjoinOptions based on the provided structure.
func Transfer2UnjoinOpts(clusterTag string) UnjoinOptions {
	return UnjoinOptions{
		ClusterName: clusterTag,
	}
}

// Transfer2InspectOpts constructs InspectOptions based on the provided structure.
func Transfer2InspectOpts(clusterTag string) InspectOptions {
	return InspectOptions{
		ClusterName: clusterTag,
	}
}

// Transfer2EditOpts constructs EditOptions based on the provided structure.
func Transfer2EditOpts(clusterTag string, tags TagOfCluster) EditOptions {
	labels := deepcopyLabels(tags.ClusterLabels)
	return EditOptions{
		ClusterName:   clusterTag,
		ClusterLabels: labels,
	}
}

// Transfer2AuthorizeOpts constructs AuthorizeOptions based on the provided structure.
func Transfer2AuthorizeOpts(identityInfo IdentityCollection) AuthorizeOptions {
	var users []*IdentityDescriptor
	for _, user := range identityInfo.UserInfo {
		users = append(users, user)
	}

	return AuthorizeOptions{
		Users: users,
	}
}

func deepcopyLabels(data map[string]string) map[string]string {
	labels := make(map[string]string)
	for key, value := range data {
		labels[key] = value
	}
	return labels
}
