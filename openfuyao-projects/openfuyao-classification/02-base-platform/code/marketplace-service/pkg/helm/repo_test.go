/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package helm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"helm.sh/helm/v3/pkg/repo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	dynamicFake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	clientSetFake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

const (
	mockOfficialRepoCRName        = "openfuyao"
	mockLocalRepoCRName           = "local"
	mockRemoteRepoCRName          = "some_random_name"
	mockRemoteRepoCRURL           = "https://some/random/url"
	mockRemoteRepoCRDisplayName   = "SOME-random-display-Name"
	mockRemoteRepoCRRepoEntryName = mockRemoteRepoCRDisplayName
)

var (
	gvr = schema.GroupVersionResource{
		Group:    constant.CRDRepoGroup,
		Version:  constant.CRDRepoVersion,
		Resource: constant.CRDRepoResource,
	}
	gvk = schema.GroupVersionKind{
		Group:   constant.CRDRepoGroup,
		Version: constant.CRDRepoVersion,
		Kind:    constant.CRDRepoListKind,
	}
)

func mockRepoResponseFromWebsiteA() string {
	result := `
apiVersion: v1
entries:
  demo-app:
  - apiVersion: v2
    appVersion: 1.16.0
    created: "2024-06-22T14:12:45.144861737Z"
    description: A Helm chart for Kubernetes
    digest: fa41c15267ba0066a03b40fa2f8be06467c3441b21588f3087e148c03db92d0a
    name: demo-app
    type: application
    urls:
    - charts/demo-app-1.0.0.tgz
    version: 1.0.0
  nginx-all-type:
  - apiVersion: v2
    appVersion: 1.16.0
    created: "2024-06-27T09:26:56.638633245Z"
    description: A Helm chart for Kubernetes
    digest: 77cb93deb49751428cea227faa2c38d5b9cffb735cdbc72609379bf6ce5bd42c
    keywords:
    - openfuyao-extension
    - fuyao-select
    - compute-power-engine-plugin
    - artificial-intelligence
    - computing
    - database
    - developer-tool
    - CI/CD
    - monitor
    - log
    - network
    - observability
    - security
    - storage
    name: nginx-all-type
    type: application
    urls:
    - charts/nginx-all-type-0.1.0.tgz
    version: 0.1.0
`
	return result
}

func mockRepoResponseFromWebsiteB() string {
	result := `
apiVersion: v1
entries:
  demo-app:
  - apiVersion: v2
    appVersion: 1.16.0
    created: "2024-06-22T14:12:45.144861737Z"
    description: A Helm chart for Kubernetes
    digest: fa41c15267ba0066a03b40fa2f8be06467c3441b21588f3087e148c03db92d0a
    name: demo-app
    type: application
    urls:
    - charts/demo-app-1.0.0.tgz
    version: 1.0.0
  nginx-all-type:
  - apiVersion: v2
    appVersion: 1.16.0
    created: "2024-06-27T09:26:56.638633245Z"
    description: A Helm chart for Kubernetes
    digest: 77cb93deb49751428cea227faa2c38d5b9cffb735cdbc72609379bf6ce5bd42c
    keywords:
    - openfuyao-extension
    - fuyao-select
    - compute-power-engine-plugin
    - artificial-intelligence
    - computing
    - database
    - developer-tool
    - CI/CD
    - monitor
    - log
    - network
    - observability
    - security
    - storage
    name: nginx-all-type
    type: application
    urls:
    - charts/nginx-all-type-0.1.0.tgz
    version: 0.1.0
`
	return result
}

func mockIndexResponseFromOfficialHarbor() string {
	resp := "apiVersion: v1\nentries:\n  application-management-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T10:42:47.933770633Z\n      description: Application-management-service involves the deployment, monitoring, and lifecycle management of applications within the ecosystem.\n      digest: 754dfd479a3d0dbf1f328033f71686f515b6016c50feb207606fdbdd2fa4b4a5\n      keywords:\n        - openfuyao-component\n      name: application-management-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/application-management-service:0.0.0-latest\n      version: 0.0.0-latest\n  atune-log-watcher: []\n  atune-test: []\n  bke-console-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T10:46:36.114047069Z\n      description: 'openFuyao console service helm chart. It works as a reverse-proxy for the whole openFuyao cluster and interacts with oauth-server to retrieves the access-token. '\n      digest: fc921afc5833a18f60cfe9955d6f51cf379d58cdd28589f8d0f8d8196ed5fd54\n      keywords:\n        - openfuyao-component\n      name: bke-console-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/bke-console-service:0.0.0-latest\n      version: 0.0.0-latest\n  bke-console-website:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T10:47:20.590057422Z\n      description: openFuyao console website\n      digest: ad73c287c238bac9e63ed93c0b66105044627401d5d05b20fc2a8f3ab6276ffa\n      keywords:\n        - openfuyao-component\n      name: bke-console-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/bke-console-website:0.0.0-latest\n      version: 0.0.0-latest\n  boltengine: []\n  bookinfo: []\n  bookinfo-envoy: []\n  bookinfo-kmesh: []\n  cache-indexer:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:31:44.617319863Z\n      description: cache-indexer is a component that exhibits the actual KV Cache distribution of inference backend nodes in a cluster. It is based on KV event mechanism in vllm.\n      digest: 6aa10d509193559adeade2776ff1829a324d685bd0c391e5ef60a266e953e1ed\n      keywords:\n        - artificial-intelligence\n      name: cache-indexer\n      urls:\n        - oci://cr.openfuyao.cn/charts/cache-indexer:0.0.0-latest\n      version: 0.0.0-latest\n  circuit-breaker: []\n  colocation-agent:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2024-08-22T02:59:14.221743023Z\n      description: openFuyao colocation agent helm chart\n      digest: cdb0f30a4e7647a4452fa52385ac730786eeb97dd7ba2522f578edaf65158f6f\n      name: colocation-agent\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/colocation-agent:0.0.0-latest\n      version: 0.0.0-latest\n  colocation-management:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:01:35.355976273Z\n      description: A Helm chart for OverQuota Resource Management\n      digest: 52fedd7afcde0b74ec8c95917aebb32276f928af43c45ace41e349c0019237d5\n      name: colocation-management\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/colocation-management:0.0.0-latest\n      version: 0.0.0-latest\n  colocation-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2024-08-21T12:57:49.784004328Z\n      description: openFuyao colocation service helm chart\n      digest: 4f09c8cc48c0b7af1c9b3189eb0c7b35f5d897f77ccf1bb0a9a48f5f36d23095\n      name: colocation-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/colocation-service:0.0.0-latest\n      version: 0.0.0-latest\n  computing-power-engine-backend:\n    - apiVersion: v2\n      appVersion: 1169801f1c8d7153750e3dd8e400ef9e685576bd\n      created: 2025-03-24T01:33:13.498230671Z\n      description: cpeb is the base of the computing power engine. it provides unified management and scheduling of optimization plugins and optimization scenarios.\n      digest: a86007e98b0570e52a80be62638be6ebbfa2d19781af8cce099bcca9ad07b868\n      keywords:\n        - openfuyao-extension\n        - network\n        - database\n        - big-data\n      name: computing-power-engine-backend\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/computing-power-engine-backend:0.0.0-latest\n      version: 0.0.0-latest\n  computing-power-engine-website:\n    - apiVersion: v2\n      appVersion: 6a136d7c9fae0b3c4f418a324d7b8f9ee5c133a5\n      created: 2025-02-25T04:00:05.287214509Z\n      description: Cpew is the website of computing power engine. It provides five pages, Overview, Console, Scenario Management, Plugin Management, and Optimization Report.\n      digest: 7a7854ba16e0ec8d127e35fc9450ecbfd830f4d06d8dc933fccf353d69406d54\n      keywords:\n        - openfuyao-extension\n      name: computing-power-engine-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/computing-power-engine-website:0.0.0-latest\n      version: 0.0.0-latest\n  computingpowerengine:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2024-08-05T09:21:32.453942739Z\n      description: The computing power engine platform\n      digest: 463e02bcbdd6e029560eafe1cba889452339f4738506e6e200aa6b9b31ade556\n      keywords:\n        - fuyao-extension\n      name: computingpowerengine\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/computingpowerengine:0.0.0-latest\n      version: 0.0.0-latest\n  console-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:03:31.637412054Z\n      description: 'openFuyao console service helm chart. It works as a reverse-proxy for the whole openFuyao cluster and interacts with oauth-server to retrieves the access-token. '\n      digest: a84b38be6350242e533f19da10547d5dc5c1fb3b76b46bc27200d40527eaa98a\n      keywords:\n        - openfuyao-component\n      name: console-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/console-service:0.0.0-latest\n      version: 0.0.0-latest\n  console-website:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:04:31.280382857Z\n      description: openFuyao console website\n      digest: 66bfac74adf8553ffc805511a42ed9e337e8a34dc22a3fa131ec0db10cbf3c4e\n      keywords:\n        - openfuyao-component\n      name: console-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/console-website:0.0.0-latest\n      version: 0.0.0-latest\n  efk-chart: []\n  envoy-kmesh-latency-test: []\n  extension-management-operator:\n    - apiVersion: v2\n      appVersion: 38d52742aa0bd9b60cebc913e6012f82353e3665\n      created: 2024-12-17T11:31:44.122798153Z\n      description: A Helm chart for Kubernetes\n      digest: b46cc3412f021cf673f93768b2bd4a46a556eca946988584f040bce27fcc1df6\n      name: extension-management-operator\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/extension-management-operator:0.0.0-latest\n      version: 0.0.0-latest\n  harbor: []\n  hermes-router:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:05:04.660057358Z\n      description: The core capability of an hermes-router is to receive users' inference requests, forward them to appropriate inference service backends, and optimize inference efficiency by perceiving global KV Cache.\n      digest: 270d8ba82624e2d52a09b69c2875e520c76ea8593edaa9cff6ebb298e63ea957\n      keywords:\n        - artificial-intelligence\n      name: hermes-router\n      urls:\n        - oci://cr.openfuyao.cn/charts/hermes-router:0.0.0-latest\n      version: 0.0.0-latest\n  installer:\n    - apiVersion: v2\n      appVersion: d56b1e1dd5d6ea436184b6688b3c7b3d4bb573dd\n      created: 2025-03-20T01:30:51.731154695Z\n      description: Custom Monitoring Dashboards provide real-time visualizations of system metrics, allowing users to track key performance indicators (KPIs) tailored to their specific needs. Users can configure widgets, charts, and alerts to monitor infrastructure, applications, and custom metrics from integrated data.\n      digest: 4b945e2710b1a124940176642ccefa1a720597d869b803e63bcd6ff9a34dffd3\n      keywords:\n        - openfuyao-component\n      name: installer\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/installer:0.0.0-latest\n      version: 0.0.0-latest\n  installer-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:07:21.991326884Z\n      description: 'A Helm chart for openFuyao installer service. The installer is used to provide a visual interface  that allows users to perform installation and deployment operations on business clusters,  including creating, deleting, scaling up, scaling down, and upgrading clusters on the management cluster. '\n      digest: eff8651ef8c35e87485ad6c7c6a9d6578103d94e6e4d2f35a609fdf3287374e3\n      keywords:\n        - openfuyao-component\n      name: installer-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/installer-service:0.0.0-latest\n      version: 0.0.0-latest\n  installer-website:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:07:20.442320836Z\n      description: Custom Monitoring Dashboards provide real-time visualizations of system metrics, allowing users to track key performance indicators (KPIs) tailored to their specific needs. Users can configure widgets, charts, and alerts to monitor infrastructure, applications, and custom metrics from integrated data.\n      digest: 793be2007f65b5a4b822b6929253ce90dafe596b4156dc78c458ab410753c80e\n      keywords:\n        - openfuyao-component\n      name: installer-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/installer-website:0.0.0-latest\n      version: 0.0.0-latest\n  istio-envoy: []\n  istio-kmesh: []\n  kae-deviceplugin: []\n  kae-operator:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2024-08-21T12:40:40.664460346Z\n      description: A Helm chart for Kubernetes\n      digest: e71396df01162854d5d3244826070bf729ee6b75f525b3ab64a627c5b464ae82\n      name: kae-operator\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/kae-operator:0.0.0-latest\n      version: 0.0.0-latest\n  kubeos: []\n  logging-package:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2024-07-31T10:29:59.969821307Z\n      description: A Helm chart for fuyao logging extension\n      digest: 7a14f41e84672b0f4032aa521a0ca4518c3b2e7af6921d04dfb0eb55804ee76d\n      keywords:\n        - fuyao-extension\n      name: logging-package\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/logging-package:0.0.0-latest\n      version: 0.0.0-latest\n  many-core-scheduler:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-06-22T14:06:04.938929661Z\n      description: A Helm chart for many-core-scheduler\n      digest: 51995b4f93f724821a5ab36768531a94355455186eb43930360b5d070c3d375c\n      name: many-core-scheduler\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/many-core-scheduler:0.0.0-latest\n      version: 0.0.0-latest\n  marketplace-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-18T02:13:33.522188257Z\n      description: openFuyao marketplace-service provides an efficient platform for browsing,discovering, and deploying applications and extension components,primarily supporting the management of Helm packages.\n      digest: f83178dc873b2ebb7496d95385f8b376b0da9d08c68b4d2ee0e956701446f4b9\n      keywords:\n        - openfuyao-component\n      name: marketplace-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/marketplace-service:0.0.0-latest\n      version: 0.0.0-latest\n  mis-management:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-13T10:17:00.833723448Z\n      description: A Helm chart for MIS Management Service\n      digest: bc80211e15e29b7fa27c0b8660771d1b911d59dff3a53c9c68def150bb850377\n      home: https://gitcode.com/openFuyao/mis-management\n      keywords:\n        - mis\n      maintainers:\n        - email: mazifan@huawei.com\n          name: Ma Zifan\n        - email: maoyouwen1@huawei.com\n          name: Mao Youwen\n        - email: yujinzheng@huawei.com\n          name: Yu Jinzheng\n        - email: wengxutao@h-partners.com\n          name: Weng Xutao\n      name: mis-management\n      sources:\n        - https://gitcode.com/openFuyao/mis-management\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/mis-management:0.0.0-latest\n      version: 0.0.0-latest\n  mis-management-website:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-16T09:44:11.511948009Z\n      description: A Helm chart for Kubernetes\n      digest: 8022fdc3479a8204b2684b943a5db0b1636794f35693aef13d8309095e12bc1a\n      keywords:\n        - openfuyao-extension\n      name: mis-management-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/mis-management-website:0.0.0-latest\n      version: 0.0.0-latest\n  monitoring-dashboard-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:18:59.436746918Z\n      description: Monitoring Dashboards provide real-time visualizations of system metrics, allowing users to track key performance indicators (KPIs) tailored to their specific needs. Users can configure widgets, charts, and alerts to monitor infrastructure, applications, and custom metrics from integrated data sources.\n      digest: 529dba319b4fe8b0d4470090d0ef5f009613cb146cc596d319713a6a3060f37c\n      keywords:\n        - openfuyao-extension\n        - monitor\n      name: monitoring-dashboard-service\n      urls:\n        - oci://cr.openfuyao.cn/charts/monitoring-dashboard-service:0.0.0-latest\n      version: 0.0.0-latest\n  monitoring-dashboard-website:\n    - apiVersion: v2\n      appVersion: 3d9d0213b70076728229b8dfae7eb3eed3d49bd6\n      created: 2025-04-13T07:38:19.736685926Z\n      dependencies:\n        - name: monitoring-dashboard-service\n          repository: https://harbor.openfuyao.com/chartrepo/openfuyao\n          version: 0.0.0-latest\n      description: Custom Monitoring Dashboards provide real-time visualizations of system metrics, allowing users to track key performance indicators (KPIs) tailored to their specific needs. Users can configure widgets, charts, and alerts to monitor infrastructure, applications, and custom metrics from integrated data.\n      digest: 587b6d3da9a60257331b7e64b6a0195f6b089033e3019ef19508ae5220fa0f77\n      keywords:\n        - openfuyao-extension\n        - monitor\n      name: monitoring-dashboard-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/monitoring-dashboard-website:0.0.0-latest\n      version: 0.0.0-latest\n  monitoring-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:20:35.975229955Z\n      description: 'A Helm chart for the OpenFuyao monitoring service. It interacts with Prometheus to offer built-in queries  for commonly used metrics and provides functionalities to view monitoring targets and alerting rules. '\n      digest: c13d60cbc99730f7df6b1632a4da032e5be10d174bd0107770c476cc0de8549a\n      keywords:\n        - openfuyao-component\n      name: monitoring-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/monitoring-service:0.0.0-latest\n      version: 0.0.0-latest\n  multi-cluster-helm:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:21:38.339379693Z\n      description: A Helm chart for Kubernetes\n      digest: ae76adc2f313498397032a3f0d361d6f4945075f6b37cf9b81b5b3eed056f70d\n      keywords:\n        - openfuyao-extension\n      name: multi-cluster-helm\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/multi-cluster-helm:0.0.0-latest\n      version: 0.0.0-latest\n  multi-cluster-service:\n    - apiVersion: v2\n      appVersion: 25d4a2ea5af08de3046c420179823756d89461b5\n      created: 2024-10-09T09:21:50.478520447Z\n      description: A Helm chart for Kubernetes\n      digest: 3de8bb0d6b86221c8c64693d892b9bbeaa1a41d9b10b1875d08b0ebf75a63de9\n      keywords:\n        - fuyao-extension\n      name: multi-cluster-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/multi-cluster-service:0.0.0-latest\n      version: 0.0.0-latest\n  nginx: []\n  npu-operator:\n    - apiVersion: v2\n      appVersion: 71248dbf51c08f131c33f7a479b50df60f1529b1\n      created: 2024-10-23T01:25:09.869030737Z\n      dependencies:\n        - condition: nfd.enabled\n          name: node-feature-discovery\n          repository: https://kubernetes-sigs.github.io/node-feature-discovery/charts\n          version: v0.16.4\n      description: A Helm chart for Kubernetes\n      digest: 1968c261d99803496216e941af196e392cf7dd39e2df15ca9d0c78aabc278607\n      name: npu-operator\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/npu-operator:0.0.0-latest\n      version: 0.0.0-latest\n  oauth-server:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:25:41.763506205Z\n      description: 'A Helm chart for openFuyao OAuth2.0 server. It interacts with console-service to provide access token for each user. Users are required to input passwords for authentication. '\n      digest: d28b10580d04efa68892e4e70950da69e60e54cf1e93bacdc679adbf47630dd1\n      keywords:\n        - openfuyao-component\n      name: oauth-server\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/oauth-server:0.0.0-latest\n      version: 0.0.0-latest\n  oauth-webhook:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:26:18.091124175Z\n      description: 'A Helm chart for openFuyao OAuth webhook. It interacts with kube-apiserver and validates user identities using webhook validation. '\n      digest: 2798253b99c4f887cf337eb8dc60583035b30b3e36b48922e8f97eec481eb258\n      keywords:\n        - openfuyao-component\n      name: oauth-webhook\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/oauth-webhook:0.0.0-latest\n      version: 0.0.0-latest\n  operator: []\n  optimization-framework:\n    - apiVersion: v2\n      appVersion: 4ff46be9439be677397204e8373a1193c0287b48\n      created: 2025-03-24T01:33:35.614385224Z\n      description: Optimization framework is an significant component of computing power engine. It manages four-step plugins by scheduling four sub-plugins(collecter, analyzer, decisionmaker,and executor).\n      digest: 03d5cc1e99868c4ed75f6ab443e7d99755fe5382a20dedd82e397756c583f41d\n      keywords:\n        - openfuyao-extension\n      name: optimization-framework\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/optimization-framework:0.0.0-latest\n      version: 0.0.0-latest\n  plugin-management-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:27:14.95871299Z\n      description: It enables the integration and orchestration of additional functionalities into the system, allowing for enhanced modularity and adaptability to specific requirements\n      digest: c2051dac959ef36f5b92e9282d8dd6782731090cece6457fb5a75011c1debb8b\n      keywords:\n        - openfuyao-component\n      name: plugin-management-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/plugin-management-service:0.0.0-latest\n      version: 0.0.0-latest\n  prometheus: []\n  ray-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:27:52.353553475Z\n      description: A backend service responsible for handling configuration related operations in the openfuyao rayservice.\n      digest: 6713606f6e96b33a736d1f9dc8b53ab6957a3835a7663c9c0c35bd711c63c339\n      name: ray-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/ray-service:0.0.0-latest\n      version: 0.0.0-latest\n  ray-website:\n    - apiVersion: v2\n      appVersion: 9863e3805f5544429ffa2373f660f660bd6a3d1a\n      created: 2025-03-06T02:15:12.721132525Z\n      description: 'A Helm chart for deploying the Ray Website,  providing a user-friendly interface for configuring, monitoring, and managing Ray workloads. '\n      digest: 56f4639e757fd07b06f9ed6da26dc27a5629c4c1a52889da777d2da98d5b72dc\n      keywords:\n        - openfuyao-extension\n      name: ray-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/ray-website:0.0.0-latest\n      version: 0.0.0-latest\n  rubik-chart: []\n  serviceentry-test: []\n  stress-runner: []\n  tenant: []\n  traffic-mirroring: []\n  user-management-operator:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:28:59.641175441Z\n      description: 'This chart is responsible for providing management (CRUD) of user custom resources and  handling changes to each user''s permissions and role bindings. '\n      digest: b18e1b4b4f335a2b1b605e2341a10d9ca0aac80c783ce55d42830c24c7d317b9\n      keywords:\n        - openfuyao-component\n      name: user-management-operator\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/user-management-operator:0.0.0-latest\n      version: 0.0.0-latest\n  volcano-config-package:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:29:25.796854478Z\n      dependencies:\n        - name: numa-exporter\n          repository: ''\n          version: 0.0.0-latest\n        - name: scheduler-config\n          repository: ''\n          version: 0.0.0-latest\n        - name: volcano\n          repository: ''\n          version: 1.9.0\n      description: A backend service responsible for handling configuration related operations in the volcano-system, ensuring consistent and reliable communication between components.\n      digest: a4a619f317af1daca3ba467c5a7db8aded8db2c7b1b30ac10173056cf09e2eb0\n      keywords:\n        - openfuyao-extension\n      name: volcano-config-package\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/volcano-config-package:0.0.0-latest\n      version: 0.0.0-latest\n  volcano-config-website:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:44:13.344252043Z\n      description: This is a user-facing interface designed for configuraing and managing the volcano system, providing a streamlined and intuitive experience for users to customize settings and monitor the system.\n      digest: a1bc76b3acc03593605fd9ff212ce61771bd758ae3ddbe770201ce4117ca32c0\n      keywords:\n        - openfuyao-extension\n      name: volcano-config-website\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/volcano-config-website:0.0.0-latest\n      version: 0.0.0-latest\n  volcano-config-website-helm:\n    - apiVersion: v2\n      appVersion: 66b7d6edb2ccacf85e4cbe5b7789a7e9a3022091\n      created: 2024-10-09T02:30:36.852926851Z\n      description: A Helm chart for volcano-config-website\n      digest: b9682d478dc6e7225b40dba276f629239eb765684a6f64af9156a0f4e1c24f27\n      keywords:\n        - fuyao-extension\n      name: volcano-config-website-helm\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/volcano-config-website-helm:0.0.0-latest\n      version: 0.0.0-latest\n  web-terminal-service:\n    - apiVersion: v2\n      appVersion: latest\n      created: 2025-09-19T11:54:50.692896836Z\n      description: 'The Helm Chart deploys openFuyao Web Terminal Service that provides terminal for users. It allows users to interact with kubernetes clusters or kubernetes containers directly  from the openFuyao web console. '\n      digest: 32e6a3f0d1cb2518cf037bf0673794d96c834525bcb988a5f1f70d79ffb3f2dc\n      keywords:\n        - openfuyao-component\n      name: web-terminal-service\n      type: application\n      urls:\n        - oci://cr.openfuyao.cn/charts/web-terminal-service:0.0.0-latest\n      version: 0.0.0-latest\n  wrk-runner: []\ngenerated: 2025-09-19T11:54:51Z\nserverInfo: {}"
	return resp
}

func mockClientset() kubernetes.Interface {
	clientset := clientSetFake.NewSimpleClientset()
	AddSecrectInMockedClientset(clientset, mockHarborSecret())
	configmap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.MarketplaceServiceConfigmap,
			Namespace: constant.MarketplaceServiceDefaultNamespace,
		},
		Data: map[string]string{
			"chart-limit":                  "100",
			"official-harbor-tags-URL":     "/v2/openfuyao-catalog-tags",
			"local-harbor-display-name":    "local",
			"local-harbor-host":            "http://local-harbor.openfuyao-system.svc.cluster.local",
			"local-harbor-project":         "/chartrepo/library",
			"official-harbor-display-name": "openFuyao",
			"official-harbor-host":         "https://cr.openfuyao.cn",
			"official-harbor-project":      "/chartrepo/openfuyao-catalog",
			"marketplace-scenes": `
            - fuyao-select
            - compute-power-engine-plugin
            - artificial-intelligence
            - computing
            - database
            - developer-tool
            - CI/CD
            - monitor
            - log
            - network
            - observability
            - security
            - storage
        `,
			"oauth-server-host":        "https://oauth-server.openfuyao-system.svc.cluster.local",
			"marketplace-service-host": "http://marketplace-service.openfuyao-system.svc.cluster.local",
			"console-website-host":     "http://console-website.openfuyao-system.svc.cluster.local",
			"alert-host":               "http://alertmanager-main.monitoring.svc.cluster.local:9093",
			"monitoring-host":          "http://monitoring-service.openfuyao-system.svc.cluster.local:80",
		},
	}
	_, err := clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Create(context.TODO(), configmap, metav1.CreateOptions{})
	if err != nil {
		zlog.Fatalf("failed to create resource: %v", err)
		return nil
	}
	return clientset
}

func updateMarketServiceConfigChartLimit(clientset kubernetes.Interface, chartLimit string) {
	configMap, err := clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Get(context.TODO(), constant.MarketplaceServiceConfigmap, metav1.GetOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	// 更新 ConfigMap 的数据字段
	configMap.Data["chart-limit"] = chartLimit

	// 应用更新
	_, err = clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Update(context.TODO(), configMap, metav1.UpdateOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	if err != nil {
		zlog.Fatalf("failed to update resource: %v", err)
	}
}

func mockDynamicClient() dynamic.Interface {

	scheme := runtime.NewScheme()

	if err := helm.AddHelmChartRepositoryToScheme(scheme); err != nil {
		zlog.Fatalf("failed to add custom resource to scheme: %v", err)
	}

	dynamicClient := dynamicFake.NewSimpleDynamicClientWithCustomListKinds(scheme, map[schema.GroupVersionResource]string{
		gvr: gvk.Kind,
	})
	return dynamicClient
}

func mockOfficialRepoCR() *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "HelmChartRepository",
			"metadata": map[string]interface{}{
				"name": mockOfficialRepoCRName,
			},
			"spec": map[string]interface{}{
				"basicAuth":             nil,
				"ca":                    nil,
				"displayName":           "openFuyao",
				"insecureSkipTLSVerify": false,
				"passCredentialsAll":    false,
				"tls":                   nil,
				"url":                   "https://helm.openfuyao.cn",
			},
		},
	}
}

func mockLocalRepoCR() *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "HelmChartRepository",
			"metadata": map[string]interface{}{
				"name": mockLocalRepoCRName,
			},
			"spec": map[string]interface{}{
				"basicAuth":             nil,
				"ca":                    nil,
				"displayName":           "local",
				"insecureSkipTLSVerify": false,
				"passCredentialsAll":    false,
				"tls":                   nil,
				"url":                   "http://local-harbor.openfuyao-system.svc",
			},
		},
	}
}

func mockRemoteRepoCR() *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "console.openfuyao.com/v1beta1",
			"kind":       "HelmChartRepository",
			"metadata": map[string]interface{}{
				"name": mockRemoteRepoCRName,
			},
			"spec": map[string]interface{}{
				"basicAuth": map[string]interface{}{
					"name": "",
				},
				"ca": map[string]interface{}{
					"name": "",
				},
				"displayName":           mockRemoteRepoCRDisplayName,
				"insecureSkipTLSVerify": false,
				"passCredentialsAll":    false,
				"tls": map[string]interface{}{
					"name": "",
				},
				"url": mockRemoteRepoCRURL,
			},
		},
	}
}

func mockDynamicClientWithRepoCR(repoCR *unstructured.Unstructured) dynamic.Interface {
	fakeDynamicClient := mockDynamicClient()

	resourceClient := fakeDynamicClient.Resource(gvr).Namespace("")
	_, err := resourceClient.Create(context.TODO(), repoCR, metav1.CreateOptions{})
	if err != nil {
		zlog.Errorf("Error creating object: %v\n", err)
		return nil
	}
	_, err = fakeDynamicClient.Resource(gvr).Namespace("").Get(context.TODO(), repoCR.GetName(), metav1.GetOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
		return nil
	}
	return fakeDynamicClient
}

func mockRepoServer(response string) *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 检查请求 URL
		if r.URL.Path == "/index.yaml" {
			w.Header().Set("Content-Type", "application/x-yaml")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(response))
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	return server
}

func deleteNestedField(obj map[string]interface{}, fields ...string) {
	unstructured.RemoveNestedField(obj, fields...)
}

func compareResponse(got, want *httputil.ResponseJson, serverURL string) bool {
	if got.Code != want.Code || got.Msg != want.Msg {
		return false
	}
	if got.Data == nil && want.Data == nil {
		return true
	}
	gotData, ok1 := got.Data.(*unstructured.Unstructured)
	wantData, ok2 := want.Data.(*unstructured.Unstructured)
	if wantData.Object != nil {
		spec, ok := wantData.Object["spec"].(map[string]interface{})
		if ok && serverURL != "" {
			spec["url"] = serverURL
			wantData.Object["spec"] = spec
		}
	}
	if !ok1 || !ok2 {
		return false
	}

	// 转换成 map[string]interface{}，方便移除字段
	gotMap := gotData.Object
	wantMap := wantData.Object

	// 移除动态字段
	deleteNestedField(gotMap, "metadata", "annotations")
	deleteNestedField(gotMap, "metadata", "creationTimestamp")
	deleteNestedField(gotMap, "metadata", "resourceVersion")
	deleteNestedField(gotMap, "metadata", "uid")
	if managedFields, found, _ := unstructured.NestedSlice(gotMap, "metadata", "managedFields"); found && len(managedFields) > 0 {
		field := managedFields[0].(map[string]interface{})
		delete(field, "time")
	}

	deleteNestedField(wantMap, "metadata", "annotations")
	deleteNestedField(wantMap, "metadata", "creationTimestamp")
	deleteNestedField(wantMap, "metadata", "resourceVersion")
	deleteNestedField(wantMap, "metadata", "uid")
	if managedFields, found, _ := unstructured.NestedSlice(wantMap, "metadata", "managedFields"); found && len(managedFields) > 0 {
		field := managedFields[0].(map[string]interface{})
		delete(field, "time")
	}
	return reflect.DeepEqual(gotMap, wantMap)
}

func Test_getRepoCRObjectMeta(t *testing.T) {
	type args struct {
		repoName string
	}
	tests := []struct {
		name string
		args args
		want metav1.ObjectMeta
	}{
		{
			name: "Test case 1: Test_getRepoCRObjectMeta",
			args: args{
				repoName: "test-repo",
			},
			want: metav1.ObjectMeta{
				Name: "test-repo",
				Annotations: map[string]string{
					fmt.Sprintf("%s/%s", constant.MarketplaceServiceDefaultOrgName,
						modificationTimestamp): time.Now().UTC().Format(time.RFC3339),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getRepoCRObjectMeta(tt.args.repoName); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getRepoCRObjectMeta() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getRepoCRTypeMeta(t *testing.T) {
	tests := []struct {
		name string
		want metav1.TypeMeta
	}{
		{
			name: "Test case 2: Test_getRepoCRTypeMeta",
			want: metav1.TypeMeta{
				Kind:       constant.CRDRepoKind,
				APIVersion: fmt.Sprintf("%s/%s", constant.CRDRepoGroup, constant.CRDRepoVersion),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getRepoCRTypeMeta(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getRepoCRTypeMeta() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_CreateRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoEntry *helm.SafeRepoEntry
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_CreateRepo_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClient(),
				clientset:     mockClientset(),
			},
			args: args{
				repoEntry: &helm.SafeRepoEntry{
					Name: mockRemoteRepoCRRepoEntryName,
					URL:  "http://example.com",
				},
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "repository created",
				Data: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "console.openfuyao.com/v1beta1",
						"kind":       "HelmChartRepository",
						"metadata": map[string]interface{}{
							"annotations": map[string]interface{}{
								"openfuyao.com/modificationTimestamp": "DYNAMIC_TIMESTAMP",
							},
							"creationTimestamp": "DYNAMIC_TIMESTAMP",
							"name":              strings.ToLower(mockRemoteRepoCRRepoEntryName),
						},
						"spec": map[string]interface{}{
							"basicAuth": map[string]interface{}{
								"name": "",
							},
							"ca": map[string]interface{}{
								"name": "",
							},
							"displayName":           mockRemoteRepoCRRepoEntryName,
							"insecureSkipTLSVerify": false,
							"passCredentialsAll":    false,
							"tls": map[string]interface{}{
								"name": "",
							},
							"url": "http://example.com",
						},
					},
				},
			},
			want1: 201,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := mockRepoServer(mockRepoResponseFromWebsiteA())
			defer server.Close()
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			tt.args.repoEntry.URL = server.URL

			got, got1 := c.CreateRepo(tt.args.repoEntry)
			if !compareResponse(got, tt.want, server.URL) {
				gotJSON, _ := json.MarshalIndent(got, "", "  ")
				wantJSON, _ := json.MarshalIndent(tt.want, "", "  ")
				t.Errorf("CreateRepo() got = %s, want %s", gotJSON, wantJSON)
			}
			if got1 != tt.want1 {
				t.Errorf("CreateRepo() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_DeleteRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_DeleteRepo_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				repoName: mockRemoteRepoCRName,
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  fmt.Sprintf("%s deleted", mockRemoteRepoCRName),
				Data: nil,
			},
			want1: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.DeleteRepo(tt.args.repoName)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("DeleteRepo() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("DeleteRepo() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_GetRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_GetRepo_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				repoName: mockRemoteRepoCRName,
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: &helm.HelmChartRepository{
					TypeMeta: metav1.TypeMeta{
						Kind:       constant.CRDRepoKind,
						APIVersion: constant.CRDRepoGroup + "/" + constant.CRDRepoVersion,
					},
					ObjectMeta: metav1.ObjectMeta{
						Name: mockRemoteRepoCRName,
					},
					Spec: helm.HelmChartRepositorySpec{
						DisplayName: mockRemoteRepoCRDisplayName,
						URL:         mockRemoteRepoCRURL,
						BasicAuth: helm.SecretReference{
							Name: "",
						},
						TLS: helm.SecretReference{
							Name: "",
						},
						CA: helm.ConfigMapReference{
							Name: "",
						},
						InsecureSkipTLSVerify: false,
						PassCredentialsAll:    false,
					},
				},
			},
			want1: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.GetRepo(tt.args.repoName)
			if !reflect.DeepEqual(got, tt.want) {
				gotJSON, _ := json.MarshalIndent(got, "", "  ")
				wantJSON, _ := json.MarshalIndent(tt.want, "", "  ")
				t.Errorf("GetRepo() got = %s, want %s", gotJSON, wantJSON)
			}
			if got1 != tt.want1 {
				t.Errorf("GetRepo() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_GetRepoSyncStatus(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_GetRepoSyncStatus_failed",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClient(),
				clientset:     mockClientset(),
			},
			args: args{
				repoName: mockRemoteRepoCRName,
			},
			want: &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  fmt.Sprintf("%s update task not found", mockRemoteRepoCRName),
			},
			want1: http.StatusBadRequest,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.GetRepoSyncStatus(tt.args.repoName)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetRepoSyncStatus() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("GetRepoSyncStatus() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_ListRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		query *param.Query
		repo  string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_ListRepo_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				query: &param.Query{
					Pagination: &param.Pagination{
						Limit:       0,
						Offset:      0,
						CurrentPage: 1,
					},
					SortBy:    "",
					Ascending: false,
				},
				repo: "",
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: &helm.ListResponse{
					TotalItems: 1,
					Items: []interface{}{
						&helm.RepoResponse{
							Name: mockRemoteRepoCRDisplayName,
							URL:  mockRemoteRepoCRURL,
						},
					},
					CurrentPage: 1,
					TotalPages:  1,
				},
			},
			want1: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.ListRepo(tt.args.query, tt.args.repo)
			if !reflect.DeepEqual(got, tt.want) {
				gotJSON, _ := json.MarshalIndent(got, "", "  ")
				wantJSON, _ := json.MarshalIndent(tt.want, "", "  ")
				t.Errorf("ListRepo() got = %s, want %s", gotJSON, wantJSON)
			}
			if got1 != tt.want1 {
				t.Errorf("ListRepo() got1 = %d, want %d", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_SyncAllRepos(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	tests := []struct {
		name   string
		fields fields
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_ListRepo_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "synchronizing repositories",
			},
			want1: http.StatusAccepted,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.SyncAllRepos()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SyncAllRepos() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("SyncAllRepos() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_SyncRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoDisplayName string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_SyncRepo_success",
			args: args{
				repoDisplayName: mockRemoteRepoCRName,
			},
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  fmt.Sprintf("syncronizing repository %s", mockRemoteRepoCRName),
			},
			want1: http.StatusAccepted,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.SyncRepo(tt.args.repoDisplayName)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SyncRepo() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("SyncRepo() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_UpdateRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoEntry *helm.SafeRepoEntry
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_UpdateRepo_success",
			args: args{
				repoEntry: &helm.SafeRepoEntry{
					Name: mockRemoteRepoCRName,
					URL:  mockRemoteRepoCRURL + "123",
				},
			},
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "repository updated",
				Data: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "console.openfuyao.com/v1beta1",
						"kind":       "HelmChartRepository",
						"metadata": map[string]interface{}{
							"annotations": map[string]interface{}{
								"openfuyao.com/modificationTimestamp": "DYNAMIC_TIMESTAMP",
							},
							"creationTimestamp": "DYNAMIC_TIMESTAMP",
							"name":              mockRemoteRepoCRName,
						},
						"spec": map[string]interface{}{
							"basicAuth": map[string]interface{}{
								"name": "",
							},
							"ca": map[string]interface{}{
								"name": "",
							},
							"displayName":           mockRemoteRepoCRDisplayName,
							"insecureSkipTLSVerify": false,
							"passCredentialsAll":    false,
							"tls": map[string]interface{}{
								"name": "",
							},
							"url": mockRemoteRepoCRURL + "123",
						},
					},
				},
			},
			want1: http.StatusOK,
		},
		{
			name: "Test_helmClient_UpdateRepo_failed",
			args: args{
				repoEntry: &helm.SafeRepoEntry{
					Name: mockRemoteRepoCRRepoEntryName,
					URL:  mockRemoteRepoCRURL,
				},
			},
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			want: &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  fmt.Sprintf("creation failed, repository：%s doesn't exist.", mockRemoteRepoCRRepoEntryName),
			},
			want1: http.StatusBadRequest,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := mockRepoServer(mockRepoResponseFromWebsiteB())
			defer server.Close()
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			tt.args.repoEntry.URL = server.URL
			got, got1 := c.UpdateRepo(tt.args.repoEntry)
			if !compareResponse(got, tt.want, server.URL) {
				gotJSON, _ := json.MarshalIndent(got, "", "  ")
				wantJSON, _ := json.MarshalIndent(tt.want, "", "  ")
				t.Errorf("UpdateRepo() got = %s, want %s", gotJSON, wantJSON)
			}
			if got1 != tt.want1 {
				t.Errorf("UpdateRepo() got1 = %d, want %d", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_asyncUpdateRepository(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repository  *helm.HelmChartRepository
		errRepoList chan<- errRepo
	}
	var tests []struct {
		name   string
		fields fields
		args   args
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			c.asyncUpdateRepository(tt.args.repository, tt.args.errRepoList)
		})
	}
}

func Test_helmClient_createRepoCR(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoEntry *helm.SafeRepoEntry
		authRef   *v1.Secret
		tlsRef    *v1.Secret
		caRef     *v1.ConfigMap
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *unstructured.Unstructured
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.createRepoCR(tt.args.repoEntry, tt.args.authRef, tt.args.tlsRef, tt.args.caRef)
			if (err != nil) != tt.wantErr {
				t.Errorf("createRepoCR() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createRepoCR() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_createRepoCRBasicAuth(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
		username []byte
		password []byte
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.Secret
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.createRepoCRBasicAuth(tt.args.repoName, tt.args.username, tt.args.password)
			if (err != nil) != tt.wantErr {
				t.Errorf("createRepoCRBasicAuth() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createRepoCRBasicAuth() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_createRepoCRCA(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
		caFile   string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.ConfigMap
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.createRepoCRCA(tt.args.repoName, tt.args.caFile)
			if (err != nil) != tt.wantErr {
				t.Errorf("createRepoCRCA() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createRepoCRCA() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_createRepoCRTLS(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
		certFile string
		keyFile  string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.Secret
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.createRepoCRTLS(tt.args.repoName, tt.args.certFile, tt.args.keyFile)
			if (err != nil) != tt.wantErr {
				t.Errorf("createRepoCRTLS() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createRepoCRTLS() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_createRepoSecretAndConfigmap(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoEntry *helm.SafeRepoEntry
	}
	var tests []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.createRepoSecretAndConfigmap(tt.args.repoEntry)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createRepoSecretAndConfigmap() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("createRepoSecretAndConfigmap() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_deleteRepoCR(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoCRName string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			if err := c.deleteRepoCR(tt.args.repoCRName); (err != nil) != tt.wantErr {
				t.Errorf("deleteRepoCR() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_helmClient_getCustomRepoByName(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *helm.HelmChartRepository
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.getCustomRepoByName(tt.args.repoName)
			if (err != nil) != tt.wantErr {
				t.Errorf("getCustomRepoByName() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getCustomRepoByName() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_isRepoModificationAllowed(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *httputil.ResponseJson
		want1   int
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1, err := c.isRepoModificationAllowed(tt.args.repoName)
			if (err != nil) != tt.wantErr {
				t.Errorf("isRepoModificationAllowed() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("isRepoModificationAllowed() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("isRepoModificationAllowed() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_listCustomRepo(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	var tests []struct {
		name    string
		fields  fields
		want    []helm.HelmChartRepository
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.listCustomRepo()
			if (err != nil) != tt.wantErr {
				t.Errorf("listCustomRepo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("listCustomRepo() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_patchRepoModificationTime(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		name string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			if err := c.patchRepoModificationTime(tt.args.name); (err != nil) != tt.wantErr {
				t.Errorf("patchRepoModificationTime() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_helmClient_repoCRtoRepoEntry(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoCR *helm.HelmChartRepository
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *repo.Entry
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.repoCRtoRepoEntry(tt.args.repoCR)
			if (err != nil) != tt.wantErr {
				t.Errorf("repoCRtoRepoEntry() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("repoCRtoRepoEntry() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateAndClearSecretData(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		secret *v1.Secret
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.Secret
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.updateAndClearSecretData(tt.args.secret)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateAndClearSecretData() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateAndClearSecretData() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateRepoCR(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		customHelmRepository *helm.HelmChartRepository
		repoEntry            *helm.SafeRepoEntry
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *unstructured.Unstructured
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.updateRepoCR(tt.args.customHelmRepository, tt.args.repoEntry)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateRepoCR() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateRepoCR() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateRepoCRBasicAuth(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repo     *helm.HelmChartRepository
		repoName string
		username []byte
		password []byte
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.Secret
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.updateRepoCRBasicAuth(tt.args.repo, tt.args.repoName, tt.args.username, tt.args.password)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateRepoCRBasicAuth() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateRepoCRBasicAuth() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateRepoCRCA(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repo     *helm.HelmChartRepository
		repoName string
		caFile   string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.ConfigMap
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.updateRepoCRCA(tt.args.repo, tt.args.repoName, tt.args.caFile)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateRepoCRCA() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateRepoCRCA() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateRepoCRTLS(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repo     *helm.HelmChartRepository
		repoName string
		certFile string
		keyFile  string
	}
	var tests []struct {
		name    string
		fields  fields
		args    args
		want    *v1.Secret
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, err := c.updateRepoCRTLS(tt.args.repo, tt.args.repoName, tt.args.certFile, tt.args.keyFile)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateRepoCRTLS() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateRepoCRTLS() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_updateRepoSecretAndConfigmap(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoEntry  *helm.SafeRepoEntry
		repository *helm.HelmChartRepository
	}
	var tests []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			got, got1 := c.updateRepoSecretAndConfigmap(tt.args.repoEntry, tt.args.repository)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("updateRepoSecretAndConfigmap() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("updateRepoSecretAndConfigmap() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_safeRepoEntryToRepoEntry(t *testing.T) {
	type args struct {
		safeRepoEntry *helm.SafeRepoEntry
	}
	var tests []struct {
		name string
		args args
		want *repo.Entry
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := safeRepoEntryToRepoEntry(tt.args.safeRepoEntry); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("safeRepoEntryToRepoEntry() = %v, want %v", got, tt.want)
			}
		})
	}
}
