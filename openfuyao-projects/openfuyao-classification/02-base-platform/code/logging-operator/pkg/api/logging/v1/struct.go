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

// Package v1 provides the HTTP request handling and interaction logic for
// a logging server designed to work within Kubernetes environments. This package
// encapsulates functionalities for making and processing HTTP requests
// to Loki, a horizontally-scalable, highly-available, multi-tenant log aggregation
// system. It aims to facilitate logging data retrieval and analysis by
// providing structured access to logs stored in Loki.
//
// The package defines the APIServer and Handler types that manage the HTTP server
// and request routing respectively. The APIServer serves as the entry point for
// incoming HTTP requests, whereas Handler implements the logic for processing
// these requests, including querying Loki for log data based on various
// parameters such as filename, time range, and log level.
//
// Key Components:
//
//   - HTTPRequest: A struct that encapsulates details of an HTTP request, including
//     URL, method, headers, and body. It serves as a foundational element for
//     constructing requests to Loki.
//
//   - Handler: The core struct that handles incoming requests, routes them to
//     appropriate processing functions, and interacts with Loki to fetch or
//     manipulate log data. It maintains a map of predefined Loki query configurations
//     and a dictionary mapping human-readable time units to their respective
//     durations for flexible time range queries.
//
//   - LogData: A struct designed to unmarshal JSON responses from Loki, specifically
//     targeting the data structure returned by Loki's series and query_range APIs.
//
// Functionalities:
//
//   - Dynamic Loki Query Construction: Allows building and executing complex Loki
//     queries based on request parameters, enabling clients to perform detailed log
//     searches and analysis.
//
//   - Time Range Calculation: Offers utilities to calculate start and end times for
//     queries based on either specific timestamps or relative time units (e.g., "last 5 minutes").
//
//   - API Endpoints: Defines RESTful endpoints for interacting with the logging server,
//     including endpoints for listing available log sources and querying logs based
//     on various filters.
//
// Usage:
// To use this package, create an instance of the Handler struct via the New function,
// then call its methods to handle incoming requests or interact with Loki. The
// package's functionality is encapsulated in the Handler type, making it the primary
// interface for clients.
//
// Note:
// This package assumes that Loki is accessible via HTTP and that clients have
// necessary permissions to query Loki. It also relies on external packages such
// as "github.com/emicklei/go-restful/v3" for RESTful routing and
// "k8s.io/apimachinery/pkg/util/runtime" for Kubernetes-related utilities.
//
// The package is designed with extensibility in mind, allowing for future
// enhancements such as additional Loki query types, improved error handling,
// and richer response parsing.
package v1

import (
	"context"
	"net/http"

	testv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
)

// HTTPRequest defines the structure of an HTTP request including its URL, method,
// headers, and body. It's designed to facilitate the construction of HTTP requests
// for various API calls.
type HTTPRequest struct {
	URL     string
	Method  string
	Headers map[string]string
	Body    []byte
}

// Handler manages HTTP requests and time unit conversions. It maintains a mapping
// of predefined HTTPRequest configurations and supports converting human-readable
// time units into durations for query parameters.
type Handler struct {
	RequestsMap     map[string]*HTTPRequest
	timeUnitsDict   map[string]string
	PodClient       v1.PodInterface
	ConfigmapClient v1.ConfigMapInterface
	SecretsClient   v1.SecretInterface
}

// HandlerTest is a struct for faking client test
type HandlerTest struct {
	PodClient PodInterface
}

// PodInterface defines the interface for Pod operations
type PodInterface interface {
	List(ctx context.Context, opts metav1.ListOptions) (*testv1.PodList, error)
}

// QueryResourcesParams defines parameters used to query resources within a Kubernetes environment.
// It encapsulates details about the specific scope of the query, allowing for precise resource management.
// Fields:
// Namespace - The Kubernetes namespace in which the resource is located. Namespaces are used to
// organize resources in isolated groups within the cluster, aiding in management and security.
// Pod       - The name of the pod within the specified namespace. Pods are the smallest deployable units
// in Kubernetes and can contain one or more containers.
// Container - The specific container within the pod that is the target of the resource query. Containers
// are running instances of Docker images and are the basic operational unit of a pod.
type QueryResourcesParams struct {
	Namespace string
	Pod       string
	Container string
}

// PodList represents a collection of Pods as typically retrieved from a Kubernetes API.
// This structure is often used to hold the results of a list operation on Pods, providing
// a convenient way to access all retrieved pods through the Items field.
type PodList struct {
	Items []Pod `json:"items"`
}

// Pod represents a single Kubernetes Pod resource. A Pod is a group of one or more containers
// (such as Docker containers), with shared storage/network resources, and a specification for
// how to run the containers. The Pod struct encapsulates all the essential information about
// the Pod, including its metadata.
type Pod struct {
	Metadata Metadata `json:"metadata"`
}

// Metadata contains the basic metadata of Kubernetes resources. It provides identifying
// information about the resource, including the name, which is unique within a given namespace
// and crucial for accessing the resource through the Kubernetes API.
type Metadata struct {
	Name string `json:"name"`
}

// VolumeSource defines various types of storage that can be mounted as volumes in a Pod.
// It supports mounting ConfigMaps, Secrets, PersistentVolumeClaims, and other types of storage
// as specified by the application's needs.
type VolumeSource struct {
	ConfigMap             *ConfigMapVolumeSource             `json:"configMap,omitempty"`
	Secret                *SecretVolumeSource                `json:"secret,omitempty"`
	PersistentVolumeClaim *PersistentVolumeClaimVolumeSource `json:"persistentVolumeClaim,omitempty"`
	// 其他类型的volume source可以根据需要添加
}

// ConfigMapVolumeSource provides a way to inject configuration data stored in ConfigMaps
// into Pods as files or environment variables.
type ConfigMapVolumeSource struct {
	Name string `json:"name"`
}

// SecretVolumeSource allows a volume to mount a Secret into the Pod, providing a secure
// storage of sensitive values such as passwords or tokens.
type SecretVolumeSource struct {
	SecretName string `json:"secretName"`
}

// PersistentVolumeClaimVolumeSource represents a request to use a Persistent Volume Claim
// as a volume in a Pod, allowing a higher level of storage abstraction that supports data
// persistence across pod restarts.
type PersistentVolumeClaimVolumeSource struct {
	ClaimName string `json:"claimName"`
}

// PodDetail represents the detailed information about a Kubernetes pod, focusing on its specification.
type PodDetail struct {
	Spec PodSpec `json:"spec"`
}

// PodSpec holds the specifications of a pod, primarily listing its containers.
type PodSpec struct {
	Containers []Container `json:"containers"`
}

// Container represents a container within a pod in Kubernetes, detailing its name and volume mounts.
type Container struct {
	Name         string        `json:"name"`
	VolumeMounts []VolumeMount `json:"volumeMounts"`
}

// VolumeMount describes a single volume mount in a container, including the mount path and other relevant details.
type VolumeMount struct {
	MountPath string `json:"mountPath"`
	Name      string `json:"name"`
	ReadOnly  bool   `json:"readOnly,omitempty"`
}

// RuleFile encapsulates a set of alerting rules for Loki, organized into rule groups.
type RuleFile struct {
	RuleGroups []RuleGroup `yaml:"loki-alerting-rules.yaml"`
}

// RuleGroup defines a group of rules within a RuleFile that share common characteristics.
type RuleGroup struct {
	Name  string `yaml:"name"`
	Rules []Rule `yaml:"rules"`
}

// Rule specifies the details of a single alerting rule in a RuleGroup.
type Rule struct {
	Alert       string            `yaml:"alert"`
	Expr        string            `yaml:"expr"`
	For         string            `yaml:"for"`
	Labels      map[string]string `yaml:"labels"`
	Annotations map[string]string `yaml:"annotations"`
}

// Labels defines key-value pairs attached to objects for identification and grouping,
// here specifically used for alert severity.
type Labels struct {
	Severity string `yaml:"severity"`
}

// Annotations provides additional information about a rule that can be used to
// describe the rule's intent and implications.
type Annotations struct {
	Description string `yaml:"description"`
	Summary     string `yaml:"summary"`
}

// LokiQueryParams structures the parameters for querying logs from Loki,
// specifying filters and search criteria.
type LokiQueryParams struct {
	Filename  string
	Namespace string
	Pod       string
	Container string
	Keyword   string
	LogLevel  string
}

// HTTPClient defines a http client for unit testing
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// ScrapeConfig defines configurations for scraping logs in Promtail.
type ScrapeConfig struct {
	JobName             string          `yaml:"job_name"`
	KubernetesSdConfigs []interface{}   `yaml:"kubernetes_sd_configs"`
	RelabelConfigs      []RelabelConfig `yaml:"relabel_configs"`
	PipelineStages      []interface{}   `yaml:"pipeline_stages"`
	StaticConfigs       []StaticConfig  `yaml:"static_configs"`
}

// RelabelConfig defines configurations for relabeling targets in Promtail.
type RelabelConfig struct {
	SourceLabels []string `yaml:"source_labels"`
	Regex        string   `yaml:"regex"`
	Action       string   `yaml:"action"`
	TargetLabel  string   `yaml:"target_label"`
	Replacement  string   `yaml:"replacement,omitempty"`
	Separator    string   `yaml:"separator,omitempty"`
}

// StaticConfig defines static targets and labels for scraping.
type StaticConfig struct {
	Targets []string          `yaml:"targets"`
	Labels  map[string]string `yaml:"labels"`
}

// PromtailConfigMap represents a Kubernetes ConfigMap for Promtail configurations.
type PromtailConfigMap struct {
	APIVersion string            `json:"apiVersion"`
	Kind       string            `json:"kind"`
	Data       map[string]string `json:"data"`
	Metadata   Metadata          `json:"metadata"`
}

// PromtailConfig defines the configuration for a Promtail instance.
type PromtailConfig struct {
	Server struct {
		HttpListenPort int    `yaml:"http_listen_port"`
		LogLevel       string `yaml:"log_level"`
		LogFormat      string `yaml:"log_format"`
	} `yaml:"server"`
	ScrapeConfigs []ScrapeConfig    `yaml:"scrape_configs"`
	Clients       []Clients         `yaml:"clients"`
	Positions     map[string]string `yaml:"positions"`
}

// Clients defines configurations for client interactions in Promtail.
type Clients struct {
	Url string `yaml:"url"`
}

// PromtailUpdateRequest represents an update request for a Promtail scraping job.
type PromtailUpdateRequest struct {
	JobName string `json:"jobName"`
	Path    string `json:"path"`
}

// LimitsConfig defines limit configurations for Promtail scraping.
type LimitsConfig struct {
	EnforceMetricName       bool   `yaml:"enforce_metric_name"`
	MaxEntriesLimitPerQuery int    `yaml:"max_entries_limit_per_query"`
	RejectOldSamples        bool   `yaml:"reject_old_samples"`
	RejectOldSamplesMaxAge  string `yaml:"reject_old_samples_max_age"`
}

// StorageConfig defines storage configurations for logs in Promtail.
type StorageConfig struct {
	BoltDBShipper BoltDBShipper `yaml:"boltdb_shipper"`
	Filesystem    Filesystem    `yaml:"filesystem"`
}

// BoltDBShipper defines configurations for the BoltDB shipper used in Loki.
type BoltDBShipper struct {
	ActiveIndexDirectory string `yaml:"active_index_directory"`
	CacheLocation        string `yaml:"cache_location"`
	CacheTTL             string `yaml:"cache_ttl"`
	SharedStore          string `yaml:"shared_store"`
}

// Filesystem defines the filesystem configuration used in storage.
type Filesystem struct {
	Directory string `yaml:"directory"`
}

// TableManager configures table management in databases.
type TableManager struct {
	RetentionDeletesEnabled bool   `yaml:"retention_deletes_enabled"`
	RetentionPeriod         string `yaml:"retention_period"`
}

// SecretConfig defines configurations related to secret management.
type SecretConfig struct {
	AuthEnabled   bool          `yaml:"auth_enabled"`
	LimitsConfig  LimitsConfig  `yaml:"limits_config"`
	StorageConfig StorageConfig `yaml:"storage_config"`
	TableManager  TableManager  `yaml:"table_manager"`
}

// ApiResponse represents a standard API response format.
type ApiResponse struct {
	Code int32       `json:"code"`
	Msg  string      `json:"msg"`
	Data interface{} `json:"data"`
}

// Summary provides an overview of processing metrics.
type Summary struct {
	BytesProcessedPerSecond               float64 `json:"bytesProcessedPerSecond"`
	ExecTime                              float64 `json:"execTime"`
	LinesProcessedPerSecond               float64 `json:"linesProcessedPerSecond"`
	TotalBytesProcessed                   int     `json:"totalBytesProcessed"`
	TotalEntriesReturned                  int     `json:"totalEntriesReturned"`
	TotalLinesProcessed                   int     `json:"totalLinesProcessed"`
	TotalPostFilterLines                  int     `json:"totalPostFilterLines"`
	TotalStructuredMetadataBytesProcessed int     `json:"totalStructuredMetadataBytesProcessed"`
}

// Response defines the structure of an api response
type Response struct {
	Status string `json:"status"` // 根据JSON中的顶层"status"字段
	Data   Data   `json:"data"`   // 匹配嵌套的"data"字段
}

// Stats contains statistical data about processing.
type Stats struct {
	Summary Summary `json:"summary"`
}

// Data holds result data from queries or operations.
type Data struct {
	Result     []LogStream `json:"result"`
	ResultType string      `json:"resultType"`
	Stats      Stats       `json:"stats"`
}

// LogStream represents a stream of log entries.
type LogStream struct {
	Stream map[string]string `json:"stream"`
	Values [][]string        `json:"values"`
}

// CombinedResult combines two searchs from one timestamp
type CombinedResult struct {
	Results []interface{} `json:"results"`
}
