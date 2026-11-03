/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	DefaultActiveDeadlineSecond = 600
)

type CommandType string

const (
	CommandBuiltIn    CommandType = "BuiltIn"
	CommandShell      CommandType = "Shell"
	CommandKubernetes CommandType = "Kubernetes"
)

type CommandPhase string

// These are the valid phases of node.
const (
	CommandPending  CommandPhase = "Pending"
	CommandRunning  CommandPhase = "Running"
	CommandComplete CommandPhase = "Completed"
	CommandSuspend  CommandPhase = "Suspend"
	CommandSkip     CommandPhase = "Skip"
	CommandFailed   CommandPhase = "Failed"
	CommandUnKnown  CommandPhase = "unKnown"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CommandSpec defines the desired state of Command
type CommandSpec struct {
	// 命令执行节点
	// +optional
	NodeName string `json:"nodeName"`
	// 挂起暂不执行，可阻止下个执行的指令
	// +optional
	Suspend bool `json:"suspend"`
	// 这里的指令会按照数组顺序执行，如果上个不成功则下个不会执行，除非设置了失败跳过
	// 对于指令书写错误的直接标识失败
	Commands []ExecCommand `json:"commands,omitempty" default:"[]"`
	// 当某个命令执行失败时， 最大重试次数
	// +optional
	BackoffLimit int `json:"backoffLimit,omitempty" default:"0"`

	// 超过此时间后，不在执行。默认600
	// 当该任务暂停后，重新启动时将重新计时
	// +optional
	ActiveDeadlineSecond int `json:"activeDeadlineSecond,omitempty" default:"600"`
	// 运行完成后，超过此清理的时间则清理该任务,不设置不删除
	// +optional
	TTLSecondsAfterFinished int `json:"ttlSecondsAfterFinished,omitempty"`
	// 选定某些节点执行，NodeName需要为空
	// +optional
	NodeSelector *metav1.LabelSelector `json:"nodeSelector,omitempty"`
}

type ExecCommand struct {
	// 每条指令都必须有唯一的ID
	// +required
	ID string `json:"id"`
	// 这里要根据命令类型进行不同的指令解析
	// Type: BuiltIn，是Agent内置实现指令，比如节点Ipv4开启等，
	// 示例[]string{ipv4, dockerStorageCapacity},将检查ipv4转发是否开启， docker目录/var/lib/docker是否大于300G
	// Type: Shell，这个是要Agent执行具体的指令
	// 示例[]string{"iptables", "--table", "nat", "--list", ">", "/tmp/iptables.rule"},获取iptables规则并写入文件
	// Type: Kubernetes，这个是要获取K8s中资源或者执行里边的指令
	// 固定格式: [configmap|secret]:ns/name:ro:/tmp/secret.json
	// 只支持[configmap|secret], ns/name标识唯一资源，只有[ro|rx|rw]三个值标识[configmap|secret]资源是[只读|执行|写入]
	// 最后一个为宿主机目录，当rx时最后一个为任意值
	// 示例[]string{"secret:ns/name:ro:/tmp/secret.json"} 获取secret/ns/name资源并写入/tmp/secret.json文件
	// 示例[]string{"configmap:ns/name:rx:shell"} 获取configmap/ns/name中的资源，在agent以shell方式执行
	// 示例[]string{"configmap:ns/name:rw:/tmp/iptables.rule"} 读取/tmp/iptables.rule中的内容并写入configmap/ns/name
	// +required
	Command []string `json:"command"`
	// 指令类型
	// +required
	Type CommandType `json:"type"`
	// 当该条指令执行失败，并且达到失败重试次数时，为true则运行跳过，默认false
	// +optional
	BackoffIgnore bool `json:"backoffIgnore,omitempty"`
	// 命令执行失败时， 重试间隔时间 默认为0
	// +optional
	BackoffDelay int `json:"backoffDelay,omitempty" default:"0"`
}

// CommandStatus defines the observed state of Command
type CommandStatus struct {
	// +optional
	Conditions []*Condition `json:"conditions,omitempty"`
	// 这个时间在两处更新，一处该CRD刚刚要被处理时，由agent来更新
	// 当该任务暂停后，在此被启动的时候要cluster-api-bke来同时更新此字段
	// spec.activeDeadlineSecond 依据此字段做判断
	// +optional
	LastStartTime *metav1.Time `json:"lastStartTime,omitempty"`

	// Represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// The completion time is only set when the job finishes successfully.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty" protobuf:"bytes,3,opt,name=completionTime"`

	// The number of pods which reached phase Succeeded.
	// +optional
	Succeeded int `json:"succeeded,omitempty" protobuf:"varint,5,opt,name=succeeded"`

	// The number of pods which reached phase Failed.
	// +optional
	Failed int `json:"failed,omitempty" protobuf:"varint,6,opt,name=failed"`
	// 执行阶段
	// +optional
	Phase CommandPhase `json:"phase,omitempty"`
	// 执行结果
	// +optional
	Status metav1.ConditionStatus `json:"status,omitempty"`
}

// 每当agent开始执行此Command时便在Status中添加condition,并且阶段为pending或者running
// 当该命令执行完成后更新condition的status为true，phase为complete
// condition和spec.command是一一对应的，只有condition中存在了才允许执行下一个command

type Condition struct {
	// 每条指令都必须有唯一的ID
	// +required
	ID string `json:"id"`
	// 该命令执行的结果
	// +optional
	Status metav1.ConditionStatus `json:"status,omitempty"`
	// 该命令所在阶段
	// +optional
	Phase CommandPhase `json:"phase,omitempty"`
	// +required
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Type=string
	// +kubebuilder:validation:Format=date-time
	LastStartTime *metav1.Time `json:"lastStartTime,omitempty"`
	// +optional
	StdOut []string `json:"stdOut,omitempty"`
	// +optional
	StdErr []string `json:"stdErr,omitempty"`
	// 执行次数
	// +optional
	Count int `json:"count,omitempty" default:"0"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=cmd
// +kubebuilder:printcolumn:name="NODENAME",type="string",JSONPath=".spec.nodeName"
// +kubebuilder:printcolumn:name="SUSPEND",type="boolean",JSONPath=".spec.suspend"
// +kubebuilder:printcolumn:name="BACKOFFLIMIT",type="integer",JSONPath=".spec.backoffLimit"
// +kubebuilder:printcolumn:name="TTLSECONDSAFTERFINISHED",type="integer",JSONPath=".spec.ttlSecondsAfterFinished"

// Command is the Schema for the commands API
type Command struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CommandSpec               `json:"spec,omitempty"`
	Status map[string]*CommandStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// CommandList contains a list of Command
type CommandList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Command `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Command{}, &CommandList{})
}
