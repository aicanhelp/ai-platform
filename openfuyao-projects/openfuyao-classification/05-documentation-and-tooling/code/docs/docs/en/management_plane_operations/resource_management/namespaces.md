---
sidebar_position: 6
---
# Namespaces

## Feature Overview

In Kubernetes, namespaces provide a mechanism for isolating and organizing resources. Namespaces divide a cluster into multiple virtual clusters, with resources in each namespace isolated from each other.

### Applicable Scenarios

Namespaces are used to isolate and manage resources of different applications, environments, or teams to ensure resource security and reliability.

### Supported Capabilities

Namespaces can contain various resources, such as pods, Services, and Deployments, and allow permission control over the resources within them.

### Highlights

Namespaces can be categorized and selected using labels and selectors, enabling flexible resource management and access control.

### Implementation Principles

As one of the core concepts in Kubernetes, namespaces are specified in resource metadata to isolate and organize resources.

### Related Features

Namespaces logically isolate and group other resources in Kubernetes. In a namespace, you can create and manage various resource objects, such as pods, Deployments, Services, and ConfigMaps. These resource objects are visible and accessible only in their owning namespaces.

## Using a Namespace

In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Namespace** > **Namespace**. The **Namespace** page is displayed.

### Creating a Namespace

#### Prerequisites

- Access permission: You must have permissions to manage the cluster or create namespaces.

- Configuration file: Ensure that the kubectl configuration file points to the correct cluster.

- Naming rule: The namespace name must comply with the Kubernetes naming rules. For example, the namespace name must not contain uppercase letters or special characters.

#### Context

Namespaces are used to logically isolate resources in a Kubernetes cluster. They enable multiple teams or projects to share a cluster without interference. Resource quotas, permission control, and network policies can be independently configured for each namespace to implement grouped and isolated resource management.

#### Restrictions

- Resource isolation: Resources in different namespaces are isolated by default. Pods in different namespaces cannot directly communicate with each other unless network policies are configured.

- Quota limit: A resource quota can be set within a namespace to limit its resource usage.

- Deletion impact: Deleting a namespace will clear all resources within it. Please proceed with caution to prevent resource loss caused by misoperations.

#### Procedure

1. Click **Create** in the upper-right corner of the **Namespace** list. The **Create Namespace** page is displayed.

2. Edit the YAML file.

3. Click **OK** to complete creation.

#### Related Operations

You can view the namespace information on the list page or by clicking the name of a namespace. You can also modify or remove a namespace by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

## Using a LimitRange

In the navigation pane of the openFuyao platform, choose **Resource Management** > **Namespace** > **LimitRange**. The **LimitRange** page is displayed.

### Creating a LimitRange

#### Prerequisites

- Namespace availability: The LimitRange must be defined in a namespace. Therefore, you need to create or specify a namespace in advance.

- Determined resource requirements: Resource allocation requirements for the namespace are clearly defined, including specific values for resource requests and limits.

- Permission requirements: You must have permissions to create a LimitRange, which are typically granted to a namespace administrator or a cluster administrator.

#### Context

A LimitRange is a policy in Kubernetes to control resource usage by pods and containers within a namespace. It allows setting resource request and limit values at the namespace level, defining upper and lower limits for resources such as CPU and memory that can be used by each pod or container. This mechanism prevents resource overuse and ensures proper allocation of cluster resources.

#### Restrictions

- Effective only within a namespace: LimitRange is a namespace-scoped configuration, and its restrictions apply only to pods and containers within the specified namespace. LimitRanges in different namespaces are independent of each other and cannot be applied across namespaces.

- Inability to limit the total resource usage of a namespace: LimitRange can only define the range of resource requests and limits for each pod or container, but cannot control the total amount of resources in a namespace. To limit the total amount of resources in a namespace, use ResourceQuota.

#### Procedure

1. Click **Create** in the upper-right corner of the LimitRange list. The **Create LimitRange** page is displayed.

2. Edit the YAML file.

3. Click **OK** to complete creation.

#### Related Operations

You can view the LimitRange information on the list page or by clicking the name of a LimitRange. You can also modify or remove a LimitRange by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

**Table 1** LimitRange-related operations

| Operation| Description|
|:----------|:----------|
| View| Switch to the **YAML** tab on the details page to view the information about the LimitRange in YAML format. You can export the YAML file.|

## Using a ResourceQuota

In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Namespace** > **ResourceQuota**. The **ResourceQuota** page is displayed.

### Creating a ResourceQuota

#### Prerequisites

- Namespace availability: A ResourceQuota takes effect at the namespace level. Therefore, ensure that the target namespace exists.

- RBAC permission: You must have permissions to create or update resources, which are configurable using RBAC.

#### Context

A ResourceQuota is used to limit the total amount of resources that can be used in a namespace. It allows administrators to allocate a fixed amount of resources to each namespace. This prevents a namespace from occupying too many cluster resources and affecting applications in other namespaces.

#### Restrictions

- Aggregate limits: A ResourceQuota limits the total amount of resources in a namespace, not for a single pod or container.

- Resource type restrictions: It can limit various resource types, such as CPU, memory, storage, and the number of pods.

- Monitoring and optimization: In practice, ResourceQuotas may need to be adjusted based on loads to ensure the normal running of applications.

#### Procedure

1. Click **Create** in the upper-right corner of the ResourceQuota list. The **Create ResourceQuota** page is displayed.

2. Edit the YAML file.

3. Click **Save** to complete creation.

#### Related Operations

You can view the ResourceQuota information on the list page or by clicking the name of a ResourceQuota. You can also modify or remove a ResourceQuota by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

**Table 2** ResourceQuota-related operations

| Operation| Description|
|:----------|:----------|
| View| Switch to the **YAML** tab on the details page to view the information about the ResourceQuota in YAML format. You can export the YAML file.|
| Modify| Click ![Input image description](figures/iconediting.png) on the **YAML** tab to modify the ResourceQuota.|
