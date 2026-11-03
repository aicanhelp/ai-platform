---
sidebar_position: 7
---
# Custom Resources

## Feature Overview

Custom resource definition (CRD) is a type of resource that can be customized by users in Kubernetes. It extends Kubernetes capabilities and enables users to define and manage their resources.

### Applicable Scenarios

CRDs can be used to describe and manage resources in specific domains, such as databases and message queues. You can define your custom resources (hereinafter referred to as CR) and use Kubernetes APIs to manage them.

### Supported Capabilities

CRDs are user-defined. You can define resource attributes, behaviors, and control logic as required.

### Highlights

CRs created based on CRDs, in coordination with the controller, can interact and integrate seamlessly with Kubernetes' core resources.

### Implementation Principles

CRDs are managed and accessed through the Kubernetes API server. You can use custom controllers to automatically manage custom resources.

### Related Features

CRD is an extension mechanism that allows users to define custom resources and manage them through Kubernetes APIs. You can create a CRD instance and use a custom controller to process it. This enables Kubernetes to better adapt to user requirements and application scenarios.

## Using a Custom Resource

In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Custom Resources**. The **Custom Resources** page is displayed.

### Creating a Custom Resource

#### Prerequisites

- Permission requirement: You must have the cluster-admin permission or the specific permission to create CRDs.

- API version compatibility: Ensure that the Kubernetes cluster version supports the required API versions (for example, apiextensions.k8s.io/v1) to prevent CRD creation failures due to version incompatibility.

- Resource planning: Specify the resource types and fields to be customized. This ensures that the CRD can accurately describe and manage customized workloads or configuration items.

#### Context

CRD is an API extension mechanism in Kubernetes. By creating CRDs, you can define new types of resources in Kubernetes, enabling clusters to support custom resources beyond native ones. CRDs apply to a wide range of application scenarios, including configuration management, controller customization, and complex application management. After a CRD is created, you can add, delete, modify, and query the CRD using kubectl commands in the same way as managing native resources.

#### Restrictions

- API stability: CRDs are extensions of Kubernetes. Their APIs are not as stable as those for native resources and may be affected by Kubernetes version changes.

- Resource management: As custom resources, CRDs require regular inspection and maintenance by cluster administrators to prevent increased management complexity due to excessive CRDs.

- Permission control: To ensure that appropriate users can access and manage custom resources, separate RBAC rules are configured to control such permissions.

- Compatibility and migration: When upgrading the Kubernetes version, you may need to migrate or upgrade existing CRDs to ensure compatibility.

#### Procedure

1. Click **Create** in the upper-right corner of the custom resource list. The **Create Custom Resource** page is displayed.

2. Edit the YAML file.

3. Click **OK** to complete creation.

#### Related Operations

- View: You can view the custom resource information on the list page or by clicking the name of a custom resource.

- Create an instance: You can switch to the **Instance** tab on the details page and click **Create** to create a custom resource instance as required.

- Modify and remove: You can modify or remove a custom resource by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

**Table 3** Operation description

| Operation| Description|
|:----------|:----------|
| View| Switch to the **YAML** tab on the details page to view the information about the custom resource in YAML format. You can export the YAML file.<br /> Choose **Instance** to view the instance information about the custom resource.|
| Modify| Click ![Input image description](figures/iconediting.png) on the **YAML** tab to modify the custom resource.|

#### Follow-up Operations

In the **Instance** tab on the custom resource details page, click **Create** to create a custom resource instance as required.
