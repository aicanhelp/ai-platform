---
sidebar_position: 3
---
# Storage

## Feature Overview

The storage management feature in Kubernetes provides a dynamic, efficient, and scalable way to manage persistent storage resources in a cluster. It allows users to request, allocate, and manage storage resources in a Kubernetes cluster, ensuring that applications can maintain data durability throughout the container lifecycle. This feature involves three core concepts: PersistentVolume (PV), PersistentVolumeClaim (PVC), and StorageClass (SC).

### Applicable Scenarios

- Persistent storage: Many applications require persistent data storage, such as databases, log storage, cache (Redis), and file storage. Persistent storage resources are required across the container lifecycle to prevent data loss after the container is restarted.

- Stateful applications: Compared with stateless applications, stateful applications need to store their status data (such as VM management and data center storage). In these scenarios, the storage management feature in Kubernetes provides functions such as dynamic expansion and persistent storage.

- Dynamic storage provisioning: In scenarios where storage resources need to be dynamically applied during runtime, the Kubernetes SC supports dynamic provisioning of storage resources, simplifying storage resource management.

### Supported Capabilities

- Flexible storage configurations: Different types of storage backends (such as cloud storage, network storage, and local storage) are supported, and the storage space and performance can be adjusted based on application requirements.

- Storage resource isolation and scheduling: PVs are bound to PVCs to ensure the isolation and proper allocation of storage resources.

### Highlights

- Automatic expansion of storage resources: Kubernetes supports automatic expansion of storage resources to ensure data scalability and durability.

- Storage resource lifecycle management: PVs and PVCs provide lifecycle management mechanisms for storage resources, including creating, modifying, and deleting storage resources.

- Access control of storage resources: Kubernetes supports multiple access modes (such as ReadWriteOnce, ReadOnlyMany, and ReadWriteMany) to ensure data access security and flexibility.

### Basic Concepts

**PV**: It is a persistent storage resource in a cluster and indicates a specific storage resource obtained from network storage (such as NFS, Ceph, and iSCSI) or local disks. An administrator creates and configures PVs, and specifies the storage type, capacity, and access mode for them.

**PVC**: It is a user's request for storage resources. The user declares the required storage space and access mode. Once a PVC is created, Kubernetes searches for a proper PV for binding based on the requirements.

**StorageClass (SC)**: It defines the attributes of a storage resource, such as the storage type and performance level. It also supports dynamic provisioning of storage resources. When an SC is specified for a PVC, Kubernetes selects a proper storage resource provisioning plug-in based on the definition of the SC to dynamically provide PVs.

### Implementation Principles

A user creates a PVC to request storage resources. Kubernetes selects a proper PV through the scheduler. If no proper PV is available, Kubernetes dynamically creates a PV based on the SC. After a PV is bound to a PVC, storage resources can be used by pods.

### Related Features

- Pod and storage: Storage resources are mounted to pods through PVCs. You need to declare PVCs for pods so that they can access persistent storage resources. To meet different storage requirements, Kubernetes provides different types of volumes to mount PVCs.

- Scheduler and storage: The Kubernetes scheduler needs to understand the availability of storage resources and ensure that pods can run on nodes with proper storage resources. Storage resources can affect pod scheduling, especially when storage resources require specific hardware support.

- Role-based access control (RBAC): The creation, modification, and deletion of storage resources are subject to permission control, especially in large-scale clusters. The RBAC mechanism restricts access to PVs, PVCs, and SCs to ensure security.

- StatefulSet: For stateful applications, Kubernetes StatefulSets are closely integrated with storage resources. StatefulSets can provide independent persistent storage for each pod through PVCs. This is ideal for scenarios such as databases and distributed cache.

- Dynamic storage management and automatic capacity expansion: The dynamic storage management function of Kubernetes supports automatic creation, deletion, and expansion of storage resources. With Horizontal Pod Autoscaler (HPA) and Cluster Autoscaler (CA) in Kubernetes, storage resources can be automatically expanded based on workloads.

## Installation

The storage management feature is installed along with the platform.

## Using a PV

A PV is a resource object in Kubernetes. It represents physical storage resources in a cluster and can be bound to a PVC. The background information is related to the storage backend, such as NFS, Ceph, and cloud storage. In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Storage** > **Persistent Volumes**. The **Persistent Volumes** page is displayed.

### Creating a PV

#### Prerequisites

- User permissions: You must have platform-admin and cluster-admin permissions.

- Storage backend configuration: Before creating a PV, ensure that supported storage backends (such as NFS, Ceph, iSCSI, and cloud storage) have been configured in the cluster. The storage system needs to be configured and managed based on specific requirements.

- Available storage resources: The physical storage referenced by the PV exists and can be accessed. For cloud storage, you need to create a volume in advance.

#### Context

A PV is an abstraction of physical storage resources in a cluster. It indicates the implementation of persistent storage and comes from volumes of different storage systems (such as block storage and file storage). PVs are configured by administrators and bound to PVCs by the cluster to provide storage services. To create a PV, you need to specify the storage capacity, access mode, storage type, and mounting mode.

#### Restrictions

- Binding restriction: A PV can be bound to only one PVC, and a PVC can be bound to only one PV. Once bound, the relationship cannot be changed unless they are manually unbound.

- Access mode restriction: Access modes (such as ReadWriteOnce, ReadOnlyMany, and ReadWriteMany) impose limitations on how the PV can be used. You must select an access mode that can meet the access requirements.

#### Procedure

1. Click **Create** in the upper-right corner of the list. The **YAML-based** page is displayed.
2. Edit the YAML file.
3. Click **OK** to complete creation.

#### Related Operations

You can view the PV information on the list page or by clicking the name of a PV. You can also modify or remove a PV by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

## Using a PVC

A PVC is a user-declared requirement for storage resources. It defines the required storage size and access mode. In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Storage** > **PersistentVolumeClaim (PVC)**. The **PersistentVolumeClaim (PVC)** page is displayed.

### Creating a PVC

#### Prerequisites

- SC definition: You need to specify an SC for a PVC to determine the storage type or match storage resources based on existing PVs.

- Available PVs: The cluster must have available PVs to meet the PVC's storage requests or support dynamic provisioning.

- PVC access mode requirements: An access mode must be specified for the PVC, which should match the required storage access permissions.

#### Context

A PVC is a user-declared request for storage resources, indicating the application's requirements for storage capacity and access modes. A user requests storage resources through a PVC, and Kubernetes binds a proper PV based on the PVC requirements. PVC is the core of storage resource allocation in Kubernetes. It abstracts storage details so that users can apply for storage in a declarative manner.

#### Restrictions

- Size limit: The storage size requested by the PVC cannot be greater than the capacity of the available PV. Once the storage size is set in the PVC, it cannot be adjusted unless you apply for a new PVC.

- Access mode restriction: The access mode requested by the PVC must match the bound PV. For example, a ReadWriteOnce PVC can only be bound to a PV that supports this access mode.

- SC restriction: If no SC is specified for a PVC, the PVC attempts to bind to a default SC or satisfy the request based on existing PVs.

#### Procedure

1. Click **Create** in the upper-right corner of the list. The **YAML-based** page is displayed.
2. Edit the YAML file.
3. Click **OK** to complete creation.

#### Related Operations

You can view the PVC information on the list page or by clicking the name of a PVC. You can also modify or remove a PVC by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.

## Using an SC

An SC defines a class of storage resources, and is used for the dynamic provisioning of PVs. It defines the attributes such as the storage type and performance. In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Storage** > **StorageClass (SC)**. The **StorageClass (SC)** page is displayed.

### Creating an SC

#### Prerequisites

- Storage provider support: Before creating an SC, ensure that supported storage provider plugins and dynamic provisioning plugins (such as volume plugins of cloud providers) are installed in the cluster.

- User permissions: You must have platform-admin and cluster-admin permissions to configure and control access to the storage backend.

- Storage parameter configuration: You need to understand the parameters of the storage backend, such as the storage type and performance requirements.

#### Context

SC is a resource object in Kubernetes. It allows administrators to define different types of storage classes and provides dynamic storage provisioning capabilities for users. Each SC is bound to a provisioner and related parameters to specify the storage resource type (such as cloud storage, NFS, and Ceph). SCs provide persistent storage for PVCs through dynamic provisioning, eliminating the need to manually configure PVs.

#### Restrictions

- Storage type restriction: The value of the **provisioner** field of an SC must match the storage plug-in supported by the cluster. For example, an AWS SC can only be used in clusters supporting AWS EBS.

- Immutability: Once an SC is created, its storage plug-in or other key information cannot be modified.

#### Procedure

1. Click **Create**. The **YAML-based** page is displayed.
2. Edit the YAML file.
3. Click **OK** to complete creation.

#### Related Operations

You can view the SC information on the list page or by clicking the name of an SC. You can also modify or remove an SC by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column on the list page or **Operation** in the upper-right corner on the details page and then selecting the desired option.
