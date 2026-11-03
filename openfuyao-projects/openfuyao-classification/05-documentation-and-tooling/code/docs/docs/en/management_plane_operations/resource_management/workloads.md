---
sidebar_position: 1
---
# Workloads

## Feature Overview

A workload is a runtime carrier of your services, running in a set of pods on Kubernetes. Several workload controllers are provided, including Deployment, StatefulSet, DaemonSet, Job, and CronJob. openFuyao provides Kubernetes-native container deployment and management capabilities and supports lifecycle management of containerized workloads, such as deployment, configuration, and removal.

### Applicable Scenarios

You can use different containers to deploy specific applications. Typical application scenarios are as follows:

- Stateless applications: For example, web services, API services, and microservices. These applications do not require persistent storage or a specific startup sequence. Deployment is suitable for managing such applications and supports autoscaling and rolling updates.
- Stateful applications: For example, databases, distributed caches, and message queues. These applications usually require persistent data storage and unique pod IDs. StatefulSet is suitable for managing them.
- Periodic tasks: For example, periodic data clearing and backup tasks. CronJob can be used to schedule tasks on a regular basis, ensuring they run automatically at specified time intervals.
- One-off tasks or batch processing tasks: For example, data processing and analysis. A Job can be used to ensure that a task runs to completion and then stops.
- Node-level applications: For example, log collectors and monitoring agents. DaemonSet can be used to ensure that each node runs a pod instance, which is suitable for services that need to be deployed on each node in a cluster.

> ![Input image description](figures/iconnote.gif)**Note:**
> 
> - The openFuyao platform does not verify the content of YAML files that you use to create or modify resources. You need to ensure the security of content in resource definition YAML files.
> - You must ensure the secure use and storage of sensitive information in applications. openFuyao does not scan, identify, or shield such sensitive information. You are advised to use Kubernetes Secrets for storing such sensitive information and configure Kubernetes by referring to the Kubernetes guidelines for encrypting confidential data at rest. This prevents secrets from being stored in plaintext in the etcd database.
> - Due to Kubernetes' mechanisms, after an application is uninstalled, the files or data generated on the host's storage device or distributed storage device during the running of the application will not be forcibly removed. You need to manually remove the files or data to free up the storage space.

### Supported Capabilities

- Autoscaling: The number of pods can be automatically increased based on resource usage to accommodate traffic changes.

- Lifecycle management: Lifecycle management of workloads is supported, including creation, update, deletion, and restart.

- High availability: The multi-replica fault tolerance mechanism ensures high availability of workloads, making application deployment in production environments feasible.

- Resource management: Resource requests and limits can be set to control the resource usage of each pod, optimizing cluster resource allocation.

### Highlights

- Multiple workload types: Workloads of different types (such as Deployment, StatefulSet, and Job) can be flexibly selected to meet requirements in different scenarios.

- Automatic O\&M: Functions such as autoscaling, health check, and self-healing are provided, making O\&M easier.

- Declarative configuration: You can define the expected status of a workload using a declarative YAML configuration file. Kubernetes automatically adjusts the cluster to match the expected status.

- High scalability: Deployment and management of applications in clusters of all sizes, from small-scale to large-scale, are supported, allowing enterprises to deploy applications in their environments.

### Implementation Principles

Kubernetes workloads are implemented based on the controller pattern.

- Declarative configuration: You can use YAML configuration files or kubectl commands to declare the expected status of a workload and define the number of replicas and resource requests.

- Management with controller: A controller (such as the Deployment controller) continuously detects the difference between the actual status and expected status of the workload. If the number or status of pods is not as expected, the controller automatically adjusts the pod quantity or status.

- Auto scheduling: The Kubernetes scheduler allocates workloads to suitable nodes based on resource requirements and node status. The scheduler preferentially selects nodes that meet resource requests and considers constraints such as affinity and anti-affinity.

- Self-healing mechanism: If a pod fails abnormally or a node becomes unavailable, the corresponding controller automatically creates a new pod replica to ensure workload availability.

- Status update: For workloads (such as Deployments) that support rolling updates, controllers gradually update pods to ensure that services are not interrupted during the updates. Rollback to a previous version is supported if updates fail.

### Related Features

- Service: Services are usually used to expose workloads for external access and provide load balancing and service discovery features. Services offer stable IP addresses and DNS names for inter-application communication.

- ConfigMap and Secret: Workloads can use ConfigMaps and Secrets to mount configuration files or sensitive data. Configurations can be dynamically loaded. For example, a Deployment can reference a ConfigMap to read application configurations, and a Secret is used to store passwords or authentication information.

- Volume and PersistentVolume: Stateful workloads (such as StatefulSets) usually require persistent storage. Volumes or PersistentVolumes can ensure data persistence after pods are restarted.

- Resource quotas and limits: Resource requests and limits of workloads can be controlled using Kubernetes' resource management mechanism, ensuring proper allocation and efficient use of cluster resources.

## Using a Pod

Pods are the smallest units of computing that you can create or deploy in Kubernetes. A pod encapsulates one or more containers, storage resources (volumes), an independent network IP address, and policies for controlling container running.

In the left navigation pane of the openFuyao platform, choose **Resource Management** > **Workloads** > **Pod**. The **Pod** page is displayed.

### Creating a Pod

#### Prerequisites

- Sufficient node resources: Ensure that the nodes in the cluster have adequate resources (CPU and memory) to run the new pod.

- Proper network and storage configurations: If the pod to be created depends on volumes or specific network configurations, you need to configure the corresponding storage and network plug-ins in advance.

- Appropriate permissions: You must have permissions to create pods in a specified namespace. The role-based access control (RBAC) mechanism is used to grant the create permission.

#### Context

A pod is a basic workload unit in Kubernetes and is usually managed by a controller (such as Deployment and StatefulSet). Creating a pod is the first step to deploy an application to a Kubernetes cluster. You can define configuration files to specify images, resource requests, environment variables, and volumes for pods. Directly creating a pod is usually used for temporary tasks or debugging. For formal application deployments, you are advised to use controllers to manage the pod lifecycle.

#### Restrictions

- Lifecycle restrictions: The lifecycle of a pod is immutable. Once a pod is terminated or removed, it cannot be restarted or restored. Long-running applications should be managed using controllers.
- Storage restrictions: The local storage of a pod is temporary. After the pod is restarted, data will be lost. If persistent data storage is required, mount a persistent volume.
- Namespace isolation: Pods are namespace-scoped resources. Pods in different namespaces are isolated and cannot directly communicate with each other.

#### Procedure

1. Click **Create**. The **YAML-based** page is displayed.

2. Edit the YAML file.

3. Click **OK** to complete creation.

> ![Input image description](figures/iconnote.gif)**Note:**<br />
> 
> The number of pods that can run on each node in a Kubernetes cluster is limited. The default value is **110**. If the number of pods on all nodes reaches the limit, new pods to be created cannot be scheduled and will remain in the pending state. Please clear unwanted pods or add nodes in a timely manner. Kubernetes does not limit the number of pods in the pending state. Therefore, you need to pay attention to the number of pods in this state in a timely manner to prevent the unlimited accumulation of such pods. Excessive pending pods consume CPU and memory resources, ultimately affecting cluster responsiveness.

#### Related Operations

You can view information about pods on the pod list page or click a pod name to view the pod details. You can also remove a pod as required by clicking ![Input image description](figures/icontableaction.png) in the **Operation** column or **Operation** in the upper-right corner of the details page and then selecting the deletion option.

#### Follow-up Operations

You can create, view, modify, or remove other workload resources, such as Deployments, StatefulSets, DaemonSets, Jobs, and CronJobs, by referring to "Using a Pod."
