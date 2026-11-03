# Container Live Migration Based on UB Shared Memory

## Feature Introduction

This feature aims to provide second-level container live migration capability. It migrates containers from one node to another without stopping the container, achieving second-level switchover with no interruption or perception for upper-layer business, improving business availability. By combining with the UB resource pooling and high-speed interconnect bus capabilities, it achieves more fine-grained and flexible orchestration and scheduling capabilities, balances the load occupation of cluster nodes, and improves the overall resource utilization of the cluster.

### Application Scenarios

- As a cluster administrator, configure node resource utilization watermarks for live migration. When high resource occupation occurs on a node, automatically identify services that can be live-migrated, and migrate some businesses to other resource-sufficient nodes by issuing live migration CRs to ensure stable operation of the entire cluster.

- When the resource usage rate of cluster nodes is generally high, new Pods may fail to be scheduled due to insufficient resources on a single node, even though the overall remaining resources of the cluster can still meet their requirements. In this case, the cluster management component can automatically identify services that support migration and dynamically schedule them to free up enough resources on suitable nodes, ensuring successful scheduling and running of new Pods.

### Capability Scope

- Live migration only supports migration between homogeneous nodes within a single cluster.
- No modification to K8s code, adding new workload types for live migration, not supporting migration of existing workloads.
- Currently does not support IP address unchanged.
- Currently does not support incremental loading of huge pages and shared memory.

### Highlight Features

Based on container live migration capability, combined with resource pooling and high-speed interconnect bus provided in UB super nodes, dynamically perceive node resource changes, trigger cluster Pod rebalancing live migration, and achieve live migration downtime <1.5s under container dirty page 200MB memory specification.

### Implementation Principle

#### Container Live Migration

Through K8s's declaration mechanism, adding live migration CRD, adding live migration Controller and Agent components to reconcile live migration CRs. The Agent component needs to interact with the runtime, issuing `checkpoint/restore` interfaces, and the runtime calls CRIU for actual `checkpoint` and `restore`.

  **New Components:**

  - Migrate Controller: Live migration controller, controlling the overall migration process, reconciling CRs, issuing next operations based on current state, container running, active-standby primary running.
  - Migrate Agent: Live migration process execution component, interacting with runtime, performing `checkpoint` on source node, transferring data to target node, performing `restore` on target node, and reporting step execution status, container running, daemonset deployment.

​        **Need to Adapt Open Source Components:**

​            Containerd: Container runtime, supporting container freezing, supporting incremental checkpoint and restore.

​            Runc: Adapting incremental checkpoint and restore interfaces.

​            CRIU: Implementing incremental checkpoint and restore.

#### UB Scheduling

To implement memory sharing in container scenarios, UBS Memory interfaces need to be exposed to containers. Container processes need to explicitly call UBS Memory interfaces for shared memory creation, loading, and read/write operations. Processes use shared memory through mmap, not restricted by cgroup's memory subsystem.

Through K8s's CSI extension mechanism, implement lifecycle management of Pods using shared memory. Through custom resource declaration, support business custom shared memory quotas. When business calls shared memory creation and deletion interfaces, actively intercept requests and call the server side for quota verification and update. In K8s cloud management system, new CSI components need to be added for shared memory device mounting and quota verification, and interact with MXE to achieve end-to-end container memory sharing process.

  **New Components:**

  MatrixController: Super node memory resource controller, responsible for deciding whether nodes should perform memory borrowing or return, create container live migration tasks, etc., based on pressure of each physical node.

  - Implement new policies to screen which Pods on which nodes in the current cluster need to be migrated.
  - Implement new Evictor to screen target nodes for these Pods that need to be migrated, issue live migration CRs, and trigger Pod live migration.

  MatrixAgent: Memory resource operation executor.

  - Report node pressure, obtain memory resource usage details from current node, memory borrowing information (numa dimension), etc., create or update CRs for reporting.
  - According to CRs issued by controller, interface with MXE interfaces, execute node memory operations, report CR execution status.

#### Interaction Flow

1. Metric reporting: matrix agent periodically obtains node/NUMA memory usage, constructs CR information and reports to kube-apiserver.

2. Pressure awareness decision: After MatrixController starts, judge whether to enable rebalancing strategy execution based on node resource utilization CR updates, obtain node information and node Pod information according to configured strategies, screen Pods that need rescheduling, send to extended Evictor; Evictor screens schedulable nodes for live migration Pods through scheduler's filter and score plugins, creating live migration CRs.

3. Migrate Controller senses migration task and starts issuing first checkpoint, updating source Pod's annotation field, marking to start checkpoint.

4. Migrate Agent on source node, after detecting that Pod needs to execute checkpoint, will obtain related information from Pod's annotation and call container runtime to initiate checkpoint operation. During this process, Agent will pre-apply for shared memory for CRIU to directly write dump data into it; while CRIU on target node reads data from the same shared memory for recovery, thus requiring source node and target node to be in the same shared domain. Finally, Agent updates Pod's migration status.

5. Controller will update CR to next step (target node restore) based on Pod migration status changes, issuing next stage restore. If target Pod does not exist, first create temporary Pod on target node, and add live migration information in annotation.

6. Kubelet on target node detects newly created Pod, starts executing container creation, calls Agent's proxied CRI interface (to avoid modifying kubelet code). Agent specifies image to use checkpoint image address based on information in annotation, issues to runtime for container creation. After success, updates Pod's migration status.

7. Controller will judge whether incremental checkpoint (dirty page ratio too high, not reaching system-set threshold) or final-checkpoint is needed based on Pod migration status changes, updating source Pod annotation.

8. Migrate Agent on source node will call container runtime to execute incremental checkpoint or final-checkpoint (this operation will freeze process) according to incoming parameters, then update source Pod's migration status.

9. After final-checkpoint completes, Controller issues last-restore, updating target node Pod annotation.

10. Migrate Agent on target node will call container runtime to execute final-restore operation according to incoming parameters. After completion, Agent updates target Pod's migration status, finally making Pod enter normal running state.

11. Finally, controller does cleanup and recovery, such as deleting source Pod, deleting target Pod's annotation and other information.


## Installation

### Prerequisites

- K8s cluster has been installed.
- UB emulation/HCC scenario has been installed.
- [openFuyao/containerd](http://gitcode.com/openFuyao/containerd/tree/of-v2.1.1) has been installed, code repository branch: of-v2.1.1.
- [openFuyao/runc](http://gitcode.com/openFuyao/runc/tree/of-v1.1.10) has been installed, code repository branch: of-v1.1.10.
- [criu](http://github.com/checkpoint-restore/criu/tree/v4.2) has been installed, code repository branch: v4.2.

### Software Deployment

1. Obtain the software package required for container live migration scenario: LiveMigrate-${version}-linux.zip.

2. Deploy MatrixAgent plugin.

   2.1 On K8s master/worker nodes, in /openFuyao/live_migrate/ directory, extract MatrixAgent image.

      ```
      unzip -d image BeiMing-KubernetesPlugin-MatrixAgent-image-${version}.image.zip
      cd image 
      gunzip -c beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar.gz > beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar
      ```

   2.2 On K8s master/worker nodes, in /openFuyao/live_migrate/ directory, import MatrixAgent image.

      ```
      ctr -n k8s.io image import beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar
      ```

   2.3 On K8s master/worker nodes, confirm import success (optional).

      Execute the following command. If the return information contains beiming-kubernetesplugin-matrixagent-image:, it indicates successful import.

      ```
      ctr -n k8s.io image list | grep beiming-kubernetesplugin-matrixagent-image
      ```

   2.4 On master node in /openFuyao/live_migrate/ directory, extract MatrixAgent chart package.

      ```
      unzip -d chart BeiMing-KubernetesPlugin-MatrixAgent-chart-${version}.image.zip
      cd chart
      tar -xvf beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz
      ```
   
      2.4.1 The directory structure after extraction is similar to:

      ```
      |-- f55fdasfdsafdsafdsafdsa1fgfd24sfsdggfposx1232fdsf113.json (randomly generated filename)
      |-- beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz
      |-- 19fdafsdafdsamfasdfapiuixnyfdteuozpmnwdsfqdsafsafdsa (randomly generated folder name)
      |-- manifest.json
      |-- repositories
      ```
      
      2.4.2 Enter the only folder in the chart folder after extracting beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz.

      ```
      cd 19fdafsdafdsamfasdfapiuixnyfdteuozpmnwdsfqdsafsafdsa
      ```

      Directory structure is as follows:

      ```
      |-- json
      |-- layer.tar.gz
      |-- VERSION
      ```

   2.5 Deploy MatrixAgent service, only execute on master node.
      
      2.5.1 Check helm deployment history.

      ```
      helm list --namespace kube-system | grep m
      ```
      
      2.5.2 If the return information contains ma kube-system, MatrixAgent service was previously deployed in this environment, execute the following command to clean up.

      ```
      helm uninstall ma -n kube-system
      ```
      
      2.5.3 Execute the following command in the above directory to deploy.

      ```
      helm install ma layer.tar.gz -n kube-system
      ```

3. Deploy MatrixController plugin.

   3.1 On K8s master/worker nodes, in /openFuyao/live_migrate/ directory, extract MatrixController image.

      ```
      unzip -d imageController BeiMing-KubernetesPlugin-MatrixController-image-${version}.image.zip
      cd imageController 
      gunzip -c beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar.gz > beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar
      ```

   3.2 On K8s master node, in /openFuyao/live_migrate/ directory, import MatrixController image.

      ```
      ctr -n k8s.io image import beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar
      ```

   3.3 On K8s master/worker nodes, confirm import success (optional).

      Execute the following command. If the return information contains beiming-kubernetesplugin-matrixcontroller-image:, it indicates successful import.

      ```
      ctr -n k8s.io image list | grep beiming-kubernetesplugin-matrixcontroller-image
      ```

   3.4 On master node in /openFuyao/live_migrate/ directory, extract MatrixController chart package.
      
      ```
      unzip -d chartController BeiMing-KubernetesPlugin-MatrixController-chart-${version}.image.zip
      cd chartController
      tar -xvf beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz
      ```
       
      3.4.1 The directory structure after extraction is similar to:

      ```
      |-- apouixgnue,zhurewlkc0smxg3ube4wytb9epiknz8adfa2q3fd41a.json (randomly generated filename)
      |-- beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz
      |-- 32fdzcdsfdlfsmfasdfapiuixnyfdteuozpmnwdspamnzierzqw (randomly generated folder name)
      |-- manifest.json
      |-- repositories
      ```

      3.4.2 Enter the only folder in the chart folder after extracting beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz.
         
      ```
      cd 32fdzcdsfdlfsmfasdfapiuixnyfdteuozpmnwdspamnzierzqw
      ```

      Directory structure is as follows:

      ```
      |-- json
      |-- layer.tar.gz
      |-- VERSION
      ```

   3.5 Deploy MatrixController service, only execute on master node.
      
      3.5.1 Check helm deployment history.

      ```
      helm list --namespace kube-system | grep m
      ```

      3.5.2 If the return information contains mc kube-system, MatrixController service was previously deployed in this environment, execute the following command to clean up.

      ```
      helm uninstall mc -n kube-system
      ```

      3.5.3 Execute the following command in the above directory to deploy.

      ```
      helm install mc layer.tar.gz -n kube-system
      ```

4. Upload software package and extract, execute on both K8s master/worker nodes.

   ```
   # /openFuyao/live_migrate is the extraction path for LiveMigrate_v1.0_linux.zip software package
   mkdir -p /openFuyao/live_migrate
   unzip LiveMigrate_1.0_linux.zip -d /openFuyao/live_migrate
   ```

5. Execute the following command to deploy crd resources.

   ```
   kubectl apply -f migrate-crd.yaml
   ```

6. Deploy migrate-agent plugin.

   6.1 On K8s master/worker nodes, in /openFuyao/live_migrate directory, import migrate-agent image.

      ```
      ctr -n k8s.io image import migrate-agent-v1.0.tar
      ```

   6.2 On K8s master/worker nodes, confirm import success (optional).

      Execute the following command. If the return information contains migrate-agent, it indicates successful import.

      ```
      ctr -n k8s.io image list | grep migrate-agent
      ```

   6.3 In /openFuyao/live_migrate directory, execute the following command to deploy:

      ```
      kubectl apply -f migrate-agent_deploy.yaml
      ```

   6.4 After migrate-agent starts, modify kubelet configuration, change container-runtime-endpoint to cri proxy sock address.

      ```
      vi /usr/lib/systemd/system/kubelet.service
      ```

      Change the value of `--container-runtime-endpoint` to:

      ```
      --container-runtime-endpoint=unix:///var/run/cri-proxy.sock
      ```

   6.5 After modification, execute the following command to restart kubelet.

      ```
      systemctl daemon-reload
      systemctl restart kubelet
      ```

7. Deploy migrate-controller plugin.

   7.1 On K8s master node, in /openFuyao/live_migrate directory, execute the following command to import migrate-controller image.

      ```
      ctr -n k8s.io image import migrate-controller-v1.0.tar
      ```

   7.2 On K8s master node, confirm import success (optional).

      Execute the following command. If the return information contains migrate-controller, it indicates successful import.

      ```
      ctr -n k8s.io image list | grep migrate-controller
      ```

   7.3 In /openFuyao/live_migrate directory, execute the following command to deploy:

      ```
      kubectl apply -f migrate-controller_deploy.yaml
      ```

## Using Container Live Migration

### Background Information

Before using this feature, you need to be familiar with Kubernetes cluster basic operation commands (such as kubectl get pods, kubectl describe node).


### Usage Limitations

- Live migration supports kernel-based Checkpoint&Restore.
- Depends on UB super node's advanced scheduling capabilities, including:
  - Topology Domain aware scheduling.
  - Super node dimension resource pooling and scheduling.
  - Dynamic modeling and intelligent migration decision based on real-time load.
### Operation Steps

1. In /openFuyao/live_migrate directory, execute the following command to install live migration test Pod (skip this step if live migration test Pod already exists).

   ``` 
   kubectl apply -f mysql_stateful.yaml
   ```

2. After source Pod starts, edit migration file migrate.yaml, fill in pod, location, transportType and dataPath field information according to actual situation.

   ```
   # migrate.yaml
   apiVersion: dhac.livemigrate.livemigrate/v1
   kind: LiveMigrateController
   metadata:
     labels:
       app.kubernetes.io/name: livemigratecontroller
       app.kubernetes.io/instance: livemigratecontroller-sample
       app.kubernetes.io/part-of: live-migration
       app.kubernetes.io/managed-by: kustomize
       app.kubernetes.io/created-by: live-migration
     name: livemigratecontroller-sample
   spec:
     # TODO(user): Add fields here
     pod: livemigratestateful-mysql-0-2wx4d   # Live migration Pod name, modify according to actual situation
     srcNode: master                          # Live migration node name, modify according to actual situation
     dstNode: worker                          # Live migration target node, modify according to actual situation
     location: "xxxxxx"                       # checkpoint image data storage directory, modify according to actual situation
     transportType: "ssh"                     # Data transmission method (supports ssh, shareStorage, pageServer)
     dataPath: "xxxxxx"                       # Local storage transmission data file directory, modify according to actual situation
   ```

   > ![icon-note](../figures/icon-note.gif) **Note:**<br />  
   >
   >  Ensure location and dataPath directories exist and migrate-agent plugin has read/write permissions to directories, otherwise live migration will fail.
   > 

3. Execute the following command to trigger migration task.

   ``` 
   kubectl apply migrate.yaml
   ```

4. Execute the following command to view migration result:

   ```
   kubectl get pod -A
   ```

   When migration completes, source Pod is successfully deleted, new Pod is created normally, and data remains consistent before and after migration, indicating this migration is successful.
