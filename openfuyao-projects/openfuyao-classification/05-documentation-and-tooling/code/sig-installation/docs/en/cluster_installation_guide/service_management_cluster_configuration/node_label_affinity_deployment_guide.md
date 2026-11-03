# Node Label and Affinity Deployment Configuration

## Background Information

When using the BKE installation and deployment tool to set up a cluster, users can achieve:

1. **Node Label Capability**: Add custom labels to nodes in the cluster to describe node characteristics (such as hardware specifications, business purposes, etc.), supporting multiple label configurations.
2. **Addon Affinity Deployment**: When deploying Addon components, perform affinity scheduling based on node labels to ensure that specific Addons are only deployed on nodes that meet the conditions.

## Prerequisites

- The bootstrap node has been initialized and the bootstrap cluster is running normally.
- The cluster configuration file has been prepared.

> ![](../figures/icon-note.gif)**Note:**
>
> In the following, bke-cluster.yaml and bke-node.yaml are the files on the bootstrap node used to configure cluster installation information, located in the `/bke/cluster/` directory. These files are used to define cluster configuration, Addon configuration, and node information, etc.

## Usage Restrictions

- The `key` and `value` of node labels must comply with the [Kubernetes Label Naming Convention](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/).
---

## Operation Steps

1. Add labels to nodes.

   In the `spec` section of the `bke-node.yaml` configuration file, add the `labels` field for the target node:

   ```yaml
   apiVersion: bke.bocloud.com/v1beta1
   kind: BKENode
   metadata:
     name: master-1
     namespace: bke-cluster
     labels:
       cluster.x-k8s.io/cluster-name: bke-cluster
   spec:
     hostname: master-1
     ip: 192.168.100.226
     password: '******'
     port: "22"
     role:
     - master/node
     - etcd
     username: root
     labels:
       - key: "machine-48"
         value: "true"
       - key: "gpu"
         value: "nvidia"
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > - Each node can be configured with multiple labels.
   > - Labels will be automatically applied to the corresponding nodes when the cluster is created.

2. Add nodeSelector support in the Addon template.

   In the Addon deployment YAML file under the `bke-manifests/kubernetes/<addon-name>/<version>/` directory, add the `nodeSelector` template variable to the Pod spec:

   ```yaml
   # Example: bke-manifests/redis/latest/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: redis
   spec:
     template:
       spec:
         nodeSelector: {{ .nodeSelector }}  # Add this line to support node selection
         containers:
         - name: redis
           image: {{ .imageRepo }}/redis:latest
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > - `{{ .nodeSelector }}` will be replaced with the JSON object configured in `bke-cluster.yaml` during deployment.
   > - Only Addon templates that require affinity scheduling need to be modified.

3. Configure Addon node affinity parameters.

   In the `addons` section of `bke-cluster.yaml`, add the `nodeSelector` parameter for the Addon that requires affinity scheduling:

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: redis
         param:
           nodeSelector: '{"machine-48":"true"}'
         version: latest
   ```

   > ![](../figures/icon-note.gif)**Note:**
   >
   > - The value of `nodeSelector` is a JSON format string and must be wrapped in single quotes.
   > - Ensure that the target node has the corresponding label configured, otherwise the Addon cannot be scheduled.

4. Create the cluster.

   Execute the cluster creation command:

   ```bash
   bke cluster create -f /path/to/bke-cluster.yaml -n /path/to/bke-node.yaml
   ```

5. Verify affinity deployment.


   After the cluster is created, verify whether the Addon is deployed on the specified node:

   ```bash
   kubectl get pod -o wide -A | grep redis
   ```

   Confirm whether the node where the Pod is located contains the configured labels:

   ```bash
   kubectl get node <node-name> --show-labels
   ```

## Configuration Example

The following is a complete configuration snippet containing node labels:

**bke-cluster.yaml:**

```yaml
spec:
  clusterConfig:
    addons:
    - name: kubeproxy
      param:
        clusterNetworkMode: calico
      version: v1.33.1
    - name: calico
      param:
        calicoMode: vxlan
      version: v3.27.3
    - name: coredns
      version: v1.10.1
    - name: redis
      param:
        nodeSelector: '{"machine-48":"true"}'
      version: latest
```

**bke-node.yaml:**

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: master-1
  namespace: bke-cluster
  labels:
    cluster.x-k8s.io/cluster-name: bke-cluster
spec:
  hostname: master-1
  ip: 192.168.100.***
  password: '******'
  port: "22"
  role:
  - master/node
  - etcd
  username: root
  labels:
    - key: "machine-48"
      value: "true"
```
