# PVC Expansion Through StatefulSet Modification

## Feature Introduction

When the product management scale expands, it is necessary to adjust business instances, resource quotas, and storage resources. However, the PVC template size of a StatefulSet does not support modification, requiring manual PVC size adjustments during expansion, which is time-consuming. This feature aims to support automatic PVC expansion by modifying the PVC template size of a StatefulSet.

### Use Cases

When the product management scale expands and business instances, resource quotas, and storage resources need to be adjusted, expansion is achieved by modifying the StatefulSet template.

### Supported Scope

StatefulSet resources with PVC configured already exist in the cluster, and the SC feature switch is enabled.

### Highlights

Enables automatic PVC expansion by modifying the PVC template size of a StatefulSet.

### Implementation Principle

- Add FeatureGate: HWStatefulSetVolumeExpansion, applying to kube-apiserver and kube-controller-manager, used to enable the capability to expand PVCs by modifying StatefulSets.

- Existing features already support expanding storage resources by modifying PVCs. Currently, we only need to monitor PVC size changes declared in the StatefulSet controller and dynamically patch the PVC's storage size to achieve dynamic storage expansion.

- The StatefulSet controller adds a listener for PVC update events to update the StatefulSet status.

- Add an extended field to StatefulSet to represent the status of volumeClaimTemplates modifications.

  ```yaml
    status: 
      availableReplicas: 3 
      currentReplicas: 3 
      currentRevision: ex1-54c5bd476c 
      readyReplicas: 3 
      replicas: 3 
      updatedReplicas: 3 
      volumeClaimTemplates: 
      - finishedReconciliationGeneration: 2 
        readyReplicas: 1 
        templateName: vol1
  ```
  
### Relationship with Related Features

None.

## Installation

No installation required.

## Using PVC Expansion

### Prerequisites

None.

### Background Information

None.

### Usage Restrictions

None.

### Procedure

Using this feature requires enabling the Local CSI `allowVolumeExpansion` functionality, and automatic expansion is achieved by modifying the PVC definition.

1. Check whether the feature is enabled:

    ```shell
    kubectl get sc -A
    ```

    1.1 Check whether the `allowVolumeExpansion` value of Local CSI is `true` (enabled).
  
    1.2 If `allowVolumeExpansion` is `false`, run the following command to enable it:

    ```shell
    kubectl edit sc <sc_name>
    ```

2. Query all StatefulSets:

    ```shell
    kubectl get sts -A
    ```

3. Modify the target StatefulSet to change the PVC size configuration:

    ```shell
    kubectl edit sts -n <namespace> <sts_name>
    ```

Find the PVC definition section and modify the storage size. The corresponding section format is as follows:

```yaml
  volumeClaimTemplates:
  - apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      # ...
    spec:
      # ...
      resource:
        requests:
          storage: 30Gi  # Modify this value
      # ...
    # ...
```
