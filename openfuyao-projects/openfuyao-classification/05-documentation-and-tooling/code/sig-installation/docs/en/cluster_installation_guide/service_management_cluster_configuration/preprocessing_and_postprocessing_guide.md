# Pre-process/Post-process Script Custom Configuration

This document introduces the pre-processing and post-processing performed during cluster installation. The former is executed during the environment initialization of cluster nodes, and the latter is executed after all addon installation tasks are distributed.

## Background Information

Clusters deployed using openFuyao require pre-processing and post-processing operations to be performed on the cluster nodes, and these operations must support user-defined content.

## Prerequisites

None.

## Usage Restrictions

- The ConfigMap data for pre-processing and post-processing operations must be placed in the user-system namespace.
- Both pre-processing and post-processing operations require a global configuration to specify the operations to be executed.

## Usage Scenarios

There are two types: pre-processing and post-processing operations. A cluster can contain either one or both.

## Pre-processing (PreProcess)

This section introduces the global configuration usage for pre-processing, applicable to scenarios where all nodes execute scripts uniformly.

### Functionality and Execution Timing

Pre-processing is executed during the `EnsureNodesEnv` phase of the cluster deployment process.
The controller checks whether a global configuration exists; if it does, the built-in command `Preprocess` is distributed to all matching nodes to execute the script.

### Global Configuration ConfigMap

The global configuration only requires creating one `ConfigMap`, and it must meet the following requirements.

- The namespace is `user-system`.
- The name is `preprocess-all-config`.
- The `data` key name is `config.json`, and the content is the global configuration for pre-processing operations.

```json
apiVersion: v1
kind: ConfigMap
metadata:
  name: preprocess-all-config
  namespace: user-system
data:
  config.json: |
    {
      "scripts": [
        {
          "scriptName": "pre-post-process-test.sh",
          "order": 10,
          "params": {
            "HTTPS_REPO": "https://repo.local"
          }
        }
      ]
    }
```

Field descriptions:

- `scripts`: Script list (required).
- `scriptName`: Script ConfigMap name (must be consistent).
- `order`: Execution order (ascending).
- `params`: Script parameters (optional).

### Pre-processing Operation ConfigMap

Each pre-processing operation must have a separate `ConfigMap` created, and it must meet the following requirements.

- The namespace is `user-system`.
- The name must be consistent with the `scriptName` field in the global configuration.
- The label is `bke.preprocess.script: "true"`.
- The `data` must contain a key with the same name, and the content is the script text.

Below is an example.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pre-post-process-test.sh
  namespace: user-system
  labels:
    bke.preprocess.script: "true"
data:
  pre-post-process-test.sh: |
    #!/bin/sh
    echo "NODE_IP=${NODE_IP}"
    echo "HTTPS_REPO=${HTTPS_REPO}"
```

### Parameter Rendering and Validation Rules

Scripts support `${PARAM_NAME}` variable substitution.

- Built-in parameter: `NODE_IP`, which can be automatically obtained from the node's IP address list.
- Custom parameters: from the `params` in the global configuration.

Parameter name rules.

- Must match: `^[a-zA-Z_][a-zA-Z0-9_]*$`.

Parameter value rules (whitelist).

- Only allowed: `a-z A-Z 0-9 - _ / . space : # =`.
- Maximum length of a single value: 4096 bytes.

If an incoming custom parameter is invalid, the script will be skipped and will not cause the overall process to fail.

### Execution and Disk Writing

The reconciler distributes the custom resource of the `Preprocess` built-in command to the APIServer, and bkeagent executes it upon detecting the command. When `NODE_IP` is not explicitly passed in, bkeagent automatically obtains the current node's IP address (the IP address bound to the first non-localhost network interface), and finally writes the rendered content to the `/etc/bkeagent/scripts` directory.

### Usage Example (Complete Workflow)

1. Prepare the yaml file for the pre-processing operation.

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: pre-post-process-test.sh
     namespace: user-system
     labels:
       bke.preprocess.script: "true"
   data:
     pre-post-process-test.sh: |
       #!/bin/sh
        echo "NODE_IP=${NODE_IP}"
        echo "HTTPS_REPO=${HTTPS_REPO}"
    ```

   

2. Prepare the yaml file for the global configuration.

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: preprocess-all-config
     namespace: user-system
   data:
     config.json: |
       {
         "scripts": [
           {
             "scriptName": "pre-post-process-test.sh",
             "order": 10,
             "params": {
               "HTTPS_REPO": "https://repo.local"
             }
           }
         ]
       }
   ```

   

3. Create the cluster's ConfigMap data.

   Execute the following command to create the namespace and ConfigMap data.

   ```bash
   # If the user-system namespace does not exist, create it first
   kubectl create namespace user-system
   
   # Assume the yaml file in step 1 is named pre_deal.yaml, and the yaml file in step 2 is pre_cfg.yaml
   kubectl apply -f pre_deal.yaml -f pre_cfg.yaml
   ```

   

4. After applying the two ConfigMaps above, all eligible nodes will execute `pre-post-process-test.sh` during the `EnsureNodesEnv` phase.

## Post-processing (postprocess)


This section only introduces the global configuration usage for post-processing, applicable to scenarios where all nodes execute scripts uniformly.

### Functionality and Execution Timing

Post-processing is executed during the `EnsureNodesPostProcess` phase of the cluster deployment process.
The controller checks whether a global configuration exists; if it does, the built-in command `Postprocess` is distributed to all matching nodes to execute the script.

### Global Configuration ConfigMap

The global configuration only requires creating one `ConfigMap`, and it must meet the following requirements.

- The namespace is `user-system`.
- The name is `postprocess-all-config`.
- The `data` key name is `config.json`, and the content is the global configuration for post-processing operations.

```json
apiVersion: v1
kind: ConfigMap
metadata:
  name: postprocess-all-config
  namespace: user-system
data:
  config.json: |
    {
      "scripts": [
        {
          "scriptName": "post-clean.sh",
          "order": 10,
          "params": {
            "LOG_DIR": "/var/log"
          }
        }
      ]
    }
```

Field descriptions for the global configuration:

- `scripts`: Script list (required).
- `scriptName`: Script ConfigMap name (must be consistent).
- `order`: Execution order (ascending).
- `params`: Script parameters (optional).

### Post-processing Operation ConfigMap

Each post-processing operation must have a separate `ConfigMap` created, and it must meet the following requirements.

- The namespace is `user-system`.
- The name must be consistent with the `scriptName` field in the global configuration.
- The label is `bke.postprocess.script: "true"`.
- The `data` must contain a key with the same name, and the content is the script text.

Below is an example.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: post-clean.sh
  namespace: user-system
  labels:
    bke.postprocess.script: "true"
data:
  post-clean.sh: |
    #!/bin/sh
    echo "post clean on ${NODE_IP}"
    echo "log dir is ${LOG_DIR}"
```

### Parameter Rendering and Validation Rules

Scripts support `${PARAM_NAME}` variable substitution.

- Built-in parameter: `NODE_IP`, which can be automatically obtained from the node's IP address list.
- Custom parameters: from the `params` in the global configuration.

Parameter name rules.

- Must match: `^[a-zA-Z_][a-zA-Z0-9_]*$`.

Parameter value rules (whitelist).

- Only allowed: `a-z A-Z 0-9 - _ / . space : # =`.
- Maximum length of a single value: 4096 bytes.

If an incoming custom parameter is invalid, the script will be skipped and will not cause the overall process to fail.

### Execution and Disk Writing

The reconciler distributes the custom resource of the `Postprocess` built-in command to the APIServer, and bkeagent executes it upon detecting the command. When `NODE_IP` is not explicitly passed in, bkeagent automatically obtains the current node's IP address (the IP address bound to the first non-localhost network interface), and finally writes the rendered content to the `/etc/bkeagent/scripts` directory.

### Usage Example (Complete Workflow)

1. Prepare the yaml file for the post-processing operation.

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: post-clean.sh
     namespace: user-system
     labels:
       bke.postprocess.script: "true"
   data:
     post-clean.sh: |
       #!/bin/sh
       echo "post clean on ${NODE_IP}"
       echo "log dir is ${LOG_DIR}"
   ```

   

2. Prepare the yaml file for the global configuration.

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: postprocess-all-config
     namespace: user-system
   data:
     config.json: |
       {
         "scripts": [
           {
             "scriptName": "post-clean.sh",
             "order": 10,
             "params": {
               "LOG_DIR": "/var/log"
             }
           }
         ]
       }
   ```

   

3. Create the cluster's ConfigMap data.

   Execute the following command to create the namespace and ConfigMap data.

   ```bash
   # If the user-system namespace does not exist, create it first
   kubectl create namespace user-system
   
   # Assume the yaml file in step 1 is named post_deal.yaml, and the yaml file in step 2 is post_cfg.yaml
   kubectl apply -f post_deal.yaml -f post_cfg.yaml
   ```

   

4. After applying the two ConfigMaps above, all eligible nodes will execute `post-clean.sh` during the `EnsureNodesPostProcess` phase.
