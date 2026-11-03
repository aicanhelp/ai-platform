# Addon Component Custom Configuration

## Background Information

The openFuyao community places addon data (in yaml and chart forms) in BKECluster, and the reconciler can perform extension installation when processing BKECluster. For an introduction to addons, see the [Appendix](./using_add-ons.md#appendix) section.

## Prerequisites

None.

## Usage Restrictions

- For chart-form addons, only the framework for declaring chart installation, uninstallation, and upgrade configuration is provided. Pre- and post-installation modifications to the host machine for chart addons must be handled by the user.  
- When a deployed chart addon contains subcharts, extracting the chart and modifying the subchart's values.yaml during deployment is not supported.  
- In offline scenarios, the user must ensure that the offline image repository and offline chart repository contain the images and charts required by the addon to be installed.  

## Operation Steps

Follow the steps below to install an addon. The operations are the same for both online and offline scenarios.

### yaml-form addon

1. Edit the bc (BKECluster) CR and add the required addon components and parameters.

   ```bash
   # View existing BKECluster
   kubectl get bc -A
   # Edit the bc of the target cluster (replace placeholders with the actual namespace and name)
   kubectl edit bc -n <namespace> <name>
   ```

   You need to declare the custom addon parameters in `Spec.ClusterConfig.Addons` of the `BKECluster` CRD to ensure the reconciler can recognize and parse these parameters. For field meanings and default values, see [Appendix: cluster-api-provider-bke Configuration Parameters](../appendix/cluster-api-provider-bke_configuration.md).

   Parameter format: Use the `Param` field to pass key-value pairs.

   Example: Customize an extension that requires specifying the image version, replica count, and custom labels:

   ```yaml
   # Configure a custom addon in the BKECluster CR
   spec:
     clusterConfig:
       addons:
       - name: "my-addon"
         version: "v1.0.0"
         param:
           image: "my-registry/my-addon:v1.0.0"
           replicas: "3"
           labels: "app=my-addon, env=prod"
         block: true # Whether to block the entire flow when deployment fails
   ```

   Parameter parsing: The code `cluster-api-provider-bke/pkg/kube` injects these parameters into the YAML template via the `prepareAddonParam` function. There is no need to modify the CRD structure; simply use the `Param` field to pass them.

2. Prepare the addon's YAML template.

   You need to place the addon's deployment YAML (such as Deployment, Service, etc.) in a specified directory to ensure the reconciler can traverse and render these templates.

   BKE reads the addon manifests via hostPath on the bootstrap/management node, with a fixed directory of `/etc/openFuyao/addons/manifests/kubernetes`. Built-in components are already placed in the format of "component name/version number/yaml" and can be referenced directly, for example:

   ```text
   /etc/openFuyao/addons/manifests/kubernetes/
     coredns/
       v1.10.1/
         coredns.yaml
     calico/
       v3.27.3/
         calico.yaml
     openfuyao-system-controller/
       latest/
         openfuyao-system-controller.yaml
   ```

   When customizing an addon, create a new `<addon-name>/<version>/` directory under the same root directory and place the yaml files in it, for example:

   ```text
   /etc/openFuyao/addons/manifests/kubernetes/
     my-addon/
       v1.0.0/
         01-deployment.yaml
         02-service.yaml
   ```

   Template variables: Dynamic parameters that need to be replaced in the YAML (such as image address and replica count) must use template placeholders, and the reconciler injects parameters via functions such as `parseFabricParam`.

    > ![icon](../figures/icon-note.gif) **Note:**<br />File name prefixes (such as `01-`, `02-`) affect the deployment order (executed in ascending order), and deletion is executed in descending order.

   The following shows the 01-deployment.yaml file, where template placeholders are injected via parameters.

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: my-addon
     namespace: {{.namespace}} # The cluster namespace injected by the reconciler
   spec:
     replicas: {{.replicas}} # References the replicas in the addon parameters
     selector:
       matchLabels:
         app: my-addon
     template:
       metadata:
         labels: {{.labels}} # References the labels in the addon parameters
       spec:
         containers:
         - name: my-addon
           image: {{.image}} # References the image in the addon parameters
   ```

3. Handle special logic for the addon (optional).

    If the addon requires pre/post operations (such as creating a Secret, configuring permissions, or depending on other components), you need to add custom logic in the reconciler code.

    - **Pre-operation**: Add a branch in `EnsureAddonDeploy.addonBeforeCreateCustomOperate` to handle pre-deployment preparations (such as creating storage directories, configuring RBAC).
    - **Post-operation**: Add a branch in `EnsureAddonDeploy.addonAfterCreateCustomOperate` to handle post-deployment wrap-up work (such as verifying status, registering resources).

4. Execute the deployment.
   
   Configure the custom addon in `Spec.ClusterConfig.Addons` of the `BKECluster` CR, and the reconciler detects the new addon in the `EnsureAddonDeploy` phase and automatically executes the deployment process.  
   4.1. Compare Spec (desired state) with `Status.AddonStatus` (actual state) and trigger deployment after a difference is detected.  
   4.2. Parse parameters and render the YAML template.  
   4.3. Deploy to the target cluster via `kube.InstallAddon` calling `kubectl apply`.  
   4.4. Update `Status.AddonStatus` to record the deployment result.  

5. Verify the deployment.

   - Check whether the corresponding resources were created in the target cluster (`kubectl get deployment my-addon -n <namespace>`).
   - View the status of `BKECluster` (`kubectl describe bkecluster <name> -n <namespace>`) and confirm that `ClusterAddonCondition` is True and `Status.AddonStatus` contains a record for `my-addon`.
  
6. Addon update and uninstallation (optional).
   
   - Update: Modify the `version` or `param` of the addon in `BKECluster`, and the reconciler automatically triggers the update process (re-renders the YAML and reconciles).
   - Uninstall: Remove the addon from `Spec.ClusterConfig.Addons`, and the reconciler executes deletion operations in descending order of YAML file names (`kubectl delete -f <file>`).

### chart-form addon

> ![icon](../figures/icon-notice.gif) **Notice:**<br/>
> This feature does not perform any security, integrity, or compliance checks on the chart package content. Please ensure the security of the chart package before deployment.

1. Edit the bc (BKECluster) CR and add the chart repo.

   ```bash
   kubectl get bc -A
   kubectl edit bc -n <namespace> <name>
   ```

   You need to declare the chart repo in `spec.clusterConfig.cluster.chartRepo` of the `BKECluster` CRD, and declare the authentication information for the chart repo.

   Example: Declare a private chart repo with a self-signed certificate, as shown below.

   ```yaml
   # Configure a custom addon in the BKECluster CR
   spec:
     clusterConfig:
       cluster:
         chartRepo: 
           domain: my.chartrepo.cn  # Domain name; when the chart repo has no domain name, an IP address can be filled in 
           ip: 192.168.0.2          # IP address
           port: "443"              # Open port number
           prefix: openfuyao        # Domain prefix for pulling charts; for example, the chart pull URL in this repo is my.chartrepo.cn/openfuyao
           insecureSkipTLSVerify: false     # Whether to skip TLS verification, default is false
           authSecretRef:                   # Chart repo authentication information; can be omitted for public chart repos
             name: chart-repo-auth-secret   # Secret name
             namespace: bke-cluster         # Namespace where the Secret resides
             usernameKey: username          # Key storing the username in the Secret; defaults to username if not specified
             passwordKey: password          # Key storing the password in the Secret; defaults to password if not specified
           tlsSecretRef:                    # Chart repo TLS authentication information; can be omitted when no TLS verification is required
             name: chart-repo-tls-secret    # Secret name
             namespace: bke-cluster         # Namespace where the Secret resides
             certKey: tls.crt               # Key storing the certificate in the Secret; defaults to tls.crt if not specified
             caKey: ca.crt                  # Key storing the CA in the Secret; defaults to ca.crt if not specified
             keyKey: tls.key                # Key storing the private key in the Secret; defaults to tls.key if not specified
   ```
   
   > ![icon](../figures/icon-note.gif) **Note:**<br /> 
   > Public chart repos, private chart repos, and chart repos with TLS authentication are supported.
   > Both OCI-format charts and traditional-format charts are supported for pulling.
   > Only one chart repo can be configured in a single bc CR; pulling charts from multiple chart repos at once is not supported.

   If connecting to a private chart repo, you need to save the login authentication information in a management cluster Secret. The Secret structure is as follows.

   ```yaml
   # Secret YAML definition corresponding to authSecretRef
   apiVersion: v1
   kind: Secret
   metadata:
      name: chart-repo-auth-secret
      namespace: bke-cluster
   type: Opaque
   data:
      username: <base64-encoded-username>  # Example: replace with the actual base64-encoded username, e.g. echo -n 'your-username' | base64
      password: <base64-encoded-password>  # Example: replace with the actual base64-encoded password, e.g. echo -n 'your-password' | base64
   ```

   If connecting to a chart repo with a self-signed certificate, you need to save the TLS certificate authentication information in a management cluster Secret. The Secret structure is as follows.

   ```yaml
   # Secret YAML definition corresponding to tlsSecretRef
   apiVersion: v1
   kind: Secret
   metadata:
     name: chart-repo-tls-secret
     namespace: bke-cluster
   type: Opaque
   data:
     tls.crt: <base64-encoded-client-certificate>  # Example: replace with the actual base64-encoded client certificate content, e.g. cat client.crt | base64 -w 0
     ca.crt: <base64-encoded-ca-certificate>       # Example: replace with the actual base64-encoded CA certificate content, e.g. cat ca.crt | base64 -w 0
     tls.key: <base64-encoded-private-key>         # Example: replace with the actual base64-encoded private key content, e.g. cat client.key | base64 -w 0
   ```

2. Edit the bc (BKECluster) CR and add the chart addon.

   ```bash
   kubectl edit bc -n <namespace> <name>
   ```

   You need to declare the chart addon configuration in `spec.clusterConfig.addons` of the `BKECluster` CRD.

   Example: Customize a logging-package addon and configure a custom values.yaml, as shown below.

   ```yaml
   # Configure a custom addon in the BKECluster CR
   spec:
     clusterConfig:
       addons:
       - name: logging-package     # Chart name; the chart will be pulled based on this name, required
         release: logging          # Release name; a release will be created based on this name, optional; default value is the addon name
         type: chart               # Addon type; for chart addons, this must be set to chart
         namespace: logging        # Namespace where the chart is installed, required
         timeout: 5                # Chart installation and upgrade timeout, optional; default value is 5 minutes
         valuesRefConfigMap:       # Chart package values.yaml reference configuration, optional; uses the chart package default values.yaml by default
           name: logging           # ConfigMap name, required
           namespace: bke-cluster  # Namespace where the ConfigMap resides, required
           valuesKey: values.yaml  # values.yaml file name, optional; default value is values.yaml
         version: 0.0.0-latest     # Chart version, required
         block: false              # Whether to block the deployment of other addons during deployment, optional; default value is false
   ```

   If you configure a custom values.yaml, you need to save the values.yaml in a management cluster ConfigMap. The ConfigMap structure is as follows.

   > ![icon](../figures/icon-note.gif) **Note:**<br />
   > The values.yaml in the example must be modified to the real configuration. You can obtain the real values.yaml from the [logging-operator repository](https://gitcode.com/openFuyao/logging-operator/blob/master/charts/logging-package/values.yaml).

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: logging
     namespace: bke-cluster
   data:
     values.yaml: |  # The values.yaml content below is an example configuration and must be modified to the real configuration
       replicaCount: 2
       image:
         repository: cr.openfuyao.cn/openfuyao
         tag: "2.4"
       service:
         type: ClusterIP
         port: 80
   ```

   > ![icon](../figures/icon-notice.gif) **Notice:**<br/> 
   > Modifying the configMap that stores the values.yaml after deploying the chart will not trigger the reconciler to update the chart addon. 
   > In an offline scenario, the user must modify the image pull address in values.yaml.

3. Handle special logic for the addon (optional).

   If the addon requires pre/post operations (such as creating a Secret, configuring permissions, or depending on other components), you need to add custom logic in the reconciler code.

    - **Pre-operation**: Add a branch in `EnsureAddonDeploy.addonBeforeCreateCustomOperate` to handle pre-deployment preparations (such as creating storage directories, configuring RBAC).
    - **Post-operation**: Add a branch in `EnsureAddonDeploy.addonAfterCreateCustomOperate` to handle post-deployment wrap-up work (such as verifying status, registering resources).

4. Execute the deployment.

   Configure the custom addon in `spec.clusterConfig.addons` of the `BKECluster` CR, and the reconciler detects the new addon in the `EnsureAddonDeploy` phase and automatically executes the deployment process.  
   4.1. Compare Spec (desired state) with `Status.AddonStatus` (actual state) and trigger deployment after a difference is detected.  
   4.2. Pull the chart package to a local temporary file via `kube.InstallAddon` calling the helm sdk.  
   4.3. Deploy the chart to the target cluster via the helm sdk.  
   4.4. Update `Status.AddonStatus` to record the deployment result.  

5. Verify the deployment.

    - Check whether the corresponding resources were created in the target cluster (`kubectl get deployment my-addon -n <namespace>`).
    - View the status of `BKECluster` (`kubectl describe bkecluster <name> -n <namespace>`) and confirm that `ClusterAddonCondition` is True and `Status.AddonStatus` contains a record for `my-addon`.

6. Addon update and uninstallation (optional).

    - Update: Modify the `version` of the addon in `BKECluster`, and the reconciler automatically triggers the update process.
    - Uninstall: Remove the addon from `spec.clusterConfig.addons`, and the reconciler automatically deletes the chart addon.

## Appendix

The following introduces common extension components.

- K8s ecosystem components

   ```yaml
   - name: kubeproxy
     version: v1.33.1-of.1
     block: true # Optional, default false. Set to true to block, meaning subsequent extensions will only be deployed after this component is started
     param: # Parameters required to deploy this component
       clusterNetworkMode: calico
   - name: calico
     version: v3.25.0
     param:
       calicoMode: bgp
   - name: coredns
     version: v1.10.1
   - name: nfs-csi
     version: v4.1.0
     block: true
     param:
       nfsServer: 172.28.8.186
   ```

   - The reconciler deploys extension components in the configured order.
   - If the installation of an extension strongly depends on services started by a preceding extension, you can set the preceding extension's block parameter to true to perform a blocking installation.
   - If you need to install non-network extensions, place them after the network extensions.

- Cluster network components

   > ![icon](../figures/icon-note.gif) **Note:**<br />1. Please select different combinations of network components based on the examples below and your actual situation, and write them at the beginning of the addon list.<br />2. To ensure that subsequent components can start normally, all addons related to cluster networking shown below must have the `block: true` parameter set.<br />3. For detailed addon parameters and parameter descriptions, see the examples below.
   
   The following introduces two common network plugins, Fabric and Calico.
   > ![icon](../figures/icon-note.gif) **Note:**<br />IPVS is only supported when using Calico with kube-proxy; Fabric does not support it.

   - Fabric

     Fabric supports both Overlay and Underlay modes.

     - Overlay mode

       Pods use an independent CIDR and communicate across nodes through VXLAN tunnel encapsulation, suitable for scenarios with restricted underlying networks.

       ```yaml
       addons:
       - name: fabric
         block: true
         param:
           fabricMode: overlay
           cidrBlock: "10.250.0.0/16" # Fill in according to actual situation; after filling, also set bkecluster.spec.clusterConfig.cluster.networking.podSubnet to be consistent
           excludeIps: "" # Fill in "" if none; supports IP ranges and single IPs, separated by ",". eg:"10.250.0.1-10.250.0.10,10.250.0.55"
           tunnelType: vxlan # geneve gre erspan vxlan
           subCIDRMask: "24" # Fill in according to actual situation
         version: 2.6.2
       - name: coredns
         block: true
         version: v1.10.1
       ```
       
     - Underlay mode

       Pods directly use VPC subnet IP addresses, relying on the underlying network for communication, with no encapsulation and high performance.

       ```yaml
       addons:
       - name: fabric
         block: true
         param:
           fabricMode: underlay
           cidrBlock: "10.250.0.0/16" # Fill in according to actual situation; after filling, also set bkecluster.spec.clusterConfig.cluster.networking.podSubnet to be consistent
           excludeIps: "" # Fill in "" if none; supports IP ranges and single IPs, separated by ",". eg:"10.250.0.1-10.250.0.10,10.250.0.55"
           vlanID: "" # Fill in according to actual situation
           gateway: "10.250.0.254" # Fill in according to actual situation
           mask: "24" # Fill in according to actual situation
           dataPlaneUnified: "eth1" # Fill in according to actual situation
         version: 2.6.2
       - name: coredns
         block: true
         version: v1.10.1
       ```

   - Calico
     
     Calico is a networking and network policy plugin for Kubernetes. Calico-Typha is its middleware caching proxy component used in large-scale clusters to reduce data storage load. By aggregating Felix connections, filtering irrelevant updates, and scaling horizontally, it significantly reduces the pressure on the Kubernetes API Server/etcd and optimizes network policy distribution efficiency. You can adjust the configuration via the following parameters according to actual business needs.
    
     - calicoMode: Used to set the network mode of Calico. Optional values are `vxlan`, `bgp`, and `ipip`.
     - proxyMode: Used to set the forwarding mode of kube-proxy. Optional values are `iptables` and `ipvs`. This parameter is optional.
     - allowTypha: Used to configure whether to enable Typha. Typha is used to reduce data storage caching. The mode selection of `calicoMode` and `proxyMode` does not affect the use of Calico-Typha.
     - typhaReplicas: Used to set the number of Typha replicas. According to the [calico official documentation](https://docs.tigera.io/calico/latest/getting-started/kubernetes/self-managed-onprem/onpremises#install-calico), it is recommended to enable `Calico-Typha` when the cluster scale exceeds 50 nodes. For every additional 200 nodes, at least 1 Typha replica should be added, and the total number of Typha replicas must be less than the total number of cluster nodes.

     ```yaml
     # The kube-proxy version follows the k8s version
     - name: kubeproxy
       param:
         proxyMode: iptables # iptables, ipvs
       version: v1.33.1-of.1
       block: true
     - name: calico
       param:
         calicoMode: vxlan # vxlan bgp ipip
         allowTypha: true
         typhaReplicas: 2 # Fill in according to actual situation
       version: v3.31.3 # Only versions 3.31.3 and above support typha-related configuration, otherwise it will not take effect.
       block: true
     - name: coredns
       version: v1.10.1
       block: true
     ```
- DNS cache component
 
  The DNS cache component is a middleware software located between the DNS client and the DNS server, used to cache DNS query results, reducing the latency and network overhead of repeated queries.
  
  - node-local-dns

    As an internal DNS resolution enhancement component for Kubernetes clusters, node-local-dns can cache DNS query results at the node level, effectively reducing CoreDNS load, improving service discovery speed, and lowering network latency. It also supports automatic retry and failover to enhance DNS resolution reliability, improving the business stability and concurrent processing capability of clusters at ultra-large scales. You can adjust the configuration via the following parameters according to actual business needs:

    - localdns: Used to configure the listening address. In IPTABLES mode, the node-local-dns Pod listens on both the kube-dns service IP address and the `<localdns>` address, so that Pods can use either IP address to query DNS records. In IPVS mode, because the interface used by IPVS load balancing has already occupied that address, the node-local-dns Pod only listens on the `<localdns>` address; therefore, the node-local-dns interface cannot bind the cluster IP address of kube-dns.
    
    ```yaml
    - name: nodelocaldns            
      param:
        localdns: 10.96.0.10 # Fill in according to actual situation; take care to avoid conflicts with ports already in use
      version: v1.26.4               
    ```
