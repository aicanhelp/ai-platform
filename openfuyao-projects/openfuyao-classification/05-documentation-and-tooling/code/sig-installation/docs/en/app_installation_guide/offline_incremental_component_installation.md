# Offline Component Incremental Installation

## Background

In an offline environment where the business cluster was installed using the openFuyao community Cluster-API method, new extension components need to be installed in the existing cluster incrementally, without re-preparing offline artifacts and reinstalling the cluster.

## Environment Preparation

- Prepare a build node with public network access (used for building component incremental installation packages in online scenarios).
- Install tar, pigz, and bkeadm. This incremental package preparation uses the `oci` strategy and does not require Docker.
    - tar can be installed via the system package manager, e.g., `yum install -y tar` or `apt-get update && apt-get install -y tar`.
    - pigz can be installed via the system package manager, e.g., `yum install -y pigz` or `apt-get update && apt-get install -y pigz`; source code is available at [madler/pigz](https://github.com/madler/pigz).
    - bkeadm can be downloaded and installed using the following commands.
      ```bash
      # Download the download.sh script
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
      # Download the download.sh checksum file and verify; successful verification outputs -: OK; if verification fails, please contact openFuyao sig-installation email installation@list.openfuyao.cn or openFuyao community email contact@openfuyao.cn
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
      sha256sum -c <(cat download.sh.sha256) < download.sh
      # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool is verified during execution
      chmod +x download.sh && ./download.sh
      ```

## Installation Steps

Follow the steps below to perform incremental component installation:

1. Build image incremental installation package.

   Use the `oci` strategy to build the incremental package: no temporary image repository is started; component dependency images are synchronized to the OCI Layout directory inside the package and packaged (Docker is not required).

   1.1 Prepare the component dependency image yaml file. The following example shows the image download configuration yaml file for the openFuyao community logging-packages component.

      ```yaml
      registry:
        imageAddress: cr.openfuyao.cn/openfuyao/registry:2.8.1
        architecture:
          - amd64
          - arm64
      repos:
        - architecture: # openfuyao logging extension component
            - amd64
            - arm64
          needDownload: true # Switch for whether to download images required by the logging extension component; false - disable download, true - enable download
          subImages:
            - sourceRepo: cr.openfuyao.cn/openfuyao
              targetRepo: openfuyao
              images:
                - name: logging-operator
                  tag:
                    - latest
                - name: logging-website
                  tag:
                    - latest
                - name: oauth-proxy
                  tag:
                    - latest
            - sourceRepo: docker.io
              targetRepo: /
              images:
                - name: grafana/loki
                  tag:
                    - 2.9.1
                - name: grafana/promtail
                  tag:
                    - 2.9.3
                - name: bats/bats
                  tag:
                    - 1.8.2
      rpms:
      debs: []
      files: []
      charts: []
      ```

   1.2 Build the image incremental installation package. After completion, the image incremental installation package patch-log.tar.gz compressed file is obtained. Use the `oci` strategy to build: no temporary image repository is started; component dependency images are synchronized to the OCI Layout directory inside the package and packaged (Docker is not required).

      ```bash
      bke build patch -f offline.yaml -t patch-log.tar.gz --strategy oci
      ```

2. Copy the offline image incremental installation package to the offline bootstrap node.

   Execute the following command on the online build node to copy the image incremental installation package to the offline bootstrap node (choose the copy method based on actual conditions).

   ```bash
   scp patch-log.tar.gz root@<offline-bootstrap-node-IP>:/root/
   ```

3. Upload the Chart package to the local repository of the application repository.

   In offline scenarios, extension components cannot be pulled online from the openFuyao official Chart repository. You need to first download the target extension component's Chart package from the openFuyao official website, then import it into the local repository of the offline business cluster's application repository.

   3.1 Download the target extension component's Chart package from the [openFuyao official website](https://www.openfuyao.cn) (e.g., the logging-package Chart package): hover over the "Download" tab on the official website, click "Application Center", search for the target extension component (e.g., logging-package) in the "Enter Chart name" search bar, click to enter the component detail page, and obtain the required version's Chart package from the "Version List".

   3.2 Enter "https://_management-plane-login-IP-address_:_openFuyao-web-service-port_" in the browser, enter the username and password to log in to the openFuyao management plane of the offline business cluster.

   3.3 In the left navigation bar, select "Application Market > Repository Config" to enter the "Repository Config" page, and confirm that a local repository exists in the application repository.

   3.4 Switch to the "Package Management" tab page, click "Add Package", select the downloaded Chart package and upload it, then click "OK" to complete the addition.

   > ![icon](figures/icon-note.gif) **Note:**<br />Uploads of Chart packages in .tgz and .tar.gz formats are supported. Due to network fluctuations and other reasons, package synchronization may not be timely after adding a package, and components within the package will not be immediately displayed in the application list. You can click "Sync" on the right side of the "Package Management" tab page to manually sync to the local repository, and proceed with component installation after synchronization is complete.

4. Perform component installation.

   4.1 Offline bootstrap node image synchronization.

      ```bash
      # Extract the installation package
      tar -xzvf patch-log.tar.gz
      # Image synchronization
      bke registry patch --source ./patch-log --target 0.0.0.0:40443
      ```

   4.2 Install components through the openFuyao management plane application management of the business cluster.

      Log in to the openFuyao management plane, select "Application Market > Applications" from the left navigation bar, search for the target extension component (e.g., logging-package) in the search box in the upper right corner, and you can see the corresponding extension component.

      Click the application card to enter the application "Details" page. You can select the version to deploy in the "Version" selector, and view the component's detailed description and YAML format parameter information in the "Details" and "Default Parameters" tab pages.

      Click "Deploy", fill in the application name in the installation information, select the application version information and namespace; fill in the Helm Chart values information in Values.yaml as needed. If using the platform's preset default parameter group, no modification is needed.

      Click "Deploy" to start deploying the extension component. After deployment is complete, the method for confirming the extension component installation results can be found in the "Result Verification" section of [Online Component Incremental Installation](./online_incremental_component_installation.md).
