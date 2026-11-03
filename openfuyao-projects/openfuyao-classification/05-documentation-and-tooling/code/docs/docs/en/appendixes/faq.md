---
sidebar_position: 4
---

# FAQ

## 1. Newly added packages not displayed correctly on the package management page

This may be caused by network fluctuations or other factors resulting in synchronization failure. Click "Synchronize" manually to resolve it.

## 2. Error message "unable to find index.yaml, please provide correct ChartMuseum project URL" displayed when adding a remote repository

The remote repository URL is incorrect. Ensure that the URL specifies the exact project name in the Harbor repository. For example, the openFuyao official repository `https://helm.openfuyao.cn` specifies the project "openfuyao-catalog".

## 3. After a platform administrator is invited to a cluster and assigned a cluster role, the corresponding permissions do not take effect

If a user is already assigned the platform admin role, assigning a cluster-level role afterwards will still result in platform admin permissions. The platform admin role has the highest priority.

## 4. A cluster admin user cannot invite other users to join the cluster

The main reason is that the user does not have the platform admin role. The platform admin role must be assigned to the user first before they can invite other users.

## 5. Deployment fails when redeploying after uninstalling an application

Application uninstallation takes some time. Clicking deploy immediately may result in deployment failure due to incomplete uninstallation. It is recommended to wait a few minutes before redeploying.

## 6. Abnormal behavior when accessing the openFuyao platform log module using Firefox

When using Firefox to access the openFuyao platform log module, if the page fails to load or data displays abnormally, manually clear the browser cache to resolve the issue.

## 7. Time differences between the browser and server may affect some monitoring data

When there is a time difference between the current browser and the server, data delays or misalignment may occur. To prevent ongoing impacts, check the local device time settings and ensure synchronization with the server time. After adjusting the browser time, it may take some time for the changes to take effect.

## 8. How to change the port exposed by the openFuyao management plane

The default port number of the openFuyao management plane is 31616. To change the default port number to another port, follow the steps below for the v25.03 version.
> ![icon](figures/icon-note.gif) **Note:**<br />This step uses the installation package fuyao-v25.03-openEuler-22.03-amd64 as an example. The procedure for other installation packages is similar.

- Operation Steps

  1. After downloading the openFuyao installation package, decompress the shell script compressed package in the asset directory.

     ```shell
     cd asset
     tar -zxvf fuyao-openEuler-22.03-amd64.tar.gz
     ```

  2. Edit the file `fuyao-openEuler-22.03-amd64/ingress/ingress-nginx/resource/ingress-nginx.yaml`, find `nodePort: 31616` around line 372, and change 31616 to another port number. The modification is as follows.

     ```shell
     367 - appProtocol: https
     368   name: https
     369   port: 443
     370   protocol: TCP
     371   targetPort: https
     372   nodePort:  31616 # Modification item
     ```

  3. After completing the modification, delete the original `fuyao-openEuler-22.03-amd64.tar.gz`, and compress `fuyao-openEuler-22.03-amd64` into `fuyao-openEuler-22.03-amd64.tar.gz`.

     ```shell
     rm -f fuyao-openEuler-22.03-amd64.tar.gz
     tar -czvf fuyao-openEuler-22.03-amd64.tar.gz ./fuyao-openEuler-22.03-amd64/*
     ```

  4. The port number modification is now complete. You can continue to install the cluster according to the [Installation Guide](https://gitcode.com/openFuyao/sig-installation/blob/master/docs/en/cluster_installation_guide/read_before_installation/installation_process.md). After the cluster installation is complete, use the modified port number to access the openFuyao management plane.

## 9. Pod startup failure with "failed to create fsnotify watcher: too many open files" in kubectl logs

This may be because the system default value of fs.inotify.max_user_instances is too small. Reset this value on the node where the pod is located. For multi-node cluster deployments, reset fs.inotify.max_user_instances on all nodes to avoid errors.

Solution 1: Temporarily set the fs.inotify.max_user_instances value.

```bash
sudo sysctl fs.inotify.max_user_instances=8192
```

> ![icon](figures/icon-note.gif)**Note:**
> 
>  The setting must be reapplied each time you re-enter the console.

Solution 2: Permanently save the fs.inotify.max_user_instances value.

```bash
echo fs.inotify.max_user_instances=8192| tee -a /etc/sysctl.conf && sudo sysctl -p
```

## 10. Insufficient space in the root partition while other partitions still have free space

For cases where the root partition has insufficient space but other partitions have free space, consider migrating container-related directories to other partitions. For example, if a partition with free space is mounted on /home, follow the steps below to migrate:

1. Create a directory under `/home`.

   ```bash
   mkdir /home/containerd
   ```

2. Stop kubelet and containerd in sequence.

   ```bash
   systemctl stop kubelet
   systemctl stop containerd
   ```

3. Modify the containerd configuration file.

   First locate the configuration file used by containerd, which defaults to /etc/containerd/config.toml.
   Edit the file, record the original root directory, and change it to `root = "/home/containerd"`.

4. Migrate the original files to the newly created directory.

   ```bash
   cp -r ${original_root}/* /home/containerd/
   ```

5. Start containerd and kubelet in sequence.

   ```bash
   systemctl start containerd
   systemctl start kubelet
   ```

> ![icon](figures/icon-note.gif)**Note:**<br />
> The feasibility and stability of this solution in production scenarios have not been verified. It is **not recommended** for use in production environments.
