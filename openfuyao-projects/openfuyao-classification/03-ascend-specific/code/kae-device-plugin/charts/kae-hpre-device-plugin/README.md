# kae-hpre-device-plugin

KAE HPRE 设备插件

## 先决条件

需要主机鲲鹏920/920+CPU且开启KAE功能，在虚机节点场景下，需要完成设备虚拟化和虚机虚拟化设备绑定；详情见[鲲鹏KAE文档](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/encryp-decryp/devg-kae/kunpengaccel_16_0007.html)

未避免安装在所有节点上，需要配置节点亲和性，默认需要在有KAE设备的节点上打标签：accelerator/kae.hpre。亦可通过values配置修改亲和性。

```shell
kubectl label node <node-name> accelerator/kae.hpre=true
```

## 安装

```shell
helm install kae-hpre-device-plugin openfuyao-kae-deviceplugin-0.1.0.tgz -n <namespace> --create-namespace
```

## Uninstall

```shell
helm uninstall kae-hpre-device-plugin -n <namespace>
```
