# 超大规模集群单集群优化代码仓

| 仓库名                                                                              | 分支                             | 描述                                                    | 备注                   |
|----------------------------------------------------------------------------------|--------------------------------|-------------------------------------------------------|----------------------|
| [sig-large-scale-cluster](https://gitcode.com/openFuyao/sig-large-scale-cluster) | master                         | 超大规模集群sig文档、信息仓                                       |                      |
| [kubernetes](https://gitcode.com/openFuyao/kubernetes)                           | of-v1.33.1\of-v1.34.3          | kubernetes优化仓。批量pod创建、绑定，controller-manager快速倒换优化在此仓库 |                      |
| [mind-cluster](https://gitcode.com/Ascend/mind-cluster)                          | master                         | 支持NPU（昇腾AI处理器）训练和推理硬件的深度学习组件                          | 7.2.RC1及以后版本支持超大规模集群 |
| [volcano-ext](https://gitcode.com/openFuyao/volcano-ext)                         | release-1.9                    | 基于Kubernetes构建的批处理系统。批量pod创建、绑定，podGroup调度策略优化在此仓库    |                      |
| [pytorch](https://gitee.com/ascend/pytorch)                                      | 2.1.0\2.5.1\2.6.0\2.7.1\master | PyTorch建链耗时优化在此仓库，包括TCPstore建链耗时和通信域创建耗时优化            | 所列四个分支均包含超大规模集群优化技术点 |
| [mindspore](https://gitee.com/mindspore/mindspore/tree/master/)                  | br_feature_large_scale         | mindspore分布式组网性能优化仓库                                  |                      |
| [ascend-deployer](https://gitcode.com/Ascend/ascend-deployer)                    | master                         | mind-cluster自动化部署工具                                   |                      |
| [clyde](https://atomgit.com/openeuler/clyde)                                     | master                         | 镜像P2P拉取加速组件                                           |                      |
| [kwok](https://gitcode.com/openFuyao/kwok)                                       | master                         | 超大规模集群模拟，list-watch请求模拟                               |                      |