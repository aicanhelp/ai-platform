bkecommon

### 概要

>- 该项目主要用于bke其他项目的公共模块，该项目并不能直接运行
>- 该项目的变更需要经过严格测试，关键的公共方法必须要配备×_test代码
>- 尽量少的引用大的项目代码，避免依赖冲突

### 简介

#### cluster

>- cluster-api的配置文件十分复杂难懂，对于未进行深入了解cluster-api的人员来说上手难度太高，经过多方考察研究发现kind的配置文件，
算是十分清晰明了的，所以该配置文件将集合Kind配置文件/kubeadm配置文件所长形成独特的bke的配置文件
>- 该模块是bkeadm部署kubernetes集群的配置文件，bkeadm/bkeagent/cluster-api-bke均需要加载该配置并解析内容
>- 该模块主要实现bkecluster.yaml解析、校验、转换等


#### other module

>- cluster
>- ntp
>- security
>- source
>- constants