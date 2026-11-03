## 内存借用示例Demo

## 使用场景

**大数据领域存在三大高内存场景**

- 复杂算子（排序、聚合、连接）
- 状态存储
- 超单机内存数据集

**传统解决方案**（数据溢出到磁盘）

- 磁盘IO瓶颈
- ms级延迟
- SSD硬件成本高

**灵衢创新方案**

- 通过远程主机借用内存，提升处理能力

**架构价值**

- 扩展内存资源，提升系统吞吐
- 降低硬件成本，提升资源利用率
- 支持清凉节点运行重负载任务

## Demo简介

```shell
flink_statebackend_demo
├── CMakeLists.txt
├── include
│   ├── FlinkTaskManager.h
│   ├── LocalMemoryAllocator.h
│   ├── MemoryManager.h
│   ├── RackMemoryAllocator.h
│   └── UBStateBackend.h
└── src
    ├── FlinkTaskManager.cpp
    ├── LocalMemoryAllocator.cpp
    ├── main.cpp
    ├── MemoryManager.cpp
    ├── RackMemoryAllocator.cpp
    └── UBStateBackend.cpp
```

示例代码为简化的Filnk调用StateBackend后端存储数据。main文件中调用2次Flink处理函数，处理函数中模拟StateBackend存储状态数据，并申请内存。第一次申请本地内存1GB，第二次申请1GB内存时，超过本地内存申请阈值，通过申请远程内存作为本地内存使用。