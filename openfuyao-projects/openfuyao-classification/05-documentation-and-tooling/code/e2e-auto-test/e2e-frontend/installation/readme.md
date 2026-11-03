# 安装部署测试用例自动化

## 自动化程度

1. 已完成 验证用户修改密码成功 自动化。
2. 已完成 rbac中创建和查询角色、角色绑定、服务账号 自动化。
3. 已完成 创建集群 自动化（1Master、1Master1Node、3Master）、删除集群自动化
4. 已完成 集群扩缩容 自动化
5. 已完成 引导节点配置测试 自动化（支持引导节点作为Master或Node）
6. 已完成 管理集群升级测试 自动化（单Master/多Master升级、配置缺失场景）

## 使用方法

| 测试用例             | 对应位置                                                     |
| -------------------- | ------------------------------------------------------------ |
| 修改密码             | test-cpw.spec.js                                             |
| 创建和查询角色       | test-rbac-create.spec.js 中 test('test create role')         |
| 创建和查询角色绑定   | test-rbac-create.spec.js 中 test('test create role binding') |
| 创建和查询服务账号   | test-rbac-create.spec.js 中 test('test create service account') |
| 创建1Master集群      | test-cluster.spec.js 中 test('test create 1Master cluster')  |
| 创建1Master1Node集群 | test-cluster.spec.js 中 test('test create 1Master1Node cluster') |
| 创建3Master集群      | test-cluster.spec.js 中 test('test create 3Master cluster')  |
| 集群扩缩容单节点     | test-cluster-manage.spec.js 中 test("【集群管理-088、089、092、093】 集群扩缩容（单节点）") |
| 集群扩缩容多节点     | test-cluster-manage.spec.js 中 test("【集群管理-090、094】 集群扩缩容（多节点批量）") |
| 引导节点测试         | test-bootstrap-node.spec.js                                  |
| 单Master集群升级     | test-upgrade.spec.js 中 test('单Master业务集群升级')         |
| 多Master集群升级     | test-upgrade.spec.js 中 test('多Master业务集群升级(3M)')     |
| 集群升级配置缺失     | test-upgrade.spec.js 中 test('单Master集群升级配置缺失')     |
| 业务集群升级         | test-service-upgrade.spec.js 中 test.describe("业务集群版本升级") |
| 管理集群升级         | test-manage-upgrade.spec.js 中 test('升级管理集群到指定版本') |
| 离线创建集群并升级   | test-offline.spec.js 中 test('上传版本yaml并创建集群并升级') |

**说明：** 创建集群时一些常量在`constants/cluster.js` 中定义，如openFuyao的版本、镜像仓库地址和Chart仓库地址。创建3Master集群时3个Master分别是`TEST_MASTERIP`、`TEST_MASTER1IP`和`TEST_NODEIP` 指定的机器。

要实现不同环境下创建集群只需要修改`.env.test`中的对应节点`ip`即可。

## 用例标签说明

安装部署自动化用例会根据以下标签进行筛选用例运行：

### 场景标签

| 标签 | 含义 |
|------|------|
| `@offline` | 离线安装场景 |
| `@custom-repo` | 自定义仓库场景 |
| `@online` | 在线安装场景 |

### 运行阶段标签

| 标签 | 含义 |
|------|------|
| `@pre-init` | 引导节点初始化前 |
| `@post-init` | 引导节点已就绪 |
| `@with-management-cluster` | 管理集群已就绪 |
| `@with-workload-cluster` | 业务集群已就绪 |

### 其他标签

| 标签 | 含义 |
|------|------|
| `@skip-temporarily` | 未就绪用例，临时跳过 |

## 未来实现

1. 删除角色、角色绑定、服务账号，将`test-rbac-create.spec.js` 重命名为`test-rbac.spec.js`在其中添加对应步骤。
3. 检查应用市场和pod状态，可以新建`test-check-state.spec.js`执行各测试用例，在`utils`下新建`check.js`保存复用操作，在`constants` 中新建`check.js`定义测试用到的常量。