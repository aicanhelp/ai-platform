# Colocation Frontend E2E

当前目录用于在离线混部特性的前端自动化测试（Playwright）。

## 用例覆盖

- 概览页面：
  - 【混部-035】侧边栏混部相关链接正确性
  - 【混部-037】点击问号图标打开/关闭抽屉 - 环境准备
  - 【混部-038】点击问号图标打开/关闭抽屉 - 工作负载部署
  - 【混部-039】链接跳转到混部策略配置页面
  - 【混部-040】链接跳转到工作负载部署页面
  - 【混部-041】链接跳转到混部监控页面

- 配置页面：
  - 【混部-025/026】混部与超卖功能验证
  - 【混部-043/044/045】搜索框与搜索结果行为
  - 【混部-048/049/050/051】刷新、排序、筛选、默认开关态
  - 【混部-052/053】节点混部开关前后端一致性
  - 【混部-055/057/058】配置弹窗打开/关闭/取消
  - 【混部-062~075】负载感知调度和真实负载阈值
  - 【混部-084~099】离线负载水位线驱逐参数

- 监控页面：
  - 【混部-111/113/115/117/119/121/123/125】集群级图表展示
  - 【混部-128/130/132/134/136/138】节点级图表展示

## 关键校验链路

当前配置页关键链路采用三层校验：

1. 前端交互校验：点击开关、填写阈值、确认弹窗。
2. API 后端校验：读取 `colocation-config` / `volcano-scheduler-configmap` 确认生效。
3. 可选 SSH 校验：在配置了 SSH 环境变量后，执行 `kubectl` 核对配置和节点状态。

说明：SSH 校验目前聚焦关键开关链路（节点混部开关、混部与超卖标签验证）。

## 前置条件

- 已完成全局登录态准备（`global.setup.ts` 由 Playwright project dependency 自动执行）。
- 可访问管理面地址并具备测试账号权限。
- 运行机可通过 SSH 连接测试节点（用于自动安装/卸载依赖组件）。

## 自动安装与卸载

colocation 项目已接入专属 setup/teardown：

- 测试开始前自动安装：
  - `volcano`（`volcano-sh/volcano`, version `1.9.0`）
  - `colocation-package`（`oci://cr.openfuyao.cn/charts/colocation-package`, version `0.0.0-latest`）
- 测试结束后自动卸载以上两个 Helm release。

对应文件：

- `colocation/global.setup.ts`
- `colocation/global.teardown.ts`
- `colocation/setup-utils.ts`

## 环境变量

必填：

- `TEST_FUYAOURL`
- `TEST_USERNAME`
- `TEST_PASSWORD`

可选（启用 SSH 校验时需要）：

- `TEST_NODE2_IP`
- `TEST_NODE2_USER`（默认 `root`）
- `TEST_NODE2_PASSWORD`

## 执行方式

运行 colocation 全量前端用例：

```bash
cd e2e-frontend
npx playwright test --project=colocation
```

仅运行配置页用例：

```bash
cd e2e-frontend
npx playwright test --project=colocation colocation/test-configuration.spec.ts
```

按标签筛选运行：

```bash
cd e2e-frontend
npx playwright test --project=colocation --grep @colocation
```

## 标签规范

colocation 前端用例统一使用：

- `@v25.06`
- `@colocation`

## 已知限制

- 监控图表基于 canvas 渲染，当前自动化以可见性和无空数据提示为主，不直接断言图元内部数据点。
- 部分强依赖资源调度/驱逐的深度后端场景仍建议在后端测试体系执行（如复杂抢占编排）。

## 与旧版迁移边界

- 旧版 `v25.09` 中纯后端流程（例如复杂 Pod 抢占、部署链路编排）不在当前前端 suite 范围。
- 当前前端 suite 优先覆盖用户可见交互，并通过 API/SSH 对关键状态做闭环核验。
