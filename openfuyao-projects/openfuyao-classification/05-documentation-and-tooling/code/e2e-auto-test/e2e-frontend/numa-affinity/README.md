# NUMA Affinity Frontend E2E

当前目录用于 NUMA 亲和插件的前端功能测试（Playwright）。

## 用例覆盖

- 【NUMA2503-046】NUMA监控-亲和性优化pod-表格基础信息展示
- 【NUMA2503-001】NUMA亲和性调度开关开启（前端点击 + 节点SSH校验）
- 【NUMA2503-002】NUMA亲和性调度开关关闭（前端点击 + 节点SSH校验）
- 【NUMA2503-017】最优NUMAdistance开关开启（前端点击 + 节点SSH校验）
- 【NUMA2503-018】最优NUMAdistance开关关闭（前端点击 + 节点SSH校验）
- 【NUMA2503-019】最优NUMAdistance开关开启（预配置场景，前端点击 + 节点SSH校验）
- 【NUMA2503-006】NUMA亲和性调度开关开启（预配置场景，前端点击 + 节点SSH校验）
- 【NUMA2503-003】NUMA亲和性调度-页面基础信息展示
- 【NUMA2503-021】最优NUMAdistance-页面基础信息展示
- 【NUMA2503-036】运行时Pod亲和优化-页面基础信息展示
- 【NUMA2503-034】运行时Pod亲和优化开关开启（前端点击 + 创建任意工作负载）
- 【NUMA2503-013】亲和策略配置-认证鉴权-未认证修改策略（前台拦截与返回校验）
- 【NUMA2503-022】最优NUMAdistance-开关打开-端到端基础功能验证（兼容虚拟机跳过深层检验）

当前临时执行策略：

- 【NUMA2503-034】仅在 ARM 架构执行；x86 环境自动 Skip。
- 【NUMA2503-034】会检查依赖 rpm（`oeaware`、`netrela`、`nriplugin`），缺失则自动 Skip。

## 前置条件

- 已在 `e2e-frontend/global.setup.ts` 中完成登录态生成（默认 `setup` 项目会自动执行）
- 运行机可访问管理面地址 `TEST_FUYAOURL`
- 运行机可 SSH 到测试节点（通常是业务集群控制节点）
- 测试节点具备 `kubectl` 与 `systemctl` 权限，用于读取 kubelet 配置和服务状态

## 环境变量配置

建议先复制模板并填值：

```bash
cd e2e-frontend
cp .env.test.template .env.test
```

NUMA 相关关键变量如下：

- `TEST_NODE2_IP`：用于 SSH 与组件安装的节点 IP
- `TEST_NODE2_PASSWORD`：用于 SSH 的节点密码
- `TEST_NODE2_USER`：可选，默认 `root`

其余 NUMA 路径、服务名、chart 信息已在 `constants.ts` 固定，不需要在 `.env.test` 额外配置。

## 执行方式

执行 NUMA 前端全部用例：

```bash
cd e2e-frontend
npx playwright test --project=numa-affinity
```

只执行策略开关相关用例：

```bash
cd e2e-frontend
npx playwright test --project=numa-affinity numa-affinity/test-policy-switch.spec.ts
```

## 开关操作链路

策略用例中，页面操作使用如下固定路径：

```ts
await page.getByText('NUMA亲和调度').click();
await page.getByRole('link', { name: '亲和策略配置' }).click();
await page.getByText('numa-aware').click();
await page.getByRole('switch').first().click();
await page.getByRole('button', { name: /确\s*定/ }).click();
await page.getByText('最优NUMA Distance').click();
await page.getByRole('switch').nth(1).click();
await page.getByRole('button', { name: /确\s*定/ }).click();
await page.getByText('运行时Pod亲和优化').click();
await page.getByRole('switch').nth(2).click();
```

当前自动化已覆盖前三项策略开关（`numa-aware`、`最优NUMA Distance`、`运行时Pod亲和优化`），并补充了虚拟机场景下的简化校验路径。

此外，NUMA 前端项目在开始执行前会自动安装 `numa-affinity-package`，并在全部 NUMA 用例结束后自动卸载。

