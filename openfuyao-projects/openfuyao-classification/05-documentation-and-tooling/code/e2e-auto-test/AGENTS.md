# CLAUDE.md

本文件为 AI 助手在本仓库编写/修改代码时的约束说明。**新增用例、文件、目录时必须与现有模块保持一致**；不确定时先打开同特性下的已有文件对照，再动手。

## 项目概述

openFuyao 端到端自动化测试 monorepo，两个独立子项目：

| 子项目 | 技术栈 | 测试对象 |
|--------|--------|----------|
| `e2e/` | Go 1.24.5 + Ginkgo v2 + Gomega + client-go | 集群/后端/API/CLI |
| `e2e-frontend/` | Playwright + TypeScript (ESM) | 管理面 Web UI |

前后端按**特性名**（kebab-case 目录）对齐，例如 `elastic-scaler`、`installation`、`colocation`、`console`。

---

## 仓库布局

```
e2e-auto-test/
├── e2e/                    # Go 后端/集群测试
│   ├── framework/          # 公共框架（env/k8s/helm/executor/pod）
│   ├── <feature>/          # 特性模块（见下方列表）
│   ├── .env.template
│   └── go.mod
├── e2e-frontend/           # Playwright 前端测试
│   ├── <feature>/          # 与后端同名或 UI 专属特性
│   ├── utils/              # 全局工具（如 common.ts）
│   ├── global.setup.ts / global.teardown.ts
│   ├── playwright.config.ts
│   ├── tsconfig.json       # paths: "@/*" → "./*"
│   └── package.json
└── CLAUDE.md
```

---

## 常用命令

### e2e/

```bash
cd e2e && go mod download

# 运行整个特性包
ginkgo -v ./installation

# 标签筛选（CI 常用）
ginkgo -v --label-filter="with-workload-cluster" ./elastic-scaler
ginkgo -v --skip-label="system-integration" ./...
ginkgo -v --skip-label="skip-temporarily" ./...
```

### e2e-frontend/

```bash
cd e2e-frontend && npm install && npx playwright install chromium

npm run test                              # 全部
npx playwright test --project=installation
npx playwright test --project=console
npx playwright test --grep @post-init     # 按 tag 筛选
```

---

## 环境变量

### e2e/.env

复制 `e2e/.env.template` → `e2e/.env`。所有 Suite 统一使用此 `/env` 文件，各模块不单独维护自己的环境变量文件。Suite 的 `init()` 通过 `framework/env.LoadEnv(".env")` 加载（从当前目录向上查找 `e2e/` 根目录）。

| 变量 | 说明 |
|------|------|
| `GUIDE_NODE_HOST` | 引导节点 SSH 地址 |
| `TEST_NODE{1-5}_IP` | 测试节点 IP |
| `TEST_USERNAME` / `TEST_PASSWORD` | 管理面账号 |
| `KUBECONFIG` | kubeconfig（默认 `~/.kube/config`） |
| `MGMT_CLUSTER_VIP_HOST` / `WORKLOAD_CLUSTER_VIP_HOST` | 管理/业务集群 VIP |

### e2e-frontend/.env.test

复制 `.env.test.template` → `.env.test`，由 `playwright.config.ts` 加载。

| 变量 | 说明 |
|------|------|
| `TEST_FUYAOURL` | 管理面地址 |
| `TEST_USERNAME` / `TEST_PASSWORD` | 登录账号 |
| `TEST_NODE<n>_IP` / `TEST_NODE<n>_PASSWORD` | 节点 SSH |

---

## e2e/ 编码与目录规范

### 目录结构（标准特性模块）

```
e2e/<feature>/
├── <feature>_suite_test.go       # 或 <topic>_suite_test.go — 唯一 Suite 入口
├── <topic>_test.go               # 测试用例（可多个，按场景拆分）
├── <topic>_helpers.go            # 包内共享 helper（无 _test 后缀，同 package）
├── utils/                        # 特性专用工具，package utils
│   ├── const.go / *_functions.go
│   └── ...
├── system-integration/           # 可选：跨组件集成测试
│   ├── system_integration_suite_test.go
│   └── system_integration_*_test.go
├── performance-test/             # 可选：性能测试（installation、eagle-eye 等）
├── testdata/                     # 可选：YAML/配置夹具（installation）
└── README.md                     # 可选
```

**命名对照**

| 层级 | 规则 | 示例 |
|------|------|------|
| 目录 | kebab-case | `elastic-scaler`、`many-core-orchestrator` |
| Go package | 下划线 | 目录 `elastic-scaler` → `package elastic_scaler` |
| Suite 文件 | `*_suite_test.go` | `elastic_scaler_suite_test.go` |
| 用例文件 | `<场景>_test.go` | `validator_test.go`、`cluster_basic_test.go` |
| Helper | `<场景>_helpers.go` 或语义名 | `cluster_test_helpers.go`（**不要** `_test.go` 后缀） |
| utils 包 | 固定 `package utils` | `utils/tidal_functions.go` |

**特殊结构**

- `infernex/`：大特性下再分子目录（`infernex/`、`elastic-scaler/`、`eagle-eye/`、`hermes-router/`），各子目录独立 Suite。
- `installation/`：体量大，含 `bke-config/`、`utils/`、`framework.go`、多份 `*_test.go` 按安装场景拆分。

### Suite 文件模板

```go
package elastic_scaler

import (
    "testing"
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    // k8s client 初始化 ...
)

var _ = BeforeSuite(func() { /* clientset / dynamicClient / ctx */ })
var _ = AfterSuite(func() { /* cleanup */ })

func TestElasticScaler(t *testing.T) {
    RegisterFailHandler(Fail)
    RunSpecs(t, "ElasticScaler Suite")
}
```

需要环境变量的模块（如 `installation`）在 Suite 文件中：

```go
func init() {
    Expect(env.LoadEnv(".env")).To(Succeed()) // 或 log.Fatalf
}
```

### 用例文件模板

```go
package elastic_scaler

import (
    . "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
)

var _ = Describe("Elastic Scaler Validator 规格校验",
    Label("elastic-scaler", "with-workload-cluster"), func() {

    It("xxx-规格校验-targetRef 为空时拒绝", Label("validator"), func() {
        // ...
    })
})
```

### 组织原则

- `Describe`/`Context`/`It` 分层；长流程可用 `Ordered` + `BeforeAll`/`AfterAll`。
- 用例名称规范合适：`[用例例号](-[子模块，如果有])-用例描述`，优先使用中文
- 标签分层与命名见下方「标签」节；**新增用例必须跟随同模块已有文件的 Label 风格**。
- 复用逻辑放 `utils/` 或 `*_helpers.go`，避免在 `It` 内堆长函数。
- import 路径：`gitcode.com/openFuyao/e2e-auto-test/e2e/...`；跨特性优先用 `framework/`。

### 标签

Ginkgo 使用 `Label(...)`（无 `@` 前缀）。新增用例在 Describe 上至少声明**特性名**，并按需叠加下列通用标签；**命名风格跟随同模块已有文件**（kebab-case 或 snake_case 均可）。

| 类别 | 标签 | 含义 |
|------|------|------|
| **特性名称** | `<feature>` | 模块/场景标识，如 `elastic-scaler`、`installation`、`colocation-deployment` |
| **CI 运行环境** | `with-workload-cluster` | 需要业务/workload 集群 |
| | `with-management-cluster` | 需要管理集群 |
| | `offline-only` / `offline` | 仅离线环境 |
| **生命周期阶段** | `pre-init` | 引导节点初始化前 |
| | `post-init` | 集群已就绪后的用例 |
| | `prerequisite` | 前置/认证类 |
| **优先级** | `P0` / `P1` / `P2` | 用例优先级（It 级为主） |
| **特性里程碑** | `25.06`, `26.03`, etc. | 该用例引入的版本里程碑 |

对于安装部署（installation），由于业务逻辑复杂性，允许有更多种类的标签

```go
// 典型组合
Label("installation", "post-init", "P0")
Label("elastic-scaler", "with-workload-cluster")
Label("prescript", "3master", "P1", "post-init")
```

### framework/ 复用（尽量不在特性内重复实现）

| 包 | 用途 |
|----|------|
| `framework/env` | 加载 `.env` |
| `framework/k8s` | K8s 客户端与校验 |
| `framework/helm` | Helm 安装/卸载 |
| `framework/executor` | SSH/本地命令执行 |
| `framework/pod` | Pod 日志与校验 |

### 新增 e2e 模块检查清单

1. 在 `e2e/` 下创建 `<feature>/`（kebab-case，与已有模块不重名）。
2. 创建 `*_suite_test.go`，`package` 用下划线形式。
3. 按场景添加 `<topic>_test.go`；共享代码放 `utils/` 或 `*_helpers.go`。
4. 所有顶层 `Describe` 加 `Label("<feature>")` 及场景所需标签。
5. 跨组件集成 → `system-integration/` 子目录 + 独立 Suite。
6. 在 `e2e/.env.template` 补充新环境变量（如有）。
7. 运行 `go test ./<feature> -v` 或 `ginkgo -v ./<feature>` 验证。

---

## e2e-frontend/ 编码与目录规范

### 目录结构（标准特性模块）

```
e2e-frontend/<feature>/
├── test-<场景>.spec.ts           # 测试用例（统一 test- 前缀）
├── <feature>.setup.ts            # 可选：特性级前置
├── <feature>.teardown.ts         # 可选：特性级后置
├── utils/                        # 特性专用工具
│   └── clusterManage.ts          # camelCase 文件名
├── constants/                    # 或根目录 consts.ts / constants.ts
│   └── common.ts
├── static/                       # 可选：YAML 夹具（console 等）
└── assets/                       # 可选：二进制/图片（marketplace）
```

**复杂特性示范**

- `console/`: 核心管理面用例，按管理面页面模块分子目录 — `workload/`、`storage/`、`network/`、`namespace/`、`rbac/`、`configuration/`、`monitoring/`；共享 `utils/common.ts`、`constants.ts`、`static/*.yaml`。
- `installation/`：安装部署前端界面用例，`utils/`（clusterManage、verify、upgrade-pre…）、`constants/common.ts`、`versionYaml/`。
- `colocation/`、`numa-affinity/`等：扩展组件，带 `<feature>.setup.ts` + teardown 用于安装和卸载组件，在 `playwright.config.ts` 中配置 dependencies。

### 用例文件模板

```typescript
import { expect, test } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/console/constants';
import { clickCreate, saveCreateOrEdit } from '@/console/utils/common';  // @/ 别名

test.describe('deployment管理', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(FUYAO_BASE_URL + '/container_platform/workload/deployment');
    });

    test('【资源管理-031】输入正确 yaml 创建 Deployment 成功', {
        tag: ['@k8s'],
    }, async ({ page }) => {
        await clickCreate(page);
        // 从 ../static/valid-deployment.yaml 填入编辑器
        await saveCreateOrEdit(page);
        await expect(page.getByText(/创建成功/)).toBeVisible();
    });
});
```

**命名与风格**

| 项 | 规则 |
|----|------|
| 用例文件 | `test-<场景>.spec.ts`（如 `test-deployment.spec.ts`） |
| setup/teardown | `<feature>.setup.ts` / `<feature>.teardown.ts` |
| utils | `utils/<语义>.ts`，camelCase（如 `clusterManage.ts`、`user-management.ts`） |
| 常量 | `constants/common.ts` 或 `consts.ts`（参考同模块已有文件） |
| 静态夹具 | `static/valid-<resource>.yaml`（console 惯例） |
| import | 同特性用相对路径；跨目录用 `@/<feature>/...`（见 `tsconfig.json`） |
| 引号 | 与同目录现有文件保持一致 |
| 测试用例名称 | `【用例例号】用例描述`，优先使用中文 |

**标签（tag）规范**

Playwright 标签以 `@` 开头，分类与 e2e 一致：

| 类别 | 标签 | 含义 |
|------|------|------|
| **特性名称** | `@<feature>` | 如 `@installation`、`@user-management` |
| **CI 运行环境** | `@with-management-cluster` | 需要管理集群 |
| | `@offline-only` | 仅离线环境 |
| **生命周期阶段** | `@post-init` | 集群已就绪后 |
| | `@prerequisite` | 前置/认证类 |
| **优先级** | （暂无统一约定） | e2e 侧用 `P0`/`P1`/`P2`；前端按模块自行补充 |
| **特性里程碑** | `25.06`, `26.03`, etc. | 该用例引入的版本里程碑 |

临时跳过：`@skip-temporarily`（`--grep-invert @skip-temporarily`）。

### playwright.config.ts 注册（新增模块必改）

新增特性时在**两处**同步添加：

1. 顶层 `testMatch` 数组
2. `projects` 数组（含 `name`、`testMatch`、`timeout`、`dependencies`）

```typescript
// 简单模块
{ name: 'my-feature', testMatch: 'my-feature/**/*.spec.ts', use: { ...devices['Desktop Chrome'] }, dependencies: ['global-setup'] },

// 需要 setup/teardown 的模块（参照 colocation、numa-affinity）
{ name: 'my-feature-setup', testMatch: 'my-feature/my-feature.setup.ts', teardown: 'my-feature-teardown', ... },
{ name: 'my-feature-teardown', testMatch: 'my-feature/my-feature.teardown.ts', ... },
{ name: 'my-feature', testMatch: 'my-feature/**/*.spec.ts', dependencies: ['global-setup', 'my-feature-setup'], ... },
```

- 长耗时模块（如 `installation`）单独设 `timeout: 120 * 60 * 1000`。
- 依赖登录态的模块加 `dependencies: ['global-setup']`；`global.setup.ts` 将认证写入 `auth.json`，`use.storageState` 全局生效。
- CI：`workers: 1`，`retries: 2`，reporter 用 `allure-playwright`。

### 新增 e2e-frontend 模块检查清单

1. 创建 `e2e-frontend/<feature>/`，目录名 kebab-case。
2. 添加 `test-<场景>.spec.ts`（**必须** `test-` 前缀 + `.spec.ts` 后缀）。
3. 抽取复用逻辑到 `utils/`、`constants/`；YAML 夹具放 `static/`。
4. 更新 `playwright.config.ts` 的 `testMatch` 与 `projects`。
5. 在 `.env.test.template` 补充环境变量（如有）。
6. 运行 `npx playwright test --project=<feature>` 验证。

---

## 流水线集成要点

### Ginkgo

| 场景 | 命令 |
|------|------|
| 跳过系统集成 | `ginkgo --skip-label="system-integration" ./...` |
| 仅业务集群用例 | `ginkgo --label-filter="with-workload-cluster" ./...` |
| 跳过临时禁用 | `ginkgo --skip-label="skip-temporarily" ./...` |

### Playwright

```bash
npx playwright test --project=installation
npx playwright test --grep @post-init
npx playwright test --grep-invert @skip-temporarily
```

- Go：各 Suite `BeforeSuite` 独立初始化客户端。
- 前端：`global.setup.ts` 统一登录，`auth.json` 持久化会话。

---

## 版本与注意事项

- **Go**：1.24.5（`e2e/go.mod`）
- **Node**：≥ 18（`e2e-frontend/package.json` engines）
- **K8s client**：v0.34.1，避免 alpha API
- **标签命名**：Go `Label` 小写，kebab-case 或 snake_case（跟随同模块）；前端 tag 以 `@` 开头。均优先使用英语。
- **不要**在 `framework/` 或 `e2e-frontend/utils/` 放特性专属逻辑；**不要**新建与现有模块同义的不同目录名
- 修改前先 `ls` 目标特性目录，**复制最近似的现有文件**再改，保持 package/import/标签风格一致

## 风格检查提示

### AI编码时

AI 在本仓库**新增或修改**测试代码时，须**优先遵守**本文档中的目录结构、命名、标签、import 路径、模板等规范；不确定时先对照同模块已有文件，不得凭习惯另起一套写法。

### 用户要求「检查格式 / 规范 / 风格」时

仅做**审查与建议**，流程如下：

1. 对照本文档相关章节，**逐条列举**不符合项（注明文件路径、位置、违反的规范条目）。
2. 对每项给出**具体修改建议**（应改成什么、参考哪个现有文件）。
3. **不得**在用户未明确要求修改的前提下，自行编辑代码或提交变更。
