# e2e-auto-test

## 仓库介绍

本仓库收录 openFuyao 各特性的端到端与自动化测试能力，主要包括：

- **`e2e/`**：基于 **Golang** + **Ginkgo v2** 的后端测试，覆盖安装部署、资源管理 API、在离线混部等特性。
- **`e2e-frontend/`**：基于 **TypeScript** + **Playwright** 的管理面与引导节点安装界面的 UI 自动化测试。

所有测试都需要在真实集群环境运行。

仓库所属 SIG：[sig-qa](https://gitcode.com/openFuyao/sig-qa)

## 项目结构

```plain
e2e-auto-test/
├── e2e/                      # Ginkgo 后端用例
├── e2e-frontend/             # Playwright 前端用例
├── docs/                     # 开发说明
├── .gitignore
├── LICENSE
└── README-*.md
```

### `e2e/`

覆盖 openFuyao 各**特性级后端集成测试**与**系统级集成测试**，使用 Golang + Ginkgo 编写。按特性划分为各测试目录，并在 `framework/` 中提供公共能力（例如从仓库根路径解析并加载 `e2e/.env` 的 `framework/env`）。

```plain
e2e/
├── framework/                # 公共测试框架（k8s 控制、环境管理、工具等）
├── colocation/               # 在离线混部相关测试用例
|   ├── system-integration/       # 在离线混部系统级集成测试
├── numa-affinity/            # NUMA 亲和相关测试用例
|   ├── system-integration/       # NUMA 亲和系统级集成测试
├── ...
├── .env.template             # 环境变量模板
├── go.mod
└── go.sum
```

**说明**：

- 每个特性目录（如 `colocation/`、`numa-affinity/` 等）下通常包含：
  - `<feature>_suite_test.go` 测试入口文件，包含当前特性测试的前后置操作。
  - `<topic>_test.go` 测试用例文件，后者提供了所有用例的代码实现。
  - `utils/`、`static/` 等工具函数与资源模板文件。
- 部分特性目录下还提供独立的 `system-integration` 文件夹，用于承载该特性的系统级集成测试用例。
- `framework/` 为所有用例提供统一的 K8s 客户端、Helm 管理、环境变量加载与通用工具，避免重复开发。
- README 文件（可选）会在各特性目录给出该特性特异性的配置说明与使用指导。

### `e2e-frontend/`（Playwright）

覆盖 openFuyao 管理面与引导安装界面的 UI 自动化测试，使用 TypeScript + Playwright 实现。按特性划分为多个 `project`。

```plain
e2e-frontend/
├── auth/                    # 认证鉴权
├── console/                 # 管理面
├── installation/            # 引导/安装部署界面
├── colocation/              # 在离线混部测试（扩展组件，含 setup/teardown）
├── ...
├── global.setup.ts          # 全局初始化
├── global.teardown.ts       # 全局清理
├── .env.test.template       # 环境变量模板
├── playwright.config.ts
├── package.json
└── tsconfig.json
```

**说明**：

- 用例文件命名：`test-<场景>.spec.ts`（如 `console/workload/test-deployment.spec.ts`）。
- 各特性目录可自带 `utils/`、`constants/`、`static/`（YAML 夹具）等。
- `console/utils/common.ts` 为管理面常用操作封装。
- 需要登录态的 project 依赖 `global-setup`；部分扩展组件另有 `<feature>.setup.ts` / `<feature>.teardown.ts`。

## 用例运行

推荐使用 VS Code 测试工具，可以提供项目结构预览、快捷运行与调试、测试结果展示与缓存等便捷能力。安装 **Playwright Test for VSCode** 扩展后，可进一步提供针对 Playwright 的辅助功能。

### 运行 `e2e`

1. 安装 Ginkgo。

   ```bash
   go install github.com/onsi/ginkgo/v2/ginkgo@latest
   ```

   > 若安装 `latest` 版本，后续执行可能因版本与 `go.mod` 中的 ginkgo 版本不一致而告警，一般不影响测试执行。若需精确控制版本，建议安装与 `go.mod` 中一致的版本。

2. **环境变量**：将 `e2e/.env.template` 复制为 `e2e/.env`，按模板与各特性 README 填写（`framework/env` 会从 `e2e` 根目录加载指定文件，例如 `.env`）。
3. **依赖**：

   ```bash
   cd e2e
   go mod download
   ```

4. **执行示例**：

   ```bash
   # 示例：运行 colocation 包下全部 suite
   ginkgo -v ./colocation

   # 示例：仅运行包含 "MyTest" 标题的用例
   ginkgo -v --focus="MyTest" ./colocation

   # 示例：按标签筛选（CI 常用）
   ginkgo -v --label-filter="with-workload-cluster" ./elastic-scaler
   ginkgo -v --skip-label="system-integration" ./...
   ginkgo -v --skip-label="skip-temporarily" ./...
   ```

### 运行 `e2e-frontend`

**环境要求**：Node.js ≥ 18。

1. 进入目录并安装依赖、安装浏览器（默认 Chromium）：

   ```bash
   cd e2e-frontend
   npm install
   npx playwright install chromium
   ```

2. 将 `.env.test.template` 复制为 **`.env.test`**，按需取消注释并填写。

   **管理面与扩展组件**（`console/`、`auth/`、`colocation/` 等）：

   | 变量 | 说明 |
   | ---- | ---- |
   | `TEST_FUYAOURL` | 管理面 Web 地址 |
   | `TEST_USERNAME` | 登录用户名 |
   | `TEST_PASSWORD` | 登录密码 |
   | `TEST_INITIAL_PASSWORD` | 初始密码（用户管理、首次改密等用例） |

   **安装部署**（`installation/`）：

   | 变量 | 说明 |
   | ---- | ---- |
   | `TEST_GUIDEURL` | 引导/安装部署前端地址 |
   | `TEST_MANAGERURL` | 管理集群升级界面地址（管理集群升级用例） |
   | `TEST_USERNAME` | 登录用户名（与管理面一致时可复用） |
   | `TEST_PASSWORD` | 登录密码（与管理面一致时可复用） |
   | `TEST_INITIAL_PASSWORD` | 初始密码（引导节点改密等用例） |
   | `TEST_NODE1_IP` | 引导节点 IP（NODE1） |
   | `TEST_NODE1_PASSWORD` | 引导节点 SSH 密码 |
   | `TEST_NODE2_IP` / `TEST_NODE2_PASSWORD` | 测试节点 1 SSH 配置 |
   | `TEST_NODE3_IP` / `TEST_NODE3_PASSWORD` | 测试节点 2 SSH 配置 |
   | `TEST_NODE4_IP` / `TEST_NODE4_PASSWORD` | 测试节点 3 SSH 配置 |
   | `TEST_NODE5_IP` / `TEST_NODE5_PASSWORD` | 测试节点 4 SSH 配置 |

   > `TEST_*URL` 均使用 `https://<ip>:<port>` 格式，**勿以 `/` 结尾**。NODE2 及以后的节点数量按具体用例（如 1Master1Node、3Master、扩缩容）增减填写。

3. 运行：

   ```bash
   npm run test                              # 全部 project
   npm run test:ui                           # UI 模式
   npx playwright test --project=console     # 仅某一特性
   npx playwright test --project=installation
   npx playwright test --grep @post-init     # 按 tag 筛选
   npx playwright test --grep-invert @skip-temporarily
   ```

## 用例开发

> 新增或修改用例时，先对照同特性目录下已有文件。

### 开发 `e2e`

#### 新增 `e2e` 模块

对于后端用例，应当根据其依赖环境、组件、测试用例间影响关系等，划分到不同的 **Golang** 包中。

建议使用 Ginkgo CLI 工具创建测试目录：

```bash
mkdir <feature>
cd <feature>
ginkgo bootstrap
```

执行后会生成以下文件：

```plain
<feature>/
├── <feature>_suite_test.go    # 测试套件入口
├── <feature>_test.go          # 测试用例模板文件
└── ...
```

#### `e2e` 文件与代码规范

- `<feature>_suite_test.go` 作为测试入口文件保持默认名称，可在 `TestXXX()` 测试入口函数或 `init()` 函数中添加该测试套件的初始化逻辑。
- 测试文件名遵循 `<topic>_test.go` 的命名规范，允许进一步根据子模块增加子目录。
- 对于通用工具函数，可创建 `<feature>_helpers.go` 文件，或统一存放到 `utils/` 路径。
- 使用 `Describe` 对测试用例进行分组（逻辑上可归为一类，或存在顺序、依赖关系的一系列用例）。
- 每个 `Describe` 和 `It` 应有清晰的描述和标签，描述与测试用例名称一致，标签**至少**提供版本和特性：
  - `It` 描述遵循 `<用例例号> - 用例描述` 的格式，优先使用中文。
  - `Label` 标签详见下文表格。

**示例代码**：

```go
var _ = Describe("Elastic Scaler Validator 规格校验",
    Ordered, Label("elastic-scaler", "with-workload-cluster"), func() {

    It("ES-001 - 规格校验-targetRef 为空时拒绝", Label("validator", "P1"), func() {
        // ...
    })
})
```

**标签（`Label` 声明）**：

| 类别 | 示例 | 含义 |
| ---- | ---- | ---- |
| 特性名 | `elastic-scaler`、`installation` | 模块标识 |
| 运行环境 | `with-workload-cluster`、`with-management-cluster`、`offline-only` | CI 按环境筛选用例 |
| 生命周期 | `pre-init`、`post-init`、`prerequisite` | 集群阶段 |
| 优先级 | `P0`、`P1`、`P2` | 用例优先级 |
| 临时跳过 | `skip-temporarily` | 本地/CI 排除 |

### 开发 `e2e-frontend`

#### 新增 `e2e-frontend` 项目

对于前端测试用例，应当根据其依赖环境、组件、测试用例间影响关系等，划分到不同的**项目**中。

新增项目时，需要在 `playwright.config.ts` 的 `testMatch` 与 `projects` 中注册，并根据需要添加前置与后置依赖：

- 需登录态的 project，加 `dependencies: ['global-setup']`。
- 需要安装/卸载组件的 project，加 `dependencies: ['<feature>.setup', '<feature>.teardown']`，并实现安装与卸载的用例逻辑。

执行步骤：

1. 在 `e2e-frontend` 目录下创建新的项目目录，并添加 `test-<场景>.spec.ts` 文件。
2. 在 `playwright.config.ts` 的 `projects` 中注册，声明路径匹配 `testMatch` 与前置后置依赖 `dependencies`。

   ```typescript
   # playwright.config.ts
   projects: [
      {
         name: 'new-module',
         testMatch: 'new-module/**/*.spec.ts',
         dependencies: ['global-setup', 'new-module.setup'],
      }
   ]
   ```

3. 运行 `npx playwright test --project=<feature> --list` 验证。

编写 Playwright 测试用例时，推荐在 Chrome 浏览器中安装 [Playwright CRX](https://chromewebstore.google.com/detail/playwright-crx/jambeljnbnfbkcpnoiaedcabbgmnnlcd) 扩展。在浏览器中打开待测试界面，点击地址栏右侧的扩展图标，即可进入 Playwright 调试模式；插件将录制前端操作并生成相应的测试代码。

#### `e2e-frontend` 文件与代码规范

- 测试文件名遵循 `test-<场景>.spec.ts` 的命名规范，允许进一步根据子模块增加子目录。
- 使用 `test.describe` 对测试用例进行分组（逻辑上可归为一类，或存在顺序、依赖关系的一系列用例）。
- 每个 `test.describe` 和 `test` 应有清晰的描述和标签，描述与测试用例名称一致，标签**至少**提供版本和特性：
  - `test` 描述遵循 `【用例例号】用例描述` 的格式，优先使用中文。
  - `tag` 标签以 `@` 开头，详见下文表格。

**示例代码**：

```typescript
test.describe('自定义资源定义管理', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + '/container_platform/customResourceDefinition');
  });

  test('【资源管理-371】 自定义资源，输入正确 yaml 创建自定义资源定义成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreate(page);
    // 从 static/valid-crd.yaml 填入编辑器
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
  });
});
```

**标签（`tag`，以 `@` 开头）**：

| 类别 | 示例 | 含义 |
| ---- | ---- | ---- |
| 特性/场景 | `@k8s`、`@installation` | 模块或场景 |
| 运行环境 | `@with-management-cluster`、`@offline-only` | 环境要求 |
| 生命周期 | `@post-init`、`@prerequisite` | 集群阶段 |
| 版本里程碑 | `@v24.06`、`@25.06` | 引入版本 |
| 临时跳过 | `@skip-temporarily` | 通过 `--grep-invert` 排除 |
