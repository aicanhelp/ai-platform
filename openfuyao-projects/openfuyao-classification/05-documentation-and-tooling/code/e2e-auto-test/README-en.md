# e2e-auto-test

## Repository Overview

This repository consolidates the end-to-end and automated testing capabilities for openFuyao, primarily including:

- **`e2e/`**: Backend tests built on **Golang** + **Ginkgo v2**, covering installation, resource management APIs, online-offline colocation, and other features.
- **`e2e-frontend/`**: UI automation tests built on **TypeScript** + **Playwright**, covering openFuyao Console and Bootstrap Node Installer.

All tests require a real cluster environment.

Repository SIG: [sig-qa](https://gitcode.com/openFuyao/sig-qa)

## Project Structure

```plain
e2e-auto-test/
├── e2e/                      # Ginkgo backend test cases
├── e2e-frontend/             # Playwright frontend test cases
├── docs/                     # Development notes
├── .gitignore
├── LICENSE
└── README-*.md
```

### `e2e/`

Covers **feature-level backend integration tests** and **system-level integration tests** for openFuyao, written in Golang + Ginkgo. Tests are organized into directories by feature, with shared capabilities provided in `framework/` (e.g., `framework/env` for resolving and loading `e2e/.env` from the repository root).

```plain
e2e/
├── framework/                # Shared test framework (k8s clients, env loading, utilities, etc.)
├── colocation/               # Colocation test cases
|   ├── system-integration/       # Colocation system-level integration tests
├── numa-affinity/            # NUMA affinity test cases
|   ├── system-integration/       # NUMA affinity system-level integration tests
├── ...
├── .env.template             # Environment variable template
├── go.mod
└── go.sum
```

**Notes**:

- Each feature directory (e.g., `colocation/`, `numa-affinity/`, etc.) typically contains:
  - `<feature>_suite_test.go` test entry file, containing setup and teardown operations for the current feature's tests.
  - `<topic>_test.go` test case file, providing the code implementation for all test cases.
  - `utils/`, `static/`, etc. utility functions and resource template files.
- Some feature directories include a standalone `system-integration` subdirectory for system-level integration test cases specific to that feature.
- `framework/` provides a unified K8s client, Helm management, environment variable loading, and common utilities for all test cases, avoiding redundant development.
- README files (optional) in each feature directory provide feature-specific configuration and usage guidance.

### `e2e-frontend/` (Playwright)

Covers UI automation tests for the openFuyao Console and Bootstrap Node Installer, implemented in TypeScript + Playwright. Tests are organized into multiple `projects` by feature.

```plain
e2e-frontend/
├── auth/                    # Authentication & authorization
├── console/                 # openFuyao Console
├── installation/            # Bootstrap Node Installer
├── colocation/              # Colocation tests (extension component, with setup/teardown)
├── ...
├── global.setup.ts          # Global setup
├── global.teardown.ts       # Global teardown
├── .env.test.template       # Environment variable template
├── playwright.config.ts
├── package.json
└── tsconfig.json
```

**Notes**:

- Test case files are named `test-<scenario>.spec.ts` (e.g., `console/workload/test-deployment.spec.ts`).
- Each feature directory may include its own `utils/`, `constants/`, and `static/` (YAML fixtures).
- `console/utils/common.ts` provides common console operations.
- Projects requiring login state depend on `global-setup`; some extension components also have `<feature>.setup.ts` / `<feature>.teardown.ts`.

## Running Tests

It is recommended to use the VS Code testing tool, which offers convenient capabilities such as project structure preview, quick test running and debugging, as well as result display and caching. Installing the **Playwright Test for VSCode** extension further enhances the Playwright testing experience with dedicated features.

### Running `e2e`

1. Install Ginkgo.

   ```bash
   go install github.com/onsi/ginkgo/v2/ginkgo@latest
   ```

   > Installing the `latest` version may trigger warnings if it differs from the Ginkgo version in `go.mod`; this generally does not affect test execution. To pin the version, install the same version as specified in `go.mod`.

   Alternatively, refer to the [Ginkgo installation guide](https://onsi.github.io/ginkgo/#installing-ginkgo).

2. **Environment variables**: Copy `e2e/.env.template` to `e2e/.env` and fill in values according to the template and each feature's README (`framework/env` loads the specified file from the `e2e` root directory, e.g., `.env`).

3. **Dependencies**:

   ```bash
   cd e2e
   go mod download
   ```

4. **Execution examples**:

   ```bash
   # Example: Run all suites under the colocation package
   ginkgo -v ./colocation

   # Example: Run only test cases with "MyTest" in the title
   ginkgo -v --focus="MyTest" ./colocation

   # Example: Filter by label (commonly used in CI)
   ginkgo -v --label-filter="with-workload-cluster" ./elastic-scaler
   ginkgo -v --skip-label="system-integration" ./...
   ginkgo -v --skip-label="skip-temporarily" ./...
   ```

### Running `e2e-frontend`

**Requirements**: Node.js ≥ 18.

1. Navigate to the directory and install dependencies and browsers (Chromium by default):

   ```bash
   cd e2e-frontend
   npm install
   npx playwright install chromium
   ```

2. Copy `.env.test.template` to **`.env.test`** and fill in values as needed.

   **Console & extension components** (`console/`, `auth/`, `colocation/`, etc.):

   | Variable | Description |
   | -------- | ----------- |
   | `TEST_FUYAOURL` | Console URL |
   | `TEST_USERNAME` | Login username |
   | `TEST_PASSWORD` | Login password |
   | `TEST_INITIAL_PASSWORD` | Initial password (user management, first-time password change, etc.) |

   **Installation** (`installation/`):

   | Variable | Description |
   | -------- | ----------- |
   | `TEST_GUIDEURL` | Bootstrap Node Installer URL |
   | `TEST_MANAGERURL` | Management cluster upgrade UI address (management cluster upgrade test cases) |
   | `TEST_USERNAME` | Login username (can be reused if same as Console) |
   | `TEST_PASSWORD` | Login password (can be reused if same as Console) |
   | `TEST_INITIAL_PASSWORD` | Initial password (guide node password change, etc.) |
   | `TEST_NODE1_IP` | Guide node IP (NODE1) |
   | `TEST_NODE1_PASSWORD` | Guide node SSH password |
   | `TEST_NODE2_IP` / `TEST_NODE2_PASSWORD` | Test node 1 SSH config |
   | `TEST_NODE3_IP` / `TEST_NODE3_PASSWORD` | Test node 2 SSH config |
   | `TEST_NODE4_IP` / `TEST_NODE4_PASSWORD` | Test node 3 SSH config |
   | `TEST_NODE5_IP` / `TEST_NODE5_PASSWORD` | Test node 4 SSH config |

   > All `TEST_*URL` values use the `https://<ip>:<port>` format and **must not** end with `/`. The number of NODE2 and subsequent nodes depends on the specific test case (e.g., 1Master1Node, 3Master, scaling).

3. Run:

   ```bash
   npm run test                              # All projects
   npm run test:ui                           # UI mode
   npx playwright test --project=console     # A single feature
   npx playwright test --project=installation
   npx playwright test --grep @post-init     # Filter by tag
   npx playwright test --grep-invert @skip-temporarily
   ```

## Test Development

> When adding or modifying test cases, refer to existing files in the same feature directory first.

### Developing `e2e`

#### Adding an `e2e` Module

Backend test cases should be organized into separate **Golang** packages based on their environment dependencies, components, and inter-test impact.

It is recommended to use the Ginkgo CLI to scaffold a test directory:

```bash
mkdir <feature>
cd <feature>
ginkgo bootstrap
```

This generates the following files:

```plain
<feature>/
├── <feature>_suite_test.go    # Test suite entry
├── <feature>_test.go          # Test case template file
└── ...
```

#### `e2e` File and Code Conventions

- Keep `<feature>_suite_test.go` as the default test entry file name; add suite initialization logic in the `TestXXX()` entry function or `init()` function.
- Test files follow the `<topic>_test.go` naming convention; subdirectories may be added for sub-modules.
- Shared helper functions go in `<feature>_helpers.go` or under `utils/`.
- Use `Describe` to group test cases (logically related cases, or cases with ordering/dependencies).
- Each `Describe` and `It` should have a clear description and labels. Descriptions should match the test case name; labels should **at least** include version and feature:
  - `It` descriptions follow the `<case-id> - case description` format; Chinese is preferred.
  - `Label` tags are detailed in the table below.

**Example code**:

```go
var _ = Describe("Elastic Scaler Validator 规格校验",
    Ordered, Label("elastic-scaler", "with-workload-cluster"), func() {

    It("ES-001 - 规格校验-targetRef 为空时拒绝", Label("validator", "P1"), func() {
        // ...
    })
})
```

**Labels (`Label` declaration)**:

| Category | Examples | Meaning |
| -------- | -------- | ------- |
| Feature | `elastic-scaler`, `installation` | Module identifier |
| Environment | `with-workload-cluster`, `with-management-cluster`, `offline-only` | CI environment filtering |
| Lifecycle | `pre-init`, `post-init`, `prerequisite` | Cluster stage |
| Priority | `P0`, `P1`, `P2` | Test case priority |
| Temporary skip | `skip-temporarily` | Excluded locally / in CI |

### Developing `e2e-frontend`

#### Adding an `e2e-frontend` Project

Frontend test cases should be organized into separate **projects** based on their environment dependencies, components, and inter-test impact.

When adding a new project, register it in both `testMatch` and `projects` in `playwright.config.ts`, and add setup/teardown dependencies as needed:

- Projects requiring login state: add `dependencies: ['global-setup']`.
- Projects requiring component install/uninstall: add `dependencies: ['<feature>.setup', '<feature>.teardown']` and implement the install/uninstall logic.

Steps:

1. Create a new project directory under `e2e-frontend` and add `test-<scenario>.spec.ts` files.
2. Register in `playwright.config.ts` `projects`, declaring the path match `testMatch` and setup/teardown dependencies `dependencies`.

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

3. Verify with `npx playwright test --project=<feature> --list`.

When writing Playwright test cases, it is recommended to install the [Playwright CRX](https://chromewebstore.google.com/detail/playwright-crx/jambeljnbnfbkcpnoiaedcabbgmnnlcd) extension in Chrome. Open the page to be tested, click the extension icon in the address bar, and enter Playwright debug mode. The plugin will record page operations and generate the corresponding test code.

#### `e2e-frontend` File and Code Conventions

- Test files follow the `test-<scenario>.spec.ts` naming convention; subdirectories may be added for sub-modules.
- Use `test.describe` to group test cases (logically related cases, or cases with ordering/dependencies).
- Each `test.describe` and `test` should have a clear description and tags. Descriptions should match the test case name; tags should **at least** include version and feature:
  - `test` descriptions follow the `【case-id】case description` format; Chinese is preferred.
  - `tag` values start with `@`; see the table below.

**Example code**:

```typescript
test.describe('自定义资源定义管理', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + '/container_platform/customResourceDefinition');
  });

  test('【资源管理-371】 自定义资源，输入正确 yaml 创建自定义资源定义成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreate(page);
    // Fill editor from static/valid-crd.yaml
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
  });
});
```

**Tags (`tag`, prefixed with `@`)**:

| Category | Examples | Meaning |
| -------- | -------- | ------- |
| Feature / scenario | `@k8s`, `@installation` | Module or scenario |
| Environment | `@with-management-cluster`, `@offline-only` | Environment requirements |
| Lifecycle | `@post-init`, `@prerequisite` | Cluster stage |
| Version milestone | `@v24.06`, `@v25.06` | Introduced version |
| Temporary skip | `@skip-temporarily` | Excluded via `--grep-invert` |
