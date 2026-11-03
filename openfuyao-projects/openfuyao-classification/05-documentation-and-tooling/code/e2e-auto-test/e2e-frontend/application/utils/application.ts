import { expect, type Browser, type Page } from "@playwright/test";
import { NodeSSH } from "node-ssh";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { newAuthorizedContext } from "@/utils/common";
import { confirmDeleteDialog, deleteResourceInList, expectResourceDeleted, replaceTextInEditor, searchResource } from "@/console/utils/common";
import {
    componentNameApp,
    buildDeployPlans,
    localPackageSource,
    packageVersion,
    serviceClusterUrl,
    upgradeTime,
    type DeployPlan,
    type DeployPlanTemplate,
    TEST_SSH_ADDRESS,
    TEST_SSH_PASSWORD,
    applicationTestNamespace,
} from "./constants";

type UninstallOptions = {
    strict?: boolean;
    quickCheckTimeoutMs?: number;
};

function escapeRegExp(text: string) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const __applicationDir = path.dirname(fileURLToPath(import.meta.url));
const testAssetsDir = path.resolve(__applicationDir, "../test-assets");
const chartsDir = path.join(testAssetsDir, "charts");

/** Same pattern as marketplace specs: newAuthorizedContext + page. */
async function withAuthorizedPage(browser: Browser, fn: (page: Page) => Promise<void>): Promise<void> {
    const context = await newAuthorizedContext(browser);
    const page = await context.newPage();
    try {
        await fn(page);
    } finally {
        await context.close();
    }
}

// =========================
// App market navigation
// =========================

async function goToAppMarket(page: Page) {
    await page.goto(serviceClusterUrl + "/container_platform/appMarket/marketCategory");
    await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory/, { timeout: 60_000 });
}

async function selectAppMarketSource(page: Page, source: string) {
    const defaultSource = page.getByRole("checkbox", { name: "openFuyao" }).first();
    if (source !== "openFuyao" && (await defaultSource.count()) > 0 && (await defaultSource.isVisible().catch(() => false))) {
        await defaultSource.uncheck().catch(() => {});
    }
    const sourceCheckbox = page.getByRole("checkbox", { name: source }).first();
    if ((await sourceCheckbox.count()) > 0 && (await sourceCheckbox.isVisible().catch(() => false))) {
        await sourceCheckbox.check();
    }
}

// =========================
// Component installation
// =========================

/** 对应在应用市场安装组件 */
async function installComponent(
    page: Page,
    componentName: string,
    appName: string,
    version?: string,
    source = localPackageSource,
    opts?: { beforeConfirm?: (page: Page) => Promise<void> },
) {
    await goToAppMarket(page);
    await page.getByRole("link", { name: "应用列表" }).click();
    const searchBox = page.getByRole("textbox", { name: /搜索.*名称/ }).first();
    await expect(searchBox).toBeVisible({ timeout: 60_000 });
    if (source) {
        await selectAppMarketSource(page, source);
    }
    await searchBox.fill(componentName);
    await page.getByRole("button", { name: "search" }).click();
    const componentTitle = page
        .locator("#root_container .application_item_title")
        .filter({ hasText: new RegExp(`^${escapeRegExp(componentName)}$`) })
        .first();
    await expect(componentTitle).toBeVisible({ timeout: 60_000 });
    await componentTitle.click();

    const selectedVersion = page.getByText(new RegExp(`${componentName}版本：`)).first();
    await expect(selectedVersion).toBeVisible({ timeout: 60_000 });
    const defaultVersionMet = (await selectedVersion.textContent())?.includes(version ?? "");

    if (version && !defaultVersionMet) {
        await selectedVersion.locator('div').nth(1).click();
        await page.getByText(`版本： ${version}`, { exact: true }).click();
    }

    await page.getByRole("button", { name: "部 署" }).click();
    // 某些时候组件在点击"部署"后会弹出风险确认框
    const appNameInput = page.getByRole("textbox", { name: "* 应用名称" });
    const deployRiskCheckbox = page.getByRole("checkbox", { name: /我已了解并确认上述风险/ }).first();
    if ((await deployRiskCheckbox.count()) > 0 && (await deployRiskCheckbox.isVisible().catch(() => false))) {
        await deployRiskCheckbox.check();
        await page.getByRole("button", { name: /继续部署/ }).first().click();
    }
    await appNameInput.fill(appName);
    const namespaceCombobox = page.getByRole("combobox", { name: /命名空间/i }).first();
    await expect(namespaceCombobox).toBeVisible({ timeout: 60_000 });
    await namespaceCombobox.click({ force: true });
    const namespaceListbox = page.locator("#form_in_modal_nameSpace_list");
    if (!(await namespaceListbox.isVisible().catch(() => false))) {
        await namespaceCombobox.focus();
        await page.keyboard.press("ArrowDown");
    }
    const defaultNamespaceOption = page.getByTitle(applicationTestNamespace).first();
    await expect(defaultNamespaceOption).toBeVisible({ timeout: 10_000 });
    await defaultNamespaceOption.click();
    if (opts?.beforeConfirm) {
        await opts.beforeConfirm(page);
    }
    const confirmButton = page.getByRole("button", { name: /确\s*定/ });
    await expect(confirmButton).toBeVisible({ timeout: 60_000 });
    await expect(confirmButton).toBeEnabled({ timeout: 60_000 });
    await confirmButton.click();
}

/** 检查组件部署成功 */
async function checkComponent(
    page: Page,
    appName: string,
    options?: {
        entry?: "extendManage" | "applicationManageHelm";
    },
) {
    const entry = options?.entry ?? "extendManage";
    const isHelm = entry === "applicationManageHelm";
    await expect(
        page.getByText(isHelm ? "部署成功，即将跳转应用管理页面" : "部署成功，即将跳转扩展组件管理页面"),
    ).toBeVisible({ timeout: 120_000 });
    await expect(page.getByRole("navigation").getByText(isHelm ? "应用管理" : "扩展组件管理")).toBeVisible();
    await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(appName);
    await page.getByRole("button", { name: "search" }).click();
    await expect(page.getByRole("cell", { name: "check-circle 部署成功" })).toBeVisible();
}

// =========================
// YAML editing
// =========================

type PullPolicy = "Always" | "IfNotPresent";

async function ensureValuesYamlEditor(page: Page) {
    await page.getByRole("tab", { name: "Values.yaml" }).click().catch(() => {});
    await expect(page.locator(".cm-editor").first()).toBeVisible({ timeout: 60_000 });
}

async function detectPullPolicy(page: Page): Promise<PullPolicy> {
    const editor = page.locator(".cm-editor").first();
    if (await editor.getByText(/pullPolicy:\s*Always/).first().isVisible().catch(() => false)) {
        return "Always";
    }
    await expect(editor.getByText(/pullPolicy:\s*IfNotPresent/).first()).toBeVisible({ timeout: 60_000 });
    return "IfNotPresent";
}

/** Flip all pullPolicy values in Values.yaml via CodeMirror search-replace. */
async function changePullPolicyLineWithWait(page: Page): Promise<{ oldLine: string; newLine: string }> {
    await ensureValuesYamlEditor(page);
    const current = await detectPullPolicy(page);
    const next: PullPolicy = current === "Always" ? "IfNotPresent" : "Always";
    const oldLine = `pullPolicy: ${current}`;
    const newLine = `pullPolicy: ${next}`;
    await replaceTextInEditor(page, oldLine, newLine);
    await expect(page.locator(".cm-editor").first().getByText(newLine).first()).toBeVisible({ timeout: 60_000 });
    return { oldLine, newLine };
}

/** 在升级页选择目标 chart 版本（不点确定）。 */
async function selectUpgradeChartVersion(page: Page, fromVersion: string, toVersion: string) {
    const fromVersionLoc = page.getByTitle(fromVersion, { exact: true });
    await expect(fromVersionLoc).toBeVisible({ timeout: 60_000 });
    await fromVersionLoc.click();
    await page.getByTitle(toVersion, { exact: true }).click();
}

/** 打开升级页并等待 Values 编辑器就绪。 */
async function openUpgradeEditor(page: Page) {
    await openUpgradePage(page);
    await expect(page.getByText("导出").or(page.getByRole("tab", { name: "Values.yaml" })).first()).toBeVisible({
        timeout: 60_000,
    });
}

/** Chart-version upgrade: select target chart version and confirm (no values edit). */
async function upgradeByChartVersion(page: Page, fromVersion: string, toVersion: string) {
    await openUpgradePage(page);
    await selectUpgradeChartVersion(page, fromVersion, toVersion);
    await page.getByRole("button", { name: /确\s*定/ }).click();
    await expect(page.getByText("升级成功")).toBeVisible({ timeout: upgradeTime });
    await expect(page.getByText("模板版本：" + toVersion)).toBeVisible({ timeout: 60_000 });
}

// =========================
// Log utilities
// =========================

/** 对应在日志界面进行筛选操作 */
async function selectLog(page: Page, instanceNamePrefix: string, containerName: string) {
    await page.getByRole("tab", { name: "日志" }).click();
    const selects = page.locator(".container-platform-select-selector");
    await expect(selects.nth(1)).toBeVisible({ timeout: 30_000 });
    if ((await selects.count()) >= 2) {
        await selects.nth(1).click();
        await page.getByText(instanceNamePrefix + "-").first().click({ timeout: 60_000 });
        await selects.nth(2).click();
        await page.getByText(containerName, { exact: true }).last().click({ timeout: 60_000 });
    }
    await page.getByRole("button", { name: /查\s*询/ }).click();
}

/** 等待日志数据就绪（不再显示「暂无日志」） */
async function waitForLogReady(page: Page) {
    await expect(page.getByText("暂无日志")).toBeHidden({ timeout: 60_000 });
}

/** 日志导出并校验下载文件 */
async function verifyLogExport(page: Page, outputDir: string) {
    await waitForLogReady(page);
    const exportBtn = page.getByText("导出").first();
    const exportIcon = page.getByRole("img", { name: "export" }).locator("svg");
    const triggerExport = async () => {
        if (await exportBtn.count()) {
            await exportBtn.click();
        } else {
            await exportIcon.click();
        }
    };

    const download = await Promise.all([
        page.waitForEvent("download", { timeout: 60_000 }),
        triggerExport(),
    ]).then(([d]) => d);

    await expect(page.getByText("导出成功")).toBeVisible({ timeout: 60_000 });
    const filename = download.suggestedFilename();
    expect(filename).toBeTruthy();
    const savePath = path.join(outputDir, filename);
    await download.saveAs(savePath);
    expect(fs.statSync(savePath).size).toBeGreaterThan(0);
}

// =========================
// Uninstall
// =========================

type UninstallKind = "extend" | "helm";

/** Shared uninstall flow for extend-manage / helm-manage list pages. */
async function uninstallManagedApp(
    page: Page,
    appName: string,
    kind: UninstallKind,
    opts: UninstallOptions = {},
) {
    const { strict = true, quickCheckTimeoutMs = 3_000 } = opts;
    await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(appName);
    await page.getByRole("button", { name: "search" }).click();
    const targetRow = page.getByRole("row").filter({ hasText: appName }).first();

    if (!strict) {
        try {
            await expect(targetRow).toBeVisible({ timeout: quickCheckTimeoutMs });
        } catch {
            return false;
        }
    } else {
        await expect(targetRow).toBeVisible({ timeout: quickCheckTimeoutMs });
    }

    await targetRow.locator('[data-icon="more"]').first().click();
    await page.getByRole("button", { name: "卸载" }).click();
    const dialogTitle = kind === "extend" ? "卸载扩展组件" : "卸载应用";
    await expect(page.getByText(dialogTitle, { exact: true })).toBeVisible();
    await page.getByRole("checkbox", { name: "已了解卸载操作无法恢复" }).check();
    await page.getByRole("button", { name: "卸 载" }).click();
    if (kind === "extend") {
        await expect(page.getByText("扩展组件" + appName + "卸载成功")).toBeVisible();
        await expect(page.getByRole("cell", { name: "Simple Empty 暂无数据" })).toBeVisible();
    } else {
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(appName);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("row").filter({ hasText: appName })).toHaveCount(0);
    }
    return true;
}

/** Batch-uninstall managed apps on an already-open page (best-effort). */
async function uninstallManagedApps(page: Page, apps: DeployPlan[]) {
    const seen = new Set<string>();
    const uniqueApps = apps.filter((item) => {
        const key = `${item.entry}:${item.appName}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    const extendApps = uniqueApps.filter((item) => item.entry === "extendManage");
    const helmApps = uniqueApps.filter((item) => item.entry === "applicationManageHelm");

    if (extendApps.length) {
        await page.goto(serviceClusterUrl + "/container_platform/extendManage");
        for (const app of extendApps) {
            await uninstallManagedApp(page, app.appName, "extend").catch(() => {});
        }
    }
    if (helmApps.length) {
        await page.goto(serviceClusterUrl + "/container_platform/applicationManageHelm");
        for (const app of helmApps) {
            await uninstallManagedApp(page, app.appName, "helm").catch(() => {});
        }
    }
}

// =========================
// Navigation
// =========================

async function openUpgradePage(page: Page) {
    await page.getByRole("tab", { name: "详情" }).click();
    await page.getByRole("button", { name: "操作 down" }).click();
    await page.getByRole("button", { name: "升级" }).click();
}

async function gotoManagedAppDetail(page: Page, app: DeployPlan) {
    const pathSuffix =
        app.entry === "applicationManageHelm"
            ? "/container_platform/applicationManageHelm"
            : "/container_platform/extendManage";
    await page.goto(serviceClusterUrl + pathSuffix);
    await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(app.appName);
    await page.getByRole("button", { name: "search" }).click();
    const row = page.getByRole("row").filter({ hasText: app.appName }).first();
    await expect(row).toBeVisible({ timeout: 60_000 });
    const detailLink = row.getByRole("link").first();
    if ((await detailLink.count()) > 0) {
        await detailLink.click();
    } else {
        await row.click();
    }
}

// =========================
// Revert
// =========================

/** 对应在升级成功界面后进行回退操作 */
async function revertComponent(page: Page, revertName: string) {
    await expect(page.getByText("基本信息")).toBeVisible({ timeout: 60_000 });
    await page.getByRole("button", { name: "操作 down" }).click();
    await page.getByRole("button", { name: "回退" }).click();
    await expect(page.getByText("应用回退")).toBeVisible();
    const revertRowFirstCell = page.getByRole("row", { name: revertName }).getByRole("cell").first();
    await expect(revertRowFirstCell).toBeVisible({ timeout: 60_000 });
    await revertRowFirstCell.click();
    await page.getByRole("button", { name: /确\s*定/ }).click();
    await expect(page.getByText("回退版本成功")).toBeVisible({ timeout: 60_000 });
}

// =========================
// Resource check
// =========================

/** 检查指定资源前缀的运行实例数 */
async function checkPrefixResourceNum(page: Page, resource: string, prefix: string, state: string, stateNum: number) {
    await page.getByText("工作负载").click();
    await page.getByRole("link", { name: resource }).click();
    await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(prefix);
    await page.getByRole("button", { name: "search" }).click();
    await expect
        .poll(
            async () => {
                await page.getByRole("button", { name: "sync" }).click();
                return page.getByText(state).count();
            },
            { timeout: 60_000, intervals: [3000] },
        )
        .toBe(stateNum);
}

// =========================
// Sort verification
// =========================

/** 点击指定列头的排序按钮 */
async function clickSortHeader(page: Page, headerName: string) {
    const header = page.getByRole("columnheader", { name: new RegExp(headerName) }).first();
    await expect(header).toBeVisible();
    const before = await header.getAttribute("aria-sort");
    await header.click();
    // 等待排序完成：以 aria-sort 变化作为结束条件
    if (before) {
        await expect
            .poll(async () => await header.getAttribute("aria-sort"), { timeout: 60_000 })
            .not.toBe(before);
    } else {
        await expect.poll(async () => await header.getAttribute("aria-sort"), { timeout: 60_000 }).not.toBeNull();
    }
}

/** 获取表格某列文本（按列号，1-based） */
async function getColumnTexts(page: Page, columnIndex: number): Promise<string[]> {
    const cells = page.locator(`tbody tr td:nth-child(${columnIndex})`);
    const texts = await cells.allTextContents();
    return texts.map((t) => (t ?? "").trim()).filter((t) => t.length > 0);
}

/** 断言升序 */
function expectSortedAsc(values: string[], message = "排序结果不是升序") {
    const normalized = values.map((v) => v.trim().toLowerCase());
    const sorted = [...normalized].sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
    expect(normalized, message).toEqual(sorted);
}

/** 断言降序 */
function expectSortedDesc(values: string[], message = "排序结果不是降序") {
    const normalized = values.map((v) => v.trim().toLowerCase());
    const sorted = [...normalized].sort((a, b) => b.localeCompare(a, "zh-Hans-CN"));
    expect(normalized, message).toEqual(sorted);
}

/** Click a column header twice and assert asc then desc order. */
async function verifyColumnSortToggle(page: Page, headerName: string, colIndex: number) {
    await clickSortHeader(page, headerName);
    const asc = await getColumnTexts(page, colIndex);
    expectSortedAsc(asc, `${headerName}第一次点击后升序排序结果不正确`);
    await clickSortHeader(page, headerName);
    const desc = await getColumnTexts(page, colIndex);
    expectSortedDesc(desc, `${headerName}第二次点击后降序排序结果不正确`);
}

async function verifyTableColumnSorts(
    page: Page,
    columns: Array<{ header: string; colIndex: number }>,
) {
    const rows = page.locator("table tbody tr");
    await expect.poll(async () => rows.count(), { timeout: 60_000 }).toBeGreaterThanOrEqual(2);
    for (const col of columns) {
        await verifyColumnSortToggle(page, col.header, col.colIndex);
    }
}

/** 概览页排序校验 */
async function verifyOverviewSort(
    page: Page,
    options: { nameHeader: string; nameColIndex: number; updateTimeHeader: string; updateTimeColIndex: number },
) {
    await verifyTableColumnSorts(page, [
        { header: options.nameHeader, colIndex: options.nameColIndex },
        { header: options.updateTimeHeader, colIndex: options.updateTimeColIndex },
    ]);
}

/** 资源标签页排序校验（默认：资源名称在第1列，类型在第3列） */
async function verifyResourceSortInResourceTab(page: Page, typeColumnIndex = 3) {
    await verifyTableColumnSorts(page, [
        { header: "资源名称", colIndex: 1 },
        { header: "类型", colIndex: typeColumnIndex },
    ]);
}

// =========================
// SSH + kubectl utilities
// =========================

const normalizeSpaces = (text: string) => (text ?? "").replace(/\s+/g, " ").trim();

async function withKubectlSsh<T>(fn: (ssh: NodeSSH) => Promise<T>): Promise<T> {
    const TEST_SSH_USERNAME = "root";
    expect(TEST_SSH_ADDRESS, "缺少环境变量 TEST_NODE2_IP").toBeTruthy();
    expect(TEST_SSH_PASSWORD, "缺少环境变量 TEST_NODE2_PASSWORD").toBeTruthy();
    const ssh = new NodeSSH();
    await ssh.connect({
        host: TEST_SSH_ADDRESS,
        username: TEST_SSH_USERNAME,
        password: TEST_SSH_PASSWORD,
    });

    try {
        return await fn(ssh);
    } finally {
        ssh.dispose();
    }
}

async function execKubectl(
    ssh: NodeSSH,
    command: string,
): Promise<{ stdout: string; stderr: string; code?: number }> {
    const result = await ssh.execCommand(command);
    const code = typeof result.code === "number" ? result.code : undefined;
    return {
        stdout: (result.stdout ?? "").toString(),
        stderr: (result.stderr ?? "").toString(),
        code,
    };
}

async function kubectlGetFirstPodByNamePrefix(
    ssh: NodeSSH,
    namespace: string,
    podNamePrefix: string,
): Promise<{ namespace: string; podName: string }> {
    const maxRetry = 30;
    const intervalMs = 2000;

    for (let i = 0; i < maxRetry; i++) {
        const { stdout, code } = await execKubectl(
            ssh,
            `kubectl get pods -n ${namespace} --no-headers -o custom-columns=NAME:.metadata.name`,
        );
        if (code === 0) {
            const names = stdout
                .trim()
                .split(/\s+/)
                .filter(Boolean);
            const match =
                names.find((n) => n.startsWith(podNamePrefix)) ??
                names.find((n) => n.includes(podNamePrefix));
            if (match) return { namespace, podName: match };
        }
        await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error(`未找到命名空间=${namespace} 下 pod 前缀=${podNamePrefix} 的 pod（尝试 ${maxRetry} 次后仍为空）`);
}

async function getUiLogText(page: Page): Promise<string> {
    const candidates = [
        page.locator("pre").first(),
        page.locator("code").first(),
        page.locator("textarea").first(),
        page.locator('[class*="log"]').first(),
        page.locator('[class*="Log"]').first(),
    ];

    for (const loc of candidates) {
        if ((await loc.count()) === 0) continue;
        const t = await loc.textContent();
        if (t && normalizeSpaces(t).length > 0 && !normalizeSpaces(t).includes("暂无日志")) {
            return t;
        }
    }

    const bodyText = await page.locator("body").innerText();
    const emptyLoc = page.getByText("暂无日志");
    const emptyVisible = await emptyLoc.isVisible().catch(() => false);
    if (emptyVisible) return "";
    return bodyText;
}

async function verifyUiLogWithKubectl(
    page: Page,
    ssh: NodeSSH,
    params: {
        kubectlNamespace: string;
        kubectlPodNamePrefix: string;
        kubectlContainerName?: string;
    },
) {
    const { namespace, podName } = await kubectlGetFirstPodByNamePrefix(
        ssh,
        params.kubectlNamespace,
        params.kubectlPodNamePrefix,
    );

    const isHeaderLikeLine = (line: string) => /Pod/.test(line) && /容器/.test(line);
    const extractMsgLikeFragments = (line: string): string[] => {
        const n = normalizeSpaces(line);
        if (!n || n.length < 8) return [];

        const out = new Set<string>();

        for (const key of ["msg", "message"]) {
            const m = n.match(new RegExp(`"${key}"\\s*:\\s*"([^"]+)"`));
            if (m?.[1]) out.add(m[1]);
        }

        const mLogfmt = n.match(/\b(?:msg|message)\s*=\s*"([^"]+)"/);
        if (mLogfmt?.[1]) out.add(mLogfmt[1]);

        const mGoQuoted = n.match(/]\s*"([^"]+)"/);
        if (mGoQuoted?.[1]) out.add(mGoQuoted[1]);

        const quoted = Array.from(n.matchAll(/"([^"]{10,})"/g)).map((m) => m[1]);
        for (const q of quoted) out.add(q);

        if (!isHeaderLikeLine(n)) {
            const tokens = Array.from(n.matchAll(/[A-Za-z0-9][A-Za-z0-9._-]{8,}/g)).map((m) => m[0]);
            for (const t of tokens) out.add(t);
        }

        return Array.from(out);
    };

    const buildCandidateKeywords = (uiText: string): string[] => {
        const lines = (uiText ?? "")
            .split(/\r?\n/)
            .map((l) => l.trim())
            .filter(Boolean);
        const fragments: string[] = [];
        for (const l of lines) fragments.push(...extractMsgLikeFragments(l));
        if (lines.length <= 1) fragments.push(...extractMsgLikeFragments(uiText));
        const normalized = fragments
            .map((s) => normalizeSpaces(String(s)))
            .filter((s) => s.length >= 8)
            .map((s) => (s.length > 80 ? s.slice(0, 80) : s));
        const unique = Array.from(new Set(normalized));
        if (unique.length > 0) return unique;
        const meaningfulLines = lines.filter((l) => !isHeaderLikeLine(l) && l.length >= 8);
        const baseLines = meaningfulLines.length > 0 ? meaningfulLines : lines.filter((l) => l.length >= 8);
        const fallbackLines = [
            baseLines[0],
            baseLines[baseLines.length - 1],
            baseLines.find((l) => l.length >= 20),
        ].filter(Boolean) as string[];
        const fallbackTokens: string[] = [];
        for (const l of fallbackLines) {
            const s1 = normalizeSpaces(l).slice(0, 120);
            const s2 = normalizeSpaces(l).slice(-120);
            if (s1.length >= 8) fallbackTokens.push(s1);
            if (s2.length >= 8 && s2 !== s1) fallbackTokens.push(s2);
        }
        return Array.from(new Set(fallbackTokens));
    };

    // 轮询等待 UI 日志真正加载出"可匹配片段"
    let uiLogText = "";
    let candidateKeywords: string[] = [];
    const maxUiRetry = 6;
    for (let i = 0; i < maxUiRetry; i++) {
        uiLogText = await getUiLogText(page);
        if (normalizeSpaces(uiLogText).length < 20) {
            await new Promise((r) => setTimeout(r, 1500));
            continue;
        }
        const uiLines = (uiLogText ?? "")
            .split(/\r?\n/)
            .map((l) => l.trim())
            .filter(Boolean);
        const meaningfulLines = uiLines.filter((l) => !isHeaderLikeLine(l) && l.length >= 8);
        // 只有 Pod/容器标题行时，继续等
        if (meaningfulLines.length === 0) {
            await new Promise((r) => setTimeout(r, 1500));
            continue;
        }
        candidateKeywords = buildCandidateKeywords(uiLogText);
        if (candidateKeywords.length > 0) break;
        await new Promise((r) => setTimeout(r, 1500));
    }
    expect(normalizeSpaces(uiLogText).length, "UI 日志为空或仍在显示暂无日志").toBeGreaterThanOrEqual(20);
    expect(candidateKeywords.length, "UI 日志中未提取到可用于匹配的关键片段").toBeGreaterThanOrEqual(1);
    const tryCommands: string[] = [];
    if (params.kubectlContainerName) {
        tryCommands.push(
            `kubectl logs -n ${namespace} ${podName} -c ${params.kubectlContainerName} --tail=500`,
        );
    }
    tryCommands.push(`kubectl logs -n ${namespace} ${podName} --tail=500`);
    tryCommands.push(`kubectl logs -n ${namespace} ${podName} --all-containers --tail=500`);

    let kubectlLogs = "";
    const maxRetry = 8;
    const intervalMs = 2000;
    for (let i = 0; i < maxRetry; i++) {
        for (const cmd of tryCommands) {
            const { stdout, code } = await execKubectl(ssh, cmd);
            if (code === 0 && stdout.trim().length > 0) {
                kubectlLogs = stdout;
                break;
            }
        }
        if (kubectlLogs.trim().length > 0) break;
        await new Promise((r) => setTimeout(r, intervalMs));
    }

    expect(kubectlLogs.trim().length, "kubectl 获取日志失败或日志为空").toBeGreaterThanOrEqual(20);
    const kubectlLogsNormalized = normalizeSpaces(kubectlLogs);
    const containsAny = candidateKeywords.some((kw) => kubectlLogsNormalized.includes(kw));
    expect(
        containsAny,
        `UI日志片段无法在 kubectl 日志中找到。\n` +
            `候选匹配关键字=${candidateKeywords.join(" | ")}\n` +
            `UI日志(前400)=${normalizeSpaces(uiLogText).slice(0, 400)}\n` +
            `kubectl日志(前400)=${kubectlLogsNormalized.slice(0, 400)}`,
    ).toBeTruthy();
}

// =========================
// YAML quick deploy
// =========================

async function openYamlQuickDeploy(page: Page) {
    await goToAppMarket(page);
    await page.getByRole("link", { name: "应用列表" }).click();
    await page.getByRole("img", { name: "appstore-add" }).locator("svg").click();
    await page.getByRole("button", { name: "前往部署" }).click();
    await expect(page).toHaveURL(/\/container_platform\/appMarket\/oneClickDeploy/);
    await expect(page.getByRole("button", { name: "部 署" })).toBeVisible();
}

async function inputYamlContent(page: Page, yamlContent: string) {
    const genericTextbox = page.getByRole("textbox").first();
    await expect(genericTextbox).toBeVisible({ timeout: 60_000 });
    await genericTextbox.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.insertText(yamlContent);
}

/** Delete Deployment + Service created by YAML quick deploy, using shared console utilities. */
async function deleteResources(page: Page, deploymentName: string, serviceName: string) {
    await page.goto(serviceClusterUrl + "/container_platform/workload/deployment/");
    await searchResource(page, deploymentName);
    await deleteResourceInList(page, deploymentName);
    await confirmDeleteDialog(page);
    await searchResource(page, deploymentName);
    await expectResourceDeleted(page, deploymentName);
    await page.goto(serviceClusterUrl + "/container_platform/network/service/");
    await searchResource(page, serviceName);
    await deleteResourceInList(page, serviceName);
    await confirmDeleteDialog(page);
    await searchResource(page, serviceName);
    await expectResourceDeleted(page, serviceName);
}

// =========================
// Component lifecycle (shared deploy / cleanup for specs)
// =========================

/** beforeAll helper: deploy shared apps; hash is generated inside buildDeployPlans.
 * Pass `outApps` to accumulate installs as they succeed, so afterAll can still
 * clean up if this hook times out mid-way.
 */
async function deployComponents(
    browser: Browser,
    templates: DeployPlanTemplate[],
    outApps?: DeployPlan[],
): Promise<DeployPlan[]> {
    const deployedApps = outApps ?? [];
    await withAuthorizedPage(browser, async (page) => {
        for (const plan of buildDeployPlans(templates)) {
            await installComponent(page, plan.componentName, plan.appName, plan.version, plan.source);
            await checkComponent(page, plan.appName, { entry: plan.entry });
            deployedApps.push(plan);
        }
    });
    return deployedApps;
}

/** afterAll helper: uninstall shared apps via a fresh authorized page. */
async function cleanupComponents(browser: Browser, apps: DeployPlan[]): Promise<void> {
    if (!apps.length) return;
    await withAuthorizedPage(browser, async (page) => {
        await uninstallManagedApps(page, apps);
    });
}

// Used by "概览-跳转" tests for bulk helm release creation via API
const helmReleaseNamespace = applicationTestNamespace;
const helmReleaseApiBase =
    `/clusters/host/rest/application-management/v1beta1/helm-releases/namespace/${helmReleaseNamespace}/release`;

/** beforeAll helper: bulk-create helm releases via API (for pagination test). */
async function deployHelmReleasesByApi(
    browser: Browser,
    templates: DeployPlanTemplate[],
    outApps?: DeployPlan[],
): Promise<DeployPlan[]> {
    const deployedApps = outApps ?? [];
    // Use shared e2e-helm-release values for API-created releases
    const appValuesPath = path.join(chartsDir, componentNameApp, "values.yaml");
    const values = fs.readFileSync(appValuesPath, "utf8");
    await withAuthorizedPage(browser, async (page) => {
        for (const plan of buildDeployPlans(templates)) {
            const response = await page.request.post(
                `${serviceClusterUrl}${helmReleaseApiBase}/${plan.appName}`,
                {
                    data: {
                        chartName: plan.componentName,
                        repoName: plan.source ?? localPackageSource,
                        version: plan.version ?? packageVersion,
                        values,
                    },
                },
            );
            expect(
                response.ok(),
                `create helm release ${plan.appName}: ${response.status()} ${await response.text()}`,
            ).toBeTruthy();
            deployedApps.push(plan);
        }
    });
    return deployedApps;
}

/** afterAll helper: bulk-delete helm releases via API. */
async function cleanupHelmReleasesByApi(browser: Browser, apps: DeployPlan[]): Promise<void> {
    if (!apps.length) return;
    await withAuthorizedPage(browser, async (page) => {
        for (const app of apps) {
            const response = await page.request.delete(
                `${serviceClusterUrl}${helmReleaseApiBase}/${app.appName}`,
            );
            if (response.status() === 404) continue;
            expect(
                response.ok(),
                `delete helm release ${app.appName}: ${response.status()} ${await response.text()}`,
            ).toBeTruthy();
        }
    });
}

export {
    installComponent,
    checkComponent,
    changePullPolicyLineWithWait,
    selectLog,
    waitForLogReady,
    openUpgradeEditor,
    selectUpgradeChartVersion,
    upgradeByChartVersion,
    gotoManagedAppDetail,
    revertComponent,
    checkPrefixResourceNum,
    verifyResourceSortInResourceTab,
    verifyOverviewSort,
    withKubectlSsh,
    verifyUiLogWithKubectl,
    verifyLogExport,
    openYamlQuickDeploy,
    inputYamlContent,
    deleteResources,
    withAuthorizedPage,
    deployComponents,
    deployHelmReleasesByApi,
    uninstallManagedApps,
    cleanupComponents,
    cleanupHelmReleasesByApi,
};
