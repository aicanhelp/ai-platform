import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import YAML from 'yaml';
import { FUYAO_BASE_URL } from '@/utils/constants';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.resolve(__dirname, '..', 'static');

/** 在列表页点击「创建」进入创建页 */
export const clickCreate = async (page: Page) => {
  await page.getByRole('button', { name: /创\s*建/ }).click();
};

/** 在角色、角色绑定列表页点击「创建」进入创建页 */
export const clickCreateType = async (page: Page, type: string) => {
  await page.getByRole('button', { name: /创\s*建/ }).click();
  await page.getByRole('button', { name: type, exact: true }).click();
};

/** 在创建/编辑页通过文件上传 YAML（若有 type=file 的 input） */
export const uploadYamlFile = async (page: Page, yamlFileName: string) => {
  const filePath = path.join(staticDir, yamlFileName);
  const input = page.locator('input[type="file"]').first();
  await input.setInputFiles(filePath);
};

/** 在创建/编辑页通过 CodeMirror 编辑器填写 YAML 内容（适用于无文件上传时） */
export const fillYamlContent = async (page: Page, yamlContent: string) => {
  const editor = page.locator('.cm-editor .cm-content, .CodeMirror').first();
  await editor.click();
  await page.keyboard.press('Control+a');
  await page.keyboard.insertText(yamlContent);
};

/** 在创建/编辑页点击「保存」或「确 定」 */
export const saveCreateOrEdit = async (page: Page) => {
  await page.getByRole('button', { name: /保\s*存|确\s*定|确\s*认/ }).first().click();
};

/** 在创建/编辑页点击「取消」 */
export const cancelCreateOrEdit = async (page: Page) => {
  await page.getByRole('button', { name: /取\s*消/ }).first().click();
};

/** 列表页：搜索框输入并查询 */
export const searchResource = async (page: Page, name: string) => {
  await page.getByRole('textbox', { name: /搜索.*名称/ }).fill(name);
  await page.getByRole('button', { name: 'search' }).click();
};

/** 列表页：对指定名称的资源点击删除（行内操作或表格操作） */
export const deleteResourceInList = async (page: Page, name: string) => {
  const row = page.getByRole('row', { name: new RegExp(name) }).first();
  await row.locator('svg').click();
  await page.getByRole('button', { name: /删\s*除/ }).click();
};

/** 删除确认弹窗：点击确定 */
export const confirmDeleteDialog = async (page: Page) => {
  await page.getByText('已了解删除操作无法恢复').click();
  await page.getByRole('dialog').getByRole('button', { name: /删\s*除/ }).click();
};

/** 删除确认弹窗：点击取消 */
export const cancelDeleteDialog = async (page: Page) => {
  await page.getByRole('dialog').getByRole('button', { name: /取\s*消/ }).click();
};

/** 列表页：点击刷新 */
export const clickRefresh = async (page: Page) => {
  await page.getByRole('button', { name: 'sync' }).click();
};

/** 列表页：点击资源名称进入资源详情页面 */
export const clickResourceNameInList = async (page: Page, name: string) => {
  const row = page.getByRole('row', { name: new RegExp(name) }).first();
  await row.getByText(name, { exact: true }).click();
};

/** 详情页：点击「详情」Tab */
export const clickDetailTab = async (page: Page) => {
  await page.getByRole('tab', { name: '详情' }).click();
};

/** 详情页：点击「YAML」Tab */
export const clickYamlTab = async (page: Page) => {
  await page.getByRole('tab', { name: 'YAML' }).click();
};

/** 详情页：点击「事件」Tab */
export const clickEventTab = async (page: Page) => {
  await page.getByRole('tab', { name: '事件' }).click();
};

export type LabelAnnotationKind = 'label' | 'annotation';
export type LabelAnnotationEntry = 'icon' | 'menu';

/** 打开标签/注解编辑弹窗 */
export const openModifyLabelAnnotationEditor = async (
  page: Page,
  kind: LabelAnnotationKind,
  entry: LabelAnnotationEntry,
): Promise<void> => {
  const successMsg = kind === 'label' ? /编辑标签成功/ : /编辑注解成功/;
  await expect(page.getByText(successMsg)).not.toBeVisible({ timeout: 5000 });

  if (entry === 'menu') {
    await page.getByRole('button', { name: '操作 down' }).click();
    await page.getByRole('button', { name: kind === 'label' ? /修改标签|编辑标签/ : /修改注解|编辑注解/ }).click();
    return;
  }

  const sectionLabel = kind === 'label' ? /^标签：$/ : /^注解：$/;
  await page.locator('div').filter({ hasText: sectionLabel }).locator('svg').click();
};

/** 详情页：操作 -> 修改注解 / 基本信息 -> 编辑注解 */
export const confirmAddLabelAnnotation = async (page: Page, key?: string, value?: string) => {
  await page.getByRole('button', { name: 'plus-circle' }).click();
  if (key && value) {
    await page.getByRole('textbox').nth(-2).fill(key);
    await page.getByRole('textbox').nth(-1).fill(value);
  }
};

export const clickDeleteLabelAnnotation = async (page: Page, key: string) => {
  const dialog = page.getByRole('dialog');
  const textboxes = dialog.getByRole('textbox');
  const count = await textboxes.count();
  // key 和 value 各占一个 textbox，一行有两个
  for (let i = 0; i < count; i += 2) {
    if (await textboxes.nth(i).inputValue() === key) {
      const rowindex = i / 2;
      await dialog.locator('[aria-label=delete]').nth(rowindex).click();
      await page.getByRole('button', { name: /确\s*定/ }).click();
      return;
    }
  }
  throw new Error(`Label/annotation row not found for key: ${key}`);
};

/** 标签/注解编辑弹窗：取消 */
export const cancelLabelAnnotationDialog = async (page: Page) => {
  await page.getByRole('dialog').getByRole('button', { name: /取\s*消/ }).click();
};

/** 标签/注解编辑弹窗：确定 */
export const confirmLabelAnnotationDialog = async (page: Page) => {
  await page.getByRole('dialog').getByRole('button', { name: /确\s*定/ }).click();
};

/** 新增标签/注解并校验成功提示与页面展示 */
export const ExpectAddLabelAnnotationSuccess = async (
  page: Page,
  kind: LabelAnnotationKind,
  key?: string,
  value?: string,
) => {
  await confirmAddLabelAnnotation(page, key, value);
  await confirmLabelAnnotationDialog(page);
  const successMsg = (kind === 'label' ? /编辑标签成功/ : /编辑注解成功/);
  await expectSuccessMessage(page, successMsg);
  await expect(page.locator('.label_item').getByText(`${key}${value}`)).toBeVisible({ timeout: 5000 });
};

/** 未修改内容直接确定，校验未修改提示 */
export const expectNoChangeLabelAnnotationMessage = async (page: Page, kind: LabelAnnotationKind) => {
  await confirmLabelAnnotationDialog(page);
  await expect(page.getByText(/未进行修改/)).toBeVisible({ timeout: 5000 });
};

/** 新增空行后提交，校验必填错误提示 */
export const expectEmptyLabelAnnotationError = async (page: Page, kind: LabelAnnotationKind) => {
  await page.getByRole('button', { name: 'plus-circle' }).click();
  await confirmLabelAnnotationDialog(page);
  const emptyErrorMsg = (kind === 'label' ? /标签键不能为空/ : /注解键不能为空/);
  await expect(page.getByText(emptyErrorMsg)).toBeVisible({ timeout: 5000 });
  await cancelLabelAnnotationDialog(page);
};

/** 删除指定 key 的标签/注解并校验成功提示 */
export const deleteLabelAnnotationAndExpectSuccess = async (
  page: Page,
  kind: LabelAnnotationKind,
  key: string,
  value: string = 'test-value',
) => {
  await clickDeleteLabelAnnotation(page, key);
  const successMsg = (kind === 'label' ? /编辑标签成功/ : /编辑注解成功/);
  await expectSuccessMessage(page, successMsg);
  await expect(page.locator('.label_item').getByText(`${key}${value}`)).not.toBeVisible({ timeout: 5000 });
};

/** 详情页：操作 -> 删除 */
export const clickDeleteInDetail = async (page: Page) => {
  await page.getByRole('button', { name: '操作 down' }).click();
  await page.getByRole('button', { name: '删除' }).click();
};

/** 详情页：容器区域点击「日志」 */
export const clickLogInContainer = async (page: Page) => {
  await page.locator('[aria-label=calendar]').click();
};

/** 详情页：容器区域点击「终端」 */
export const clickTerminalInContainer = async (page: Page) => {
  await page.getByRole('table').locator('[aria-label="code"]').first().click();
};

export const closeTerminal = async (page: Page) => {
  await page.locator('#root_container div').filter({ hasText: /容器：/ }).locator('svg').nth(2).click();
  await page.getByRole('button', { name: /确\s*定/ }).click();
};

/** 详情页：点击「监控大屏」 */
export const clickMonitorDashboard = async (page: Page) => {
  await page.getByRole('link', { name: /监控大屏/ }).or(page.getByRole('button', { name: /监控大屏/ })).first().click();
};

/** YAML Tab：点击导出 */
export const clickExportYaml = async (page: Page) => {
  await page.getByText('导出').click();
};

/** YAML Tab：点击查找 */
export const clickFindInYaml = async (page: Page) => {
  const editor = page.locator('.cm-editor').first();
  await editor.click();
  await editor.press('ControlOrMeta+f');
};

/** YAML Tab：点击复制 */
export const clickCopyYaml = async (page: Page) => {
  await page.getByText('复制').click();
};

/** 事件 Tab：点击查询 */
export const eventFilterQuery = async (page: Page, option: string) => {
  await page.locator('.container-platform-select').last().click();
  await page.getByTitle(option).locator('div').click();
  await page.getByRole('button', { name: /查\s*询/ }).click();
};

/** 事件 Tab：点击重置 */
export const eventFilterReset = async (page: Page) => {
  await page.getByRole('button', { name: /重\s*置/ }).click();
};

/** 列表页：对指定资源点击「修改」进入编辑 */
export const clickEditResourceInList = async (page: Page, name: string) => {
  const row = page.getByRole('row', { name: new RegExp(name) }).first();
  await row.locator('svg').click();
  await page.getByRole('button', { name: /修\s*改/ }).click();
};

/** 列表页：对指定自定义资源点击「修改」进入编辑，前端实现与其他资源不统一，临时专用函数 */
export const clickEditCustomResourceInList = async (page: Page, name: string) => {
  const row = page.getByRole('row', { name: new RegExp(name) }).first();
  await row.locator('svg').click();
  await page.getByText(/修\s*改/).click();
};

/** 校验当前页出现成功/创建成功类提示 */
export const expectSuccessMessage = async (page: Page, pattern?: RegExp, timeout = 10000) => {
  const patternText = pattern || /创建成功|保存成功|修改成功|删除成功/;
  await expect(page.getByText(patternText)).toBeVisible({ timeout });
};

/** 校验当前页出现错误提示（格式错误、字段错误等） */
export const expectErrorMessage = async (page: Page, pattern?: string | RegExp, timeout = 10000) => {
  const patternText = pattern || /创建失败|保存失败|修改失败|删除失败/;
  await expect(page.getByText(patternText)).toBeVisible({ timeout });
};

/** 
 * 编辑器替换文本 
 * always use string as find */
export const replaceTextInEditor = async (page: Page, find: string, replace: string) => {
  const editor = page.locator('.cm-editor').first();
  await editor.click();
  await editor.press('ControlOrMeta+f');
  await editor.getByRole('checkbox', { name: 'regexp' }).check();
  await editor.getByRole('textbox', { name: 'Find' }).fill(find);
  await editor.getByRole('textbox', { name: 'Replace' }).fill(replace);
  await editor.getByRole('button', { name: 'replace all' }).click();
};

/** 确认资源进入状态 - 列表页*/
export const expectResourceStatus = async (
  page: Page, name: string, status: string, maxRetry = 10) => {
  for (let i = 0; i < maxRetry; i++) {
    await clickRefresh(page);
    if (await page.getByRole('row', { name: new RegExp(`${name}.*${status}`) }).first().isVisible()) {
      return;
    }
    await page.waitForTimeout(3000);
  }
  throw new Error(`Resource ${name} not found with status: ${status} after ${maxRetry} attempts`);
};

/** 确认资源已删除 with 轮询 */
export const expectResourceDeleted = async (page: Page, name: string, maxRetry = 10) => {
  for (let i = 0; i < maxRetry; i++) {
    await clickRefresh(page);
    if (await page.getByText(/暂无数据/).isVisible()) {
      return;
    }
    await page.waitForTimeout(3000);
  }
  throw new Error(`Resource ${name} not deleted after ${maxRetry} attempts`);
};

/** 前端 URL 中 CRD 全名将 `.` 编码为 `~`，如 testresources.test.org → testresources~test~org */
export const encodeCrdNameForUrl = (crdName: string): string => crdName.replace(/\./g, '~');

const K8S_PROXY_PREFIX = '/clusters/host/api/kubernetes';

type YamlObject = {
  apiVersion?: string;
  kind?: string;
  metadata?: {
    name?: string;
    namespace?: string;
  };
};

const parseApiVersion = (apiVersion?: string): { apiGroup?: string; version: string } => {
  if (!apiVersion || apiVersion === 'v1') {
    return { version: 'v1' };
  }
  const slash = apiVersion.lastIndexOf('/');
  return {
    apiGroup: apiVersion.slice(0, slash),
    version: apiVersion.slice(slash + 1),
  };
};

const k8sResourceUrl = (options: {
  apiGroup?: string;
  version?: string;
  namespace?: string;
  plural: string;
  name?: string;
  propagationPolicy?: 'Background' | 'Foreground' | 'Orphan';
}): string => {
  const version = options.version ?? 'v1';
  const apiPath = options.apiGroup
    ? `/apis/${options.apiGroup}/${version}`
    : `/api/${version}`;
  const nsPath = options.namespace ? `/namespaces/${options.namespace}` : '';
  const base = `${FUYAO_BASE_URL}${K8S_PROXY_PREFIX}${apiPath}${nsPath}/${options.plural}`;
  const url = options.name ? `${base}/${options.name}` : base;
  return options.propagationPolicy
    ? `${url}?propagationPolicy=${options.propagationPolicy}`
    : url;
};

const assertApiOk = async (
  response: Awaited<ReturnType<APIRequestContext['get']>>,
  label: string,
  allowNotFound = false,
): Promise<void> => {
  if (allowNotFound && response.status() === 404) {
    return;
  }
  if (!response.ok()) {
    throw new Error(
      `${label} failed: status=${response.status()} ${response.statusText()}; body=${await response.text()}`,
    );
  }
};

const nameCache = new Map<string, string>();

export const uniqueResourceName = (prefix: string): string => {
  const key = `${test.info().testId}::${prefix}`;
  const cached = nameCache.get(key);
  if (cached) {
    return cached;
  }
  const suffix = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 4)}`;
  let name = `${prefix}-${suffix}`.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-');
  if (name.length > 63) {
    name = name.slice(0, 63);
  }
  name = name.replace(/-+$/g, '');
  nameCache.set(key, name);
  return name;
};

export type CrdIdentity = {
  name: string;
  group: string;
  plural: string;
  singular: string;
  kind: string;
  shortName: string;
};

export const uniqueCrdIdentity = (prefix = 'r'): CrdIdentity => {
  const suffix = uniqueResourceName(prefix).replace(/-/g, '').slice(0, 10);
  const plural = `testrsrc${suffix}`.slice(0, 20);
  const kind = plural.charAt(0).toUpperCase() + plural.slice(1);
  const group = 'test.org';
  return {
    name: `${plural}.${group}`,
    group,
    plural,
    singular: plural,
    kind,
    shortName: `t${suffix}`.slice(0, 8),
  };
};

export const applyCrdIdentity = (yaml: string, crd: CrdIdentity): string =>
  yaml
    .replace(/name: testresources\.test\.org/g, `name: ${crd.name}`)
    .replace(/group: test.org/g, `group: ${crd.group}`)
    .replace(/plural: testresources/g, `plural: ${crd.plural}`)
    .replace(/singular: testresource/g, `singular: ${crd.singular}`)
    .replace(/kind: TestResource/g, `kind: ${crd.kind}`)
    .replace(/- testres\b/g, `- ${crd.shortName}`);

export const applyCrIdentity = (yaml: string, crd: CrdIdentity, crName: string): string =>
  yaml
    .replace(/apiVersion: test\.org\/v1/g, `apiVersion: ${crd.group}/v1`)
    .replace(/kind: TestResource/g, `kind: ${crd.kind}`)
    .replace(/name: test-testresource/g, `name: ${crName}`);

export const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export const applyMetadataName = (yaml: string, fromName: string, toName: string): string =>
  yaml.replace(new RegExp(`name:\\s*${escapeRegExp(fromName)}\\b`, 'g'), `name: ${toName}`);

export const readStaticYaml = (yamlFileName: string): string =>
  fs.readFileSync(path.join(staticDir, yamlFileName), 'utf-8');

const yamlWithName = (yaml: string, name: string, extraEdit?: (content: string) => string): string => {
  const fromName = (YAML.parse(yaml) as YamlObject | null)?.metadata?.name;
  if (!fromName) {
    throw new Error('yaml missing metadata.name');
  }
  const named = applyMetadataName(yaml, fromName, name);
  return extraEdit ? extraEdit(named) : named;
};

export const fillYaml = async (
  page: Page,
  yaml: string,
  name: string,
  extraEdit?: (content: string) => string,
): Promise<void> => {
  await fillYamlContent(page, yamlWithName(yaml, name, extraEdit));
};

export const createCrdFromYaml = async (page: Page, crd: CrdIdentity): Promise<void> => {
  await clickCreate(page);
  await fillYamlContent(page, applyCrdIdentity(readStaticYaml('valid-crd.yaml'), crd));
  await saveCreateOrEdit(page);
  await expectSuccessMessage(page);
};

export const createCrFromYaml = async (page: Page, crd: CrdIdentity, crName: string): Promise<void> => {
  await clickCreate(page);
  await fillYamlContent(page, applyCrIdentity(readStaticYaml('valid-cr.yaml'), crd, crName));
  await saveCreateOrEdit(page);
  await expectSuccessMessage(page);
};

export type ApiResourceOptions = {
  plural: string;
  name: string;
  namespace?: string;
  apiGroup?: string;
  version?: string;
  propagationPolicy?: 'Background' | 'Foreground' | 'Orphan';
};

export type CreateFromYamlViaApiOptions = {
  yaml: string;
  name: string;
  plural: string;
  namespace?: string;
  apiGroup?: string;
  version?: string;
  extraEdit?: (content: string) => string;
};

export const deleteResourceViaApi = async (
  apiContext: APIRequestContext,
  options: ApiResourceOptions,
): Promise<void> => {
  const response = await apiContext.delete(k8sResourceUrl(options));
  await assertApiOk(response, `DELETE ${options.plural}/${options.namespace ?? ''}/${options.name}`, true);
};

export const createFromYamlViaApi = async (
  apiContext: APIRequestContext,
  options: CreateFromYamlViaApiOptions,
): Promise<void> => {
  const body = YAML.parse(yamlWithName(options.yaml, options.name, options.extraEdit)) as YamlObject;
  const parsed = parseApiVersion(body.apiVersion);
  const response = await apiContext.post(
    k8sResourceUrl({
      apiGroup: options.apiGroup ?? parsed.apiGroup,
      version: options.version ?? parsed.version,
      namespace: options.namespace ?? body.metadata?.namespace,
      plural: options.plural,
    }),
    { data: body },
  );
  await assertApiOk(response, `POST ${options.plural}/${options.namespace ?? ''}/${options.name}`);
};

export const waitForCrdApiViaProxy = async (
  apiContext: APIRequestContext,
  crd: Pick<CrdIdentity, 'group' | 'plural'>,
  timeoutMs = 60_000,
): Promise<void> => {
  const url = k8sResourceUrl({
    apiGroup: crd.group,
    version: 'v1',
    namespace: 'default',
    plural: crd.plural,
  });
  await expect.poll(async () => {
    const response = await apiContext.get(url);
    return response.ok();
  }, { timeout: timeoutMs, intervals: [1000] }).toBeTruthy();
};
