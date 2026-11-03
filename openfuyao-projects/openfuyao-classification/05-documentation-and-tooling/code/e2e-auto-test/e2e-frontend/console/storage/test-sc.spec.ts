import { expect, test, type APIRequestContext } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  clickCreate,
  saveCreateOrEdit,
  cancelCreateOrEdit,
  searchResource,
  deleteResourceInList,
  confirmDeleteDialog,
  cancelDeleteDialog,
  clickRefresh,
  clickYamlTab,
  openModifyLabelAnnotationEditor,
  ExpectAddLabelAnnotationSuccess,
  deleteLabelAnnotationAndExpectSuccess,
  expectEmptyLabelAnnotationError,
  expectNoChangeLabelAnnotationMessage,
  clickDeleteInDetail,
  clickExportYaml,
  clickFindInYaml,
  clickCopyYaml,
  clickEditResourceInList,
  expectSuccessMessage,
  expectErrorMessage,
  replaceTextInEditor,
  expectResourceDeleted,
  uniqueResourceName,
  readStaticYaml,
  fillYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
} from '@/console/utils/common'

const LIST_PATH = '/container_platform/storage/sc';
const SC_YAML = readStaticYaml('valid-sc.yaml');
const NAME_PREFIX = 'test-sc';
const COPY_PREFIX = 'test-sc-copy';

const createScViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await createFromYamlViaApi(apiContext, {
    yaml: SC_YAML,
    name,
    plural: 'storageclasses',
    apiGroup: 'storage.k8s.io',
  });
};

const deleteScViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await deleteResourceViaApi(apiContext, {
    name,
    plural: 'storageclasses',
    apiGroup: 'storage.k8s.io',
  });
};

test.describe('创建storageclass', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await deleteScViaApi(request, uniqueResourceName(NAME_PREFIX));
  });

  test('【资源管理-699】 存储-StorageClass，取消创建StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, SC_YAML, name);
    await cancelCreateOrEdit(page);
    await searchResource(page, name);
    await expect(page.getByText('暂无数据')).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-695】 存储-StorageClass，输入格式有误的yaml创建StorageClass失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreate(page);
    await fillYaml(page, SC_YAML, uniqueResourceName(NAME_PREFIX), (yaml) => yaml.replace(/^\s+/gm, ''));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范|创建失败/);
  });

  test('【资源管理-696】 存储-StorageClass，输入字段有误的yaml创建StorageClass失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, SC_YAML, name, (content) => (
      content.replace(/Retain/g, 'Invalid')
    ));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`StorageClass\\.storage\\.k8s\\.io "${name}" is invalid`));
  });

  test('【资源管理-694】 存储-StorageClass，输入正确yaml创建StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, SC_YAML, name);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-700】 存储-StorageClass，短时间连续点击创建按钮时，查看页面提示', {
    tag: ['@v24.06', '@k8s', '@skip-temporarily'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, SC_YAML, name);
    await saveCreateOrEdit(page);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await expectErrorMessage(page, /storageclasses.storage.k8s.io ".*" already exists/);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('查询、修改、删除storageclass', () => {
  test.beforeEach(async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await createScViaApi(request, name);
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await deleteScViaApi(request, uniqueResourceName(NAME_PREFIX));
  });

  test('【资源管理-697】 存储-StorageClass，同一个命名空间下已存在相同的StorageClass名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, SC_YAML, name);
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /storageclasses.storage.k8s.io ".*" already exists/);
  });

  test('【资源管理-702】 存储-StorageClass，搜索框输入正确的StorageClass名称，可正确查询出StorageClass信息', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-703】 存储-StorageClass，搜索框异常场景验证', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name.toUpperCase());
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
    await searchResource(page, 'a'.repeat(64));
    await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-713】 存储-StorageClass，取消删除StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await deleteResourceInList(page, name);
    await cancelDeleteDialog(page);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-712】 存储-StorageClass，删除StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await deleteResourceInList(page, name);
    await confirmDeleteDialog(page);
    await expect(page.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
    await searchResource(page, name);
    await expectResourceDeleted(page, name);
  });

  test('【资源管理-715】 存储-StorageClass，刷新按钮可正常点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickRefresh(page);
  });

  test('【资源管理-705】 存储-StorageClass，修改StorageClass时输入正确yaml修改StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'example.com/created-by: e2e', 'example.com/created-by: e2e-test');
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
  });

  test('【资源管理-706】 存储-StorageClass，修改StorageClass时输入格式有误的yaml修改StorageClass失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, '^\\s+', '');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范/);
  });

  test('【资源管理-707】 存储-StorageClass，修改StorageClass时输入字段有误的yaml修改StorageClass失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'Retain', 'Invalid');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`StorageClass\\.storage\\.k8s\\.io "${name}" is invalid`));
  });

  test('【资源管理-708】 存储-StorageClass，修改StorageClass时同一个命名空间下已存在相同的StorageClass名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    const copyName = uniqueResourceName(COPY_PREFIX);
    try {
      await clickCreate(page);
      await fillYaml(page, SC_YAML, copyName);
      await saveCreateOrEdit(page);
      await expectSuccessMessage(page);

      await searchResource(page, name);
      await clickEditResourceInList(page, name);
      await replaceTextInEditor(page, `name: ${name}`, `name: ${copyName}`);
      await saveCreateOrEdit(page);
      await expectErrorMessage(page);
    } finally {
      await deleteScViaApi(request, copyName);
    }
  });

  test('【资源管理-710】 存储-StorageClass，取消修改StorageClass成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await cancelCreateOrEdit(page);
  });

  test('【资源管理-716】 存储-StorageClass，StorageClass详情中的详情Tab页面信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await page.waitForLoadState('networkidle');
    for (const val of await page.locator('.base_value').all()) {
      expect(await val.textContent()).not.toMatch(/^-?$/);
    }
  });

  test('【资源管理-717】 存储-StorageClass，StorageClass详情的操作-修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    const key = '000-test-key';
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await ExpectAddLabelAnnotationSuccess(page, 'label', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await expectEmptyLabelAnnotationError(page, 'label');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await expectNoChangeLabelAnnotationMessage(page, 'label');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await deleteLabelAnnotationAndExpectSuccess(page, 'label', key, value);
    });
  });

  test('【资源管理-718】 存储-StorageClass，StorageClass详情的操作-修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    const key = '000-test-key';
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await ExpectAddLabelAnnotationSuccess(page, 'annotation', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await expectEmptyLabelAnnotationError(page, 'annotation');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await expectNoChangeLabelAnnotationMessage(page, 'annotation');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await deleteLabelAnnotationAndExpectSuccess(page, 'annotation', key, value);
    });
  });

  test('【资源管理-720】 存储-StorageClass，StorageClass详情中的详情Tab-基本信息中修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    const key = '000-test-key';
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'menu');
      await ExpectAddLabelAnnotationSuccess(page, 'label', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'menu');
      await expectEmptyLabelAnnotationError(page, 'label');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'menu');
      await expectNoChangeLabelAnnotationMessage(page, 'label');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'menu');
      await deleteLabelAnnotationAndExpectSuccess(page, 'label', key, value);
    });
  });

  test('【资源管理-721】 存储-StorageClass，StorageClass详情中的详情Tab-基本信息中修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    const key = '000-test-key';
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'menu');
      await ExpectAddLabelAnnotationSuccess(page, 'annotation', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'menu');
      await expectEmptyLabelAnnotationError(page, 'annotation');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'menu');
      await expectNoChangeLabelAnnotationMessage(page, 'annotation');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'menu');
      await deleteLabelAnnotationAndExpectSuccess(page, 'annotation', key, value);
    });
  });

  test('【资源管理-722】 存储-StorageClass，StorageClass详情中的YAML Tab页面展示正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await clickYamlTab(page);
    expect(await page.getByText(/apiVersion|kind:\s*SC/).first().textContent()).not.toBeNull();
  });

  test('【资源管理-723】 存储-StorageClass，StorageClass详情中的YAML Tab页面导出按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await clickYamlTab(page);
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 3000 }),
      clickExportYaml(page),
    ]);
    expect(download).toBeTruthy();
  });

  test('【资源管理-724】 存储-StorageClass，StorageClass详情中的YAML Tab页面查找按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await clickYamlTab(page);
    await clickFindInYaml(page);
    await expect(page.getByRole('textbox').or(page.getByPlaceholder(/查找|搜索/))).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-725】 存储-StorageClass，StorageClass详情中的YAML Tab页面复制按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await clickYamlTab(page);
    await clickCopyYaml(page);
    await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-719】 存储-StorageClass，StorageClass详情的操作-删除功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/storage/sc/${name}`);
    await clickDeleteInDetail(page);
    await confirmDeleteDialog(page);
    await expectSuccessMessage(page, /删除成功！/);
    expect(page.url()).toContain('/container_platform/storage/sc');
    await searchResource(page, name);
    await expectResourceDeleted(page, name);
  });
});
