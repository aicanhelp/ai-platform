import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  clickCreate,
  saveCreateOrEdit,
  cancelCreateOrEdit,
  searchResource,
  deleteResourceInList,
  confirmDeleteDialog,
  cancelDeleteDialog,
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
  clickEditCustomResourceInList,
  clickResourceNameInList,
  expectSuccessMessage,
  expectErrorMessage,
  replaceTextInEditor,
  expectResourceDeleted,
  encodeCrdNameForUrl,
  uniqueCrdIdentity,
  uniqueResourceName,
  applyCrdIdentity,
  applyCrIdentity,
  fillYamlContent,
  readStaticYaml,
  createCrFromYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
  waitForCrdApiViaProxy,
  type CrdIdentity,
} from '@/console/utils/common'

const CRD_LIST_PATH = '/container_platform/customResourceDefinition';
const CRD_YAML = readStaticYaml('valid-crd.yaml');
const CR_YAML = readStaticYaml('valid-cr.yaml');
const CR_PREFIX = 'test-testresource';

const gotoInstanceTab = async (page: Page, crd: CrdIdentity) => {
  await page.goto(FUYAO_BASE_URL + `${CRD_LIST_PATH}/${encodeCrdNameForUrl(crd.name)}`);
  await page.getByRole('tab', { name: '实例' }).click();
};

test.describe('自定义资源实例管理', () => {
  let crd: CrdIdentity;

  test.beforeAll(async ({ request }) => {
    crd = uniqueCrdIdentity();
    await createFromYamlViaApi(request, {
      yaml: applyCrdIdentity(CRD_YAML, crd),
      name: crd.name,
      plural: 'customresourcedefinitions',
      apiGroup: 'apiextensions.k8s.io',
    });
    await waitForCrdApiViaProxy(request, crd);
  });

  test.afterAll(async ({ request }) => {
    await deleteResourceViaApi(request, {
      name: crd.name,
      plural: 'customresourcedefinitions',
      apiGroup: 'apiextensions.k8s.io',
    });
  });

  test.describe('创建自定义资源实例', () => {
    test.beforeEach(async ({ page }) => {
      await gotoInstanceTab(page, crd);
    });

    test.afterEach(async ({ request }) => {
      await deleteResourceViaApi(request, {
        name: uniqueResourceName(CR_PREFIX),
        plural: crd.plural,
        namespace: 'default',
        apiGroup: crd.group,
      });
    });

    test('【资源管理-591】 自定义资源-实例，取消创建实例功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name));
      await cancelCreateOrEdit(page);
      await page.getByRole('tab', { name: '实例' }).click();
      await searchResource(page, name);
      await expect(page.getByText('暂无数据')).toBeVisible({ timeout: 5000 });
    });

    test('【资源管理-587】 自定义资源-实例，输入格式有误的yaml创建实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      await clickCreate(page);
      await fillYamlContent(page, CR_YAML.replace(/^\s+/gm, ''));
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, /YAML格式不规范|创建失败/);
    });

    test('【资源管理-588】 自定义资源-实例，输入字段有误的yaml创建实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name).replace(/field2: 1/g, 'field2: xxx'));
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, new RegExp(`${crd.kind}.${crd.group} "${name}" is invalid`));
    });

    test('【资源管理-590】 自定义资源-实例，使用不存在的namespace创建实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name).replace(/namespace: default/g, 'namespace: non-existing-ns'));
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, 'namespaces "non-existing-ns" not found');
    });

    test('【资源管理-592】 自定义资源-实例，创建实例页面重置yaml内容功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      await clickCreate(page);
      const originalText = await page.locator('.cm-editor').first().textContent() || '';
      const name = uniqueResourceName(CR_PREFIX);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name));
      await page.getByRole('button', { name: /重\s*置/ }).click();
      await expect(page.locator('.cm-editor').first()).toContainText(originalText);
    });

    test('【资源管理-593】 自定义资源-实例，创建实例页面导出功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name));
      const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 3000 }),
        clickExportYaml(page),
      ]);
      expect(download).toBeTruthy();
    });

    test('【资源管理-594】 自定义资源-实例，创建实例页面查询功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name));
      await clickFindInYaml(page);
      await expect(page.getByRole('textbox', { name: 'Find' })).toBeVisible({ timeout: 3000 });
      await expect(page.getByRole('textbox', { name: 'Replace' })).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-595】 自定义资源-实例，创建实例页面复制功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name));
      await clickCopyYaml(page);
      await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-586】 自定义资源-实例，输入正确yaml创建实例正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await createCrFromYaml(page, crd, name);
      await gotoInstanceTab(page, crd);
      await searchResource(page, name);
      await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
    });
  });

  test.describe('查询、修改、删除自定义资源实例', () => {
    test.beforeEach(async ({ page, request }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await createFromYamlViaApi(request, {
        yaml: applyCrIdentity(CR_YAML, crd, name),
        name,
        plural: crd.plural,
        namespace: 'default',
        apiGroup: crd.group,
      });
      await gotoInstanceTab(page, crd);
    });

    test.afterEach(async ({ request }) => {
      await deleteResourceViaApi(request, {
        name: uniqueResourceName(CR_PREFIX),
        plural: crd.plural,
        namespace: 'default',
        apiGroup: crd.group,
      });
    });

    test('【资源管理-589】 自定义资源-实例，同一个命名空间下已存在相同的实例名称，应提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await clickCreate(page);
      await fillYamlContent(page, applyCrIdentity(CR_YAML, crd, name).replace(/field1: test/g, 'field1: updated'));
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, new RegExp(`${crd.plural}.${crd.group} ".*" already exists`));
    });

    test('【资源管理-596】 自定义资源-实例，实例展示正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
    });

    test('【资源管理-602】 自定义资源-实例，实例取消删除功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await deleteResourceInList(page, name);
      await cancelDeleteDialog(page);
      await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 5000 });
    });

    test('【资源管理-597】 自定义资源-实例，使用正确yaml修改实例成功', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickEditCustomResourceInList(page, name);
      await replaceTextInEditor(page, 'field1: test', 'field1: e2e-updated');
      await saveCreateOrEdit(page);
      await expectSuccessMessage(page);
    });

    test('【资源管理-598】 自定义资源-实例，输入格式有误的yaml修改实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickEditCustomResourceInList(page, name);
      await replaceTextInEditor(page, '^\\s+', '');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page);
    });

    test('【资源管理-599】 自定义资源-实例，输入字段有误的yaml修改实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickEditCustomResourceInList(page, name);
      await replaceTextInEditor(page, 'field2: 1', 'field2: xxx');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, new RegExp(`${crd.kind}.${crd.group} "${name}" is invalid`));
    });

    test('【资源管理-600】 自定义资源-实例，使用不存在的namespace创修改实例失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickEditCustomResourceInList(page, name);
      await replaceTextInEditor(page, 'namespace: default', 'namespace: non-existing-ns');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, /修改失败！/);
    });

    test('【资源管理-604】 自定义资源-实例，实例详情-详情Tab页面，页面信息展示正确', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickEditCustomResourceInList(page, name);
      for (const val of await page.locator('.base_value').all()) {
        expect(await val.textContent()).not.toMatch(/^-+$/);
      }
      await cancelCreateOrEdit(page);
    });

    test('【资源管理-605】 自定义资源-实例，实例详情-详情Tab页面，操作-编辑标签功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
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

    test('【资源管理-606】 自定义资源-实例，实例详情-详情Tab页面，操作-编辑注解功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
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

    test('【资源管理-608】 自定义资源-实例，实例详情-详情Tab页面，基本信息中修改标签功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
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

    test('【资源管理-609】 自定义资源-实例，实例详情-详情Tab页面，基本信息中修改注解功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
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

    test('【资源管理-610】 自定义资源-实例，实例详情-YAML Tab页面，导出功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
      await clickYamlTab(page);
      const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 3000 }),
        clickExportYaml(page),
      ]);
      expect(download).toBeTruthy();
    });

    test('【资源管理-611】 自定义资源-实例，实例详情-YAML Tab页面，查找功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
      await clickYamlTab(page);
      await clickFindInYaml(page);
      await expect(page.getByRole('textbox').or(page.getByPlaceholder(/查找|搜索/))).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-612】 自定义资源-实例，实例详情-YAML Tab页面，复制功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
      await clickYamlTab(page);
      await clickCopyYaml(page);
      await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-601】 自定义资源-实例，实例删除功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await deleteResourceInList(page, name);
      await confirmDeleteDialog(page);
      await expect(page.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
      await gotoInstanceTab(page, crd);
      await searchResource(page, name);
      await expectResourceDeleted(page, name);
    });

    test('【资源管理-607】 自定义资源-实例，实例详情-详情Tab页面，操作-删除功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const name = uniqueResourceName(CR_PREFIX);
      await searchResource(page, name);
      await clickResourceNameInList(page, name);
      await clickDeleteInDetail(page);
      await confirmDeleteDialog(page);
      await expectSuccessMessage(page, /删除成功！/);
      await gotoInstanceTab(page, crd);
      await searchResource(page, name);
      await expectResourceDeleted(page, name);
    });
  });
});
