import { expect, test, type APIRequestContext } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
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
  clickCreateType,
  uniqueResourceName,
  readStaticYaml,
  fillYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
} from '@/console/utils/common'

const LIST_PATH = '/container_platform/userManage/roleBinding';
const ROLEBINDING_YAML = readStaticYaml('valid-rolebinding.yaml');
const NAME_PREFIX = 'test-rolebinding';
const COPY_PREFIX = 'test-rolebinding-copy';
const NS = 'default';

const createRoleBindingViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await createFromYamlViaApi(apiContext, {
    yaml: ROLEBINDING_YAML,
    name,
    plural: 'rolebindings',
    namespace: NS,
    apiGroup: 'rbac.authorization.k8s.io',
  });
};

const deleteRoleBindingViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await deleteResourceViaApi(apiContext, {
    name,
    plural: 'rolebindings',
    namespace: NS,
    apiGroup: 'rbac.authorization.k8s.io',
  });
};

test.describe('创建角色绑定', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await deleteRoleBindingViaApi(request, uniqueResourceName(NAME_PREFIX));
  });

  test('【资源管理-536】 RBAC管理-角色绑定，取消创建角色绑定成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name);
    await cancelCreateOrEdit(page);
    await searchResource(page, name);
    await expect(page.getByText('暂无数据')).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-532】 RBAC管理-角色绑定，输入格式有误的yaml创建角色绑定失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, uniqueResourceName(NAME_PREFIX), (yaml) => yaml.replace(/^\s+/gm, ''));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范|创建失败/);
  });

  test('【资源管理-533】 RBAC管理-角色绑定，输入字段有误的yaml创建角色绑定失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name, (content) => (
      content.replace(/- kind:/, '- kindddd:')
    ));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`RoleBinding\\.rbac\\.authorization\\.k8s\\.io "${name}" is invalid`));
  });

  test('【资源管理-535】 RBAC管理-角色绑定，使用不存在的namespace创建，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name, (content) => (
      content.replace(/namespace: default/g, 'namespace: non-existing-ns')
    ));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, 'namespaces "non-existing-ns" not found');
  });

  test('【资源管理-531】 RBAC管理-角色绑定，输入正确yaml创建角色绑定成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-537】 RBAC管理-角色绑定，短时间连续点击创建按钮时，查看页面提示', {
    tag: ['@v24.06', '@k8s', '@skip-temporarily'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name);
    await saveCreateOrEdit(page);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await expectErrorMessage(page, /rolebindings\.rbac\.authorization\.k8s\.io ".*" already exists/);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('查询、修改、删除角色绑定', () => {
  test.beforeEach(async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await createRoleBindingViaApi(request, name);
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await deleteRoleBindingViaApi(request, uniqueResourceName(NAME_PREFIX));
  });

  test('【资源管理-534】 RBAC管理-角色绑定，同一个命名空间下已存在相同的角色绑定名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreateType(page, '角色绑定');
    await fillYaml(page, ROLEBINDING_YAML, name);
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /rolebindings\.rbac\.authorization\.k8s\.io ".*" already exists/);
  });

  test('【资源管理-539】 RBAC管理-角色绑定，搜索框输入正确的角色绑定名称，可正确查询出角色绑定信息', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-540】 RBAC管理-角色绑定，搜索框异常场景验证', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name.toUpperCase());
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 10000 });
    await searchResource(page, 'a'.repeat(64));
    await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-550】 RBAC管理-角色绑定，取消删除角色绑定成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await deleteResourceInList(page, name);
    await cancelDeleteDialog(page);
    await expect(page.getByRole('row', { name }).first()).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-549】 RBAC管理-角色绑定，删除角色绑定成功', {
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

  test('【资源管理-552】 RBAC管理-角色绑定，刷新按钮可正常点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickRefresh(page);
  });

  test('【资源管理-542】 RBAC管理-角色绑定，修改角色绑定时输入正确yaml修改角色绑定成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'example.com/created-by: e2e', 'example.com/created-by: e2e-test');
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
  });

  test('【资源管理-543】 RBAC管理-角色绑定，修改角色绑定时输入格式有误的yaml修改角色绑定失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, '^\\s+', '');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范/);
  });

  test('【资源管理-544】 RBAC管理-角色绑定，修改角色绑定时输入字段有误的yaml修改角色绑定失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, '- kind:', '- kindddd:');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`RoleBinding\\.rbac\\.authorization\\.k8s\\.io "${name}" is invalid`));
  });

  test('【资源管理-545】 RBAC管理-角色绑定，修改角色绑定时同一个命名空间下已存在相同的角色绑定名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    const copyName = uniqueResourceName(COPY_PREFIX);
    try {
      await clickCreateType(page, '角色绑定');
      await fillYaml(page, ROLEBINDING_YAML, copyName);
      await saveCreateOrEdit(page);
      await expectSuccessMessage(page);

      await searchResource(page, name);
      await clickEditResourceInList(page, name);
      await replaceTextInEditor(page, `name: ${name}`, `name: ${copyName}`);
      await saveCreateOrEdit(page);
      await expectErrorMessage(page);
    } finally {
      await deleteRoleBindingViaApi(request, copyName);
    }
  });

  test('【资源管理-546】 RBAC管理-角色绑定，修改时使用不存在的namespace，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'namespace: default', 'namespace: non-existing-ns');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /修改失败/);
  });

  test('【资源管理-547】 RBAC管理-角色绑定，取消修改角色绑定成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await cancelCreateOrEdit(page);
  });

  test('【资源管理-553】 RBAC管理-角色绑定，角色绑定详情中的详情Tab页面信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await page.waitForLoadState('networkidle');
    for (const val of await page.locator('.base_value').all()) {
      expect(await val.textContent()).not.toMatch(/^-?$/);
    }
  });

  test('【资源管理-554】 RBAC管理-角色绑定，角色绑定详情的操作-修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
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

  test('【资源管理-555】 RBAC管理-角色绑定，角色绑定详情的操作-修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
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

  test('【资源管理-557】 RBAC管理-角色绑定，角色绑定详情中的详情Tab-基本信息中修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
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

  test('【资源管理-558】 RBAC管理-角色绑定，角色绑定详情中的详情Tab-基本信息中修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
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

  test('【资源管理-559】 RBAC管理-角色绑定，角色绑定详情中的YAML Tab页面展示正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await clickYamlTab(page);
    expect(await page.getByText(/apiVersion|kind:\s*RoleBinding/).first().textContent()).not.toBeNull();
  });

  test('【资源管理-560】 RBAC管理-角色绑定，角色绑定详情中的YAML Tab页面导出按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await clickYamlTab(page);
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 3000 }),
      clickExportYaml(page),
    ]);
    expect(download).toBeTruthy();
  });

  test('【资源管理-561】 RBAC管理-角色绑定，角色绑定详情中的YAML Tab页面查找按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await clickYamlTab(page);
    await clickFindInYaml(page);
    await expect(page.getByRole('textbox').or(page.getByPlaceholder(/查找|搜索/))).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-562】 RBAC管理-角色绑定，角色绑定详情中的YAML Tab页面复制按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await clickYamlTab(page);
    await clickCopyYaml(page);
    await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-556】 RBAC管理-角色绑定，角色绑定详情的操作-删除功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/userManage/roleBinding/${NS}/${name}`);
    await clickDeleteInDetail(page);
    await confirmDeleteDialog(page);
    await expectSuccessMessage(page, /删除成功！/);
    expect(page.url()).toContain('/container_platform/userManage/roleBinding');
    await searchResource(page, name);
    await expectResourceDeleted(page, name);
  });
});
