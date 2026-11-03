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
  encodeCrdNameForUrl,
  uniqueCrdIdentity,
  applyCrdIdentity,
  fillYamlContent,
  readStaticYaml,
  createCrdFromYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
  type CrdIdentity,
} from '@/console/utils/common'

const LIST_PATH = '/container_platform/customResourceDefinition';
const CRD_YAML = readStaticYaml('valid-crd.yaml');

const crdDetailUrl = (name: string) =>
  `${LIST_PATH}/${encodeCrdNameForUrl(name)}`;

const createCrdViaApi = async (apiContext: APIRequestContext, crd: CrdIdentity): Promise<void> => {
  await createFromYamlViaApi(apiContext, {
    yaml: applyCrdIdentity(CRD_YAML, crd),
    name: crd.name,
    plural: 'customresourcedefinitions',
    apiGroup: 'apiextensions.k8s.io',
  });
};

const deleteCrdViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await deleteResourceViaApi(apiContext, {
    name,
    plural: 'customresourcedefinitions',
    apiGroup: 'apiextensions.k8s.io',
  });
};

const cleanupCrdAliases = async (apiContext: APIRequestContext) => {
  await deleteCrdViaApi(apiContext, uniqueCrdIdentity().name);
  await deleteCrdViaApi(apiContext, uniqueCrdIdentity('copy').name);
};

test.describe('创建自定义资源定义', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await cleanupCrdAliases(request);
  });

  test('【资源管理-376】 自定义资源，取消创建自定义资源定义成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const crd = uniqueCrdIdentity();
    await clickCreate(page);
    await fillYamlContent(page, applyCrdIdentity(CRD_YAML, crd));
    await cancelCreateOrEdit(page);
    await searchResource(page, crd.name);
    await expect(page.getByText('暂无数据')).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-372】 自定义资源，输入格式有误的yaml创建自定义资源定义失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreate(page);
    await fillYamlContent(page, CRD_YAML.replace(/^\s+/gm, ''));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范|创建失败/);
  });

  test('【资源管理-373】 自定义资源，输入字段有误的yaml创建自定义资源定义失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const crd = uniqueCrdIdentity();
    await clickCreate(page);
    await fillYamlContent(page, applyCrdIdentity(CRD_YAML, crd).replace(/type: string/g, 'type: wrongtype'));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`CustomResourceDefinition.apiextensions.k8s.io "${crd.name}" is invalid`));
  });

  test('【资源管理-375】 自定义资源，创建时必填字段不存在，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const crd = uniqueCrdIdentity();
    await clickCreate(page);
    await fillYamlContent(page, applyCrdIdentity(CRD_YAML, crd).replace(/^\s*scope:\s*Namespaced\s*$/m, ''));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /创建失败！/);
  });

  test('【资源管理-371】 自定义资源，输入正确yaml创建自定义资源定义成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const crd = uniqueCrdIdentity();
    await createCrdFromYaml(page, crd);
    await searchResource(page, crd.name);
    await expect(page.getByRole('row', { name: crd.name }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-377】 自定义资源，短时间连续点击创建按钮时，查看页面提示', {
    tag: ['@v24.06', '@k8s', '@skip-temporarily'],
  }, async ({ page }) => {
    const crd = uniqueCrdIdentity();
    await clickCreate(page);
    await fillYamlContent(page, applyCrdIdentity(CRD_YAML, crd));
    await saveCreateOrEdit(page);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await expectErrorMessage(page, /customresourcedefinitions.apiextensions.k8s.io ".*" already exists/);
    await searchResource(page, crd.name);
    await expect(page.getByRole('row', { name: crd.name }).first()).toBeVisible({ timeout: 10000 });
  });
});

test.describe('查询、修改、删除自定义资源定义', () => {
  test.beforeEach(async ({ page, request }) => {
    const crd = uniqueCrdIdentity();
    await createCrdViaApi(request, crd);
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }) => {
    await cleanupCrdAliases(request);
  });

    test('【资源管理-374】 自定义资源，使用已存在的自定义资源定义名称创建，应提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await clickCreate(page);
      await fillYamlContent(page, applyCrdIdentity(CRD_YAML, crd));
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, /customresourcedefinitions.apiextensions.k8s.io ".*" already exists/);
    });

    test('【资源管理-379】 自定义资源，搜索框输入正确的自定义资源定义名称，可正确查询出自定义资源定义信息', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await expect(page.getByRole('row', { name: crd.name }).first()).toBeVisible({ timeout: 10000 });
    });

    test('【资源管理-380】 自定义资源，搜索框异常场景验证', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name.toUpperCase());
      await expect(page.getByRole('row', { name: crd.name }).first()).toBeVisible({ timeout: 10000 });
      await searchResource(page, 'a'.repeat(64));
      await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
    });

    test('【资源管理-390】 自定义资源，取消删除自定义资源定义成功', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await deleteResourceInList(page, crd.name);
      await cancelDeleteDialog(page);
      await expect(page.getByRole('row', { name: crd.name }).first()).toBeVisible({ timeout: 5000 });
    });

    test('【资源管理-389】 自定义资源，删除自定义资源定义成功', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await deleteResourceInList(page, crd.name);
      await confirmDeleteDialog(page);
      await expect(page.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
      await searchResource(page, crd.name);
      await expectResourceDeleted(page, crd.name);
    });

  test('【资源管理-392】 自定义资源，刷新按钮可正常点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickRefresh(page);
  });
    test('【资源管理-382】 自定义资源，修改自定义资源定义时输入正确yaml修改自定义资源定义成功', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await clickEditResourceInList(page, crd.name);
      await replaceTextInEditor(page, `- ${crd.shortName}`, `- ${crd.shortName}x`);
      await saveCreateOrEdit(page);
      await expectSuccessMessage(page);
    });

    test('【资源管理-383】 自定义资源，修改自定义资源定义时输入格式有误的yaml修改自定义资源定义失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await clickEditResourceInList(page, crd.name);
      await replaceTextInEditor(page, '^\\s+', '');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, /error parsing|YAML格式不规范/);
    });

    test('【资源管理-384】 自定义资源，修改自定义资源定义时输入字段有误的yaml修改自定义资源定义失败有提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await clickEditResourceInList(page, crd.name);
      await replaceTextInEditor(page, 'type: string', 'type: wrongtype');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, new RegExp(`CustomResourceDefinition.apiextensions.k8s.io "${crd.name}" is invalid`));
    });

    test('【资源管理-385】 自定义资源，修改自定义资源定义时使用已存在相同的自定义资源名称，应提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page, request }) => {
      const crd = uniqueCrdIdentity();
      const copy = uniqueCrdIdentity('copy');
      try {
        await createCrdFromYaml(page, copy);

        await searchResource(page, crd.name);
        await clickEditResourceInList(page, crd.name);
        await replaceTextInEditor(page, `name: ${crd.name}`, `name: ${copy.name}`);
        await saveCreateOrEdit(page);
        await expectErrorMessage(page, /修改失败！/);
      } finally {
        await deleteCrdViaApi(request, copy.name);
      }
    });

    test('【资源管理-386】 自定义资源，修改时必填字段不存在，应提示', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await clickEditResourceInList(page, crd.name);
      await replaceTextInEditor(page, 'scope: Namespaced', 'scope:');
      await saveCreateOrEdit(page);
      await expectErrorMessage(page, /修改失败！/);
    });

    test('【资源管理-387】 自定义资源，取消修改自定义资源定义成功', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await searchResource(page, crd.name);
      await clickEditResourceInList(page, crd.name);
      await cancelCreateOrEdit(page);
    });

    test('【资源管理-393】 自定义资源，自定义资源定义详情中的详情Tab页面信息展示正确', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await page.waitForLoadState('networkidle');
      for (const val of await page.locator('.base_value').all()) {
        expect(await val.textContent()).not.toMatch(/^-+$/);
      }
    });

    test('【资源管理-394】 自定义资源，自定义资源定义详情的操作-修改标签功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
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

    test('【资源管理-395】 自定义资源，自定义资源定义详情的操作-修改注解功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
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

    test('【资源管理-397】 自定义资源，自定义资源定义详情中的详情Tab-基本信息中修改标签功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
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

    test('【资源管理-398】 自定义资源，自定义资源定义详情中的详情Tab-基本信息中修改注解功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
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

    test('【资源管理-399】 自定义资源，自定义资源定义详情中的YAML Tab页面展示正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await clickYamlTab(page);
      expect(await page.getByText(/apiVersion|kind:\s*CustomResourceDefinition/).first().textContent()).not.toBeNull();
    });

    test('【资源管理-400】 自定义资源，自定义资源定义详情中的YAML Tab页面导出按钮正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await clickYamlTab(page);
      const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 3000 }),
        clickExportYaml(page),
      ]);
      expect(download).toBeTruthy();
    });

    test('【资源管理-401】 自定义资源，自定义资源定义详情中的YAML Tab页面查找按钮正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await clickYamlTab(page);
      await clickFindInYaml(page);
      await expect(page.getByRole('textbox').or(page.getByPlaceholder(/查找|搜索/))).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-402】 自定义资源，自定义资源定义详情中的YAML Tab页面复制按钮正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await clickYamlTab(page);
      await clickCopyYaml(page);
      await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
    });

    test('【资源管理-396】 自定义资源，自定义资源定义详情的操作-删除功能正常', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const crd = uniqueCrdIdentity();
      await page.goto(FUYAO_BASE_URL + crdDetailUrl(crd.name));
      await clickDeleteInDetail(page);
      await confirmDeleteDialog(page);
      await expectSuccessMessage(page, /删除成功！/);
      expect(page.url()).toContain(LIST_PATH);
      await searchResource(page, crd.name);
      await expectResourceDeleted(page, crd.name);
    });
});
