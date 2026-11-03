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
  clickMonitorDashboard,
  clickEventTab,
  eventFilterQuery,
  eventFilterReset,
  expectResourceStatus,
  uniqueResourceName,
  readStaticYaml,
  fillYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
} from '@/console/utils/common'
import { dumpWorkloadOnFailure } from '@/console/utils/workload-debug';

const LIST_PATH = '/container_platform/workload/daemonSet';
const DAEMONSET_YAML = readStaticYaml('valid-daemonset.yaml');
const NAME_PREFIX = 'test-daemonset';
const COPY_PREFIX = 'test-daemonset-copy';
const NS = 'default';

const createDaemonSetViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await createFromYamlViaApi(apiContext, {
    yaml: DAEMONSET_YAML,
    name,
    plural: 'daemonsets',
    namespace: NS,
    apiGroup: 'apps',
  });
};

const deleteDaemonSetViaApi = async (apiContext: APIRequestContext, name: string): Promise<void> => {
  await deleteResourceViaApi(apiContext, {
    name,
    plural: 'daemonsets',
    namespace: NS,
    apiGroup: 'apps',
  });
};

test.describe('创建daemonset', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
  });

  test.afterEach(async ({ request }, testInfo) => {
    await deleteDaemonSetViaApi(request, uniqueResourceName(NAME_PREFIX));
    await dumpWorkloadOnFailure(request, 'daemonset', 'default', uniqueResourceName(NAME_PREFIX), testInfo);
  });

  test('【资源管理-108】 工作负载-Daemonset，取消创建Daemonset成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name);
    await cancelCreateOrEdit(page);
    await searchResource(page, name);
    await expect(page.getByText('暂无数据')).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-104】 工作负载-Daemonset，输入格式有误的yaml创建Daemonset失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, uniqueResourceName(NAME_PREFIX), (yaml) => yaml.replace(/^\s+/gm, ''));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范|创建失败/);
  });

  test('【资源管理-105】 工作负载-Daemonset，输入字段有误的yaml创建Daemonset失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name, (content) => (
      content.replace(/containers/g, 'containerssssssss')
    ));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`DaemonSet\\.apps "${name}" is invalid`));
  });

  test('【资源管理-107】 工作负载-Daemonset，使用不存在的namespace创建，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name, (content) => (
      content.replace(/namespace: default/g, 'namespace: non-existing-ns')
    ));
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, 'namespaces "non-existing-ns" not found');
  });

  test('【资源管理-103】 工作负载-Daemonset，输入正确yaml创建Daemonset成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await searchResource(page, name);
    await expectResourceStatus(page, name, 'Active');
  });

  test('【资源管理-109】 工作负载-Daemonset，短时间连续点击创建按钮时，查看页面提示', {
    tag: ['@v24.06', '@k8s', '@skip-temporarily'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name);
    await saveCreateOrEdit(page);
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
    await expectErrorMessage(page, /daemonsets.apps ".*" already exists/);
    await searchResource(page, name);
    await expectResourceStatus(page, name, 'Active');
  });
});

test.describe('查询、修改、删除daemonset', () => {
  test.beforeEach(async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await createDaemonSetViaApi(request, name);
    await page.goto(FUYAO_BASE_URL + LIST_PATH);
    await expectResourceStatus(page, name, 'Active');
  });

  test.afterEach(async ({ request }, testInfo) => {
    await deleteDaemonSetViaApi(request, uniqueResourceName(NAME_PREFIX));
    await dumpWorkloadOnFailure(request, 'daemonset', 'default', uniqueResourceName(NAME_PREFIX), testInfo);
  });

  test('【资源管理-106】 工作负载-Daemonset，同一个命名空间下已存在相同的Daemonset名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await clickCreate(page);
    await fillYaml(page, DAEMONSET_YAML, name);
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /daemonsets.apps ".*" already exists/);
  });

  test('【资源管理-111】 工作负载-Daemonset，搜索框输入正确的Daemonset名称，可正确查询出Daemonset信息', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await expectResourceStatus(page, name, 'Active');
  });

  test('【资源管理-112】 工作负载-Daemonset，搜索框异常场景验证', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name.toUpperCase());
    await expectResourceStatus(page, name, 'Active');
    await searchResource(page, 'a'.repeat(64));
    await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-122】 工作负载-Daemonset，取消删除Daemonset成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await deleteResourceInList(page, name);
    await cancelDeleteDialog(page);
    await expectResourceStatus(page, name, 'Active');
  });

  test('【资源管理-121】 工作负载-Daemonset，删除Daemonset成功', {
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

  test('【资源管理-124】 工作负载-Daemonset，刷新按钮可正常点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickRefresh(page);
  });

  test('【资源管理-114】 工作负载-Daemonset，修改Daemonset时输入正确yaml修改Daemonset成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'example.com/created-by: e2e', 'example.com/created-by: e2e-test');
    await saveCreateOrEdit(page);
    await expectSuccessMessage(page);
  });

  test('【资源管理-115】 工作负载-Daemonset，修改Daemonset时输入格式有误的yaml修改Daemonset失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, '^\\s+', '');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /YAML格式不规范/);
  });

  test('【资源管理-116】 工作负载-Daemonset，修改Daemonset时输入字段有误的yaml修改Daemonset失败有提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'containers', 'containerssssssss');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, new RegExp(`DaemonSet\\.apps "${name}" is invalid`));
  });

  test('【资源管理-117】 工作负载-Daemonset，修改Daemonset时同一个命名空间下已存在相同的Daemonset名称，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page, request }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    const copyName = uniqueResourceName(COPY_PREFIX);
    try {
      await clickCreate(page);
      await fillYaml(page, DAEMONSET_YAML, copyName);
      await saveCreateOrEdit(page);
      await expectSuccessMessage(page);

      await searchResource(page, name);
      await clickEditResourceInList(page, name);
      await replaceTextInEditor(page, `name: ${name}`, `name: ${copyName}`);
      await saveCreateOrEdit(page);
      await expectErrorMessage(page);
    } finally {
      await deleteDaemonSetViaApi(request, copyName);
    }
  });

  test('【资源管理-118】 工作负载-Daemonset，修改时使用不存在的namespace，应提示', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await replaceTextInEditor(page, 'namespace: default', 'namespace: non-existing-ns');
    await saveCreateOrEdit(page);
    await expectErrorMessage(page, /修改失败/);
  });

  test('【资源管理-119】 工作负载-Daemonset，取消修改Daemonset成功', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await searchResource(page, name);
    await clickEditResourceInList(page, name);
    await cancelCreateOrEdit(page);
  });

  test('【资源管理-125】 工作负载-Daemonset，Daemonset详情中的详情Tab页面信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await page.waitForLoadState('networkidle');
    for (const val of await page.locator('.base_value').all()) {
      expect(await val.textContent()).not.toMatch(/^-?$/);
    }
  });

  test('【资源管理-126】 工作负载-Daemonset，Daemonset详情中的详情Tab跳转到监控大屏功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickMonitorDashboard(page);
    await page.waitForLoadState('networkidle');
    expect(page.url()).toContain('/container_platform/monitor/monitorDashboard');
  });

  test('【资源管理-127】 工作负载-Daemonset，Daemonset详情的操作-修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
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

  test('【资源管理-128】 工作负载-Daemonset，Daemonset详情的操作-修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
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

  test('【资源管理-130】 工作负载-Daemonset，Daemonset详情中的详情Tab-基本信息中修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
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

  test('【资源管理-131】 工作负载-Daemonset，Daemonset详情中的详情Tab-基本信息中修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
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

  test('【资源管理-132】 工作负载-Daemonset，Daemonset详情中的YAML Tab页面展示正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickYamlTab(page);
    expect(await page.getByText(/apiVersion|kind:\s*DaemonSet/).first().textContent()).not.toBeNull();
  });

  test('【资源管理-133】 工作负载-Daemonset，Daemonset详情中的YAML Tab页面导出按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickYamlTab(page);
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 3000 }),
      clickExportYaml(page),
    ]);
    expect(download).toBeTruthy();
  });

  test('【资源管理-134】 工作负载-Daemonset，Daemonset详情中的YAML Tab页面查找按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickYamlTab(page);
    await clickFindInYaml(page);
    await expect(page.getByRole('textbox').or(page.getByPlaceholder(/查找|搜索/))).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-135】 工作负载-Daemonset，Daemonset详情中的YAML Tab页面复制按钮正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickYamlTab(page);
    await clickCopyYaml(page);
    await expect(page.getByText(/复制成功/)).toBeVisible({ timeout: 3000 });
  });

  test('【资源管理-136】 工作负载-Daemonset，Daemonset详情中的事件Tab页面事件列表展示正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickEventTab(page);
    await page.waitForTimeout(1000);
    await expect(page.locator('.event_timeLine li').first().or(page.getByText('暂无内容'))).toBeVisible();
  });

  test('【资源管理-137】 工作负载-Daemonset，Daemonset详情中的事件Tab页面事件列表按筛选条件查询正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickEventTab(page);
    await eventFilterQuery(page, '正常');
    await expect(page.locator('.event_timeLine li').first()).toBeVisible();
  });

  test('【资源管理-138】 工作负载-Daemonset，Daemonset详情中的事件Tab页面事件列表重置功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickEventTab(page);
    await eventFilterQuery(page, '警告');
    await expect(page.locator('.event_timeLine li').first()).not.toBeVisible();
    await eventFilterReset(page);
    await expect(page.locator('.event_timeLine li').first()).toBeVisible();
  });

  test('【资源管理-129】 工作负载-Daemonset，Daemonset详情的操作-删除功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const name = uniqueResourceName(NAME_PREFIX);
    await page.goto(FUYAO_BASE_URL + `/container_platform/workload/daemonSet/${NS}/${name}`);
    await clickDeleteInDetail(page);
    await confirmDeleteDialog(page);
    await expectSuccessMessage(page, /删除成功！/);
    expect(page.url()).toContain('/container_platform/workload/daemonSet');
    await searchResource(page, name);
    await expectResourceDeleted(page, name);
  });
});
