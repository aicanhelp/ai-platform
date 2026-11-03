import { expect, test, type Page } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';

const OVERVIEW_PATH = '/container_platform/overview';

test.describe('概览', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + OVERVIEW_PATH);
    await page.waitForLoadState('networkidle');
  });

  test('【资源管理-613】概览，各卡片数据展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page.locator('.overview_flex_node_progress').first()).toBeVisible();
    await expect(page.locator('.overview_flex_namespace_group_count').first()).toBeVisible();
    await expect(page.locator('.overview_flex_user_group_count').first()).toBeVisible();
    await expect(page.locator('.overview_flex_author_group_count').first()).toBeVisible();
    await expect(page.locator('.overview_flex_pod_progress').first()).toBeVisible();
    await expect(page.locator('.overview_flex_permission_group_alert_heigh').first()).toBeVisible();
    await expect(page.locator('.overview_flex_permission_group_alert_middle').first()).toBeVisible();
    await expect(page.locator('.overview_flex_permission_group_alert_low').first()).toBeVisible();
    await expect(page.locator('.overview_flex_application_donutchart').first()).toBeVisible();
    await expect(page.locator('.overview_flex_workload_chart').first()).toBeVisible();
    await expect(page.locator('.overview_flex_resource_title').first()).toBeVisible();
    await expect(page.locator('.overview_flex_resource_box').getByText('CPU', { exact: true }).first()).toBeVisible();
    await expect(page.locator('.overview_flex_resource_box').getByText('内存', { exact: true }).first()).toBeVisible();
    await expect(page.locator('.overview_flex_resource_box').getByText('存储', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('网络负载Top 5').first()).toBeVisible();
    await expect(page.getByText('流入流量').first()).toBeVisible();
    await expect(page.getByText('流出流量').first()).toBeVisible();
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-614】概览，各卡片点击后跳转展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await test.step('资源类卡片', async () => {
      await clickOverviewResorceAndExpect(
        page,
        () => page.getByRole('paragraph').filter({ hasText: /^节点$/ }).click(),
        /nodeManage/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.locator('.overview_flex_namespace_group_count').click(),
        /namespace\/namespaceManage/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.locator('.overview_flex_user_group_count').click(),
        /userManage\/serviceAccount/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.locator('.overview_flex_author_group_count').click(),
        /userManage\/role/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.getByRole('paragraph').filter({ hasText: /^Pod$/ }).click(),
        /workload\/pod/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.getByRole('paragraph').filter({ hasText: /^应用$/ }).click(),
        /applicationManageHelm/,
      );
      await clickOverviewResorceAndExpect(
        page,
        () => page.getByRole('paragraph').filter({ hasText: /^扩展组件$/ }).click(),
        /extendManage/,
      );
    });

    await test.step('告警卡片', async () => {
      await clickOverviewAlertAndExpect(
        page,
        () => page.locator('.overview_flex_permission_group_alert_heigh').first().click(),
        '严重',
      );
      await clickOverviewAlertAndExpect(
        page,
        () => page.locator('.overview_flex_permission_group_alert_middle').first().click(),
        '警告',
      );
      await clickOverviewAlertAndExpect(
        page,
        () => page.locator('.overview_flex_permission_group_alert_low').first().click(),
        '提示',
      );
    });

    await test.step('网络负载TOP5卡片', async () => {
      await clickOverviewNetworkAndExpect(page,'left');
      await clickOverviewNetworkAndExpect(page,'right');
    });
  });

  test('【资源管理-615】概览-资源统计卡片，切换使用率、请求率、限制率，TOP5的图随之显示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const resourceBox = page.locator('.overview_flex_resource_box');
    for (const metric of ['使用率', '请求率', '限制率']) {
      await resourceBox.getByText(metric, { exact: true }).first().click();
      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });
    }
    await resourceBox.getByText('内存', { exact: true }).first().click();
    for (const metric of ['使用率', '请求率', '限制率']) {
      await resourceBox.getByText(metric, { exact: true }).nth(1).click();
      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });
    }
    await resourceBox.getByText('存储', { exact: true }).first().click();
    for (const metric of ['磁盘空间使用率', 'inode使用率']) {
      await resourceBox.getByText(metric, { exact: true }).first().click();
      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });
    }
  });

  test('【资源管理-616】概览-资源统计卡片，切换时间筛选项，使用率、请求率、限制率，TOP5的图随之显示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.locator('.overview_flex_resource_title').getByText('近10分钟').click();
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await page.locator('.overview_flex_resource_box').getByText('使用率', { exact: true }).first().click();
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });

    await page.locator('.overview_flex_resource_title').getByText(/近\d+分钟/).click();
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await page.locator('.overview_flex_resource_box').getByText('请求率', { exact: true }).first().click();
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-617】概览-网络负载TOP5卡片，切换流入流量、流出流量功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const netBox = page.locator('.overview_flex_net_box');
    await netBox.getByText('流入流量', { exact: true }).click();
    await expect(page.locator('.overview_flex_net_left_nodeListItem').first()).toBeVisible({ timeout: 10000 });
    await netBox.getByText('流出流量', { exact: true }).click();
    await expect(page.locator('.overview_flex_net_right_podListItem').first()).toBeVisible({ timeout: 10000 });
  });
});

// /workload/pod/cluster-system/capi-controller-manager-dbb4488df-pn79f

async function clickOverviewResorceAndExpect(
  page: Page,
  clickFn: () => Promise<void>,
  urlPattern: RegExp,
): Promise<void> {
  await page.goto(FUYAO_BASE_URL + OVERVIEW_PATH);
  await page.waitForLoadState('networkidle');
  await clickFn();
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(urlPattern);
}

async function clickOverviewAlertAndExpect(
  page: Page,
  clickFn: () => Promise<void>,
  expectLevel: string,
): Promise<void> {
  await page.goto(FUYAO_BASE_URL + OVERVIEW_PATH);
  await page.waitForLoadState('networkidle');
  await clickFn();
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/alert\/alertIndex/);
  await expect(page.getByText(`告警等级${expectLevel}`)).toBeVisible();
}

async function clickOverviewNetworkAndExpect(
  page: Page,
  direction: 'left' | 'right',
): Promise<void> {
  await page.goto(FUYAO_BASE_URL + OVERVIEW_PATH);
  await page.waitForLoadState('networkidle');
  if (direction === 'left') {
    const targetLink = page.locator('.overview_flex_net_left_nodeListItem').getByRole('link').first();
    const targetName = await targetLink.textContent();
    await targetLink.click();
    await page.waitForLoadState('networkidle');
    await expect(page).toHaveURL(new RegExp(`\/nodeManage\/${targetName}`));

  } else {
    const targetLink = page.locator('.overview_flex_net_right_podListItem_ip').getByRole('link').first();
    const targetName = await targetLink.textContent();
    await targetLink.click();
    await page.waitForLoadState('networkidle');
    await expect(page).toHaveURL(new RegExp(`\/workload\/pod\/[\\w-]+\/${targetName}`));
  }
}