import { test, expect, Page } from '@playwright/test';
import {
  visitColocationOverviewPage,
  getSidemenuLinkOverview,
  getSidemenuLinkConfiguration,
  getSidemenuLinkMonitor,
  doClickInfo1,
  getDrawerTitle1,
  doClickInfo3,
  getDrawerTitle3,
  doCloseDrawer,
  doClickLink2,
  doClickLink3,
  doClickLink4,
} from './utils/common';
import { COLOCATION_CONFIG_PAGE_URI, COLOCATION_MONITOR_PAGE_URI, COLOCATION_OVERVIEW_PAGE_URI, WORKLOAD_DEPLOYMENT_PAGE_URI } from './constants';

/**
 * @description Test colocation overview page sidebar links
 */
test.describe('混部概览页面 - 侧边栏', () => {
  test.beforeEach(async ({ page }: { page: Page }) => {
    await visitColocationOverviewPage(page);
  });

  /**
   * Test 35: Verify sidebar links exist and work correctly
   */
  test('【混部-035】侧边栏混部相关链接正确性', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await expect(getSidemenuLinkOverview(page)).toBeVisible();
    await expect(getSidemenuLinkConfiguration(page)).toBeVisible();
    await expect(getSidemenuLinkMonitor(page)).toBeVisible();

    await getSidemenuLinkConfiguration(page).click();
    await expect(page.url()).toContain(COLOCATION_CONFIG_PAGE_URI);

    await getSidemenuLinkMonitor(page).click();
    await expect(page.url()).toContain(COLOCATION_MONITOR_PAGE_URI);

    await getSidemenuLinkOverview(page).click();
    await expect(page.url()).toContain(COLOCATION_OVERVIEW_PAGE_URI);
  });
});

/**
 * @description Test colocation overview page main functionality
 */
test.describe('混部概览页面 - 主要功能', () => {

  test.beforeEach(async ({ page }: { page: Page }) => {
    await visitColocationOverviewPage(page);
  });

  /**
   * Test 37: Open and close drawer by clicking question marks - part 1 (环境准备)
   */
  test('【混部-037】点击问号图标打开/关闭抽屉 - 环境准备', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doClickInfo1(page);
    await expect(getDrawerTitle1(page)).toBeVisible();
    await doCloseDrawer(page);
    await expect(getDrawerTitle1(page)).not.toBeVisible();
  });

  /**
   * Test 38: Open and close drawer by clicking question marks - part 3 (工作负载部署)
   */
  test('【混部-038】点击问号图标打开/关闭抽屉 - 工作负载部署', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doClickInfo3(page);
    await expect(getDrawerTitle3(page)).toBeVisible();
    await doCloseDrawer(page);
    await expect(getDrawerTitle3(page)).not.toBeVisible();
  });

  /**
   * Test 39: Link to colocation configuration page
   */
  test('【混部-039】链接跳转到混部策略配置页面', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doClickLink2(page);
    await page.waitForLoadState('networkidle');
    await expect(page.url()).toContain(COLOCATION_CONFIG_PAGE_URI);
  });

  /**
   * Test 40: Link to workload deployment page
   */
  test('【混部-040】链接跳转到工作负载部署页面', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doClickLink3(page);
    await page.waitForLoadState('networkidle');
    await expect(page.url()).toContain(WORKLOAD_DEPLOYMENT_PAGE_URI);
  });

  /**
   * Test 41: Link to colocation monitor page
   */
  test('【混部-041】链接跳转到混部监控页面', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doClickLink4(page);
    await page.waitForLoadState('networkidle');
    await expect(page.url()).toContain(COLOCATION_MONITOR_PAGE_URI);
  });
});
