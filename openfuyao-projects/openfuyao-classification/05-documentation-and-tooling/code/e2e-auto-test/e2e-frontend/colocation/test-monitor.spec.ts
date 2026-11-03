import { test, expect, Page, APIRequestContext } from '@playwright/test';
import {
  visitColocationMonitorPage,
  doSwitchToClusterMonitor,
  doSwitchToNodeMonitor,
  getMonitorParagraphs,
  waitForCheck,
} from './utils/common';

/**
 * @description Test colocation monitor page chart loading
 * ECharts are rendered as <canvas> elements, so we can't verify data correctness via DOM
 */
test.describe('混部监控页面', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage, request }: { page: Page; request: APIRequestContext }) => {
    page = testPage;
    await visitColocationMonitorPage(page);
  });

  /**
   * Test 111, 113, 115, 117, 119, 121, 123, 125: Cluster data display
   * 需要混部节点已开启和BE资源已部署才能看到完整的集群监控数据
   */
  test('【混部-111/113/115/117/119/121/123/125】集群级混部监控数据展示', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page, request }: { page: Page; request: APIRequestContext }) => {
    // 前置环境检查：确保至少一个混部节点已开启状态下的监控数据显示

    await doSwitchToClusterMonitor(page);

    // 等待图表完全加载，包括数据初始化延迟
    await waitForCheck('cluster monitor charts should stabilize', async () => {
      const paragraphs = await getMonitorParagraphs(page);
      if (paragraphs.length === 0) {
        throw new Error('monitor chart titles not loaded yet');
      }
    }, 60000, 2000);

    const paragraphs = await getMonitorParagraphs(page);

    // 获取图表标题和对应的"暂无内容"标记
    const noContentCharts: string[] = [];
    for (let i = 0; i < paragraphs.length; i++) {
      const text = await paragraphs[i].textContent();

      // 检查下一个元素是否为"暂无内容"
      if (i + 1 < paragraphs.length) {
        const nextText = await paragraphs[i + 1].textContent();
        if (nextText === '暂无内容') {
          noContentCharts.push(text || '未知');
        }
      }
    }

    const allowedNoDataCharts = ['集群超卖CPU申请量', '集群超卖内存申请量'];
    const unexpectedNoDataCharts = noContentCharts.filter(
      (chart) => !allowedNoDataCharts.some((allowed) => chart.includes(allowed))
    );

    if (unexpectedNoDataCharts.length > 0) {
      expect(unexpectedNoDataCharts, `以下图表没有预期的数据: ${unexpectedNoDataCharts.join(', ')}`).toHaveLength(0);
    }
  });

  /**
   * Test 128, 130, 132, 134, 136, 138: Node data display
   */
  test('【混部-128/130/132/134/136/138】节点级混部监控数据展示', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doSwitchToNodeMonitor(page);

    // 等待图表完全加载
    await waitForCheck('node monitor charts should stabilize', async () => {
      const paragraphs = await getMonitorParagraphs(page);
      if (paragraphs.length === 0) {
        throw new Error('monitor chart titles not loaded yet');
      }
    }, 60000, 2000);

    const paragraphs = await getMonitorParagraphs(page);

    const noContentCharts: string[] = [];
    for (let i = 0; i < paragraphs.length; i++) {
      const text = await paragraphs[i].textContent();

      // 检查下一个元素是否为"暂无内容"
      if (i + 1 < paragraphs.length) {
        const nextText = await paragraphs[i + 1].textContent();
        if (nextText === '暂无内容') {
          noContentCharts.push(text || '未知');
        }
      }
    }

    expect(noContentCharts, `以下图表没有数据: ${noContentCharts.join(', ')}`).toHaveLength(0);
  });
});
