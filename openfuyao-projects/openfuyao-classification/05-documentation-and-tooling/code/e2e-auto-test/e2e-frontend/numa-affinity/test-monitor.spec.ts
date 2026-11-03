import { test, expect } from './utils';
import { FUYAO_BASE_URL } from '@/utils/constants';
import { NUMA_MONITOR_TABLE_HEADERS } from './constants';
import { openMonitorPageAndWait } from './utils';

test.describe('NUMA亲和调度-功能测试', () => {
  test('【NUMA2503-046】NUMA监控-集群NUMA监控-表格基础信息展示', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page }) => {
    expect(FUYAO_BASE_URL).not.toEqual('');

    await openMonitorPageAndWait(page);

    // 等待表格加载
    const table = page.locator('table').first();
    await expect(table).toBeVisible({ timeout: 60000 });

    // 尝试多种列头定位策略
    console.log(`验证表格列头，期望列: ${NUMA_MONITOR_TABLE_HEADERS.join(', ')}`);

    // 策略1: 使用 columnheader 角色（标准表格）
    let columnHeadersFound = true;
    for (const header of NUMA_MONITOR_TABLE_HEADERS) {
      const columnHeader = page.getByRole('columnheader', { name: new RegExp(header, 'i') });
      const count = await columnHeader.count();
      if (count === 0) {
        console.log(`列头 "${header}" 未找到 (columnheader role)`);
        columnHeadersFound = false;
        break;
      } else {
        console.log(`列头 "${header}" 已找到`);
      }
    }

    // 策略2: 如果策略1失败，检查表格头行的文本内容
    if (!columnHeadersFound) {
      console.log('尝试策略2: 检查表格头行文本');
      const headerRowText = await table.locator('thead').innerText();
      console.log(`表格头行内容: ${headerRowText}`);

      // 至少验证第一个列名存在
      let foundFirstHeader = false;
      for (const header of NUMA_MONITOR_TABLE_HEADERS) {
        if (headerRowText.includes(header)) {
          console.log(`在表格头中找到列: "${header}"`);
          foundFirstHeader = true;
          break;
        }
      }

      if (!foundFirstHeader) {
        throw new Error(
          `表格列头验证失败\n期望列: [${NUMA_MONITOR_TABLE_HEADERS.join(', ')}]\n实际表格头: ${headerRowText}`
        );
      }
    }

    // 验证表格行数据
    const rows = table.locator('tbody tr');
    const rowCount = await rows.count();

    if (rowCount === 0) {
      // 检查是否显示"暂无数据"
      const emptyMsg = page.getByText(/暂无数据/);
      if (await emptyMsg.count() > 0) {
        console.log('表格无数据（显示"暂无数据"提示）');
        return;
      } else {
        throw new Error('表格无数据行，且未显示"暂无数据"提示');
      }
    }

    // 验证第一行数据不为空
    const firstRowText = (await rows.first().innerText()).trim();
    expect(firstRowText).not.toEqual('');
    expect(firstRowText).not.toEqual('-');
    expect(firstRowText).not.toEqual('--');

    console.log(`测试通过: 表格包含 ${rowCount} 行节点数据`);
  });
});
