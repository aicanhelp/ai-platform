// Batch i18n refactoring script for colocation-website
// This script processes all remaining files with hardcoded Chinese strings
import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = join(__dirname, '..');

// Workload page configurations
const workloadPages = [
  { file: 'pages/workload/StatefulSetPage.jsx', type: 'StatefulSet', hasCloseIcon: true, hasIpSort: false, hasOversold: true },
  { file: 'pages/workload/CronJobPage.jsx', type: 'CronJob', hasCloseIcon: true, hasIpSort: false, hasOversold: true },
  { file: 'pages/workload/JobPage.jsx', type: 'Job', hasCloseIcon: true, hasIpSort: false, hasOversold: true },
  { file: 'pages/workload/DaemonSetPage.jsx', type: 'DaemonSet', hasCloseIcon: false, hasIpSort: false, hasOversold: false },
];

function processWorkloadPage(content, config) {
  const { type, hasCloseIcon, hasOversold } = config;
  let result = content;

  // 1. Add useIntl import (after openinula import)
  if (!result.includes("import { useIntl } from 'inula-intl'")) {
    result = result.replace(
      "import { useEffect, useState, useCallback } from 'openinula';",
      "import { useEffect, useState, useCallback } from 'openinula';\nimport { useIntl } from 'inula-intl';"
    );
    // For files without useCallback
    if (!result.includes("import { useIntl } from 'inula-intl'")) {
      result = result.replace(
        "import { useEffect, useState } from 'openinula';",
        "import { useEffect, useState } from 'openinula';\nimport { useIntl } from 'inula-intl';"
      );
    }
  }

  // 2. Add intl hook and translateState helper inside component
  const funcRegex = new RegExp(`export default function \\w+Page\\(\\) \\{\\n`);
  const funcMatch = result.match(funcRegex);
  if (funcMatch && !result.includes('const intl = useIntl()')) {
    result = result.replace(
      funcRegex,
      `export default function ${funcMatch[0].match(/function (\w+)/)[1]}() {\n  const intl = useIntl();\n  const translateState = (s) => s === '在线' ? intl.formatMessage({ id: 'workload.status.online' }) : s === '离线' ? intl.formatMessage({ id: 'workload.status.offline' }) : s === '超卖' ? intl.formatMessage({ id: 'workload.status.oversold' }) : s;\n`
    );
  }

  // 3. Replace filter status options - translate text, keep values
  result = result.replace(
    /\{ text: '在线', value: '在线' \}/g,
    "{ text: intl.formatMessage({ id: 'workload.status.online' }), value: '在线' }"
  );
  result = result.replace(
    /\{ text: '离线', value: '离线' \}/g,
    "{ text: intl.formatMessage({ id: 'workload.status.offline' }), value: '离线' }"
  );
  result = result.replace(
    /\{ text: '超卖', value: '超卖' \}/g,
    "{ text: intl.formatMessage({ id: 'workload.status.oversold' }), value: '超卖' }"
  );

  // DaemonSet statusOption with label
  result = result.replace(
    /value: '在线',\s*\n\s*label: '在线'/g,
    "value: '在线',\n      label: intl.formatMessage({ id: 'workload.status.online' })"
  );
  result = result.replace(
    /value: '离线',\s*\n\s*label: '离线'/g,
    "value: '离线',\n      label: intl.formatMessage({ id: 'workload.status.offline' })"
  );

  // 4. Replace useState('在线') for choseState
  result = result.replace(
    /const \[choseState, setChoseState\] = useState\('在线'\)/g,
    "const [choseState, setChoseState] = useState(intl.formatMessage({ id: 'workload.status.online' }))"
  );

  // 5. Replace setChoseState('离线') and setChoseState('超卖')
  result = result.replace(
    /setChoseState\('离线'\)/g,
    "setChoseState(intl.formatMessage({ id: 'workload.status.offline' }))"
  );
  result = result.replace(
    /setChoseState\('超卖'\)/g,
    "setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }))"
  );

  // 6. Replace type === '离线' comparison
  result = result.replace(
    /if \(type === '离线'\)/g,
    "if (type === intl.formatMessage({ id: 'workload.status.offline' }))"
  );

  // 7. Replace all message strings
  result = result.replace(/messageApi\.success\('非混部设置成功'\)/g, "messageApi.success(intl.formatMessage({ id: 'message.nonColocationSetSuccess' }))");
  result = result.replace(/messageApi\.error\('非混部设置失败'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.nonColocationSetFailed' }))");
  result = result.replace(/messageApi\.error\('接口异常'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.apiError' }))");
  result = result.replace(/messageApi\.success\('修改成功'\)/g, "messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }))");
  result = result.replace(/messageApi\.error\('修改失败'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }))");
  result = result.replace(/messageApi\.error\('至少需要选择1个工作负载'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }))");
  result = result.replace(/messageApi\.error\('一次最多只能添加10个工作负载'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }))");

  // Type-specific success/failure messages
  const typeLower = type.toLowerCase();
  result = result.replace(
    new RegExp(`messageApi\\.success\\('混部${typeLower}添加成功'\\)`, 'gi'),
    `messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: '${type}' }))`
  );
  result = result.replace(
    new RegExp(`messageApi\\.error\\('混部${typeLower}添加失败'\\)`, 'gi'),
    `messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: '${type}' }))`
  );

  // 8. Replace column titles
  result = result.replace(/title: '负载名称'/g, "title: intl.formatMessage({ id: 'table.column.workloadName' })");
  result = result.replace(/title: 'Pod名称'/g, "title: intl.formatMessage({ id: 'table.column.podName' })");
  result = result.replace(/title: '命名空间'/g, "title: intl.formatMessage({ id: 'table.column.namespace' })");
  result = result.replace(/title: '所属混部节点'/g, "title: intl.formatMessage({ id: 'table.column.colocationNode' })");
  result = result.replace(/title: 'Pod IP地址'/g, "title: intl.formatMessage({ id: 'table.column.podIP' })");
  result = result.replace(/title: '业务属性'/g, "title: intl.formatMessage({ id: 'table.column.businessAttr' })");
  result = result.replace(/title: '混部属性'/g, "title: intl.formatMessage({ id: 'table.column.businessAttr' })");
  result = result.replace(/title: '期望业务属性'/g, "title: intl.formatMessage({ id: 'table.column.expectedBusinessAttr' })");
  result = result.replace(/title: '创建时间'/g, "title: intl.formatMessage({ id: 'table.column.createTime' })");
  result = result.replace(/title: '操作'/g, "title: intl.formatMessage({ id: 'common.action' })");

  // 9. Replace button text
  result = result.replace(/\{`修改为离线`\}/g, "{intl.formatMessage({ id: 'action.changeToOffline' })}");
  result = result.replace(/ 修改为离线/g, " {intl.formatMessage({ id: 'action.changeToOffline' })}");
  result = result.replace(/ 设置为非混部/g, " {intl.formatMessage({ id: 'action.setNonColocation' })}");

  // 10. Replace tooltip
  result = result.replace(
    /title='请前往其他更高级的控制器进行管理'/g,
    "title={intl.formatMessage({ id: 'message.higherControllerTip' })}"
  );

  // 11. Replace QoS tip
  result = result.replace(
    /在线属性的工作负载比离线属性会有更高的QoS等级，超卖属性的工作负载会使用超卖资源。/g,
    "{intl.formatMessage({ id: 'message.qosTip' })}"
  );

  // 12. Replace search placeholders
  result = result.replace(/placeholder="搜索Pod名称"/g, "placeholder={intl.formatMessage({ id: 'search.podName' })}");
  result = result.replace(/placeholder="搜索负载名称"/g, "placeholder={intl.formatMessage({ id: 'search.workloadName' })}");
  result = result.replace(/placeholder="按标签搜索  例：key：value"/g, "placeholder={intl.formatMessage({ id: 'search.byLabel' })}");

  // 13. Replace "添加混部" button
  result = result.replace(
    new RegExp(`添加混部${type}`, 'g'),
    `{intl.formatMessage({ id: 'button.addColocation' }, { type: '${type}' })}`
  );

  // 14. Replace pagination showTotal
  result = result.replace(/showTotal: \(total\) => `共\$\{total\}条`/g, "showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total })");

  // 15. Replace modal titles and content
  result = result.replace(/title=\{`修改为\$\{choseState\}`\}/g, "title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}");
  result = result.replace(/title="设置为非混部工作负载"/g, "title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}");

  // Delete modal content with variable name
  result = result.replace(
    /content=\{\[\s*`确定将混部工作负载 \$\{delete\w+Name\} 设置为非混部工作负载吗\？`,\s*\]\}/g,
    (match) => {
      const nameMatch = match.match(/delete\w+Name/);
      return `content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: ${nameMatch[0]} })]}`;
    }
  );

  // 16. Replace footer buttons
  result = result.replace(/>取消</g, ">{intl.formatMessage({ id: 'common.cancel' })}<");
  result = result.replace(/>重置</g, ">{intl.formatMessage({ id: 'common.reset' })}<");
  result = result.replace(/>确定</g, ">{intl.formatMessage({ id: 'common.confirm' })}<");

  // 17. Replace selected workloads count
  result = result.replace(
    /\{`已选择\$\{selectedRows\.length\}个工作负载`\}/g,
    "{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}"
  );

  return result;
}

// Process workload pages
for (const config of workloadPages) {
  const filePath = join(srcDir, config.file);
  try {
    let content = readFileSync(filePath, 'utf-8');
    content = processWorkloadPage(content, config);
    writeFileSync(filePath, content, 'utf-8');
    console.log(`✓ Processed ${config.file}`);
  } catch (e) {
    console.error(`✗ Error processing ${config.file}: ${e.message}`);
  }
}

// Process PodPage.jsx (same pattern but Pod-specific)
try {
  const podPath = join(srcDir, 'pages/workload/PodPage.jsx');
  let content = readFileSync(podPath, 'utf-8');

  // PodPage still has remaining Chinese - fix the render section
  content = content.replace(/messageApi\.error\('接口异常'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.apiError' }))");
  content = content.replace(/messageApi\.success\('修改成功'\)/g, "messageApi.success(intl.formatMessage({ id: 'message.modifySuccess' }))");
  content = content.replace(/messageApi\.error\('修改失败'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.modifyFailed' }))");
  content = content.replace(/messageApi\.error\('至少需要选择1个工作负载'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.selectAtLeastOneWorkload' }))");
  content = content.replace(/messageApi\.error\('一次最多只能添加10个工作负载'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.maxTenWorkloads' }))");
  content = content.replace(/messageApi\.success\('混部pod添加成功'\)/g, "messageApi.success(intl.formatMessage({ id: 'message.addColocationSuccess' }, { type: 'Pod' }))");
  content = content.replace(/messageApi\.error\('混部pod添加失败'\)/g, "messageApi.error(intl.formatMessage({ id: 'message.addColocationFailed' }, { type: 'Pod' }))");

  // choseState
  content = content.replace(/setChoseState\('离线'\)/g, "setChoseState(intl.formatMessage({ id: 'workload.status.offline' }))");
  content = content.replace(/setChoseState\('超卖'\)/g, "setChoseState(intl.formatMessage({ id: 'workload.status.oversold' }))");
  content = content.replace(/if \(type === '离线'\)/g, "if (type === intl.formatMessage({ id: 'workload.status.offline' }))");

  // QoS tip
  content = content.replace(/在线属性的工作负载比离线属性会有更高的QoS等级，超卖属性的工作负载会使用超卖资源。/g, "{intl.formatMessage({ id: 'message.qosTip' })}");

  // Search placeholders
  content = content.replace(/placeholder="搜索Pod名称"/g, "placeholder={intl.formatMessage({ id: 'search.podName' })}");
  content = content.replace(/placeholder="按标签搜索  例：key：value"/g, "placeholder={intl.formatMessage({ id: 'search.byLabel' })}");

  // Add button
  content = content.replace(/添加混部Pod/g, "{intl.formatMessage({ id: 'button.addColocation' }, { type: 'Pod' })}");

  // Pagination
  content = content.replace(/showTotal: \(total\) => `共\$\{total\}条`/g, "showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total })");

  // Modal title
  content = content.replace(/title=\{`修改为\$\{choseState\}`\}/g, "title={intl.formatMessage({ id: 'message.changeToState' }, { state: choseState })}");
  content = content.replace(/title="设置为非混部工作负载"/g, "title={intl.formatMessage({ id: 'action.setNonColocationWorkload' })}");
  content = content.replace(
    /content=\{\[\s*`确定将混部工作负载 \$\{deleteName\} 设置为非混部工作负载吗\？`,\s*\]\}/g,
    "content={[intl.formatMessage({ id: 'message.confirmSetNonColocation' }, { name: deleteName })]}"
  );

  // Footer buttons
  content = content.replace(/>取消</g, ">{intl.formatMessage({ id: 'common.cancel' })}<");
  content = content.replace(/>重置</g, ">{intl.formatMessage({ id: 'common.reset' })}<");
  content = content.replace(/>确定</g, ">{intl.formatMessage({ id: 'common.confirm' })}<");

  // Selected count
  content = content.replace(
    /\{`已选择\$\{selectedRows\.length\}个工作负载`\}/g,
    "{intl.formatMessage({ id: 'message.selectedWorkloads' }, { count: selectedRows.length })}"
  );

  writeFileSync(podPath, content, 'utf-8');
  console.log('✓ Processed pages/workload/PodPage.jsx');
} catch (e) {
  console.error(`✗ Error processing PodPage.jsx: ${e.message}`);
}

console.log('\nBatch processing complete!');
