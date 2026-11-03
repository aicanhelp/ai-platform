/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useMemo, useCallback } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Link } from 'inula-router';
import { ResponseCode, namespaceStatusOptions } from '@/common/constants';
import { listNamespaces, deleteNamespace, getCPUAndMemUsageFromAllPods } from '@/api/containerApi';
import { firstAlphabetUp, sorterFirstAlphabet } from '@/tools/utils';
import Dayjs from 'dayjs';
import ResourceList from '@/components/layouts/ResourceList';
import '@/styles/pages/workload.less';
import Big from 'big.js';
import { message } from 'antd';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

const numericSortValue = (value) => {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : -1;
};

export default function Namespace() {
  const { t, i18n } = useTranslation('namespace');

  const filterNamespaceStatus = useMemo(() =>
    Object.keys(namespaceStatusOptions).map(k => ({
      text: namespaceStatusOptions[k],
      value: k,
    })), []);

  const getNamespaceCpuAndMemoryData = useCallback(async () => {
    const namespaceUsage = {};
    try {
      const res = await getCPUAndMemUsageFromAllPods();
      if (res.status === ResponseCode.OK) {
        res.data.results.forEach(result => {
          const type = result.metricName;
          if (type === 'pod_cpu_usage') {
            result?.data?.result?.forEach((res) => {
              if (!res?.labels) { return; }
              const { labels: { namespace }, sample: { value = 0 } = {} } = res;
              if (!namespaceUsage[namespace]) {
                namespaceUsage[namespace] = { cpuUsage: new Big(0), memoryUsage: new Big(0) };
              }
              namespaceUsage[namespace].cpuUsage = namespaceUsage[namespace].cpuUsage.plus(new Big(value));
            });
          } else if (type === 'pod_memory_usage') {
            result?.data?.result?.forEach((res) => {
              if (!res?.labels) { return; }
              const { labels: { namespace }, sample: { value = 0 } = {} } = res;
              if (!namespaceUsage[namespace]) {
                namespaceUsage[namespace] = { cpuUsage: new Big(0), memoryUsage: new Big(0) };
              }
              namespaceUsage[namespace].memoryUsage = namespaceUsage[namespace].memoryUsage.plus(new Big(value));
            });
          }
        });
        for (const usage of Object.values(namespaceUsage)) {
          const parseCpu = parseFloat(usage.cpuUsage.toFixed(2));
          const parseMem = parseFloat((usage.memoryUsage.div(1024 * 1024)).toFixed(2));
          if (usage.cpuUsage.eq(0)) {
            usage.cpuUsage = '0';
          } else if (parseCpu === 0) {
            usage.cpuUsage = '< 0.01';
          } else {
            usage.cpuUsage = parseCpu;
          }
          if (usage.memoryUsage.eq(0)) {
            usage.memoryUsage = '0';
          } else if (parseMem === 0) {
            usage.memoryUsage = '< 0.01';
          } else {
            usage.memoryUsage = parseMem;
          }
        }
      }
    } catch (e) {
      message.error(t('messages.getCpuMemoryFailed', { reason: e.message }));
      return namespaceUsage;
    }
    return namespaceUsage;
  }, [t]);

  const getNamespaceListForResourceList = useCallback(async (_namespace) => {
    const res = await listNamespaces();
    if (res.status === ResponseCode.OK) {
      const namespaceUsage = await getNamespaceCpuAndMemoryData();
      res.data.items.forEach(obj => {
        const n = obj.metadata.name;
        if (namespaceUsage[n]) {
          obj.metadata.cpuUsage = namespaceUsage[n].cpuUsage;
          obj.metadata.memoryUsage = namespaceUsage[n].memoryUsage;
        }
      });
    }
    return res;
  }, [getNamespaceCpuAndMemoryData]);

  const namespaceColumns = useMemo(() => [
    {
      title: t('namespace.nameLabel'),
      key: 'namespace_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/namespace/namespaceManage/${record.metadata.name}`}>{record.metadata.name}</Link>
      ),
    },
    {
      title: t('namespace.status'),
      key: 'namespace_status',
      width: 220,
      sorter: (a, b) => sorterFirstAlphabet(firstAlphabetUp(a.status.phase), firstAlphabetUp(b.status.phase)),
      render: (_, record) => (
        <div className="status_group">
          <span className={record.status.phase === 'Active' ? 'running_circle' : 'terminating_circle'} />
          <span>{firstAlphabetUp(record.status.phase)}</span>
        </div>
      ),
      filters: filterNamespaceStatus,
      onFilter: (value, record) => record.status.phase.toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('namespace.memory'),
      key: 'namespace_memory',
      sorter: (a, b) => numericSortValue(a.metadata.memoryUsage) - numericSortValue(b.metadata.memoryUsage),
      render: (_, record) => record.metadata.memoryUsage || '--',
    },
    {
      title: t('namespace.cpu'),
      key: 'namespace_cpu',
      sorter: (a, b) => numericSortValue(a.metadata.cpuUsage) - numericSortValue(b.metadata.cpuUsage),
      render: (_, record) => record.metadata.cpuUsage || '--',
    },
    {
      title: t('namespace.createdAt'),
      key: 'namespace_created_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [t, i18n.language, filterNamespaceStatus]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/namespaceManage`, disabled: true },
          { title: 'Namespace', path: '/' },
        ]}
      />
      <ResourceList
        resourceType="Namespace"
        columns={namespaceColumns}
        getResourceFn={getNamespaceListForResourceList}
        deleteResourceFn={(record) => deleteNamespace(record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/namespace/namespaceManage/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
