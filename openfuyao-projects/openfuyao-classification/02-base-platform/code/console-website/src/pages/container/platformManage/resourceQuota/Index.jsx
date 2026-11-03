/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *
 *       http://license.coscl.org.cn/MulanPSL2
 *
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { useMemo } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Link } from 'inula-router';
import { Tag } from 'antd';
import { listResourceQuota, deleteResourceQuota } from '@/api/containerApi';
import Dayjs from 'dayjs';
import ResourceList from '@/components/layouts/ResourceList';
import { solveResourceQuota, sorterFirstAlphabet } from '@/tools/utils';
import '@/styles/pages/workload.less';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

export default function ResourceQuota() {
  const { t, i18n } = useTranslation('namespace');

  const filterResourceQuotaStatus = useMemo(() => [
    { text: t('resourceQuota.statusQuotaExceeded'), value: 'exist' },
    { text: t('resourceQuota.statusQuotaNormal'), value: 'cantExist' },
  ], [t]);

  const resourceQuotaColumns = useMemo(() => [
    {
      title: t('resourceQuota.nameLabel'),
      key: 'resourceQuota_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/namespace/resourceQuota/${record.metadata.namespace}/${record.metadata.name}`}>
          {record.metadata.name}
        </Link>
      ),
    },
    {
      title: t('resourceQuota.namespace'),
      key: 'resourceQuota_namespace',
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('resourceQuota.labels'),
      key: 'resourceQuota_memory',
      width: 200,
      render: (_, record) => (
        <div className="table_label">
          {record.metadata.labels
            ? Object.keys(record.metadata.labels).map(item => (
              <Tag key={item} color="#CFE7FF" className="label_tag_key">{`${item}:${record.metadata.labels[item]}`}</Tag>
            ))
            : '--'}
        </div>
      ),
    },
    {
      title: t('resourceQuota.annotations'),
      key: 'resourceQuota_cpu',
      width: 300,
      render: (_, record) => (
        <div className="table_label">
          {record.metadata.annotations
            ? Object.keys(record.metadata.annotations).map(item => (
              <Tag key={item} color="#CFE7FF" className="label_tag_key">{`${item}:${record.metadata.annotations[item]}`}</Tag>
            ))
            : '--'}
        </div>
      ),
    },
    {
      title: t('resourceQuota.status'),
      key: 'resourceQuota_status',
      width: 220,
      render: (_, record) => (
        <div className="status_group">
          <span className={solveResourceQuota(record.status) === 'exist' ? 'failed_circle' : 'running_circle'} />
          <span>{solveResourceQuota(record.status) === 'exist' ? t('resourceQuota.statusQuotaExceeded') : t('resourceQuota.statusQuotaNormal')}</span>
        </div>
      ),
      filters: filterResourceQuotaStatus,
      onFilter: (value, record) => (solveResourceQuota(record.status) === 'exist' ? 'exist' : 'cantExist') === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('resourceQuota.createdAt'),
      key: 'resourceQuota_created_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [t, i18n.language, filterResourceQuotaStatus]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/resourceQuota`, disabled: true },
          { title: 'ResourceQuota', path: '/' },
        ]}
      />
      <ResourceList
        resourceType="ResourceQuota"
        columns={resourceQuotaColumns}
        getResourceFn={listResourceQuota}
        deleteResourceFn={(record) => deleteResourceQuota(record.metadata.namespace, record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/namespace/resourceQuota/${record.metadata.namespace}/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
