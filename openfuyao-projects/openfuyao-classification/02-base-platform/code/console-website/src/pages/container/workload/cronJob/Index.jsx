/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { cronJobStatus } from '@/common/constants';
import { listCronJobs, deleteCronJob } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { firstAlphabetUp, sorterFirstAlphabet, getCronJobStatus } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { containerRouterPrefix } from '@/constant.js';

export default function CronJob() {
  const { t, i18n } = useTranslation('workload');

  const filterStatus = useMemo(() => {
    return cronJobStatus.map(item => ({ text: firstAlphabetUp(item), value: item }));
  }, []);
  // 列表项
  const cronJobColumns = useMemo(() => [
    {
      title: t('cronJob.name'),
      key: 'cronJob_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/workload/cronJob/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('cronJob.status'),
      key: 'status',
      width: 220,
      sorter: (a, b) => sorterFirstAlphabet(getCronJobStatus(a.status, a.spec?.suspend), getCronJobStatus(b.status, b.spec?.suspend)),
      render: (_, record) => <p className={`resource_status ${getCronJobStatus(record.status, record.spec?.suspend).toLowerCase()}_circle`}>
        {getCronJobStatus(record.status, record.spec?.suspend)}
      </p>,
      filters: filterStatus,
      onFilter: (value, record) => getCronJobStatus(record.status, record.spec?.suspend).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('cronJob.namespace'),
      key: 'cronJob_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('cronJob.schedule'),
      key: 'cronJob_scheduling',
      sorter: (a, b) => sorterFirstAlphabet(a.spec.schedule, b.spec.schedule),
      render: (_, record) => record.spec.schedule,
    },
    {
      title: t('cronJob.concurrencyPolicy'),
      key: 'concurrency_strategy',
      sorter: (a, b) => sorterFirstAlphabet(a.spec.concurrencyPolicy, b.spec.concurrencyPolicy),
      render: (_, record) => record.spec.concurrencyPolicy,
    },
    {
      title: t('pod.created'),
      key: 'cronJob_create_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [filterStatus, t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom
      className="create_bread"
      items={[
        { title: t('module.title'), disabled: true },
        { title: t('cronJob.title') },
      ]}
    />
    <ResourceList
      resourceType={'CronJob'}
      columns={cronJobColumns}
      getResourceFn={listCronJobs}
      deleteResourceFn={(record) => deleteCronJob(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/workload/cronJob/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}