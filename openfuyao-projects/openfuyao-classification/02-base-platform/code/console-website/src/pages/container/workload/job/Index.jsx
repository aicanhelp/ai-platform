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
import { jobStatus } from '@/common/constants';
import { listJobs, deleteJob } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { firstAlphabetUp, sorterFirstAlphabet, getJobStatus } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { containerRouterPrefix } from '@/constant.js';

export default function Job() {
  const { t, i18n } = useTranslation('workload');

  const filterStatus = useMemo(() => {
    return jobStatus.map(item => ({ text: firstAlphabetUp(item), value: item }));
  }, []);
  // 列表项
  const jobColumns = useMemo(() => [
    {
      title: t('job.name'),
      key: 'job_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/workload/job/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('job.status'),
      key: 'status',
      width: 220,
      filters: filterStatus,
      onFilter: (value, record) => getJobStatus(record.status).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      sorter: (a, b) => sorterFirstAlphabet(getJobStatus(a.status), getJobStatus(b.status)),
      render: (_, record) => <p className={`resource_status ${getJobStatus(record.status).toLowerCase()}_circle`}>
        {getJobStatus(record.status)}
      </p>,
    },
    {
      title: t('job.namespace'),
      key: 'job_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('deployment.instances'),
      key: 'job_example',
      render: (_, record) => `${record.status.succeeded || 0}/${(record.status.succeeded || 0) + (record.status.ready || 0)}`,
    },
    {
      title: t('job.executionTime'),
      key: 'completionTime',
      sorter: (a, b) => Dayjs(a.status.completionTime) - Dayjs(b.status.completionTime),
      render: (_, record) => record.status.completionTime ? Dayjs(record.status.completionTime).format('YYYY-MM-DD HH:mm') : '--',
    },
    {
      title: t('pod.created'),
      key: 'job_create_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [filterStatus, t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('job.title') },
    ]} />
    <ResourceList
      resourceType={'Job'}
      columns={jobColumns}
      getResourceFn={listJobs}
      deleteResourceFn={(record) => deleteJob(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/workload/job/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}