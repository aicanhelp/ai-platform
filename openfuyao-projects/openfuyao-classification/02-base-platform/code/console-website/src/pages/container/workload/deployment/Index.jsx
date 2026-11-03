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
import { workloadStatus } from '@/common/constants';
import { listDeployments, deleteDeployment } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { firstAlphabetUp, sorterFirstAlphabet, getWorkloadStatusJudge } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { containerRouterPrefix } from '@/constant.js';

export default function Deployment() {
  const { t, i18n } = useTranslation('workload');

  const filterStatus = useMemo(() => {
    return workloadStatus.map(item => ({ text: firstAlphabetUp(item), value: item }));
  }, []);
  // 列表项
  const deploymentColumns = useMemo(() => [
    {
      title: t('deployment.name'),
      key: 'deployment_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/workload/deployment/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('deployment.status'),
      key: 'status',
      width: 220,
      sorter: (a, b) => sorterFirstAlphabet(getWorkloadStatusJudge(a.status), getWorkloadStatusJudge(b.status)),
      render: (_, record) => <p className={`resource_status ${getWorkloadStatusJudge(record.status).toLowerCase()}_circle`}>
        {getWorkloadStatusJudge(record.status)}
      </p>,
      filters: filterStatus,
      onFilter: (value, record) => getWorkloadStatusJudge(record.status).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('deployment.namespace'),
      key: 'deployment_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('deployment.instances'),
      key: 'deployment_example',
      render: (_, record) => `${record.status.readyReplicas || 0}/${record.status.replicas || 0}`,
    },
    {
      title: t('pod.created'),
      key: 'deployment_create_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [filterStatus, t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('deployment.title') },
    ]} />
    <ResourceList
      resourceType={'Deployment'}
      columns={deploymentColumns}
      getResourceFn={listDeployments}
      deleteResourceFn={(record) => deleteDeployment(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/workload/deployment/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}