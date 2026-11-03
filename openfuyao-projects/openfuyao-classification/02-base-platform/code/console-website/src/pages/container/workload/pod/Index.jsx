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
import { podStatus } from '@/common/constants';
import { deletePod, listPods } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { firstAlphabetUp, sorterFirstAlphabet, solveHostIpAddress, solvePodIpAddress } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { containerRouterPrefix } from '@/constant.js';

export default function Pod() {
  const { t, i18n } = useTranslation('workload');

  const filterPodStatus = useMemo(() => {
    return podStatus.map(item => ({ text: firstAlphabetUp(item), value: item }));
  }, []);
  const sortPod = (a, b) => {
    let firstTime = a.metadata.annotations?.lastUpdateTime ? a.metadata.annotations.lastUpdateTime : a.metadata.creationTimestamp;
    let endTime = b.metadata.annotations?.lastUpdateTime ? b.metadata.annotations.lastUpdateTime : b.metadata.creationTimestamp;
    return Dayjs(firstTime) - Dayjs(endTime);
  };

  // 列表项
  const podColumns = useMemo(() => [
    {
      title: t('pod.name'),
      key: 'pod_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/workload/pod/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('pod.status'),
      key: 'status',
      width: 220,
      sorter: (a, b) => sorterFirstAlphabet(a.status.phase, b.status.phase),
      render: (_, record) => <p className={`resource_status ${(record.status.phase).toLowerCase()}_circle`}>
        {record.status.phase}
      </p>,
      filters: filterPodStatus,
      onFilter: (value, record) => record.status.phase.toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('pod.namespace'),
      key: 'pod_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('pod.workNode'),
      key: 'pod_work_node',
      sorter: (a, b) => sorterFirstAlphabet(solveHostIpAddress(a.status), solveHostIpAddress(b.status)),
      render: (_, record) => solveHostIpAddress(record.status),
    },
    {
      title: t('pod.ip'),
      key: 'pod_ip',
      sorter: (a, b) => sorterFirstAlphabet(solvePodIpAddress(a.status), solvePodIpAddress(b.status)),
      render: (_, record) => solvePodIpAddress(record.status),
    },
    {
      title: t('pod.updateTime'),
      key: 'pod_update_time',
      sorter: (a, b) => sortPod(a, b),
      render: (_, record) =>
        Dayjs(record.metadata.annotations?.lastUpdateTime ? record.metadata.annotations.lastUpdateTime : record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [filterPodStatus, t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('pod.title') },
    ]} />
    <ResourceList
      resourceType={'Pod'}
      columns={podColumns}
      getResourceFn={listPods}
      deleteResourceFn={(record) => deletePod(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/workload/pod/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}