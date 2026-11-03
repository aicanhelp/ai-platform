/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { useMemo, useCallback, useState } from 'openinula';
import { containerRouterPrefix } from '@/constant';
import '@/styles/pages/nodeManage.less';
import { ResponseCode, nodeTypeOptions } from '@/common/constants';
import { listNodes } from '@/api/containerApi';
import { Link, useLocation } from 'inula-router';
import ManageNodeIcon from '@/assets/images/manageNodeIcon.png';
import WorkerNodeIcon from '@/assets/images/workerNodeIcon.png';
import {
  solveNodeReadyStatus,
  solveNodeBaseInfo,
  solveNodeIpAddress,
  solveNodeType,
  sorterFirstAlphabet,
  nodeDetailIsBusy,
} from '@/tools/utils';
import ResourceList from '@/components/layouts/ResourceList';
import { useTranslation } from 'react-i18next';

export default function Node() {
  const { t, i18n } = useTranslation('node');
  const { state } = useLocation();
  const [nodeInfo, setNodeInfo] = useState({}); // 节点情况

  const filterNodeType = useMemo(() =>
    Object.keys(nodeTypeOptions).map(k => ({
      text: t(`type.${k === 'manageNode' ? 'manageNode' : 'workerNode'}`),
      value: k,
    })), [t, i18n.language]);

  const filterBusyStatus = useMemo(() => [
    { text: t('busyStatus.yes'), value: 'True' },
    { text: t('busyStatus.no'), value: 'False' },
  ], [t, i18n.language]);

  const nodeColumns = useMemo(() => [
    {
      title: t('table.name'),
      key: 'node_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/nodeManage/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('table.ip'),
      key: 'node_ip',
      sorter: (a, b) => sorterFirstAlphabet(solveNodeIpAddress(a.status.addresses), solveNodeIpAddress(b.status.addresses)),
      render: (_, record) => solveNodeIpAddress(record.status.addresses),
    },
    {
      title: t('table.type'),
      filters: filterNodeType,
      defaultFilteredValue: state?.nodeType ? [state.nodeType] : undefined,
      filterMultiple: false,
      onFilter: (value, record) => solveNodeType(record.metadata.labels) === value,
      sorter: (a, b) => sorterFirstAlphabet(
        t(`type.${solveNodeType(a.metadata.labels)}`),
        t(`type.${solveNodeType(b.metadata.labels)}`),
      ),
      key: 'nodeType',
      render: (_, record) => t(`type.${solveNodeType(record.metadata.labels)}`),
    },
    {
      title: t('table.status'),
      key: 'nodeStatus',
      width: 220,
      render: (_, record) => {
        const nodeStatus = solveNodeReadyStatus(record);
        let statusClass = 'unknown_circle';
        let statusText = t('status.unknown');
        if (nodeStatus === 'Ready') {
          statusClass = 'running_circle';
          statusText = t('status.normal');
        } else if (nodeStatus === 'NotReady') {
          statusClass = 'down_circle';
          statusText = t('status.abnormal');
        }
        return <div className={'status_group'}>
          <span className={statusClass}></span>
          <span>{statusText}</span>
        </div>;
      },
    },
    {
      title: t('table.isBusy'),
      filters: filterBusyStatus,
      defaultFilteredValue: state?.isBusy ? [state.isBusy] : undefined,
      onFilter: (value, record) =>
        value === 'True'
          ? record.status.conditions.some(item => (item.type !== 'Ready' && item.status === 'True') || (item.type === 'Ready' && item.status === 'False'))
          : record.status.conditions.every(item => (item.type === 'Ready' && item.status === 'True') || (item.type !== 'Ready' && item.status === 'False')),
      filterMultiple: false,
      sorter: (a, b) => sorterFirstAlphabet(
        nodeDetailIsBusy(a.status.conditions) ? t('busyStatus.yes') : t('busyStatus.no'),
        nodeDetailIsBusy(b.status.conditions) ? t('busyStatus.yes') : t('busyStatus.no'),
      ),
      key: 'isNodeBusy',
      render: (_, record) => nodeDetailIsBusy(record.status.conditions) ? t('busyStatus.yes') : t('busyStatus.no'),
    },
    {
      title: t('table.os'),
      key: 'node_os',
      sorter: (a, b) => sorterFirstAlphabet(a.status.nodeInfo.osImage, b.status.nodeInfo.osImage),
      render: (_, record) => record.status.nodeInfo.osImage || '--',
    },
  ], [t, filterNodeType, filterBusyStatus, state, i18n.language]);

  const getNodeListForResourceList = useCallback(async () => {
    const res = await listNodes();
    if (res.status === ResponseCode.OK) {
      setNodeInfo(solveNodeBaseInfo(res.data.items));
    }
    return res;
  }, []);

  return <div className="child_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/nodeManage`, disabled: true }]} />
    <div className="node_header node_manage_card_header">
      <div className="no_image_flex">
        <div className="node_title">
          <span>{nodeInfo.unnormalNode}</span>
          <span>/</span>
          <span>{nodeInfo.totalNode}</span>
        </div>
        <p>{t('header.abnormalNode')}</p>
      </div>
      <div className="node_item">
        <img src={ManageNodeIcon} />
        <div className="node_title">
          <div className="node_count">{nodeInfo.manageNode}</div>
          <p>{t('header.manageNode')}</p>
        </div>
      </div>
      <div className="node_item">
        <img src={WorkerNodeIcon} />
        <div className="node_title">
          <div className="node_count">{nodeInfo.workerNode}</div>
          <p>{t('header.workerNode')}</p>
        </div>
      </div>
    </div>
    <ResourceList
      resourceType={t('module.title')}
      columns={nodeColumns}
      getResourceFn={getNodeListForResourceList}
      hideOperations={true}
      hideCreate={true}
    />
  </div>;
}