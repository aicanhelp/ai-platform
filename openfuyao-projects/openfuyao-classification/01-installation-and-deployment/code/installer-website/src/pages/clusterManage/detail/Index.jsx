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
import '@/styles/pages/clusterDetail.less';
import { useLocation, useParams, useHistory } from 'inula-router';
import { clusterStatus, ResponseCode } from '@/common/constants';
import { CheckCircleFilled, ExclamationCircleFilled } from '@ant-design/icons';
import { useState, useCallback, useEffect } from 'openinula';
import { Button, message, Tabs } from 'antd';
import { requestResource, clusterStatusValueToText } from '@/tools/utils';
import { upgradeCluster, getCluster } from '@/api/clusterApi';
import UpgradeClusterModal from '@/components/UpgradeClusterModal';
import ClusterDetail from '@/pages/clusterManage/detail/ClusterDetail';
import NodeDetail from '@/pages/clusterManage/detail/NodeDetail';
import useLocalStorage from '@/hooks/useLocalStorage';
import { useIntl } from '../../../i18n';

export default function ClusterDetailPage() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const { clusterName } = useParams();
  const [clusterDetailTabKey, setClusterDetailTabKey] = useState('node');
  const [status, setStatus] = useState('');
  const [upgradeModalOpen, setUpgradeModalOpen] = useState(false);
  const [clusterDetail, setClusterDetail] = useState({}); // 详情数据
  const [messageApi, contextHolder] = message.useMessage();
  const [hiddenPage, setHiddenPage] = useState(false);
  const [nodeLoading, setNodeLoading] = useState(false); // 加载中

  const handleOpenUpgradeModal = () => {
    setUpgradeModalOpen(true);
  };

  const handleUpgrade = async (clusterUpgradeForm) => {
    await clusterUpgradeForm.validateFields();
    const version = clusterUpgradeForm.getFieldValue('openFuyaoUpgradeVersion');
    const [error, res] = await requestResource(upgradeCluster(clusterName, { version }));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'cluster.upgradeFailed' }));
    } else if (res) {
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'cluster.upgrading' }));
        setUpgradeModalOpen(false);
        getClusterDescription();
      }
    }
  };

  const handleCancelUpgrade = (clusterUpgradeForm) => {
    clusterUpgradeForm.resetFields();
    setUpgradeModalOpen(false);
  };

  const items = [
    {
      key: 'node',
      label: intl.formatMessage({ id: 'cluster.detail.nodeTab' }),
      children: <NodeDetail clusterDetail={clusterDetail} nodeLoading={nodeLoading} />,
    },
    {
      key: 'cluster',
      label: intl.formatMessage({ id: 'cluster.detail.clusterTab' }),
      children: <ClusterDetail clusterDetail={clusterDetail} />,
    },
  ];

  const handleSetClusterDetailTabKey = (key) => {
    setClusterDetailTabKey(key);
  };

  const handlePassMessage = (event) => {
    if (event.data.type === 'clusterDetailPageDestory') {
      if (event.data.isDestory) {
        setHiddenPage(true);
        // 清理定时器
        const timeId = localStorage.getItem('clusterTimeDetailId') || '';
        clearTimeout(timeId);
        localStorage.removeItem('clusterTimeDetailId');
      } else {
        setHiddenPage(false);
      }
    }
  };

  const getClusterDescription = async () => {
    setNodeLoading(true);
    const [error, res] = await requestResource(getCluster(clusterName));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'cluster.detail.getInfoFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        setClusterDetail(res.data.data);
        setStatus(res.data.data?.clusterStatus || '');
      }
    }
    setNodeLoading(false);
  };

  useEffect(() => {
    getClusterDescription();

    // 添加轮询，每10秒获取一次集群详情
    const intervalId = setInterval(() => {
      getClusterDescription();
    }, 10000);

    // 清理函数，组件卸载时清除定时器
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);

  return (
    <div className="child_content">
      {contextHolder}
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'nav.localMenu' }), path: `/cluster` },
          { title: intl.formatMessage({ id: 'cluster.detail.detailTitle' }), path: '/detail' },
        ]}
      />
      <div
        className="pod_title cluster_desc"
        style={{ border: currentTheme !== 'light' && 'none', backgroundColor: currentTheme !== 'light' && '#2a2d34' }}
      >
        <div className="left_cluster">
          <h3>{clusterName}</h3>
          <div className={`cluster_status cluster_status_bg_${clusterStatus[status] || 'unknown'}`}>
            <span>{clusterStatus[status] === 'healthy' ? <CheckCircleFilled /> : <ExclamationCircleFilled />}</span>
            <span>{clusterStatusValueToText(status, intl)}</span>
          </div>
          <span className="cluster_ha_bg">
            {clusterDetail?.isHA
              ? intl.formatMessage({ id: 'cluster.detail.haCluster' })
              : intl.formatMessage({ id: 'cluster.detail.nonHaCluster' })}
          </span>
        </div>
        <Button
          className={status === 'Healthy' && 'cluster_primary_btn'}
          disabled={status !== 'Healthy'}
          onClick={handleOpenUpgradeModal}
        >
          {intl.formatMessage({ id: 'cluster.detail.upgradeCluster' })}
        </Button>
      </div>
      {upgradeModalOpen && (
        <UpgradeClusterModal
          open={upgradeModalOpen}
          clusterInfo={{ name: clusterName, openFuyaoVersion: clusterDetail?.openFuyaoVersion }}
          okFn={handleUpgrade}
          cancelFn={handleCancelUpgrade}
        />
      )}
      {!hiddenPage && (
        <Tabs
          items={items}
          onChange={handleSetClusterDetailTabKey}
          activeKey={clusterDetailTabKey}
          destroyInactiveTabPane={true}
        ></Tabs>
      )}
    </div>
  );
}
