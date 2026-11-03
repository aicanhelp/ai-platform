/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import Dayjs from 'dayjs';
import { InfoCircleFilled, CheckCircleFilled, EditOutlined, FileSearchOutlined } from '@ant-design/icons';
import { useCallback, useEffect, useMemo, useState, useStore } from 'openinula';
import { ResponseCode } from '@/common/constants';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import DiskNodeIcon from '@/assets/images/diskNodeIcon.png';
import MemoryIcon from '@/assets/images/memoryIcon.png';
import NetworkIsUseNodeIcon from '@/assets/images/networkIsUseNodeIcon.png';
import NetworkSendNodeIcon from '@/assets/images/networkSendNodeIcon.png';
import AnnotationModal from '@/components/AnnotationModal';
import { message, Progress, Tooltip } from 'antd';
import {
  solveAnnotationOrLabelDiff,
  solveNodeIpAddress,
  solveNodeReadyStatus,
  solveK8sQuantityToBytes,
  nodeDetailIsBusy,
  transformUnifiedUnit,
  forbiddenMsg,
  showSuccessThen,
} from '@/tools/utils';
import { getResourceControllMonitor } from '@/api/monitorApi';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import { useTranslation } from 'react-i18next';

export default function Detail({ nodeName, nodeDetailDataProps, refreshFn }) {
  const { t } = useTranslation(['node', 'common']);
  const [messageApi, contextHolder] = message.useMessage();
  // 对话框展示
  const [isNodeDetailLabelModalOpen, setIsNodeDetailLabelModalOpen] = useState(false);
  const [isNodeDetailAnnotationModalOpen, setIsNodeDetailAnnotationModalOpen] = useState(false);
  const oldNodeDetailAnnotations = nodeDetailDataProps?.metadata?.annotations || [];
  const oldNodeDetailLabels = nodeDetailDataProps?.metadata?.labels || [];
  const [networkCpuRate, setNetwworkCpuRate] = useState({
    cpuUsage: 0,
    cpuTotal: 0,
    cpuUtilisation: 0,
  }); // 监控cpu使用率
  const [networkMemoryRate, setNetwworkMemoryRate] = useState({
    memoryUsage: 0,
    memoryTotal: 0,
    memoryUtilisation: 0,
  }); // 监控内存使用率
  const [clusterPodCount, setClusterPodCount] = useState(0); // pod数
  const themeStore = useStore('theme');

  // 标签成功回调
  const handleLabelOk = async (nodeDetailTemporyData) => {
    if (JSON.stringify(oldNodeDetailLabels) === JSON.stringify(nodeDetailTemporyData)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsNodeDetailLabelModalOpen(false);
    } else {
      const keyArr = [];
      nodeDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldNodeDetailLabels, nodeDetailTemporyData, 'label');
        try {
          const res = await editAnnotationsOrLabels('node', '', nodeName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsNodeDetailLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };
  // 标签失败回调
  const handleLabelCancel = () => {
    setIsNodeDetailLabelModalOpen(false);
  };

  // 注解成功回调
  const handleAnnotationOk = async (nodeDetailTemporyData) => {
    if (JSON.stringify(oldNodeDetailAnnotations) === JSON.stringify(nodeDetailTemporyData)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsNodeDetailAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      nodeDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldNodeDetailAnnotations, nodeDetailTemporyData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('node', '', nodeName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsNodeDetailAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsNodeDetailAnnotationModalOpen(false);
  };

  // 获取监控信息
  const getNetworkInfo = useCallback(async () => {
    try {
      const clusterRes = await getResourceControllMonitor({ cluster: 'main', value: 'cluster' });
      if (clusterRes.status === ResponseCode.OK) {
        clusterRes.data.results.map(item => {
          if (item.metricName === 'cluster_pods_count') {
            if (item.data.result && item.data.result.length) {
              const [filterObj] = item.data.result.filter(filterItem => filterItem.labels?.node === nodeName);
              setClusterPodCount(filterObj?.sample?.value);
            }
          }
        });
      }
      const nodeRes = await getResourceControllMonitor({ cluster: 'main', value: 'node', start: '', end: '', conditions: { instance: nodeName } });
      if (nodeRes.status === ResponseCode.OK) {
        let nodeCpuInfo = { ...networkCpuRate };
        let nodeMemoryInfo = { ...networkMemoryRate };
        nodeRes.data.results.map(item => {
          if (item.metricName === 'node_cpu_usage') {
            const [nodeInfo] = item.data.result ?? [];
            nodeCpuInfo.cpuUsage = nodeInfo?.sample?.value;
          }
          if (item.metricName === 'node_cpu_total') {
            const [nodeInfo] = item.data.result ?? [];
            nodeCpuInfo.cpuTotal = nodeInfo?.sample?.value;
          }
          if (item.metricName === 'node_cpu_utilisation') {
            const [nodeInfo] = item.data.result ?? [];
            nodeCpuInfo.cpuUtilisation = nodeInfo?.sample?.value;
          }
          if (item.metricName === 'node_memory_usage') {
            const [nodeInfo] = item.data.result ?? [];
            nodeMemoryInfo.memoryUsage = nodeInfo?.sample?.value / 1024 / 1024 / 1024;
          }
          if (item.metricName === 'node_memory_total') {
            const [nodeInfo] = item.data.result ?? [];
            nodeMemoryInfo.memoryTotal = nodeInfo?.sample?.value / 1024 / 1024 / 1024;
          }
          if (item.metricName === 'node_memory_utilisation') {
            const [nodeInfo] = item.data.result ?? [];
            nodeMemoryInfo.memoryUtilisation = nodeInfo?.sample?.value;
          }
        });
        setNetwworkCpuRate({ ...nodeCpuInfo });
        setNetwworkMemoryRate({ ...nodeMemoryInfo });
      }
    } catch (error) {
      if (error.response?.status !== ResponseCode.OK) {
        messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  }, [nodeName, t, messageApi]);

  useEffect(() => {
    getNetworkInfo();
  }, [getNetworkInfo]);

  const nodeBaseInfoItems = useMemo(() => [
    { key: t('fields.nodeName'), value: nodeDetailDataProps?.metadata?.name },
    {
      key: t('fields.nodeSpec'),
      value: nodeDetailDataProps?.status?.capacity
        ? `${nodeDetailDataProps?.status?.capacity?.cpu}${t('unit.cores')} ${(solveK8sQuantityToBytes(nodeDetailDataProps?.status?.capacity?.memory) / 1024 ** 3).toFixed(2)}GB`
        : '--',
    },
    {
      key: t('fields.nodeStatus'),
      value: (() => {
        const readyStatus = solveNodeReadyStatus(nodeDetailDataProps);
        if (readyStatus === 'Ready') { return <span className='running_circle'>{t('status.normal')}</span>; }
        if (readyStatus === 'NotReady') { return <span className='down_circle'>{t('status.abnormal')}</span>; }
        return <span className='unknown_circle'>{t('status.unknown')}</span>;
      })(),
    },
    { key: t('fields.nodeIpAddress'), value: solveNodeIpAddress(nodeDetailDataProps?.status?.addresses) },
    { key: t('fields.os'), value: nodeDetailDataProps?.status?.nodeInfo?.osImage || '--' },
    { key: t('fields.nodeId'), value: nodeDetailDataProps?.metadata?.uid },
    { key: t('fields.createdAt'), value: Dayjs(nodeDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    { key: t('fields.isBusy'), value: nodeDetailIsBusy(nodeDetailDataProps?.status?.conditions) ? t('busyStatus.yes') : t('busyStatus.no') },
  ], [nodeDetailDataProps, t]);

  return <div className='tab_container container_margin_box normal_container_height'>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className="detail_card">
      <h3>{t('info.basicInfo')}</h3>
      <div className="detail_info_box">
        <DetailBaseInfoList items={nodeBaseInfoItems} columns={2} />
        <div className="annotation">
          <div className="ann_title">
            <p>{t('info.labels')}</p>
            <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsNodeDetailLabelModalOpen(true)} />
          </div>
          <AnnotationModal open={isNodeDetailLabelModalOpen} type="label" dataList={oldNodeDetailLabels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
          <div className="key_value">
            {oldNodeDetailLabels?.length ? oldNodeDetailLabels.map(item =>
              <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />,
            ) :
              <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
          </div>
        </div>
        <div className="annotation">
          <div className="ann_title">
            <p>{t('info.annotations')}</p>
            <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsNodeDetailAnnotationModalOpen(true)} />
          </div>
          <AnnotationModal open={isNodeDetailAnnotationModalOpen} type="annotation" dataList={oldNodeDetailAnnotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
          <div className="key_value">
            {oldNodeDetailAnnotations?.length ? oldNodeDetailAnnotations.map(item =>
              <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />,
            ) :
              <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
          </div>
        </div>
      </div>
    </div>
    <div className="detail_card container_card_add">
      <h3>{t('info.healthStatus')}</h3>
      <div className="resource_list node_resource_list">
        <div className="resource_item">
          <img className="resource_icon" src={NetworkIsUseNodeIcon} />
          <div className="resource_word">
            <p>{t('health.networkAvailability')}</p>
            <p>
              {!nodeDetailDataProps?.healthy?.networkUnavailable ? <CheckCircleFilled style={{ color: '#09aa71' }} /> : <InfoCircleFilled style={{ color: '#e7434a' }} />}
              <span className="node_healthy_status">{!nodeDetailDataProps?.healthy?.networkUnavailable ? t('health.healthy') : t('health.abnormal')}</span>
              {nodeDetailDataProps?.healthy?.networkUnavailable && <Tooltip title={nodeDetailDataProps?.healthy.networkReason}><FileSearchOutlined className="unset_svg" /></Tooltip>}
            </p>
          </div>
        </div>

        <div className="resource_item">
          <img className="resource_icon" src={MemoryIcon} />
          <div className="resource_word">
            <p>{t('health.memoryPressure')}</p>
            <p>
              {!nodeDetailDataProps?.healthy?.memoryPressure ? <CheckCircleFilled style={{ color: '#09aa71' }} /> : <InfoCircleFilled style={{ color: '#e7434a' }} />}
              <span className="node_healthy_status">{!nodeDetailDataProps?.healthy?.memoryPressure ? t('health.healthy') : t('health.abnormal')}</span>
              {nodeDetailDataProps?.healthy?.memoryPressure && <Tooltip title={nodeDetailDataProps?.healthy.memoryReason}><FileSearchOutlined className="unset_svg" /></Tooltip>}
            </p>
          </div>
        </div>

        <div className="resource_item">
          <img className="resource_icon" src={DiskNodeIcon} />
          <div className="resource_word">
            <p>{t('health.diskPressure')}</p>
            <p>
              {!nodeDetailDataProps?.healthy?.diskPressure ? <CheckCircleFilled style={{ color: '#09aa71' }} /> : <InfoCircleFilled style={{ color: '#e7434a' }} />}
              <span className="node_healthy_status">{!nodeDetailDataProps?.healthy?.diskPressure ? t('health.healthy') : t('health.diskPressureStatus')}</span>
              {nodeDetailDataProps?.healthy?.diskPressure && <Tooltip title={nodeDetailDataProps?.healthy.diskReason}><FileSearchOutlined className="unset_svg" /></Tooltip>}
            </p>
          </div>
        </div>

        <div className="resource_item">
          <img className="resource_icon" src={NetworkSendNodeIcon} />
          <div className="resource_word">
            <p>{t('health.acceptPods')}</p>
            <p>
              {nodeDetailDataProps?.healthy?.ready ? <CheckCircleFilled style={{ color: '#09aa71' }} /> : <InfoCircleFilled style={{ color: '#e7434a' }} />}
              <span className="node_healthy_status">{nodeDetailDataProps?.healthy?.ready ? t('health.healthy') : t('health.abnormal')}</span>
              {!nodeDetailDataProps?.healthy?.ready && <Tooltip title={nodeDetailDataProps?.healthy.readyReason}><FileSearchOutlined className="unset_svg" /></Tooltip>}
            </p>
          </div>
        </div>
      </div>
    </div>
    <div className="detail_card container_card_add">
      <h3>{t('info.resourceQuota')}</h3>
      <div className="echarts_list" style={{ justifyContent: 'space-around' }}>
        <div className="echarts_item">
          <p className="ecahrts_title">{t('resource.cpu')}</p>
          <Progress type="circle"
            percent={networkCpuRate.cpuUtilisation ? ((networkCpuRate.cpuUtilisation) * 100).toFixed(2) : 0}
            strokeColor={networkCpuRate.cpuUtilisation >= 1 ? '#e7434a' : '#09aa71'}
            format={(_percent) => `${_percent}%`}></Progress>
          <div className="word_flex">
            <p>
              <span>{t('resource.usage')}</span>
              <span>{!transformUnifiedUnit(networkCpuRate.cpuUsage) ? (networkCpuRate.cpuUsage).toFixed(2) : '--'} cores</span>
            </p>
            <p>
              <span>{t('resource.max')}</span>
              <span>{!transformUnifiedUnit(networkCpuRate.cpuTotal) ? networkCpuRate.cpuTotal : '--'} cores</span>
            </p>
          </div>
        </div>
        <div className="echarts_item">
          <p className="ecahrts_title">{t('resource.memory')}</p>
          <Progress type="circle"
            percent={((networkMemoryRate.memoryUtilisation) * 100).toFixed(2)}
            strokeColor={networkMemoryRate.memoryUtilisation >= 1 ? '#e7434a' : '#09aa71'}
            format={(_percent) => `${_percent}%`}></Progress>
          <div className="word_flex">
            <p>
              <span>{t('resource.usage')}</span>
              <span>{!transformUnifiedUnit(networkMemoryRate.memoryUsage) ? (networkMemoryRate.memoryUsage).toFixed(2) : '--'}Gi</span>
            </p>
            <p>
              <span>{t('resource.max')}</span>
              <span>{!transformUnifiedUnit(networkMemoryRate.memoryTotal) ? (networkMemoryRate.memoryTotal).toFixed(2) : '--'}Gi</span>
            </p>
          </div>
        </div>
        <div className="echarts_item">
          <p className="ecahrts_title">{t('resource.pods')}</p>
          <Progress
            type="circle"
            percent={nodeDetailDataProps?.selfResourceQuota?.podMax
              ? ((clusterPodCount / nodeDetailDataProps.selfResourceQuota.podMax) * 100).toFixed(2)
              : 0
            }
            strokeColor={
              nodeDetailDataProps?.selfResourceQuota?.podMax && ((clusterPodCount / nodeDetailDataProps.selfResourceQuota.podMax) * 100) >= 100
                ? '#e7434a'
                : '#09aa71'
            }
            format={(_percent) => `${_percent}%`}></Progress>
          <div className="word_flex">
            <p>
              <span>{t('resource.usage')}</span>
              <span>{!transformUnifiedUnit(clusterPodCount) ? clusterPodCount : '--'}</span>
            </p>
            <p>
              <span>{t('resource.max')}</span>
              <span>{!transformUnifiedUnit(nodeDetailDataProps?.selfResourceQuota?.podMax) ? nodeDetailDataProps.selfResourceQuota.podMax : '--'}</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>;
}