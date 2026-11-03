/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:

 *       http://license.coscl.org.cn/MulanPSL2

 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { Fragment, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import { EditOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import AnnotationModal from '@/components/AnnotationModal';
import { Table, message } from 'antd';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { editResourceInstanceAnnotationOrlabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import { disabledModifyMonitorServiceCr } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function Detail({ serviceMonitorName, prefixObjProps, serviceMonitorNamespace, serviceMonitorDetailDataProps, refreshFn }) {
  const { t } = useTranslation('monitoring');
  const [messageApi, contextHolder] = message.useMessage();
  const [isMonitorGoalManageAnnotationModalOpen, setIsMonitorGoalManageAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = serviceMonitorDetailDataProps?.metadata?.annotations || [];
  const oldLabels = serviceMonitorDetailDataProps?.metadata?.labels || [];
  const containerData = serviceMonitorDetailDataProps?.status?.conditions || [];
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.annotationNotModified'));
      setIsMonitorGoalManageAnnotationModalOpen(false);
    } else {
      const monitorGoalManageKeyArr = [];
      data.map(item => monitorGoalManageKeyArr.push(item.key));
      if (monitorGoalManageKeyArr.filter((item, index) => monitorGoalManageKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey'));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editResourceInstanceAnnotationOrlabels(
            { ...prefixObjProps, namespace: serviceMonitorNamespace },
            serviceMonitorName,
            addAnnotationList,
          );
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.annotationEditSuccess'), () => {
              refreshFn();
              setIsMonitorGoalManageAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.annotationEditFailed')}${error.response.data.message}`);
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsMonitorGoalManageAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.labelNotModified'));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey'));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的标签
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editResourceInstanceAnnotationOrlabels({ ...prefixObjProps, namespace: serviceMonitorNamespace }, serviceMonitorName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.labelEditSuccess'), () => {
              refreshFn();
              setIsLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.labelEditFailed')}${error.response.data.message}`);
          }
        }
      }
    }
  };
  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const containerColumns = [
    {
      title: t('serviceMonitor.detail.type'),
      key: 'container_type',
      sorter: (a, b) => a.type.slice(0, 1).charCodeAt(0) - b.type.slice(0, 1).charCodeAt(0),
      render: (_, record) => record.type,
    },
    {
      title: t('serviceMonitor.detail.status'),
      key: 'container_status',
      filters: [{
        text: 'True',
        value: 'True',
      },
      {
        text: 'False',
        value: 'False',
      }],
      showSorterTooltip: { target: 'full-header' },
      width: 220,
      onFilter: (value, record) => record.status.indexOf(value) === 0,
      sorter: (a, b) => a.status.slice(0, 1).charCodeAt(0) - b.status.slice(0, 1).charCodeAt(0),
      render: (_, record) => <div className={'status_group'}>
        <span className={record.status === 'True' ? 'running_circle' : 'pending_circle'}></span>
        <span>{record.status}</span>
      </div>,
    },
    {
      title: t('serviceMonitor.detail.updateTime'),
      key: 'container_updatedTime',
      sorter: (a, b) => Dayjs(a.lastTransitionTime) - Dayjs(b.lastTransitionTime),
      render: (_, record) => Dayjs(record.lastTransitionTime).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: t('serviceMonitor.detail.reason'),
      key: 'container_reason',
      render: (_, _record) => serviceMonitorDetailDataProps?.status?.reason || '--',
    },
    {
      title: t('serviceMonitor.detail.message'),
      key: 'container_message',
      render: (_, _record) => serviceMonitorDetailDataProps?.status?.message || '--',
    },
  ];

  const monitorGoalBaseInfoItems = [
    { key: t('serviceMonitor.detail.name'), value: serviceMonitorDetailDataProps?.metadata?.name },
    { key: t('serviceMonitor.detail.namespace'), value: serviceMonitorDetailDataProps?.metadata?.namespace },
    { key: t('serviceMonitor.detail.createTime'), value: Dayjs(serviceMonitorDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ];

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card">
        <h3>{t('serviceMonitor.detail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={monitorGoalBaseInfoItems} columns={1} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('serviceMonitor.detail.labels')}</p>
              {!disabledModifyMonitorServiceCr.includes(serviceMonitorName) && <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />}
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={oldLabels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {oldLabels?.length ? oldLabels?.map(item =>
                <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('serviceMonitor.detail.labelCount', { count: 0 })}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('serviceMonitor.detail.annotations')}</p>
              {!disabledModifyMonitorServiceCr.includes(serviceMonitorName) && <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsMonitorGoalManageAnnotationModalOpen(true)} />}
            </div>
            <AnnotationModal open={isMonitorGoalManageAnnotationModalOpen} type="annotation" dataList={oldAnnotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {oldAnnotations?.length ? oldAnnotations?.map(item =>
                <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('serviceMonitor.detail.labelCount', { count: 0 })}</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="detail_card container_card_add">
        <h3>{t('serviceMonitor.detail.statusInfo')}</h3>
        <Table className="table_padding" style={{ paddingRight: '0px' }} dataSource={containerData} columns={containerColumns}
          pagination={{
            className: 'page',
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showTotal: total => t('table.total', { total }),
          }} />
      </div>
    </div>
  </Fragment>;
}
