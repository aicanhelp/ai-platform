/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant';
import { Fragment, useState, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { CloseOutlined, EditOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import { message } from 'antd';
import { solveAnnotationOrLabelDiff, getDaemonSetStatus, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import CpuIcon from '@/assets/images/cpuIcon.png';
import MemoryIcon from '@/assets/images/memoryIcon.png';
import AnnotationModal from '@/components/AnnotationModal';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import ToolTipComponent from '@/components/ToolTipComponent';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function DaemonSetInfo({ daemonSetName, daemonSetNamespace, daemonSetDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation('common');
  const [isShow, setIsShow] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [isDaemonSetDetailAnnotationModalOpen, setIsDaemonSetDetailAnnotationModalOpen] = useState(false);
  const [isDaemonSetDetailLabelModalOpen, setIsDaemonSetDetailLabelModalOpen] = useState(false);
  
  const oldDaemonSetDetailLabels = daemonSetDetailDataProps?.metadata?.labels ?? [];
  const oldDaemonSetDetailAnnotations = daemonSetDetailDataProps?.metadata?.annotations ?? [];
  const themeStore = useStore('theme');

  const daemonSetBaseInfoItems = useMemo(() => {
    const d = daemonSetDetailDataProps;
    const statusText = getDaemonSetStatus(d.status);
    return [
      { key: t('resourceDetail.workloadName'), value: d?.metadata.name, descriptionClassName: 'daemonsetDetail' },
      { key: t('resourceDetail.workloadType'), value: d?.kind },
      { key: t('resourceDetail.workloadStatus'), value: <span className={`${statusText.toLowerCase()}_circle`}>{statusText}</span> },
      { key: t('resourceDetail.createdAt'), value: Dayjs(d?.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') },
      { key: t('resourceDetail.namespace'), value: d?.metadata.namespace },
      {
        key: t('resourceDetail.instances'),
        value: `${d.status.currentNumberScheduled}/${d.status.desiredNumberScheduled}`,
        keyStyle: { width: 'auto' },
        valueStyle: { marginLeft: '12px' },
      },
      { key: t('resourceDetail.containerImage'), value: d.spec?.template?.spec?.containers[0]?.image },
    ];
  }, [daemonSetDetailDataProps, t, i18n.language]);

  const handleShowChange = () => {
    setIsShow(false);
  };

  // 注解成功回调
  const handleAnnotationOk = async (daemonSetDetailTemporyData) => {
    if (JSON.stringify(oldDaemonSetDetailAnnotations) === JSON.stringify(daemonSetDetailTemporyData)) {
      messageApi.info(t('messages.annotationUnchanged', { ns: 'common' }));
      setIsDaemonSetDetailAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      daemonSetDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldDaemonSetDetailAnnotations, daemonSetDetailTemporyData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('daemonset', daemonSetNamespace, daemonSetName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsDaemonSetDetailAnnotationModalOpen(false);
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
    setIsDaemonSetDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (daemonSetDetailTemporyData) => {
    if (JSON.stringify(oldDaemonSetDetailLabels) === JSON.stringify(daemonSetDetailTemporyData)) {
      messageApi.info(t('messages.labelUnchanged', { ns: 'common' }));
      setIsDaemonSetDetailLabelModalOpen(false);
    } else {
      const keyArr = [];
      daemonSetDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldDaemonSetDetailLabels, daemonSetDetailTemporyData, 'label');
        try {
          const res = await editAnnotationsOrLabels('daemonset', daemonSetNamespace, daemonSetName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsDaemonSetDetailLabelModalOpen(false);
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
    setIsDaemonSetDetailLabelModalOpen(false);
  };

  return <Fragment>
    <div className="daemonsetMessage" style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={`${isShow ? '' : 'cant_show'}`}>
      <ToolTipComponent>
        <div className="daemonsetDetail" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className='promot-box'>
            {t('resourceDetail.monitorBanner')}
            <Link to={`/${containerRouterPrefix}/monitor/monitorDashboard`}>{t('resourceDetail.monitorLink')}</Link>
            {t('resourceDetail.monitorSuffix')}
          </div>
          <CloseOutlined onClick={handleShowChange} />
        </div>
      </ToolTipComponent>
    </div>
    <div className={`daemonsetDetail tab_container container_margin_box ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
      <div className="detail_card">
        <h3>{t('resourceDetail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={daemonSetBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.labelsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsDaemonSetDetailLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isDaemonSetDetailLabelModalOpen} type="label" dataList={daemonSetDetailDataProps?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {daemonSetDetailDataProps.metadata?.labels?.length ?
                daemonSetDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.annotationsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsDaemonSetDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isDaemonSetDetailAnnotationModalOpen} type="annotation" dataList={daemonSetDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {daemonSetDetailDataProps.metadata?.annotations?.length ?
                daemonSetDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add daemonsetDetail">
        <h3>{t('resourceDetail.resourceUsage')}</h3>
        <div className="resource_list">
          <div className="resource_item">
            <img className="resource_icon" src={CpuIcon} />
            <div className="resource_word">
              <p>{t('resourceDetail.cpuCore')}</p>
              <p>{daemonSetDetailDataProps?.spec?.template?.spec?.containers[0]?.resources?.requests?.cpu || '--'}</p>
            </div>
          </div>

          <div className="resource_item">
            <img className="resource_icon" src={MemoryIcon} />
            <div className="resource_word">
              <p>{t('resourceDetail.memoryMB')}</p>
              <p>{daemonSetDetailDataProps?.spec?.template?.spec?.containers[0]?.resources?.requests?.memory || '--'}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}