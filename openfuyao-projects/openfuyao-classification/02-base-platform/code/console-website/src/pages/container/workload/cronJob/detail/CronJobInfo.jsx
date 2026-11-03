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
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant';
import { Fragment, useState, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { CloseOutlined, EditOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import AnnotationModal from '@/components/AnnotationModal';
import { Table, Tag, message } from 'antd';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import ToolTipComponent from '@/components/ToolTipComponent';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function CronJobInfo({ cronJobName, cronJobNamespace, cronJobDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation('common');
  const [isShow, setIsShow] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [isCronJobDetailAnnotationModalOpen, setIsCronJobDetailAnnotationModalOpen] = useState(false);
  const [isCronJobDetailLabelModalOpen, setIsCronJobDetailLabelModalOpen] = useState(false);
  
  const oldCronJobDetailLabels = cronJobDetailDataProps?.metadata?.labels ?? [];
  const oldCronJobDetailAnnotations = cronJobDetailDataProps?.metadata?.annotations ?? [];
  const podData = cronJobDetailDataProps?.spec?.jobTemplate?.spec?.template?.spec?.containers ?? [];
  const themeStore = useStore('theme');

  const cronJobBaseInfoItems = useMemo(() => {
    const c = cronJobDetailDataProps;
    const startingSec = c?.spec?.startingDeadlineSeconds;
    return [
      { key: t('resourceDetail.workloadName'), value: c?.metadata.name, descriptionClassName: 'cornjobDetail' },
      { key: t('resourceDetail.lastScheduleTime'), value: c?.status?.lastScheduleTime ? Dayjs(c.status.lastScheduleTime).format('YYYY-MM-DD HH:mm') : '--' },
      { key: t('resourceDetail.scheduleColon'), value: c?.spec.schedule },
      { key: t('resourceDetail.concurrencyPolicyColon'), value: c?.spec.concurrencyPolicy },
      { key: t('resourceDetail.namespace'), value: c?.metadata.namespace },
      { key: t('resourceDetail.createdAt'), value: Dayjs(c.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') },
      { key: t('resourceDetail.suspendColon'), value: c?.spec.suspend ? 'true' : 'false' },
      { key: t('resourceDetail.startingDeadline'), value: startingSec !== undefined ? t('resourceDetail.secondsCount', { count: startingSec }) : undefined },
    ];
  }, [cronJobDetailDataProps, t, i18n.language]);

  const handleShowChange = () => {
    setIsShow(false);
  };
  // 注解成功回调
  const handleAnnotationOk = async (cronJobTemporyData) => {
    if (JSON.stringify(oldCronJobDetailAnnotations) === JSON.stringify(cronJobTemporyData)) {
      messageApi.info(t('messages.annotationUnchanged', { ns: 'common' }));
      setIsCronJobDetailAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      cronJobTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldCronJobDetailAnnotations, cronJobTemporyData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('cronjob', cronJobNamespace, cronJobName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsCronJobDetailAnnotationModalOpen(false);
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
    setIsCronJobDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (cronJobTemporyData) => {
    if (JSON.stringify(oldCronJobDetailLabels) === JSON.stringify(cronJobTemporyData)) {
      messageApi.info(t('messages.labelUnchanged', { ns: 'common' }));
      setIsCronJobDetailLabelModalOpen(false);
    } else {
      const keyArr = [];
      cronJobTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldCronJobDetailLabels, cronJobTemporyData, 'label');
        try {
          const res = await editAnnotationsOrLabels('cronjob', cronJobNamespace, cronJobName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsCronJobDetailLabelModalOpen(false);
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
    setIsCronJobDetailLabelModalOpen(false);
  };

  const podColumns = useMemo(() => [
    {
      key: 'pod_name',
      title: t('common.name'),
      render: (_, record) => record.name,
    },
    {
      key: 'pod_image',
      title: t('resourceDetail.image'),
      render: (_, record) => record.image,
    },
    {
      key: 'pod_resources',
      title: t('resourceDetail.resourceLimits'),
      render: (_, record) => <div style={{ display: 'flex', alignItems: 'center' }}>
        {(record.resources && record.resources.requests) ?
          Object.entries(record.resources.requests).map(([resKey, resVal]) => (
            <div className="label_item_table cornjobDetail" key={resKey}>
              <Tag
                color={themeStore.$s.theme === 'light' ? '#cce6ff' : '#234c9e'}
                className="label_tag_key cornjobDetail"
              >
                {resKey}:{resVal}
              </Tag>
            </div>
          )) : '--'
        }
      </div>,
    },
    {
      key: 'pod_port',
      title: t('resourceDetail.port'),
      render: (_, record) => {
        const ports = (record.ports || []).map(item => item?.containerPort).filter(port => port !== undefined && port !== null && port !== '');
        return ports.length ? ports.join(', ') : '--';
      },
    },
  ], [t, i18n.language, themeStore.$s.theme]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={`${isShow ? '' : 'cant_show'}`}>
      <ToolTipComponent>
        <div className="cornjobDetail" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className='promot-box'>
            {t('resourceDetail.monitorBanner')}
            <Link to={`/${containerRouterPrefix}/monitor/monitorDashboard`}>{t('resourceDetail.monitorLink')}</Link>
            {t('resourceDetail.monitorSuffix')}
          </div>
          <CloseOutlined onClick={handleShowChange} />
        </div>
      </ToolTipComponent>
    </div>
    <div className={`cornjobDetail tab_container container_margin_box ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
      <div className="detail_card">
        <h3>{t('resourceDetail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={cronJobBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.labelsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsCronJobDetailLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isCronJobDetailLabelModalOpen} type="label" dataList={cronJobDetailDataProps?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {cronJobDetailDataProps.metadata?.labels?.length ?
                cronJobDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.annotationsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsCronJobDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isCronJobDetailAnnotationModalOpen} type="annotation" dataList={cronJobDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {cronJobDetailDataProps.metadata?.annotations?.length ?
                cronJobDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="detail_card container_card_add cornjobDetail">
        <h3>{t('resourceDetail.podHeading')}</h3>
        <Table className="table_padding cornjobDetail" style={{ paddingRight: '0px' }} dataSource={podData} rowKey={(record) => record.name ?? record.image} columns={podColumns}
          pagination={{
            className: 'page',
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showTotal: total => t('resourceDetail.paginationTotal', { total }),
          }} />
      </div>
    </div>
  </Fragment>;
}