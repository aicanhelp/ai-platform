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
import AnnotationModal from '@/components/AnnotationModal';
import { Table, Tag, message } from 'antd';
import { getJobStatus, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import ToolTipComponent from '@/components/ToolTipComponent';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function JobInfo({ jobName, jobNamespace, jobDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation('common');
  const [isShow, setIsShow] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [isJobDetailAnnotationModalOpen, setIsJobDetailAnnotationModalOpen] = useState(false);
  const [isJobDetailLabelModalOpen, setIsJobDetailLabelModalOpen] = useState(false);
  
  const oldJobDetailAnnotations = jobDetailDataProps?.metadata?.annotations ?? [];
  const oldJobDetailLabels = jobDetailDataProps?.metadata?.labels ?? [];
  const podData = jobDetailDataProps?.spec?.template?.spec?.containers ?? [];
  const conditionsData = jobDetailDataProps.status?.conditions || [];
  const themeStore = useStore('theme');

  const jobBaseInfoItems = useMemo(() => {
    const j = jobDetailDataProps;
    return [
      { key: t('resourceDetail.workloadName'), value: j?.metadata.name, descriptionClassName: 'jobDetail' },
      { key: t('resourceDetail.workloadType'), value: j?.kind },
      { key: t('resourceDetail.namespace'), value: j?.metadata.namespace },
      { key: t('resourceDetail.createdAt'), value: Dayjs(j?.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    ];
  }, [jobDetailDataProps, t, i18n.language]);

  const jobStatusInfoItems = useMemo(() => {
    const j = jobDetailDataProps;
    const statusText = getJobStatus(j.status);
    return [
      { key: t('resourceDetail.status'), value: <span className={`${statusText.toLowerCase()}_circle`}>{statusText}</span> },
      {
        key: t('resourceDetail.completionTime'),
        value: j.status.completionTime ? Dayjs(j.status.completionTime).format('YYYY-MM-DD HH:mm') : undefined,
      },
      { key: t('resourceDetail.startTime'), value: j?.status?.startTime ? Dayjs(j.status.startTime).format('YYYY-MM-DD HH:mm') : '--' },
    ];
  }, [jobDetailDataProps, t, i18n.language]);

  const handleShowChange = () => {
    setIsShow(false);
  };

  // 注解成功回调
  const handleAnnotationOk = async (jobDetailTemporyData) => {
    if (JSON.stringify(oldJobDetailAnnotations) === JSON.stringify(jobDetailTemporyData)) {
      messageApi.info(t('messages.annotationUnchanged', { ns: 'common' }));
      setIsJobDetailAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      jobDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldJobDetailAnnotations, jobDetailTemporyData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('job', jobNamespace, jobName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsJobDetailAnnotationModalOpen(false);
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
    setIsJobDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (jobDetailTemporyData) => {
    if (JSON.stringify(oldJobDetailLabels) === JSON.stringify(jobDetailTemporyData)) {
      messageApi.info(t('messages.labelUnchanged', { ns: 'common' }));
      setIsJobDetailLabelModalOpen(false);
    } else {
      const keyArr = [];
      jobDetailTemporyData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的标签
        const addLabelList = solveAnnotationOrLabelDiff(oldJobDetailLabels, jobDetailTemporyData, 'label');
        try {
          const res = await editAnnotationsOrLabels('job', jobNamespace, jobName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsJobDetailLabelModalOpen(false);
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
    setIsJobDetailLabelModalOpen(false);
  };

  const podColumns = useMemo(() => [
    {
      title: t('common.name'),
      key: 'pod_name',
      render: (_, record) => record.name,
    },
    {
      title: t('resourceDetail.image'),
      key: 'pod_image',
      render: (_, record) => record.image,
    },
    {
      title: t('resourceDetail.resourceLimits'),
      key: 'pod_resources',
      render: (_, record) => <div style={{ display: 'flex', alignItems: 'center' }}>
        {(record.resources && record.resources.requests) ?
          Object.entries(record.resources.requests).map(([resKey, resVal]) => (
            <div className="label_item_table" key={resKey}>
              <Tag
                color={themeStore.$s.theme === 'light' ? '#cce6ff' : '#234c9e'}
                className="label_tag_key jobDetail"
              >
                {resKey}:{resVal}
              </Tag>
            </div>
          )) : '--'
        }
      </div>,
    },
    {
      title: t('resourceDetail.port'),
      key: 'pod_port',
      render: (_, record) => {
        const ports = (record.ports || []).map(item => item?.containerPort).filter(port => port !== undefined && port !== null && port !== '');
        return ports.length ? ports.join(', ') : '--';
      },
    },
  ], [t, i18n.language, themeStore.$s.theme]);

  const containerColumns = useMemo(() => [
    {
      title: t('resourceDetail.conditionType'),
      key: 'container_type',
      render: (_, record) => record.type,
    },
    {
      title: t('common.status'),
      key: 'container_status',
      render: (_, record) => <div className="status_group">
        <span className={record.status === 'True' ? 'running_circle' : 'pending_circle'}></span>
        <span>{record.status}</span>
      </div>,
    },
    {
      title: t('resourceDetail.updatedAtColon'),
      key: 'container_updatedTime',
      render: (_, record) => Dayjs(record.lastTransitionTime).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: t('resourceDetail.reason'),
      key: 'container_reason',
      render: (_, _record) => jobDetailDataProps.status.reason || '--',
    },
    {
      title: t('resourceDetail.messageColon'),
      key: 'container_message',
      render: (_, _record) => jobDetailDataProps.status.message || '--',
    },
  ], [t, i18n.language, jobDetailDataProps.status.reason, jobDetailDataProps.status.message]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={`${isShow ? '' : 'cant_show'}`}>
      <ToolTipComponent>
        <div className="jobDetail" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className='promot-box'>
            {t('resourceDetail.monitorBanner')}
            <Link to={`/${containerRouterPrefix}/monitor/monitorDashboard`}>{t('resourceDetail.monitorLink')}</Link>
            {t('resourceDetail.monitorSuffix')}
          </div>
          <CloseOutlined onClick={handleShowChange} />
        </div>
      </ToolTipComponent>
    </div>
    <div className={`jobDetail tab_container container_margin_box ${isShow ? 'tooltip_container_height' : 'normal_container_height'}`}>
      <div className="detail_card">
        <h3>{t('resourceDetail.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={jobBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.labelsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsJobDetailLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isJobDetailLabelModalOpen} type="label" dataList={jobDetailDataProps?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {jobDetailDataProps.metadata?.labels?.length ?
                jobDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('resourceDetail.annotationsHeading')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsJobDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isJobDetailAnnotationModalOpen} type="annotation" dataList={jobDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {jobDetailDataProps.metadata?.annotations?.length ?
                jobDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('resourceDetail.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add jobDetail">
        <h3>{t('resourceDetail.jobStatusTitle')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={jobStatusInfoItems} columns={2} />
        </div>
      </div>

      <div className="detail_card container_card_add">
        <h3>{t('resourceDetail.podHeading')}</h3>
        <Table className="table_padding jodDetail" style={{ paddingRight: '0px' }} dataSource={podData} rowKey={(record) => record.name ?? record.image} columns={podColumns}
          pagination={{
            className: 'page',
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showTotal: total => t('resourceDetail.paginationTotal', { total }),
          }} />
      </div>

      <div className="detail_card container_card_add">
        <h3>{t('resourceDetail.conditionStatusTitle')}</h3>
        <Table className="table_padding" style={{ paddingRight: '0px' }} dataSource={conditionsData} rowKey={(record) => record.type ?? record.status} columns={containerColumns}
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