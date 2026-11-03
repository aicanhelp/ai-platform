/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useMemo, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import Dayjs from 'dayjs';
import AnnotationModal from '@/components/AnnotationModal';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { EditOutlined } from '@ant-design/icons';
import { Progress, message } from 'antd';
import { ResponseCode } from '@/common/constants';
import { solveAnnotationOrLabelDiff, solveResourceEcharts, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function ResourceQuotaInfo({ resourceQuotaName, resourceQuotaNamespace, resourceQuotaDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['namespace', 'common']);
  const [messageApi, contextHolder] = message.useMessage();
  const [isResourceQuotaAnnotationModalOpen, setIsResourceQuotaAnnotationModalOpen] = useState(false);
  const [isResourceQuotaLabelModalOpen, setIsResourceQuotaLabelModalOpen] = useState(false);
  const resourceQuotaEchartsData = solveResourceEcharts(resourceQuotaDetailDataProps?.status);
  const oldAnnotations = resourceQuotaDetailDataProps?.metadata?.annotations || [];
  const oldLabels = resourceQuotaDetailDataProps?.metadata?.labels || [];
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleResourceQuotaAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsResourceQuotaAnnotationModalOpen(false);
    } else {
      const resourceQuotaAnnKeyArr = [];
      data.map(item => resourceQuotaAnnKeyArr.push(item.key));
      if (resourceQuotaAnnKeyArr.filter((item, index) => resourceQuotaAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addResourceQuotaAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('resourcequota', resourceQuotaNamespace, resourceQuotaName, addResourceQuotaAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsResourceQuotaAnnotationModalOpen(false);
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
  const handleResourceQuotaAnnotationCancel = () => {
    setIsResourceQuotaAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleResourceQuotaLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsResourceQuotaLabelModalOpen(false);
    } else {
      const resourceQuotaLabelKeyArr = [];
      data.map(item => resourceQuotaLabelKeyArr.push(item.key));
      if (resourceQuotaLabelKeyArr.filter((item, index) => resourceQuotaLabelKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addResourceQuotaLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('resourcequota', resourceQuotaNamespace, resourceQuotaName, addResourceQuotaLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsResourceQuotaLabelModalOpen(false);
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
  const handleResourceQuotaLabelCancel = () => {
    setIsResourceQuotaLabelModalOpen(false);
  };

  const resourceQuotaBaseInfoItems = useMemo(() => [
    { key: `${t('fields.name')}：`, value: resourceQuotaDetailDataProps?.metadata?.name },
    { key: `${t('fields.namespace')}：`, value: resourceQuotaDetailDataProps?.metadata?.namespace },
    { key: `${t('fields.createdAt')}：`, value: Dayjs(resourceQuotaDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [resourceQuotaDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={resourceQuotaBaseInfoItems} columns={1} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.labels')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsResourceQuotaLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isResourceQuotaLabelModalOpen} type="label" dataList={oldLabels} callbackOk={handleResourceQuotaLabelOk} callbackCancel={handleResourceQuotaLabelCancel} />
            <div className="key_value">
              {oldLabels?.length ? oldLabels.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsResourceQuotaAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isResourceQuotaAnnotationModalOpen} type="annotation" dataList={oldAnnotations} callbackOk={handleResourceQuotaAnnotationOk} callbackCancel={handleResourceQuotaAnnotationCancel} />
            <div className="key_value">
              {oldAnnotations?.length ? oldAnnotations.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
      <div className="detail_card container_card_add">
        <h3>{t('info.quotaDetails')}</h3>
        <div className="echarts_list">
          {resourceQuotaEchartsData.map(item => (
            <div className="echarts_item" key={item.title}>
              <p className="ecahrts_title">{item.title}</p>
              <Progress
                type="circle"
                percent={item.rate}
                strokeColor={item.rate >= 100 ? '#e7434a' : '#09aa71'}
                format={(_percent) => `${item.rate}%`}></Progress>
              <div className="word_flex">
                <p>
                  <span>{t('info.usedQuota')}</span>
                  <span>{item.used}</span>
                </p>
                <p>
                  <span>{t('info.quotaLimit')}</span>
                  <span>{item.max}</span>
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  </Fragment>;
}
