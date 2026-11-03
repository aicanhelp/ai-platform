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
import { EditOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import { message } from 'antd';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import { useTranslation } from 'react-i18next';

export default function ServiceAccountInfo({ serviceAccountName, serviceAccountNamespace, serviceAccountDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['rbac', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const [isServiceAccountDetailAnnotationModalOpen, setIsServiceAccountDetailAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = serviceAccountDetailDataProps?.metadata?.annotations ?? [];
  const oldLabels = serviceAccountDetailDataProps?.metadata?.labels ?? [];
  const themeStore = useStore('theme');

  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsServiceAccountDetailAnnotationModalOpen(false);
    } else {
      const serviceAccountDetailKeyArr = [];
      data.map(item => serviceAccountDetailKeyArr.push(item.key));
      if (serviceAccountDetailKeyArr.filter((item, index) => serviceAccountDetailKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('serviceaccount', serviceAccountNamespace, serviceAccountName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsServiceAccountDetailAnnotationModalOpen(false);
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

  const handleAnnotationCancel = () => {
    setIsServiceAccountDetailAnnotationModalOpen(false);
  };

  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('serviceaccount', serviceAccountNamespace, serviceAccountName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsLabelModalOpen(false);
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

  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const serviceAccountBaseInfoItems = useMemo(() => [
    { key: t('fields.serviceAccountName'), value: serviceAccountDetailDataProps?.metadata?.name },
    { key: t('fields.namespace'), value: serviceAccountDetailDataProps?.metadata?.namespace },
    { key: t('fields.createdAt'), value: Dayjs(serviceAccountDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [serviceAccountDetailDataProps, t, i18n.language]);

  if (!serviceAccountDetailDataProps?.metadata) {
    return null;
  }

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={serviceAccountBaseInfoItems} columns={2} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.labels')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={serviceAccountDetailDataProps.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />
            <div className="key_value">
              {serviceAccountDetailDataProps.metadata?.labels?.length ?
                serviceAccountDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsServiceAccountDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isServiceAccountDetailAnnotationModalOpen} type="annotation" dataList={serviceAccountDetailDataProps?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />
            <div className="key_value">
              {serviceAccountDetailDataProps.metadata?.annotations?.length ?
                serviceAccountDetailDataProps.metadata.annotations.map(item =>
                  <LabelTag labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}