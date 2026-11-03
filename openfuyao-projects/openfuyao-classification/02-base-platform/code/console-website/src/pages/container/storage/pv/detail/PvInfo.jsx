/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Fragment, useState, useStore, useMemo } from 'openinula';
import { ResponseCode } from '@/common/constants';
import Dayjs from 'dayjs';
import { EditOutlined } from '@ant-design/icons';
import { message } from 'antd';
import AnnotationModal from '@/components/AnnotationModal';
import { editPvLabelOrAnnotation } from '@/api/containerApi';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import '@/styles/pages/podDetail.less';
import { useTranslation } from 'react-i18next';

export default function PvInfo({ pvDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['storage', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const oldAnnotations = pvDetailDataProps.metadata?.annotations ?? [];

  const oldLabels = pvDetailDataProps.metadata?.labels ?? [];

  const themeStore = useStore('theme');

  const [isPvLabelModalOpen, setIsPvLabelModalOpen] = useState(false);
  const [isPvAnnotationModalOpen, setIsPvAnnotationModalOpen] = useState(false);

  const handlePvLabelOk = async (data) => {
    const pvLabKeyArr = [];
    data.map(item => pvLabKeyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsPvLabelModalOpen(false);
    } else {
      if (pvLabKeyArr.filter((item, index) => pvLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editPvLabelOrAnnotation(pvDetailDataProps.metadata.name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsPvLabelModalOpen(false);
            });
          }
        } catch (e) {
          if (e.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, e);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: e.response?.data?.message }));
          }
        }
      }
    }
  };

  const handlePvLabelCancel = () => {
    setIsPvLabelModalOpen(false);
  };

  const handlePvAnnotationOk = async (data) => {
    const pvAnnKeyArr = [];
    data.map(item => pvAnnKeyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsPvAnnotationModalOpen(false);
    } else {
      if (pvAnnKeyArr.filter((item, index) => pvAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editPvLabelOrAnnotation(pvDetailDataProps.metadata.name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsPvAnnotationModalOpen(false);
            });
          }
        } catch (e) {
          if (e.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, e);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: e.response?.data?.message }));
          }
        }
      }
    }
  };

  const handlePvAnnotationCancel = () => {
    setIsPvAnnotationModalOpen(false);
  };

  const pvBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: pvDetailDataProps?.metadata?.name },
    { key: t('fields.createdAt'), value: Dayjs(pvDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    { key: t('fields.status'), value: pvDetailDataProps?.status?.phase },
    { key: t('fields.reclaimPolicy'), value: pvDetailDataProps?.spec?.persistentVolumeReclaimPolicy },
    { key: t('fields.capacity'), value: pvDetailDataProps?.spec?.capacity?.storage },
    { key: t('fields.accessModes'), value: pvDetailDataProps?.spec?.accessModes },
    { key: t('fields.storageClass'), value: pvDetailDataProps?.spec?.storageClassName },
    { key: t('fields.boundPvc'), value: pvDetailDataProps?.spec?.claimRef?.name },
  ], [pvDetailDataProps, t, i18n.language]);

  return <Fragment>
    {contextHolder}
    <div className="tab_container container_margin_box normal_container_height">
      <div className="detail_card" style={{ padding: '32px 32px 0 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div>
          <h3>{t('info.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={pvBaseInfoItems} columns={2} />
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.labels')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsPvLabelModalOpen(true)} />
              </div>
              <AnnotationModal open={isPvLabelModalOpen} type="label" dataList={pvDetailDataProps?.metadata?.labels} callbackOk={handlePvLabelOk} callbackCancel={handlePvLabelCancel} />
              <div className="key_value">
                {pvDetailDataProps.metadata?.labels?.length ?
                  pvDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.annotations')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsPvAnnotationModalOpen(true)} />
              </div>
              <AnnotationModal open={isPvAnnotationModalOpen} type="annotation" dataList={pvDetailDataProps?.metadata?.annotations} callbackOk={handlePvAnnotationOk} callbackCancel={handlePvAnnotationCancel} />
              <div className="key_value">
                {pvDetailDataProps.metadata?.annotations?.length ?
                  pvDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </Fragment>;
}