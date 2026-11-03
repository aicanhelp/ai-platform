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
import { editPvcLabelOrAnnotation } from '@/api/containerApi';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import '@/styles/pages/podDetail.less';
import { useTranslation } from 'react-i18next';

export default function PvcInfo({ pvcDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['storage', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const oldAnnotations = pvcDetailDataProps.metadata?.annotations ?? [];

  const oldLabels = pvcDetailDataProps.metadata?.labels ?? [];

  const [isPvcLabelModalOpen, setIsPvcLabelModalOpen] = useState(false);
  const [isPvcAnnotationModalOpen, setIsPvcAnnotationModalOpen] = useState(false);

  const themeStore = useStore('theme');

  const handlePvcLabelOk = async (data) => {
    const pvcLabKeyArr = [];
    data.map(item => pvcLabKeyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsPvcLabelModalOpen(false);
    } else {
      if (pvcLabKeyArr.filter((item, index) => pvcLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addPvcLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editPvcLabelOrAnnotation(pvcDetailDataProps.metadata.namespace, pvcDetailDataProps.metadata.name, addPvcLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsPvcLabelModalOpen(false);
            });
          }
        } catch (pvce) {
          if (pvce.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, pvce);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: pvce.response?.data?.message }));
          }
        }
      }
    }
  };

  const handlePvcLabelCancel = () => {
    setIsPvcLabelModalOpen(false);
  };

  const handlePvcAnnotationOk = async (data) => {
    const pvcAnnKeyArr = [];
    data.map(item => pvcAnnKeyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsPvcAnnotationModalOpen(false);
    } else {
      if (pvcAnnKeyArr.filter((item, index) => pvcAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addPvcAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editPvcLabelOrAnnotation(pvcDetailDataProps.metadata.namespace, pvcDetailDataProps.metadata.name, addPvcAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsPvcAnnotationModalOpen(false);
            });
          }
        } catch (pvce) {
          if (pvce.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, pvce);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: pvce.response?.data?.message }));
          }
        }
      }
    }
  };

  const handlePvcAnnotationCancel = () => {
    setIsPvcAnnotationModalOpen(false);
  };

  const pvcBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: pvcDetailDataProps?.metadata?.name },
    { key: t('fields.createdAt'), value: Dayjs(pvcDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    { key: t('fields.status'), value: pvcDetailDataProps?.status?.phase },
    { key: t('fields.namespace'), value: pvcDetailDataProps?.metadata?.namespace },
    { key: t('fields.capacity'), value: pvcDetailDataProps?.spec?.resources?.requests?.storage },
    { key: t('fields.accessModes'), value: pvcDetailDataProps?.spec?.accessModes },
    { key: t('fields.storageClass'), value: pvcDetailDataProps?.spec?.storageClassName },
  ], [pvcDetailDataProps, t, i18n.language]);

  return <Fragment>
    {contextHolder}
    <div className="tab_container container_margin_box tooltip_container_height">
      <div className="detail_card" style={{ padding: '32px 32px 0 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div>
          <h3>{t('info.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={pvcBaseInfoItems} columns={2} />
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.labels')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsPvcLabelModalOpen(true)} />
              </div>
              <AnnotationModal open={isPvcLabelModalOpen} type="label" dataList={pvcDetailDataProps?.metadata?.labels} callbackOk={handlePvcLabelOk} callbackCancel={handlePvcLabelCancel} />
              <div className="key_value">
                {pvcDetailDataProps.metadata?.labels?.length ?
                  pvcDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.annotations')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsPvcAnnotationModalOpen(true)} />
              </div>
              <AnnotationModal open={isPvcAnnotationModalOpen} type="annotation" dataList={pvcDetailDataProps?.metadata?.annotations} callbackOk={handlePvcAnnotationOk} callbackCancel={handlePvcAnnotationCancel} />
              <div className="key_value">
                {pvcDetailDataProps.metadata?.annotations?.length ?
                  pvcDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </Fragment>;
}