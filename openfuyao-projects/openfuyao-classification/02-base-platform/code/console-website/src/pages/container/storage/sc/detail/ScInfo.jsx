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
import { editScLabelOrAnnotation } from '@/api/containerApi';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import '@/styles/pages/podDetail.less';
import { useTranslation } from 'react-i18next';

export default function ScInfo({ scDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['storage', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const oldAnnotations = scDetailDataProps.metadata?.annotations ?? [];

  const oldLabels = scDetailDataProps.metadata?.labels ?? [];

  const [isScLabelModalOpen, setIsScLabelModalOpen] = useState(false);
  const [isScAnnotationModalOpen, setIsScAnnotationModalOpen] = useState(false);

  const themeStore = useStore('theme');

  const handleScLabelOk = async (data) => {
    const scLabKeyArr = [];
    data.map(item => scLabKeyArr.push(item.key));
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsScLabelModalOpen(false);
    } else {
      if (scLabKeyArr.filter((item, index) => scLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addScLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editScLabelOrAnnotation(scDetailDataProps.metadata.name, addScLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsScLabelModalOpen(false);
            });
          }
        } catch (scError) {
          if (scError.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, scError);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: scError.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleScLabelCancel = () => {
    setIsScLabelModalOpen(false);
  };

  const handleScAnnotationOk = async (data) => {
    const scAnnKeyArr = [];
    data.map(item => scAnnKeyArr.push(item.key));
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsScAnnotationModalOpen(false);
    } else {
      if (scAnnKeyArr.filter((item, index) => scAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addScAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editScLabelOrAnnotation(scDetailDataProps.metadata.name, addScAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsScAnnotationModalOpen(false);
            });
          }
        } catch (scError) {
          if (scError.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, scError);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: scError.response?.data?.message }));
          }
        }
      }
    }
  };

  const handleScAnnotationCancel = () => {
    setIsScAnnotationModalOpen(false);
  };

  const getScVolumeExpansionData = (type) => {
    switch (type) {
      case true:
        return t('values.yes');
      case false:
        return t('values.no');
      default:
        return '--';
    }
  };

  const scBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: scDetailDataProps?.metadata?.name },
    { key: t('fields.createdAt'), value: Dayjs(scDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
    { key: t('fields.reclaimPolicy'), value: scDetailDataProps?.reclaimPolicy },
    { key: t('fields.volumeBindingMode'), value: scDetailDataProps?.volumeBindingMode },
    { key: t('fields.provisioner'), value: scDetailDataProps?.provisioner },
    { key: t('fields.allowVolumeExpansion'), value: scDetailDataProps?.allowVolumeExpansion !== undefined ? getScVolumeExpansionData(scDetailDataProps.allowVolumeExpansion) : undefined },
  ], [scDetailDataProps, t, i18n.language]);

  return <Fragment>
    {contextHolder}
    <div className="tab_container container_margin_box normal_container_height">
      <div className="detail_card" style={{ padding: '32px 32px 0 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div>
          <h3>{t('info.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={scBaseInfoItems} columns={2} />

            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.labels')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsScLabelModalOpen(true)} />
              </div>
              <AnnotationModal open={isScLabelModalOpen} type="label" dataList={scDetailDataProps?.metadata?.labels} callbackOk={handleScLabelOk} callbackCancel={handleScLabelCancel} />
              <div className="key_value">
                {scDetailDataProps.metadata?.labels?.length ?
                  scDetailDataProps.metadata.labels.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
            <div className="annotation">
              <div className="ann_title">
                <p>{t('info.annotations')}</p>
                <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsScAnnotationModalOpen(true)} />
              </div>
              <AnnotationModal open={isScAnnotationModalOpen} type="annotation" dataList={scDetailDataProps?.metadata?.annotations} callbackOk={handleScAnnotationOk} callbackCancel={handleScAnnotationCancel} />
              <div className="key_value">
                {scDetailDataProps.metadata?.annotations?.length ?
                  scDetailDataProps.metadata.annotations.map(item => <LabelTag labelKey={item.key} labelValue={item.value} />) :
                  <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </Fragment>;
}