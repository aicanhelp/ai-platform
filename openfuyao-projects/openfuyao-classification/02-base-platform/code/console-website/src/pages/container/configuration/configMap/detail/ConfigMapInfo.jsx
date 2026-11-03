/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { message } from 'antd';
import '@/styles/pages/configuration.less';
import { useMemo, useRef, useState, useStore } from 'openinula';
import { updateConfigMapsLabelAnnotation } from '@/api/containerApi';
import Dayjs from 'dayjs';
import { EditOutlined } from '@ant-design/icons';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import { ResponseCode } from '@/common/constants';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import { useTranslation } from 'react-i18next';

export default function ConfigMapInfo({
  configurationData,
  configurationNameSpace,
  dataInfo,
  configurationName,
  refreshFn,
}) {
  const { t, i18n } = useTranslation(['configuration', 'common']);
  const childCodeMirrorRef = useRef(null);
  const oldAnnotations = configurationData?.metadata?.annotations ?? [];
  const oldLabels = configurationData?.metadata?.labels ?? [];
  const themeStore = useStore('theme');
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  if (!configurationData?.metadata) {
    return null;
  }

  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsLabelModalOpen(false);
      return;
    }
    const keyArrDetailLabel = [];
    data.map(item => keyArrDetailLabel.push(item.key));
    if (keyArrDetailLabel.filter((item, index) => keyArrDetailLabel.indexOf(item) !== index).length) {
      messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
    } else {
      const addLabelListDetail = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
      try {
        await updateConfigMapsLabelAnnotation(configurationNameSpace, addLabelListDetail, configurationName);
        showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
          refreshFn();
          setIsLabelModalOpen(false);
        });
      } catch (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          forbiddenMsg(messageApi, error);
        } else {
          messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
        }
      }
    }
  };

  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsAnnotationModalOpen(false);
      return;
    }
    const keyArrDetailAnnotation = [];
    data.map(item => keyArrDetailAnnotation.push(item.key));
    if (keyArrDetailAnnotation.filter((item, index) => keyArrDetailAnnotation.indexOf(item) !== index).length) {
      messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      return;
    }
    const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
    try {
      await updateConfigMapsLabelAnnotation(configurationNameSpace, addAnnotationList, configurationName);
      showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
        refreshFn();
        setIsAnnotationModalOpen(false);
      });
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const configMapBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: configurationData?.metadata?.name },
    { key: t('fields.namespace'), value: configurationData?.metadata?.namespace },
    { key: t('fields.createdAt'), value: Dayjs(configurationData?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm:ss') },
  ], [configurationData, t, i18n.language]);

  return (
    <div className='tab_container container_margin_box normal_container_height ConfigurationDetail'>
      {contextHolder}
      <div className='detail_card' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <h3>{t('info.basicInfo')}</h3>
        <div className='detail_info_box ConfigurationDetail'>
          <DetailBaseInfoList items={configMapBaseInfoItems} columns={1} />
          <div className='annotation ConfigurationDetail'>
            <div className='ann_title'>
              <p>{t('info.labels')}</p>
              <EditOutlined className='primary_icon common_antd_icon ConfigurationDetail' onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type='label' dataList={configurationData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={() => setIsLabelModalOpen(false)} />
            <div className='key_value'>
              {configurationData.metadata?.labels?.length ?
                configurationData.metadata.labels.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={configurationData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={() => setIsAnnotationModalOpen(false)} />
            <div className="key_value">
              {configurationData.metadata?.annotations?.length ?
                configurationData.metadata.annotations.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
        <h3>{t('info.data')}</h3>
        <div className='detail_info_box'>
          {dataInfo && (
            <CodeMirrorEditor ref={childCodeMirrorRef} yamlData={dataInfo} isEdit={false} />
          )}
        </div>
      </div>
    </div>
  );
}
