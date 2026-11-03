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
import { updateSecretLabelAnnotation } from '@/api/containerApi';
import Dayjs from 'dayjs';
import { EditOutlined } from '@ant-design/icons';
import AnnotationModal from '@/components/AnnotationModal';
import { solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import LabelTag from '@/components/LabelTag';
import { ResponseCode } from '@/common/constants';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import { useTranslation } from 'react-i18next';

export default function SecretInfo({
  secretData,
  secretNameSpace,
  secretName,
  dataInfo,
  refreshFn,
}) {
  const { t, i18n } = useTranslation(['configuration', 'common']);
  const childCodeMirrorRef = useRef(null);
  const oldAnnotations = secretData?.metadata?.annotations ?? [];
  const oldLabels = secretData?.metadata?.labels ?? [];
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isSecretLabelModalOpen, setIsSecretLabelModalOpen] = useState(false);
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();

  if (!secretData?.metadata) {
    return null;
  }

  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsSecretLabelModalOpen(false);
      return;
    }
    const keyArrSecretDetailLable = [];
    data.map(item => keyArrSecretDetailLable.push(item.key));
    if (keyArrSecretDetailLable.filter((item, index) => keyArrSecretDetailLable.indexOf(item) !== index).length) {
      messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      return;
    }
    const addLabelListSecretDetail = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
    try {
      await updateSecretLabelAnnotation(secretNameSpace, addLabelListSecretDetail, secretName);
      showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
        refreshFn();
        setIsSecretLabelModalOpen(false);
      });
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsAnnotationModalOpen(false);
      return;
    }
    const keyArrSecretDetailAnnotation = [];
    data.map(item => keyArrSecretDetailAnnotation.push(item.key));
    if (keyArrSecretDetailAnnotation.filter((item, index) => keyArrSecretDetailAnnotation.indexOf(item) !== index).length) {
      messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      return;
    }
    const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
    try {
      await updateSecretLabelAnnotation(secretNameSpace, addAnnotationList, secretName);
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

  const secretBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: secretData?.metadata?.name },
    { key: t('fields.namespace'), value: secretData?.metadata?.namespace },
    { key: t('fields.createdAt'), value: Dayjs(secretData?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm:ss') },
  ], [secretData, t, i18n.language]);

  return (
    <div className='tab_container container_margin_box normal_container_height SecretDetail'>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='detail_card' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <h3>{t('info.basicInfo')}</h3>
        <div className='detail_info_box SecretDetail'>
          <DetailBaseInfoList items={secretBaseInfoItems} columns={1} />
          <div className='annotation SecretDetail'>
            <div className='ann_title'>
              <p>{t('info.labels')}</p>
              <EditOutlined className='primary_icon common_antd_icon SecretDetail' onClick={() => setIsSecretLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isSecretLabelModalOpen} type='label' dataList={secretData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={() => setIsSecretLabelModalOpen(false)} />
            <div className='key_value'>
              {secretData.metadata?.labels?.length ?
                secretData.metadata.labels.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isAnnotationModalOpen} type="annotation" dataList={secretData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={() => setIsAnnotationModalOpen(false)} />
            <div className="key_value">
              {secretData.metadata?.annotations?.length ?
                secretData.metadata.annotations.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
        <h3>{t('info.data')}</h3>
        <div className='detail_info_box'>
          {dataInfo && (
            <CodeMirrorEditor ref={childCodeMirrorRef} yamlData={dataInfo} isEdit={false} className='secret' />
          )}
        </div>
      </div>
    </div>
  );
}
