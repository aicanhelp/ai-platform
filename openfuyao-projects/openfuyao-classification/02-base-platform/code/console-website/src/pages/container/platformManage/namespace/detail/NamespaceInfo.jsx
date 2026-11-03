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
import { firstAlphabetUp, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import { EditOutlined } from '@ant-design/icons';
import { message } from 'antd';
import { editAnnotationsOrLabels } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import LabelTag from '@/components/LabelTag';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

export default function NamespaceInfo({ namespaceName, namespaceDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['namespace', 'common']);
  const [messageApi, contextHolder] = message.useMessage();
  // 对话框展示
  const [isNamespaceAnnotationModalOpen, setIsNamespaceAnnotationModalOpen] = useState(false);
  // 对话框展示
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = namespaceDetailDataProps?.metadata?.annotations || [];
  const oldLabels = namespaceDetailDataProps?.metadata?.labels || [];
  const themeStore = useStore('theme');
  // 注解成功回调
  const handleNamespaceAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsNamespaceAnnotationModalOpen(false);
    } else {
      const namespaceDetailAnnKeyArr = [];
      data.map(item => namespaceDetailAnnKeyArr.push(item.key));
      if (namespaceDetailAnnKeyArr.filter((item, index) => namespaceDetailAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addNamespaceAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('namespace', '', namespaceName, addNamespaceAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsNamespaceAnnotationModalOpen(false);
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
  const handleNamespaceAnnotationCancel = () => {
    setIsNamespaceAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleNamespaceLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsLabelModalOpen(false);
    } else {
      const namespaceDetailLabKeyArr = [];
      data.map(item => namespaceDetailLabKeyArr.push(item.key));
      if (namespaceDetailLabKeyArr.filter((item, index) => namespaceDetailLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addNamespaceLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('namespace', '', namespaceName, addNamespaceLabelList);
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
  // 标签失败回调
  const handleNamespaceLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const namespaceBaseInfoItems = useMemo(() => [
    { key: `${t('fields.name')}：`, value: namespaceDetailDataProps?.metadata?.name },
    {
      key: `${t('fields.status')}：`,
      value: <span className={`${namespaceDetailDataProps?.status?.phase === 'Active' ? 'running' : 'terminating'}_circle`}>
        {namespaceDetailDataProps?.status?.phase ? firstAlphabetUp(namespaceDetailDataProps.status.phase) : '--'}
      </span>,
    },
    { key: `${t('fields.createdAt')}：`, value: Dayjs(namespaceDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [namespaceDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className="detail_card">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={namespaceBaseInfoItems} columns={1} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.labels')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={oldLabels} callbackOk={handleNamespaceLabelOk} callbackCancel={handleNamespaceLabelCancel} />
            <div className="key_value">
              {oldLabels?.length ? oldLabels.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}：</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsNamespaceAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isNamespaceAnnotationModalOpen} type="annotation" dataList={oldAnnotations} callbackOk={handleNamespaceAnnotationOk} callbackCancel={handleNamespaceAnnotationCancel} />
            <div className="key_value">
              {oldAnnotations?.length ? oldAnnotations.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}
