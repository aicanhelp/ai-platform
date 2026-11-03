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

export default function Detail({ roleBindName, roleBindDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['rbac', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const [isClusterRoleBindDetailAnnotationModalOpen, setIsClusterRoleBindDetailAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = roleBindDetailDataProps?.metadata?.annotations;
  const oldLabels = roleBindDetailDataProps?.metadata?.labels;
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleRoleBindAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsClusterRoleBindDetailAnnotationModalOpen(false);
    } else {
      const roleBindAnnKeyArr = [];
      data.map(item => roleBindAnnKeyArr.push(item.key));
      if (roleBindAnnKeyArr.filter((item, index) => roleBindAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addRoleBindAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('clusterrolebinding', '', roleBindName, addRoleBindAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsClusterRoleBindDetailAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editAnnotationFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleRoleBindAnnotationCancel = () => {
    setIsClusterRoleBindDetailAnnotationModalOpen(false);
  };

  // 标签失败回调
  const handleRoleBindLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  // 标签成功回调
  const handleRoleBindLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const roleBindLabKeyArr = [];
      data.map(item => roleBindLabKeyArr.push(item.key));
      if (roleBindLabKeyArr.filter((item, index) => roleBindLabKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addRoleBindLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('clusterrolebinding', '', roleBindName, addRoleBindLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(t('messages.editLabelFailed', { ns: 'common', message: error.response?.data?.message }));
          }
        }
      }
    }
  };

  const clusterRoleBindingBaseInfoItems = useMemo(() => [
    { key: t('fields.bindName'), value: roleBindDetailDataProps?.metadata?.name },
    { key: t('fields.bindingType'), value: t('roleBinding.clusterRoleBindingType') },
    { key: t('fields.createdAt'), value: Dayjs(roleBindDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [roleBindDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card clusterRoleBindDetail">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={clusterRoleBindingBaseInfoItems} columns={1} />
          <div className="annotation clusterRoleBindDetail">
            <div className="ann_title">
              <p>{t('info.labels')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={roleBindDetailDataProps.metadata?.labels} callbackOk={handleRoleBindLabelOk} callbackCancel={handleRoleBindLabelCancel} />
            <div className="key_value">
              {roleBindDetailDataProps.metadata?.labels?.length ?
                roleBindDetailDataProps.metadata?.labels?.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span className='clusterRoleBindDetail' style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation clusterRoleBindDetail">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsClusterRoleBindDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isClusterRoleBindDetailAnnotationModalOpen} type="annotation" dataList={roleBindDetailDataProps?.metadata?.annotations} callbackOk={handleRoleBindAnnotationOk} callbackCancel={handleRoleBindAnnotationCancel} />
            <div className="key_value">
              {roleBindDetailDataProps.metadata?.annotations?.length ?
                roleBindDetailDataProps.metadata?.annotations?.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span className='clusterRoleBindDetail' style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}