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

export default function RoleInfo({ roleName, roleNamespace, roleDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['rbac', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const [isRoleDetailAnnotationModalOpen, setIsRoleDetailAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const oldAnnotations = roleDetailDataProps?.metadata?.annotations;
  const oldLabels = roleDetailDataProps?.metadata?.labels;
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleRoleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsRoleDetailAnnotationModalOpen(false);
    } else {
      const roleDetailKeyArr = [];
      data.map(item => roleDetailKeyArr.push(item.key));
      if (roleDetailKeyArr.filter((item, index) => roleDetailKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('role', roleNamespace, roleName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsRoleDetailAnnotationModalOpen(false);
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
  const handleRoleAnnotationCancel = () => {
    setIsRoleDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleRoleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsLabelModalOpen(false);
    } else {
      const roleDetailAnnKeyArr = [];
      data.map(item => roleDetailAnnKeyArr.push(item.key));
      if (roleDetailAnnKeyArr.filter((item, index) => roleDetailAnnKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('role', roleNamespace, roleName, addLabelList);
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

  // 标签失败回调
  const handleRoleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const roleBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: roleDetailDataProps?.metadata?.name },
    { key: t('fields.roleType'), value: t('role.roleType') },
    { key: t('fields.namespace'), value: roleDetailDataProps?.metadata?.namespace },
    { key: t('fields.createdAt'), value: Dayjs(roleDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [roleDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={roleBaseInfoItems} columns={1} />
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.labels')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isLabelModalOpen} type="label" dataList={roleDetailDataProps.metadata?.labels} callbackOk={handleRoleLabelOk} callbackCancel={handleRoleLabelCancel} />
            <div className="key_value">
              {roleDetailDataProps.metadata?.labels?.length ?
                roleDetailDataProps.metadata?.labels?.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsRoleDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isRoleDetailAnnotationModalOpen} type="annotation" dataList={roleDetailDataProps?.metadata?.annotations} callbackOk={handleRoleAnnotationOk} callbackCancel={handleRoleAnnotationCancel} />
            <div className="key_value">
              {roleDetailDataProps.metadata?.annotations?.length ?
                roleDetailDataProps.metadata?.annotations?.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  </Fragment>;
}