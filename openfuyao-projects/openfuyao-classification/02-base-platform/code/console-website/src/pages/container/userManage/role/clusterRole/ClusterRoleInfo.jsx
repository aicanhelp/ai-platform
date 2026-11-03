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

export default function ClusterRoleInfo({ roleName, roleDetailDataProps, refreshFn }) {
  const { t, i18n } = useTranslation(['rbac', 'common']);
  const [messageApi, contextHolder] = message.useMessage();

  const [isClusterRoleDetailAnnotationModalOpen, setIsClusterRoleDetailAnnotationModalOpen] = useState(false);
  const [isClusterRoleLabelModalOpen, setIsClusterRoleLabelModalOpen] = useState(false);
  const oldAnnotations = roleDetailDataProps?.metadata?.annotations;
  const oldLabels = roleDetailDataProps?.metadata?.labels;
  const themeStore = useStore('theme');

  // 注解成功回调
  const handleClusterRoleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsClusterRoleDetailAnnotationModalOpen(false);
    } else {
      const clusterDetailKeyArr = [];
      data.map(item => clusterDetailKeyArr.push(item.key));
      if (clusterDetailKeyArr.filter((item, index) => clusterDetailKeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('clusterrole', '', roleName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsClusterRoleDetailAnnotationModalOpen(false);
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
  const handleClusterRoleAnnotationCancel = () => {
    setIsClusterRoleDetailAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleClusterRoleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsClusterRoleLabelModalOpen(false);
    } else {
      const ClusterRolekeyArr = [];
      data.map(item => ClusterRolekeyArr.push(item.key));
      if (ClusterRolekeyArr.filter((item, index) => ClusterRolekeyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addClusterRoleLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels('clusterrole', '', roleName, addClusterRoleLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              refreshFn();
              setIsClusterRoleLabelModalOpen(false);
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
  const handleClusterRoleLabelCancel = () => {
    setIsClusterRoleLabelModalOpen(false);
  };

  const clusterRoleBaseInfoItems = useMemo(() => [
    { key: t('fields.name'), value: roleDetailDataProps?.metadata?.name },
    { key: t('fields.roleType'), value: roleDetailDataProps?.kind },
    { key: t('fields.createdAt'), value: Dayjs(roleDetailDataProps?.metadata?.creationTimestamp).format('YYYY-MM-DD HH:mm') },
  ], [roleDetailDataProps, t, i18n.language]);

  return <Fragment>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className={'tab_container container_margin_box normal_container_height'}>
      <div className="detail_card cluster_role_card1">
        <h3>{t('info.basicInfo')}</h3>
        <div className="detail_info_box">
          <DetailBaseInfoList items={clusterRoleBaseInfoItems} columns={2} />
          <div className="annotation cluster_role_card1_lab">
            <div className="ann_title">
              <p>{t('info.labels')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsClusterRoleLabelModalOpen(true)} />
            </div>
            <AnnotationModal open={isClusterRoleLabelModalOpen} type="label" dataList={roleDetailDataProps.metadata?.labels} callbackOk={handleClusterRoleLabelOk} callbackCancel={handleClusterRoleLabelCancel} />
            <div className="key_value">
              {roleDetailDataProps.metadata?.labels?.length ?
                roleDetailDataProps.metadata?.labels?.map(item => <LabelTag key={item.key} labelKey={item.key} labelValue={item.value} />) :
                <span style={{ marginTop: '13px' }}>{t('info.countZero')}</span>}
            </div>
          </div>
          <div className="annotation cluster_role_card1_ann">
            <div className="ann_title">
              <p>{t('info.annotations')}</p>
              <EditOutlined className="primary_icon common_antd_icon" onClick={() => setIsClusterRoleDetailAnnotationModalOpen(true)} />
            </div>
            <AnnotationModal open={isClusterRoleDetailAnnotationModalOpen} type="annotation" dataList={roleDetailDataProps?.metadata?.annotations} callbackOk={handleClusterRoleAnnotationOk} callbackCancel={handleClusterRoleAnnotationCancel} />
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