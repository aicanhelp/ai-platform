/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import RoleInfo from '@/pages/container/userManage/roleBinding/roleBinding/RoleInfo';
import RoleBindYaml from '@/pages/container/userManage/roleBinding/roleBinding/RoleBindYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getRoleBindDetailDescription, editAnnotationsOrLabels, deleteRoleBind } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function RoleBindDetail() {
  const { t } = useTranslation(['rbac', 'common']);
  const { roleBindNamespace, roleBindName, activeKey } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [detailLoaded, setDetailLoded] = useState(false);
  const [roleBindDetailTabKey, setRoleBindDetailTabKey] = useState(activeKey || 'detail');
  const [roleBindDetailData, setRoleBindDetailData] = useState({});
  const [roleBindPopOpen, setRoleBindPopOpen] = useState(false);
  const [roleBindDelModalOpen, setRoleBindDelModalOpen] = useState(false);
  const [isRoleBindDelCheck, setIsRoleBindDelCheck] = useState(false);
  const [isReadyOnly, setIsReadyOnly] = useState(true);
  const [isRoleBindAnnotationModalOpen, setIsRoleBindAnnotationModalOpen] = useState(false);
  const [isRoleBindLabelModalOpen, setIsRoleBindLabelModalOpen] = useState(false);
  const [oldRoleBindAnnotations, setOldRoleBindAnnotations] = useState();
  const [oldRoleBindLabels, setOldRoleBindLabels] = useState();
  const themeStore = useStore('theme');
  const [roleBindYaml, setRoleBindYaml] = useState('');

  const handleRoleBindPopOpenChange = (open) => {
    setRoleBindPopOpen(open);
  };

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleDeleteRoleBind = () => {
    setRoleBindPopOpen(false);
    setRoleBindDelModalOpen(true);
  };

  const handleSetRoleBindDetailTabKey = (key) => {
    setRoleBindDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleRoleBindCheckFn = (e) => {
    setIsRoleBindDelCheck(e.target.checked);
  };

  const handleDeleteRoleBindCancel = () => {
    setRoleBindDelModalOpen(false);
  };

  const handleDeleteRoleBindConfirm = async () => {
    try {
      const res1 = await deleteRoleBind(roleBindNamespace, roleBindName);
      if (res1.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setRoleBindDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/userManage/roleBinding`);
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const detailReqSeqRef = useRef(0);

  const getRoleBindDetailInfo = useCallback(async () => {
    if (roleBindName && roleBindNamespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getRoleBindDetailDescription(roleBindNamespace, roleBindName);
        if (res.status === ResponseCode.OK) {
          setRoleBindYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldRoleBindAnnotations([...res.data.metadata.annotations]);
          setOldRoleBindLabels([...res.data.metadata.labels]);
          setRoleBindDetailData(res.data);
        }
      } catch (error) {
        if (seq !== detailReqSeqRef.current) {
          return;
        }
        if (error.response?.status !== ResponseCode.OK) {
          messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
        }
      } finally {
        if (seq === detailReqSeqRef.current) {
          setDetailLoded(true);
        }
      }
    }
  }, [roleBindName, roleBindNamespace]);

  const handleModify = () => {
    setRoleBindDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const handleAnnotationOk = async (roleBindData) => {
    if (JSON.stringify(oldRoleBindAnnotations) === JSON.stringify(roleBindData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsRoleBindAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      roleBindData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldRoleBindAnnotations, roleBindData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('rolebinding', roleBindNamespace, roleBindName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getRoleBindDetailInfo();
              setIsRoleBindAnnotationModalOpen(false);
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

  const handleAnnotationCancel = () => {
    setIsRoleBindAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsRoleBindAnnotationModalOpen(true);
    setRoleBindPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsRoleBindLabelModalOpen(true);
    setRoleBindPopOpen(false);
  };

  const handleLabelOk = async (roleBindData) => {
    if (JSON.stringify(oldRoleBindLabels) === JSON.stringify(roleBindData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsRoleBindLabelModalOpen(false);
    } else {
      const keyArr = [];
      roleBindData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldRoleBindLabels, roleBindData, 'label');
        try {
          const res = await editAnnotationsOrLabels('rolebinding', roleBindNamespace, roleBindName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getRoleBindDetailInfo();
              setIsRoleBindLabelModalOpen(false);
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

  const handleLabelCancel = () => {
    setIsRoleBindLabelModalOpen(false);
  };

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (roleBindDetailTabKey === 'detail' || (activeKey && isInitialMountRef.current)) {
      getRoleBindDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [roleBindDetailTabKey, getRoleBindDetailInfo]);

  const items = [
    {
      label: t('actions.detail'),
      key: 'detail',
      children: <RoleInfo
        roleBindName={roleBindName}
        roleBindNamespace={roleBindNamespace}
        roleBindDetailDataProps={roleBindDetailData}
        refreshFn={getRoleBindDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <RoleBindYaml
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        roleBindYamlProps={roleBindYaml}
        refreshFn={getRoleBindDetailInfo} />,
    },
  ];

  return <div className="child_content withBread_content RoleBindDetail">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), disabled: true },
      { title: t('roleBinding.title'), path: `/${containerRouterPrefix}/userManage/roleBinding` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title RoleBindDetail' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{roleBindName}</h3>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
            <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteRoleBind}>{t('actions.delete')}</Button>
          </Space>
        }
        open={roleBindPopOpen}
        onOpenChange={handleRoleBindPopOpenChange}>
        <Button className='primary_btn RoleBindDetail'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isRoleBindAnnotationModalOpen} type="annotation" dataList={roleBindDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isRoleBindLabelModalOpen} type="label" dataList={roleBindDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('roleBinding.roleBindingTitle') })}
        open={roleBindDelModalOpen}
        isCheck={isRoleBindDelCheck}
        showCheck={true}
        checkFn={handleRoleBindCheckFn}
        cancelFn={handleDeleteRoleBindCancel}
        content={[
          t('actions.deleteWarning'),
          t('actions.deleteConfirmMessage', { resource: t('roleBinding.roleBindingTitle'), name: roleBindName }),
        ]}
        confirmFn={handleDeleteRoleBindConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetRoleBindDetailTabKey} activeKey={roleBindDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
