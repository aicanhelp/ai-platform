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
import ClusterRoleBindingInfo from '@/pages/container/userManage/roleBinding/clusterRoleBind/ClusterRoleBindingInfo';
import ClusterRoleBindingYaml from '@/pages/container/userManage/roleBinding/clusterRoleBind/ClusterRoleBindingYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getClusterRoleBindingDetailDescription, editAnnotationsOrLabels, deleteClusterRoleBinding } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ClusterRoleBindDetail() {
  const { t } = useTranslation(['rbac', 'common']);
  const { roleBindName, activeKey } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [roleBindDelModalOpen, setRoleBindDelModalOpen] = useState(false);
  const [isRoleBindDelCheck, setIsRoleBindDelCheck] = useState(false);
  const [detailLoaded, setDetailLoded] = useState(false);
  const [roleBindDetailTabKey, setRoleBindDetailTabKey] = useState(activeKey || 'detail');
  const [roleBindDetailData, setRoleBindDetailData] = useState({});
  const [roleBindPopOpen, setRoleBindPopOpen] = useState(false);
  const [isReadyOnly, setIsReadyOnly] = useState(true);
  const [isClusterRoleBindAnnotationModalOpen, setIsClusterRoleBindAnnotationModalOpen] = useState(false);
  const [isClusterRoleBindLabelModalOpen, setIsClusterRoleBindLabelModalOpen] = useState(false);
  const [oldClusterRoleBindAnnotations, setOldClusterRoleBindAnnotations] = useState();
  const [oldClusterRoleBindLabels, setOldClusterRoleBindLabels] = useState();
  const [roleBindYaml, setRoleBindYaml] = useState('');
  const themeStore = useStore('theme');

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleRoleBindPopOpenChange = (open) => {
    setRoleBindPopOpen(open);
  };

  const handleSetRoleBindDetailTabKey = (key) => {
    setRoleBindDetailTabKey(key);
    setIsReadyOnly(true);
  };

  const handleDeleteRoleBind = () => {
    setRoleBindPopOpen(false);
    setRoleBindDelModalOpen(true);
  };

  const handleDeleteRoleBindCancel = () => {
    setRoleBindDelModalOpen(false);
  };

  const handleDeleteRoleBindConfirm = async () => {
    try {
      const res = await deleteClusterRoleBinding(roleBindName);
      if (res.status === ResponseCode.OK) {
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

  const handleRoleBindCheckFn = (e) => {
    setIsRoleBindDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getRoleBindDetailInfo = useCallback(async () => {
    if (roleBindName) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getClusterRoleBindingDetailDescription(roleBindName);
        if (res.status === ResponseCode.OK) {
          setRoleBindYaml(jsonToYaml(JSON.stringify(res.data)));
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldClusterRoleBindAnnotations([...res.data.metadata.annotations]);
          setOldClusterRoleBindLabels([...res.data.metadata.labels]);
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
  }, [roleBindName]);

  const handleModify = () => {
    setRoleBindDetailTabKey('yaml');
    setIsReadyOnly(false);
  };

  const handleAnnotationOk = async (clusterRoleBindData) => {
    if (JSON.stringify(oldClusterRoleBindAnnotations) === JSON.stringify(clusterRoleBindData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsClusterRoleBindAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      clusterRoleBindData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addAnnotationList = solveAnnotationOrLabelDiff(oldClusterRoleBindAnnotations, clusterRoleBindData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('clusterrolebinding', '', roleBindName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getRoleBindDetailInfo();
              setIsClusterRoleBindAnnotationModalOpen(false);
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
    setIsClusterRoleBindAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsClusterRoleBindAnnotationModalOpen(true);
    setRoleBindPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsClusterRoleBindLabelModalOpen(true);
    setRoleBindPopOpen(false);
  };

  const handleLabelOk = async (clusterRoleBindData) => {
    if (JSON.stringify(oldClusterRoleBindLabels) === JSON.stringify(clusterRoleBindData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common' }));
      setIsClusterRoleBindLabelModalOpen(false);
    } else {
      const keyArr = [];
      clusterRoleBindData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        const addLabelList = solveAnnotationOrLabelDiff(oldClusterRoleBindLabels, clusterRoleBindData, 'label');
        try {
          const res = await editAnnotationsOrLabels('clusterrolebinding', '', roleBindName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getRoleBindDetailInfo();
              setIsClusterRoleBindLabelModalOpen(false);
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
    setIsClusterRoleBindLabelModalOpen(false);
  };

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (roleBindDetailTabKey === 'detail' || (activeKey && isInitialMountRef.current)) {
      getRoleBindDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [roleBindDetailTabKey, getRoleBindDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'detail',
      label: t('actions.detail'),
      children: <ClusterRoleBindingInfo
        roleBindName={roleBindName}
        roleBindDetailDataProps={roleBindDetailData}
        refreshFn={getRoleBindDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ClusterRoleBindingYaml
        roleBindYamlProps={roleBindYaml}
        handleEditFn={handleReadyOnly}
        readOnly={isReadyOnly}
        refreshFn={getRoleBindDetailInfo} />,
    },
  ];

  return <div className="child_content withBread_content ClusterRoleBindDetail">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), disabled: true },
      { title: t('roleBinding.clusterTitle'), path: `/${containerRouterPrefix}/userManage/roleBinding` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title ClusterRoleBindDetail' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
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
        <Button className='primary_btn ClusterRoleBindDetail'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isClusterRoleBindAnnotationModalOpen} type="annotation" dataList={roleBindDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isClusterRoleBindLabelModalOpen} type="label" dataList={roleBindDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('roleBinding.clusterTitle') })}
        open={roleBindDelModalOpen}
        cancelFn={handleDeleteRoleBindCancel}
        content={[
          t('actions.deleteWarning'),
          t('actions.deleteConfirmMessage', { resource: t('roleBinding.clusterTitle'), name: roleBindName }),
        ]}
        isCheck={isRoleBindDelCheck}
        showCheck={true}
        checkFn={handleRoleBindCheckFn}
        confirmFn={handleDeleteRoleBindConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetRoleBindDetailTabKey} activeKey={roleBindDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
