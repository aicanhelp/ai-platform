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
import ClusterRoleInfo from '@/pages/container/userManage/role/clusterRole/ClusterRoleInfo';
import ClusterRoleYaml from '@/pages/container/userManage/role/clusterRole/ClusterRoleYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useRef, useStore } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getClusterRoleDetailDescription, editAnnotationsOrLabels, deleteClusterRole } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

export default function ClusterRoleDetail() {
  const { t } = useTranslation(['rbac', 'common']);
  const { roleName, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [roleDelModalOpen, setRoleDelModalOpen] = useState(false); // 删除对话框展示
  const [isRoleDelCheck, setIsRoleDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成

  const [roleDetailTabKey, setRoleDetailTabKey] = useState(activeKey || 'detail');

  const [roleDetailData, setRoleDetailData] = useState({}); // 详情数据

  const [rolePopOpen, setRolePopOpen] = useState(false); // 气泡悬浮

  const [isCrReadyOnly, setIsCrReadyOnly] = useState(true); // 是否只读

  const [isClusterRoleAnnotationModalOpen, setIsClusterRoleAnnotationModalOpen] = useState(false);
  const [isClusterRoleLabelModalOpen, setIsClusterRoleLabelModalOpen] = useState(false);
  const [oldClusterRoleAnnotations, setOldClusterRoleAnnotations] = useState();
  const [oldClusterRoleLabels, setOldClusterRoleLabels] = useState();

  const [clusterRoleYaml, setClusterRoleYaml] = useState(''); // 传递yaml

  const themeStore = useStore('theme');

  // 处理只读
  const handleClusterRoleReadyOnly = (bool) => {
    setIsCrReadyOnly(bool);
  };

  // 气泡
  const handleClusterRolePopOpenChange = (open) => {
    setRolePopOpen(open);
  };

  const handleSetClusterRoleDetailTabKey = (key) => {
    setRoleDetailTabKey(key);
    setIsCrReadyOnly(true);
  };

  // 删除按钮
  const handleDeleteClusterRole = () => {
    setRolePopOpen(false); // 气泡框
    setRoleDelModalOpen(true); // 打开弹窗
  };

  const handleClusterRoleCancel = () => {
    setRoleDelModalOpen(false);
  };

  const handleClusterRoleConfirm = async () => {
    try {
      const res = await deleteClusterRole(roleName);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setRoleDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/userManage/role`);
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

  const handleClusterRoleCheckFn = (e) => {
    setIsRoleDelCheck(e.target.checked);
  };

  const detailReqSeqRef = useRef(0);

  const getRoleDetailInfo = useCallback(async () => {
    if (!roleName) {
      return;
    }
    if (roleName) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getClusterRoleDetailDescription(roleName);
        if (res.status === ResponseCode.OK) {
          setClusterRoleYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldClusterRoleAnnotations([...res.data.metadata.annotations]);
          setOldClusterRoleLabels([...res.data.metadata.labels]);
          setRoleDetailData(res.data);
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
  }, [roleName]);

  const handleClusterRoleModify = () => {
    setRoleDetailTabKey('yaml'); // 跳向yaml
    setIsCrReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleAnnotationOk = async (clusterRoleData) => {
    if (JSON.stringify(oldClusterRoleAnnotations) === JSON.stringify(clusterRoleData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common', resource: t('messages.annotation', { ns: 'common' }) }));
      setIsClusterRoleAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      clusterRoleData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldClusterRoleAnnotations, clusterRoleData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('clusterrole', '', roleName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getRoleDetailInfo();
              setIsClusterRoleAnnotationModalOpen(false);
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
  const handleAnnotationCancel = () => {
    setIsClusterRoleAnnotationModalOpen(false);
  };

  const handleEditClusterRoleAnnotation = () => {
    setIsClusterRoleAnnotationModalOpen(true);
    setRolePopOpen(false);
  };

  const handleEditClusterRoleLabel = () => {
    setIsClusterRoleLabelModalOpen(true);
    setRolePopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (clusterRoleData) => {
    if (JSON.stringify(oldClusterRoleLabels) === JSON.stringify(clusterRoleData)) {
      messageApi.info(t('messages.noChanges', { ns: 'common', resource: t('messages.label', { ns: 'common' }) }));
      setIsClusterRoleLabelModalOpen(false);
    } else {
      const keyArr = [];
      clusterRoleData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldClusterRoleLabels, clusterRoleData, 'label');
        try {
          const res = await editAnnotationsOrLabels('clusterrole', '', roleName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getRoleDetailInfo();
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
  const handleLabelCancel = () => {
    setIsClusterRoleLabelModalOpen(false);
  };

  const isInitialMountRef = useRef(true);

  useEffect(() => {
    if (roleDetailTabKey === 'detail' || (activeKey && isInitialMountRef.current)) {
      getRoleDetailInfo();
    }
    isInitialMountRef.current = false;
  }, [roleDetailTabKey, getRoleDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsCrReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'detail',
      label: t('actions.detail'),
      children: <ClusterRoleInfo
        roleName={roleName}
        roleDetailDataProps={roleDetailData}
        refreshFn={getRoleDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <ClusterRoleYaml
        clusterRoleYamlProps={clusterRoleYaml}
        readOnly={isCrReadyOnly}
        handleEditFn={handleClusterRoleReadyOnly}
        refreshFn={getRoleDetailInfo} />,
    },
  ];

  return <div className="child_content withBread_content">
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom items={[
      { title: t('module.title'), disabled: true },
      { title: t('role.clusterTitle'), path: `/${containerRouterPrefix}/userManage/role` },
      { title: t('actions.detail') },
    ]} />
    <div className='pod_title cluster_role_op' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{roleName}</h3>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleClusterRoleModify}>{t('actions.modify')}</Button>
            <Button type="link" onClick={handleEditClusterRoleLabel}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditClusterRoleAnnotation}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteClusterRole}>{t('actions.delete')}</Button>
          </Space>
        }
        open={rolePopOpen}
        onOpenChange={handleClusterRolePopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isClusterRoleAnnotationModalOpen} type="annotation" dataList={roleDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isClusterRoleLabelModalOpen} type="label" dataList={roleDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: t('role.clusterTitle') })}
        open={roleDelModalOpen}
        cancelFn={handleClusterRoleCancel}
        content={[
          t('actions.deleteWarning'),
          t('actions.deleteConfirmMessage', { resource: t('role.clusterTitle'), name: roleName }),
        ]}
        isCheck={isRoleDelCheck}
        showCheck={true}
        checkFn={handleClusterRoleCheckFn}
        confirmFn={handleClusterRoleConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetClusterRoleDetailTabKey} activeKey={roleDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}