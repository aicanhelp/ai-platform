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
import CronJobEvents from '@/pages/container/workload/cronJob/detail/CronJobEvents';
import CronJobInfo from '@/pages/container/workload/cronJob/detail/CronJobInfo';
import CronJobYaml from '@/pages/container/workload/cronJob/detail/CronJobYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore, useRef } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getCronJob, editAnnotationsOrLabels, deleteCronJob } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { useTranslation } from 'react-i18next';

export default function CronJobDetail() {
  const { t } = useTranslation(['workload', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [cronJobDelModalOpen, setCronJobDelModalOpen] = useState(false); // 删除对话框展示
  const [isCronJobDelCheck, setIsCronJobDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成
  const detailReqSeqRef = useRef(0);

  const [cronJobDetailTabKey, setCronJobDetailTabKey] = useState(activeKey || 'info');

  const [cronJobDetailData, setCronJobDetailData] = useState({}); // 详情数据

  const [cronJobPopOpen, setCronJobPopOpen] = useState(false); // 气泡悬浮

  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [isCronJobAnnotationModalOpen, setIsCronJobAnnotationModalOpen] = useState(false);
  const [isCronJobLabelModalOpen, setIsCronJobLabelModalOpen] = useState(false);
  const [oldCronJobAnnotations, setOldCronJobAnnotations] = useState();
  const [oldCronJobLabels, setOldCronJobLabels] = useState();

  const [cronJobYaml, setCronJobYaml] = useState(''); // 传递yaml

  const themeStore = useStore('theme');

  // 气泡
  const handleCronJobPopOpenChange = (open) => {
    setCronJobPopOpen(open);
  };

  const handleSetCronJobDetailTabKey = (key) => {
    setCronJobDetailTabKey(key);
    setIsReadyOnly(true);
  };

  // 删除按钮
  const handleDeleteCronJob = () => {
    setCronJobPopOpen(false); // 气泡框
    setCronJobDelModalOpen(true); // 打开弹窗
  };

  const handleDeleteCronJobCancel = () => {
    setCronJobDelModalOpen(false);
  };

  const handleDeleteCronJobConfirm = async () => {
    try {
      const res = await deleteCronJob(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setCronJobDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/workload/cronJob`);
        });
      }
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('messages.deleteFailed', { ns: 'common', message: error.response?.data?.message }));
      }
    }
  };

  const handleCronJobDetailCheckFn = (e) => {
    setIsCronJobDelCheck(e.target.checked);
  };

  const handleModify = () => {
    setCronJobDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleAnnotationOk = async (cronJobData) => {
    if (JSON.stringify(oldCronJobAnnotations) === JSON.stringify(cronJobData)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsCronJobAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      cronJobData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldCronJobAnnotations, cronJobData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('cronjob', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getCronJobDetailInfo();
              setIsCronJobAnnotationModalOpen(false);
            });
          }
        } catch (error) {
          if (error?.response?.status === ResponseCode.Forbidden) {
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
    setIsCronJobAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsCronJobAnnotationModalOpen(true);
    setCronJobPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsCronJobLabelModalOpen(true);
    setCronJobPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (cronJobData) => {
    if (JSON.stringify(oldCronJobLabels) === JSON.stringify(cronJobData)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsCronJobLabelModalOpen(false);
    } else {
      const keyArr = [];
      cronJobData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldCronJobLabels, cronJobData, 'label');
        try {
          const res = await editAnnotationsOrLabels('cronjob', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getCronJobDetailInfo();
              setIsCronJobLabelModalOpen(false);
            });
          }
        } catch (error) {
          if (error?.response?.status === ResponseCode.Forbidden) {
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
    setIsCronJobLabelModalOpen(false);
  };

  const getCronJobDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getCronJob(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setCronJobYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldCronJobAnnotations([...res.data.metadata.annotations]);
          setOldCronJobLabels([...res.data.metadata.labels]);
          setCronJobDetailData(res.data);
        }
      } catch (error) {
        if (error.response?.status !== ResponseCode.OK) {
          messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
        }
      } finally {
        if (seq === detailReqSeqRef.current) {
          setDetailLoded(true);
        }
      }
    }
  }, [name, namespace]);

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  useEffect(() => {
    if (cronJobDetailTabKey === 'info' || activeKey) {
      getCronJobDetailInfo();
    }
  }, [cronJobDetailTabKey, getCronJobDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <CronJobInfo
        cronJobName={name}
        cronJobNamespace={namespace}
        cronJobDetailDataProps={cronJobDetailData}
        refreshFn={getCronJobDetailInfo}
      />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <CronJobYaml
        cronJobYamlProps={cronJobYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getCronJobDetailInfo} />,
    },
    {
      key: 'events',
      label: t('actions.events'),
      children: <CronJobEvents
        cronJobName={name}
        cronJobNamespace={namespace} />,
    },
  ];

  return <div className="child_content withBread_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/workload`, disabled: true },
      { title: t('cronJob.title'), path: '/cronJob' },
      { title: t('actions.detail'), path: '/detail' },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{name}</h3>
      <Popover placement='bottom'
        content={
          <Space className='column_pop'>
            <Button type="link" onClick={handleModify}>{t('actions.modify')}</Button>
            <Button type="link" onClick={handleEditLabel}>{t('actions.editLabel')}</Button>
            <Button type="link" onClick={handleEditAnnotation}>{t('actions.editAnnotation')}</Button>
            <Button type="link" onClick={handleDeleteCronJob}>{t('actions.delete')}</Button>
          </Space>
        }
        open={cronJobPopOpen}
        onOpenChange={handleCronJobPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isCronJobAnnotationModalOpen} type="annotation" dataList={cronJobDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isCronJobLabelModalOpen} type="label" dataList={cronJobDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: 'CronJob' })}
        open={cronJobDelModalOpen}
        cancelFn={handleDeleteCronJobCancel}
        content={[
          t('actions.deleteWarning', { resource: 'CronJob' }),
          t('actions.deleteConfirmMessage', { resource: 'CronJob', name }),
        ]}
        isCheck={isCronJobDelCheck}
        showCheck={true}
        checkFn={handleCronJobDetailCheckFn}
        confirmFn={handleDeleteCronJobConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetCronJobDetailTabKey} activeKey={cronJobDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}