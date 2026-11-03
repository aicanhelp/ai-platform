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
import JobEvents from '@/pages/container/workload/job/detail/JobEvents';
import JobInfo from '@/pages/container/workload/job/detail/JobInfo';
import JobYaml from '@/pages/container/workload/job/detail/JobYaml';
import { Tabs, Button, Popover, Space, message } from 'antd';
import { useState, useEffect, useCallback, useStore, useRef } from 'openinula';
import { useHistory, useParams } from 'inula-router';
import { DownOutlined } from '@ant-design/icons';
import '@/styles/pages/podDetail.less';
import { getJob, deleteJob, editAnnotationsOrLabels } from '@/api/containerApi';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { ResponseCode } from '@/common/constants';
import AnnotationModal from '@/components/AnnotationModal';
import { jsonToYaml, solveAnnotation, solveAnnotationOrLabelDiff, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { useTranslation } from 'react-i18next';

export default function JobDetail() {
  const { t } = useTranslation(['workload', 'common']);
  const { name, namespace, activeKey } = useParams();

  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();

  const [jobDelModalOpen, setJobDelModalOpen] = useState(false); // 删除对话框展示
  const [isJobDelCheck, setIsJobDelCheck] = useState(false); // 是否选中
  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成
  const detailReqSeqRef = useRef(0);

  const [jobDetailTabKey, setJobDetailTabKey] = useState(activeKey || 'info');

  const [jobDetailData, setJobDetailData] = useState({}); // 详情数据

  const [jobPopOpen, setJobPopOpen] = useState(false); // 气泡悬浮

  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读

  const [isJobAnnotationModalOpen, setIsJobAnnotationModalOpen] = useState(false);
  const [isJobLabelModalOpen, setIsJobLabelModalOpen] = useState(false);
  const [oldJobAnnotations, setOldJobAnnotations] = useState();
  const [oldJobLabels, setOldJobLabels] = useState();

  const [jobYaml, setJobYaml] = useState(''); // 传递yaml

  const themeStore = useStore('theme');

  // 处理只读
  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  // 气泡
  const handleJobPopOpenChange = (open) => {
    setJobPopOpen(open);
  };

  const handleSetJobDetailTabKey = (key) => {
    setJobDetailTabKey(key);
    setIsReadyOnly(true);
  };

  // 删除按钮
  const handleDeleteJob = () => {
    setJobPopOpen(false); // 气泡框
    setJobDelModalOpen(true); // 打开弹窗
  };

  const handleDeleteJobCancel = () => {
    setJobDelModalOpen(false);
  };

  const handleDeleteJobConfirm = async () => {
    try {
      const res = await deleteJob(namespace, name);
      if (res.status === ResponseCode.OK) {
        showSuccessThen(messageApi, t('messages.deleteSuccess', { ns: 'common' }), () => {
          setJobDelModalOpen(false);
          history.push(`/${containerRouterPrefix}/workload/job`);
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

  const handleJobCheckFn = (e) => {
    setIsJobDelCheck(e.target.checked);
  };

  const getJobDetailInfo = useCallback(async () => {
    if (name && namespace) {
      const seq = ++detailReqSeqRef.current;
      setDetailLoded(false);
      try {
        const res = await getJob(namespace, name);
        if (seq !== detailReqSeqRef.current) { return; }
        if (res.status === ResponseCode.OK) {
          setJobYaml(jsonToYaml(JSON.stringify(res.data))); // 先传递保持元数据clean
          res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
          res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
          setOldJobAnnotations([...res.data.metadata.annotations]);
          setOldJobLabels([...res.data.metadata.labels]);
          setJobDetailData(res.data);
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

  const handleModify = () => {
    setJobDetailTabKey('yaml'); // 跳向yaml
    setIsReadyOnly(false); // 退出只读模式
  };

  // 注解成功回调
  const handleAnnotationOk = async (jobData) => {
    if (JSON.stringify(oldJobAnnotations) === JSON.stringify(jobData)) {
      messageApi.info(t('messages.annotation', { ns: 'common' }) + t('messages.noChanges', { ns: 'common' }));
      setIsJobAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      jobData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldJobAnnotations, jobData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('job', namespace, name, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getJobDetailInfo();
              setIsJobAnnotationModalOpen(false);
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
    setIsJobAnnotationModalOpen(false);
  };

  const handleEditAnnotation = () => {
    setIsJobAnnotationModalOpen(true);
    setJobPopOpen(false);
  };

  const handleEditLabel = () => {
    setIsJobLabelModalOpen(true);
    setJobPopOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (jobData) => {
    if (JSON.stringify(oldJobLabels) === JSON.stringify(jobData)) {
      messageApi.info(t('messages.label', { ns: 'common' }) + t('messages.noChanges', { ns: 'common' }));
      setIsJobLabelModalOpen(false);
    } else {
      const keyArr = [];
      jobData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的标签
        const addLabelList = solveAnnotationOrLabelDiff(oldJobLabels, jobData, 'label');
        try {
          const res = await editAnnotationsOrLabels('job', namespace, name, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getJobDetailInfo();
              setIsJobLabelModalOpen(false);
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
    setIsJobLabelModalOpen(false);
  };

  useEffect(() => {
    if (jobDetailTabKey === 'info' || activeKey) {
      getJobDetailInfo();
    }
  }, [jobDetailTabKey, getJobDetailInfo]);

  useEffect(() => {
    if (activeKey) {
      setIsReadyOnly(false);
    }
  }, [activeKey]);

  const items = [
    {
      key: 'info',
      label: t('actions.detail'),
      children: <JobInfo
        jobName={name}
        jobNamespace={namespace}
        jobDetailDataProps={jobDetailData}
        refreshFn={getJobDetailInfo} />,
    },
    {
      key: 'yaml',
      label: t('actions.yaml'),
      children: <JobYaml
        jobYamlProps={jobYaml}
        readOnly={isReadyOnly}
        handleEditFn={handleReadyOnly}
        refreshFn={getJobDetailInfo} />,
    },
    {
      key: 'events',
      label: t('actions.events'),
      children: <JobEvents
        jobName={name}
        jobNamespace={namespace} />,
    },
  ];

  return <div className="child_content withBread_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/workload`, disabled: true },
      { title: t('job.title'), path: '/job' },
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
            <Button type="link" onClick={handleDeleteJob}>{t('actions.delete')}</Button>
          </Space>
        }
        open={jobPopOpen}
        onOpenChange={handleJobPopOpenChange}>
        <Button className='primary_btn'>{t('actions.operation')} <DownOutlined className='small_margin_adjust' /></Button>
      </Popover>
      {detailLoaded && <AnnotationModal open={isJobAnnotationModalOpen} type="annotation" dataList={jobDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isJobLabelModalOpen} type="label" dataList={jobDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
      <DeleteInfoModal
        title={t('actions.deleteConfirm', { resource: 'Job' })}
        open={jobDelModalOpen}
        cancelFn={handleDeleteJobCancel}
        content={[
          t('actions.deleteWarning', { resource: 'Job' }),
          t('actions.deleteConfirmMessage', { resource: 'Job', name }),
        ]}
        isCheck={isJobDelCheck}
        showCheck={true}
        checkFn={handleJobCheckFn}
        confirmFn={handleDeleteJobConfirm} />
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetJobDetailTabKey} activeKey={jobDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}