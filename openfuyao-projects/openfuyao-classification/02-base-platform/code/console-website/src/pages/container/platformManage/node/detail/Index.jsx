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
import NodeInfo from '@/pages/container/platformManage/node/detail/NodeInfo';
import NodePod from '@/pages/container/platformManage/node/detail/NodePod';
import { Tabs, message } from 'antd';
import { useState, useCallback, useEffect, useStore, useRef } from 'openinula';
import { useParams } from 'inula-router';
import '@/styles/pages/nodeDetail.less';
import '@/styles/pages/podDetail.less';
import AnnotationModal from '@/components/AnnotationModal';
import { ResponseCode } from '@/common/constants';
import { editAnnotationsOrLabels, getNode } from '@/api/containerApi';
import { 
  solveAnnotation,
  solveAnnotationOrLabelDiff,
  solveNodeHealthyStatus,
  solveNodeResource,
  forbiddenMsg,
  showSuccessThen,
} from '@/tools/utils';
import { useTranslation } from 'react-i18next';

export default function NodeDetail() {
  const { t } = useTranslation(['node', 'common']);
  const { nodeName } = useParams();

  const [messageApi, contextHolder] = message.useMessage();
  const [nodeDetailTabKey, setNodeDetailTabKey] = useState('1');
  const [isNodeAnnotationModalOpen, setIsNodeAnnotationModalOpen] = useState(false);
  const [isNodeLabelModalOpen, setIsNodeLabelModalOpen] = useState(false);

  const [detailLoaded, setDetailLoded] = useState(false); // 加载完成

  const [nodeDetailData, setNodeDetailData] = useState();

  const [oldNodeAnnotations, setOldNodeAnnotations] = useState();
  const [oldNodeLabels, setOldNodeLabels] = useState();
  const themeStore = useStore('theme');

  const handleSetNodeDetailTabKey = (key) => {
    setNodeDetailTabKey(key);
  };

  // 注解成功回调
  const handleAnnotationOk = async (nodeData) => {
    if (JSON.stringify(oldNodeAnnotations) === JSON.stringify(nodeData)) {
      messageApi.info(`${t('messages.annotation', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsNodeAnnotationModalOpen(false);
    } else {
      const keyArr = [];
      nodeData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldNodeAnnotations, nodeData, 'annotation');
        try {
          const res = await editAnnotationsOrLabels('node', '', nodeName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editAnnotationSuccess', { ns: 'common' }), () => {
              getNodeDetailInfo();
              setIsNodeAnnotationModalOpen(false);
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
    setIsNodeAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (nodeData) => {
    if (JSON.stringify(oldNodeLabels) === JSON.stringify(nodeData)) {
      messageApi.info(`${t('messages.label', { ns: 'common' })}${t('messages.noChanges', { ns: 'common' })}`);
      setIsNodeLabelModalOpen(false);
    } else {
      const keyArr = [];
      nodeData.map(item => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(t('messages.duplicateKey', { ns: 'common' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldNodeLabels, nodeData, 'label');
        try {
          const res = await editAnnotationsOrLabels('node', '', nodeName, addLabelList);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.editLabelSuccess', { ns: 'common' }), () => {
              getNodeDetailInfo();
              setIsNodeLabelModalOpen(false);
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
    setIsNodeLabelModalOpen(false);
  };

  const detailReqSeqRef = useRef(0);

  const getNodeDetailInfo = useCallback(async () => {
    const seq = ++detailReqSeqRef.current;
    setDetailLoded(false);
    try {
      const res = await getNode(nodeName);
      if (seq !== detailReqSeqRef.current) {
        return;
      }
      if (res.status === ResponseCode.OK) {
        res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
        res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
        // 修改健康状态
        res.data.healthy = solveNodeHealthyStatus(res.data.status.conditions);
        res.data.selfResourceQuota = solveNodeResource(res.data.status.allocatable, res.data.status.capacity);
        setOldNodeAnnotations([...res.data.metadata.annotations]);
        setOldNodeLabels([...res.data.metadata.labels]);
        setNodeDetailData(res.data);
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
  }, [nodeName]);

  useEffect(() => {
    if (nodeDetailTabKey === '1') {
      getNodeDetailInfo();
    }
  }, [nodeDetailTabKey, getNodeDetailInfo]);

  const items = [
    {
      key: '1',
      label: t('detail.tabDetail'),
      children: <NodeInfo
        nodeName={nodeName}
        nodeDetailDataProps={nodeDetailData}
        refreshFn={getNodeDetailInfo}
      />,
    },
    {
      key: '2',
      label: t('detail.tabPod'),
      children: <NodePod nodeName={nodeName} />,
    },
  ];

  return <div className="child_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/nodeManage` },
      { title: t('module.detail'), path: '/detail' },
    ]} />
    <div className='pod_title' style={{ border: themeStore.$s.theme !== 'light' && 'none', backgroundColor: themeStore.$s.theme !== 'light' && '#2a2d34' }}>
      <h3>{nodeName}</h3>
      {detailLoaded && <AnnotationModal open={isNodeAnnotationModalOpen} type="annotation" dataList={nodeDetailData?.metadata?.annotations} callbackOk={handleAnnotationOk} callbackCancel={handleAnnotationCancel} />}
      {detailLoaded && <AnnotationModal open={isNodeLabelModalOpen} type="label" dataList={nodeDetailData?.metadata?.labels} callbackOk={handleLabelOk} callbackCancel={handleLabelCancel} />}
    </div>
    {detailLoaded && <Tabs items={items} onChange={handleSetNodeDetailTabKey} activeKey={nodeDetailTabKey} destroyInactiveTabPane={true}></Tabs>}
  </div>;
}
