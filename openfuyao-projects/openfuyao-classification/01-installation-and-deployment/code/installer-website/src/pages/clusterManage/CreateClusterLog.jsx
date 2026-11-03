/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button } from 'antd';
import { FullscreenOutlined, FullscreenExitOutlined } from '@ant-design/icons';
import { useEffect, useRef, useState } from 'openinula';
import useWebSocket from '@/hooks/useWebSocket';
import { useIntl } from '../../i18n';

export default function CreateClusterLog({ title, clusterName, open, okFn, cancelFn }) {
  const intl = useIntl();
  const [currentLog, setCurrentLog] = useState('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const logContentRef = useRef(null);
  const isNearBottomRef = useRef(true); // 用户是否处于日志底部附近，决定新日志到达时是否自动跟随滚动
  const [ws, wsData] = useWebSocket(`/ws/cluster/v1/clusters/${clusterName}/logs`);

  const scrollToBottom = () => {
    if (logContentRef.current) {
      logContentRef.current.scrollTop = logContentRef.current.scrollHeight;
    }
  };

  const handleLogScroll = () => {
    if (!logContentRef.current) {
      return;
    }
    const { scrollHeight, scrollTop, clientHeight } = logContentRef.current;
    // 距底部40px内视为在底部：继续自动跟随；向上翻阅历史日志时暂停跟随，拖回底部后恢复
    isNearBottomRef.current = scrollHeight - scrollTop - clientHeight <= 40;
  };

  useEffect(() => {
    if (wsData) {
      if (wsData !== 'Error...' && typeof wsData === 'string') {
        setCurrentLog((current) => {
          return `${current}${wsData}`;
        });
      }
    }
  }, [wsData]);

  useEffect(() => {
    if (isNearBottomRef.current) {
      scrollToBottom();
    }
  }, [currentLog]);

  return (
    <Modal
      getContainer={() => document.querySelector('#cluster_root .container_content')}
      title={title || intl.formatMessage({ id: 'log.title' })}
      className={`pod_log_container ${isFullscreen ? 'pod_log_fullscreen' : ''}`}
      wrapClassName={isFullscreen ? 'pod_log_modal_fullscreen' : ''}
      open={open}
      afterOpenChange={(visible) => visible && setTimeout(scrollToBottom, 0)}
      onCancel={cancelFn}
      width={isFullscreen ? '100%' : 1600}
      footer={[
        <Button
          className="cluster_primary_btn"
          icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
          onClick={() => setIsFullscreen(!isFullscreen)}
        >
          {isFullscreen ? intl.formatMessage({ id: 'log.restore' }) : intl.formatMessage({ id: 'log.fullscreen' })}
        </Button>,
        <Button
          className="cluster_primary_btn"
          style={{ marginLeft: '16px' }}
          onClick={() => {
            ws.close();
            okFn();
          }}
        >
          {intl.formatMessage({ id: 'log.exitLog' })}
        </Button>,
      ]}
      destroyOnHidden={true}
    >
      <div
        ref={logContentRef}
        className="clusterLogContent"
        onScroll={handleLogScroll}
        style={{
          marginBottom: '20px',
          height: isFullscreen ? 'calc(100vh - 260px)' : 580,
        }}
      >
        <pre>{currentLog}</pre>
      </div>
    </Modal>
  );
}
