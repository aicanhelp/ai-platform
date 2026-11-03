/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Terminal } from 'xterm';
import { FitAddon } from '@xterm/addon-fit';
import 'xterm/css/xterm.css';
import { Modal, message, Button } from 'antd';
import { CloseOutlined, ExclamationCircleFilled, InfoCircleFilled } from '@ant-design/icons';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, useStore, useCallback } from 'openinula';
import useWebSocket from '@/hooks/useWebSocket';
import { getWsPrefix } from '@/common/constants';
import dayjs from 'dayjs';
import { useTranslation } from 'react-i18next';

const maxWaitMinute = 20;
const modalCloseMinute = 5;

const TerminalConsole = forwardRef((props, ref) => {
  const { t } = useTranslation('common');
  const chaltCountRef = useRef(0);
  const gainTerminalFirstRef = useRef('');
  const fitAddonRef = useRef(null);
  const termRef = useRef(null);
  const resizeHandlerRef = useRef(null);
  const wsMessageHandlerRef = useRef(null);
  const terminalRef = useRef(null);
  const [term, setTerm] = useState();
  const terminalStore = useStore('terminal');
  const userStore = useStore('user');
  const [lastInteractionTime, setLastInteractionTime] = useState(new Date());
  const [messageApi, contextHolder] = message.useMessage();
  const timeIDRef = useRef(0);
  const recycleTimeIDRef = useRef(0);
  const openInfoTimeIDRef = useRef(0);
  const [infoModal, setInfoModal] = useState(false);
  const [ws, _, wsState] = useWebSocket(`${getWsPrefix().terminalUrl}${terminalStore.$s.type === 'cluster' ?
    `/user/${userStore.$s.user.name}/terminal`
    : `/namespace/${terminalStore.$s.podDescription.namespace}/pod/${terminalStore.$s.podDescription.pod}/container/${terminalStore.$s.name}/terminal`}`);

  useImperativeHandle(ref, () => ({
    resizeScreenConsole,
  }));

  const disablePrint = (disabled = true) => {
    const textarea = terminalRef.current?.textarea;
    if (textarea) {
      textarea.disabled = disabled;
    }
  };

  const getTerminalSize = () => {
    const box = terminalRef.current;
    return {
      cols: Math.ceil(box.offsetWidth / 9),
      rows: Math.ceil(box.offsetHeight / 17),
    };
  };

  const resizeRemoteTerminal = () => {
    const { cols, rows } = getTerminalSize();
    if (wsState === 'open') {
      ws.send(JSON.stringify({ Op: 'resize', Cols: cols, Rows: rows }));
    }
  };

  const initConsole = useCallback((dom) => {
    if (ws) {
      // 移除上一次会话残留的监听器，避免累积与写入已销毁的终端
      if (resizeHandlerRef.current) {
        window.removeEventListener('resize', resizeHandlerRef.current);
        resizeHandlerRef.current = null;
      }
      if (wsMessageHandlerRef.current) {
        ws.removeEventListener('message', wsMessageHandlerRef.current);
        wsMessageHandlerRef.current = null;
      }
      chaltCountRef.current = 0;
      gainTerminalFirstRef.current = '';
      const currentTerm = new Terminal({
        rendererType: 'dom',
        cursorBlink: true,
        convertEol: true,
        ...getTerminalSize(),
        theme: {
          foreground: '#F7F7F7',
          background: '#2A2D34',
        },
      });
      resizeRemoteTerminal();
      currentTerm.write(`connection(updated at ${dayjs().format('YYYY-MM-DD HH:mm:ss')})...`);
      disablePrint();
      fitAddonRef.current = new FitAddon();
      currentTerm.loadAddon(fitAddonRef.current);
      fitAddonRef.current.fit();
      currentTerm.open(dom);
      currentTerm.onData(data => {
        clearTimeout(timeIDRef.current);
        clearTimeout(openInfoTimeIDRef.current);
        clearTimeout(recycleTimeIDRef.current);
        setLastInteractionTime(new Date());
        if (gainTerminalFirstRef.current) {
          ws.send(JSON.stringify({ Op: 'stdin', Data: data }));
        }
      });
      resizeHandlerRef.current = () => {
        const { cols, rows } = getTerminalSize();
        currentTerm.resize(cols, Math.max(rows - 2, 1));
        fitAddonRef.current?.fit();
        resizeRemoteTerminal();
      };
      window.addEventListener('resize', resizeHandlerRef.current);
      wsMessageHandlerRef.current = ({ data }) => {
        try {
          const outData = JSON.parse(data);
          if (!chaltCountRef.current) {
            currentTerm.write('\r\n');
            gainTerminalFirstRef.current = outData.Data;
          }
          currentTerm.write(outData.Data);
          chaltCountRef.current++;
        } catch {
          // 忽略非 JSON 格式消息
        }
      };
      ws.addEventListener('message', wsMessageHandlerRef.current);
      termRef.current = currentTerm;
      setTerm(currentTerm);
    }
  }, [ws]);

  const resizeScreenConsole = () => {
    if (term) {
      const { cols, rows } = getTerminalSize();
      term.resize(cols, rows);
      resizeRemoteTerminal();
    }
  };

  useEffect(() => {
    if (terminalRef.current) {
      if (term) {
        term.dispose();
        termRef.current = null;
      }
      initConsole(terminalRef.current);
    }
    return () => {
      if (resizeHandlerRef.current) {
        window.removeEventListener('resize', resizeHandlerRef.current);
        resizeHandlerRef.current = null;
      }
      if (wsMessageHandlerRef.current) {
        ws?.removeEventListener('message', wsMessageHandlerRef.current);
        wsMessageHandlerRef.current = null;
      }
      termRef.current?.dispose();
      termRef.current = null;
    };
  }, [terminalRef, terminalStore.$s.name, initConsole]);

  useEffect(() => {
    if (wsState === 'closed' || wsState === 'fail') {
      term?.write(`\r\n\x1B[1;3;31m${t('terminal.connectionClosed')}\x1B[0m`);
    }
    if (wsState !== 'open') {
      disablePrint();
    }
  }, [wsState, term, ws]);

  useEffect(() => {
    return () => ws.close();
  }, [ws]);

  const handleCloseTerminalInfo = () => {
    messageApi.destroy('infoTerminal');
  };

  useEffect(() => {
    const checkInactivity = () => {
      const currentTime = new Date();
      const inactivityDuration = (currentTime - lastInteractionTime) / 1000 / 60;
      if (inactivityDuration >= (maxWaitMinute + modalCloseMinute)) {
        handleCloseTerminalInfo();
        setInfoModal(true);
      } else {
        if (inactivityDuration > maxWaitMinute) {
          messageApi.open({
            type: 'info',
            className: `${terminalStore.$s.screenSize}_terminal_msg terminal_info_msg`,
            key: 'infoTerminal',
            icon: <InfoCircleFilled />,
            content: <div>
              {t('terminal.timeoutWarning', { maxWait: maxWaitMinute, closeIn: modalCloseMinute })}
              <CloseOutlined style={{ color: '#f7f7f7', fontSize: '14px', marginLeft: '20px' }} onClick={handleCloseTerminalInfo} />
            </div>,
            duration: 0,
          });
          openInfoTimeIDRef.current = setTimeout(() => {
            handleCloseTerminalInfo();
            if (terminalStore.$s.screenSize !== 'small') {
              setInfoModal(true);
            }
          }, modalCloseMinute * 60 * 1000);
        } else {
          handleCloseTerminalInfo();
          recycleTimeIDRef.current = setTimeout(() => checkInactivity(), 1000);
        }
      }
    };

    timeIDRef.current = setTimeout(checkInactivity, 1000);

    return () => {
      clearTimeout(timeIDRef.current);
      clearTimeout(openInfoTimeIDRef.current);
      clearTimeout(recycleTimeIDRef.current);
    };
  }, [lastInteractionTime, terminalStore.$s.screenSize]);

  const handleCloseTerminal = () => {
    setInfoModal(false);
    terminalStore.$a.setTerminalOpen(false);
    terminalStore.$a.setTerminalScreenSize('middle');
  };

  return <>
    <div ref={terminalRef} className="terminal_console_box" style={{ width: '100%' }}>{contextHolder}</div>
    <Modal
      className='modal_flex_delete terminal_console_modal'
      open={infoModal}
      title={t('actions.info')}
      getContainer={false}
      onCancel={handleCloseTerminal}
      footer={
        <Button className='cancel_btn' onClick={handleCloseTerminal}>{t('actions.confirm')}</Button>
      }>
      <div className='modal_delete_content'>
        <ExclamationCircleFilled className='warn_icon' />
        <div className='word_tograry'>
          <div className='word_tograry_alert'>
            <p className='bread_word_delete'>{t('terminal.timeoutClose', { maxWait: maxWaitMinute })}</p>
          </div>
        </div>
      </div>
    </Modal>
  </>;
});

export default TerminalConsole;
