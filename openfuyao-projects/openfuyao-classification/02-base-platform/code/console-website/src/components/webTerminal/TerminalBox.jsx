/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import {
  LineOutlined,
  ExpandOutlined,
  CloseOutlined,
  ExpandAltOutlined,
  CompressOutlined,
  CaretRightOutlined,
  RightOutlined,
  CaretLeftOutlined,
  ExclamationCircleFilled,
} from '@ant-design/icons';
import '@/styles/components/webTerminal.less';
import { useRef, useState, useStore, useEffect } from 'openinula';
import TerminalConsole from '@/components/webTerminal/TerminalConsole';
import { Modal, Button } from 'antd';
import '@/styles/components/deleteInfoModal.less';
import { publicLink } from '@/common/constants';
import { useTranslation } from 'react-i18next';

export default function TerminalBox() {
  const { t } = useTranslation('common');
  const terminalStore = useStore('terminal');
  const terminalRef = useRef(null);
  const [collapseTerminalMenu, setCollapseTerminalMenu] = useState(false);
  const [closeTerminalOpen, setCloseTerminalOpen] = useState(false);
  const [isShowModal, setIsShowModal] = useState(true);

  const handleList = [
    {
      name: t('terminal.viewAllPods'),
      example: 'kubectl get pods',
    },
    {
      name: t('terminal.viewNodeDetail'),
      example: 'kubectl describe nodes <node-name>',
    },
  ];

  const closeContent = [
    t('terminal.closeContent1'),
    t('terminal.closeContent2'),
  ];

  const handleCloseTerminalTempory = () => {
    setCloseTerminalOpen(true);
  };

  const handleCollapseTerminalMenu = () => {
    setCollapseTerminalMenu(!collapseTerminalMenu);
    setTimeout(() => {
      if (terminalRef.current) {
        terminalRef.current.resizeScreenConsole();
      }
    });
  };

  const handleChangeScreenSize = (size) => {
    terminalStore.$a.setTerminalScreenSize(size);
    setTimeout(() => {
      if (terminalRef.current) {
        terminalRef.current.resizeScreenConsole();
      }
    });
  };

  const handleCloseTerminalModal = () => {
    setCloseTerminalOpen(false);
  };

  const handleConfirmTerminalModal = () => {
    setCloseTerminalOpen(false);
    terminalStore.$a.setTerminalOpen(false);
    terminalStore.$a.setTerminalScreenSize('middle');
  };

  useEffect(() => {
    if (terminalStore.$s.screenSize !== 'small') {
      setIsShowModal(true);
    } else {
      setIsShowModal(false);
    }
  }, [terminalStore.$s.screenSize]);

  return <div className={`terminal_box ${terminalStore.$s.screenSize}_screen_window`}>
    <div className='terminal_header'>
      <div className='terminal_host_name'>
        <span>
          {t(terminalStore.$s.type === 'container' ? 'terminal.containerTitlePrefix' : 'terminal.clusterTitlePrefix')}
          <span className='host_desc'>{terminalStore.$s.name}</span>
        </span>
      </div>
      <div className='terminal_tools'>
        {terminalStore.$s.screenSize !== 'small' ? <LineOutlined onClick={() => handleChangeScreenSize('small')} /> : <ExpandAltOutlined onClick={() => handleChangeScreenSize('middle')} />}
        {terminalStore.$s.screenSize !== 'large' ? <ExpandOutlined onClick={() => handleChangeScreenSize('large')} /> : <CompressOutlined onClick={() => handleChangeScreenSize('middle')} />}
        <CloseOutlined onClick={handleCloseTerminalTempory} />
      </div>
    </div>
    <div className='terminal_content'>
      <div className='terminal_console_actually'>
        <TerminalConsole ref={terminalRef} />
      </div>
      {terminalStore.$s.type === 'cluster' && <div className={`terminal_example_menu  ${collapseTerminalMenu ? 'collapse_terminal_example_menu' : ''}`}>
        {collapseTerminalMenu ?
          <CaretLeftOutlined className='terminal_handle_menu' onClick={handleCollapseTerminalMenu} />
          : <CaretRightOutlined className='terminal_handle_menu' onClick={handleCollapseTerminalMenu} />}
        <div className='terminal_menu_content'>
          {handleList.map(item => <div className='terminal_handle_box' key={item.name}>
            <div className='desc_word'>
              <p>{t('terminal.operationGuide')}</p>
              <p>{item.name}</p>
            </div>
            <div className='desc_word'>
              <p>{t('terminal.commandExample')}</p>
              <p>{item.example}</p>
            </div>
          </div>)}
          <p className='more_terminal_aim'>{t('terminal.kubectlReference')}<span><a href={publicLink.kubectlLink} target='_blank'>{t('terminal.learnMore')}<RightOutlined /></a></span></p>
        </div>
      </div>}
    </div>
    <Modal
      className='modal_flex_delete terminal_console_modal'
      open={closeTerminalOpen}
      title={t('terminal.closeTerminal')}
      getContainer={isShowModal ? false : () => document.body}
      onCancel={handleCloseTerminalModal}
      footer={
        <>
          <Button className='cancel_btn' onClick={handleCloseTerminalModal}>{t('actions.cancel')}</Button>
          <Button className='confirm_btn primary_btn' onClick={handleConfirmTerminalModal}>{t('actions.confirm')}</Button>
        </>
      }>
      <div className='modal_delete_content'>
        <ExclamationCircleFilled className='warn_icon' />
        <div className='word_tograry'>
          <div className='word_tograry_alert'>
            {closeContent.map(item => (
              <p className='bread_word_delete' key={item}>{item}</p>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  </div>;
}
