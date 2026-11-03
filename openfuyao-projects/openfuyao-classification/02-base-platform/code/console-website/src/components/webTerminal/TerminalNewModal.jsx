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
import { ExclamationCircleFilled } from '@ant-design/icons';
import '@/styles/components/deleteInfoModal.less';
import { Fragment, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function TerminalNewModal({
  open,
  cancelFn,
  handleOk,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('common');
  return <Modal className={`modal_flex_delete ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`} open={open} title={t('actions.info')} onCancel={cancelFn} destroyOnClose
    footer={[
      <Fragment>
        <Button className='cancel_btn' onClick={cancelFn}>{t('actions.cancel')}</Button>
        <Button className='primary_btn' onClick={handleOk}>{t('actions.start')}</Button>
      </Fragment>,
    ]}>
    <div className='modal_delete_content'>
      <ExclamationCircleFilled className='warn_icon' />
      <div className='word_tograry'>
        <div className='word_tograry_alert'>
          <p className='bread_word_delete'>{t('terminal.newTerminalWarning')}</p>
        </div>
      </div>
    </div>
  </Modal>;
}
