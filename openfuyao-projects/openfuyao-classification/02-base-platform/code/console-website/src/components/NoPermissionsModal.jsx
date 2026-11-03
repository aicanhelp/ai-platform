/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import NoPermission from '@/assets/images/noPermission.png';
import { Modal, Button } from 'antd';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import '@/styles/components/noPermissionsModal.less';

export default function NoPermissionsModal({
  open,
  handleOpen,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('common');
  return <Modal width={682} open={open} className={`no_permission_modal  ${themeStore.$s.theme === 'light' ? '' : 'dark_per_modal'}`} footer={null} onCancel={handleOpen}>
    <div className='content_per'>
      <img src={NoPermission} alt='no permissions' style={{ width: '293px', height: '234px', marginBottom: '32px' }} />
      <p style={{ fontSize: '16px', color: '#89939B' }}>{t('error.noAccessPermission')}</p>
      <p style={{ fontSize: '16px', color: '#89939B', lineHeight: '24px' }}>{t('error.contactAdmin')}</p>
      <Button className='confirm_per_btn' onClick={handleOpen}>{t('actions.confirm')}</Button>
    </div>
  </Modal>;
} 