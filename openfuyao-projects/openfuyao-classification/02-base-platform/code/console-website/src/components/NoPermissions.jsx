/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import NoContent from '@/assets/images/noPermission.png';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function NoPermissions() {
  const themeStore = useStore('theme');
  const { t } = useTranslation('common');
  return <div style={{
    display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column',
    background: themeStore.$s.theme === 'light' ? '#f7f7f7' : '#171a1f',
    height: 'calc(100vh - 64px)',
  }}
  >
    <img src={NoContent} alt='no permissions' style={{ width: '293px', height: '234px', marginBottom: '32px' }} />
    <p style={{ fontSize: '16px', color: '#89939B' }}>{t('error.noAccessPermission')}</p>
    <p style={{ fontSize: '16px', color: '#89939B', lineHeight: '24px' }}>{t('error.contactAdmin')}</p>
  </div>;
} 