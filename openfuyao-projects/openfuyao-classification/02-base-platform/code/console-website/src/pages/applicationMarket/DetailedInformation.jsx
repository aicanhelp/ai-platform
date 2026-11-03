/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useStore } from 'openinula';
import MdComponent from '@/pages/applicationMarket/component/MdComponent';
import NoContent from '@/assets/images/noContent.png';
import { Skeleton } from 'antd';
import { useTranslation } from 'react-i18next';

export default function DetailedInformation({ detailInfo, loading }) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('appMarket');

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: themeStore.$s.theme === 'light' ? '#f7f7f7' : '#2a2d34ff' }}>
        <Skeleton active />
      </div>
    );
  }

  // 当 detailInfo 不存在时显示无内容状态
  if (!detailInfo) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', height: '100%', padding: '1px 32px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', alignItems: 'center', minHeight: '400px', justifyContent: 'center' }}>
          <div><img src={NoContent} alt={t('messages.noDetailInfo')} /></div>
          <div style={{ color: '#89939b', fontSize: '14px' }}>{t('messages.noDetailInfo')}</div>
        </div>
      </div>
    );
  }

  // 当 detailInfo 存在时渲染详情信息
  return (
    <div style={{ display: 'flex', flexDirection: 'column', padding: '1px 32px', minHeight: '400px', justifyContent: 'center', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <MdComponent markdownContent={detailInfo.toString()} />
    </div>
  );
}
