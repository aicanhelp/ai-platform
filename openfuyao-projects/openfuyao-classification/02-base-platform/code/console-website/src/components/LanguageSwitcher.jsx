/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { useTranslation } from 'react-i18next';
import { Tooltip } from 'antd';
import { GlobalOutlined } from '@ant-design/icons';
import '@/styles/components/languageSwitcher.less';

/**
 * Language Switcher Component
 * Globe icon with floating language label at bottom-right
 */
export default function LanguageSwitcher() {
  const { i18n, t } = useTranslation();

  const currentLang = i18n.language || 'zh-CN';
  const isChinese = currentLang === 'zh-CN';

  const handleToggle = () => {
    const newLang = isChinese ? 'en-US' : 'zh-CN';
    i18n.changeLanguage(newLang);
    localStorage.setItem('i18nextLng', newLang);
  };

  return (
    <div className="tool_theme">
      <Tooltip
        title={t('language.toggle', { ns: 'common' })}
        placement="bottom"
      >
        <div
          className="language-switcher-container tool_icon"
          onClick={handleToggle}
        >
          <GlobalOutlined className="language-switcher-globe" />
          <span className="language-switcher-label">
            {isChinese ? 'EN' : '中'}
          </span>
        </div>
      </Tooltip>
    </div>
  );
}
