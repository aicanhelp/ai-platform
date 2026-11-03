/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState, useStore } from 'openinula';
import { containerRouterPrefix } from '@/constant.js';
import { useHistory } from 'inula-router';
import extensionManagement from '@/assets/images/extensionManagement.png';
import applicationExtendDefault from '@/assets/images/application_extend_default.png';
import { useTranslation } from 'react-i18next';
import { convertToRawUrl } from '@/tools/utils';

export default function ApplicationRecommendItem({ itemMeta, repo, types }) {
  const history = useHistory();
  const themeStore = useStore('theme');
  const { t } = useTranslation('appMarket');
  const repoType = (type) => {
    switch (type) {
      case 'openFuyao':
        return t('details.providedBy.official');
      case 'local':
        return t('details.providedBy.local');
      default:
        return t('details.providedBy.user');
    }
  };
  const isSelectExtensionComponent = itemMeta?.keywords
    ? itemMeta?.keywords?.includes('openfuyao-extension')
    : false;
  const handleViewDetails = () => {
    history.push(
      `/${containerRouterPrefix}/appMarket/marketCategory/ApplicationDetails/${itemMeta.name}/${repo}/${itemMeta.version}`,
    );
  };
  const [imgSrc, setImgSrc] = useState(() => {
    const initialUrl = itemMeta.icon || applicationExtendDefault;
    return convertToRawUrl(initialUrl);
  });

  const handleImgError = () => {
    setImgSrc(applicationExtendDefault);
  };

  useEffect(() => {
    const initialUrl = itemMeta.icon || applicationExtendDefault;
    setImgSrc(convertToRawUrl(initialUrl));
  }, [itemMeta]);

  return (
    <div className='application_item' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34' }} onClick={handleViewDetails}>
      <div>
        <div className='application_item_header'>
          <div className='application_item_logo'>
            <img src={imgSrc} alt='' onError={handleImgError} />
          </div>
          <div>
            <div className='application_item_title clamp_2' style={{ color: themeStore.$s.theme === 'light' ? 'black' : '#f7f7f7' }}>{itemMeta.name}</div>
            <div className='application_item_text'>{repoType(types ? repo : itemMeta.repo)}</div>
          </div>
        </div>
        <div className='application_item_text clamp_3'>{itemMeta.description}</div>
      </div>
      {isSelectExtensionComponent && (
        <div className='extension_label'>
          <img src={extensionManagement} />
          <div className='application_item_text'>{t('details.extension')}</div>
        </div>
      )}
    </div>
  );
}