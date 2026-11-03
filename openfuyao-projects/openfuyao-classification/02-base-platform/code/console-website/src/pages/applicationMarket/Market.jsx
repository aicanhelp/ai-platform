/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Banner } from './Banner';
import { ConfigProvider, Input, message, theme } from 'antd';

const { Search } = Input;
import '@/styles/applicationMarket/index.less';
import { useState, useEffect, useStore } from 'openinula';
import { getRecommendHelmChartList, getExtensionHelmChartList } from '@/api/applicationMarketApi';
import MarketCategoryItem from '@/pages/applicationMarket/MarketCategoryItem';
import aiml from '@/assets/images/marketplaceIcon/aiml.svg';
import computing from '@/assets/images/marketplaceIcon/computing.svg';
import database from '@/assets/images/marketplaceIcon/database.svg';
import devTool from '@/assets/images/marketplaceIcon/devtool.svg';
import cicd from '@/assets/images/marketplaceIcon/cicd.svg';
import monitoring from '@/assets/images/marketplaceIcon/monitoring.svg';
import logging from '@/assets/images/marketplaceIcon/logging.svg';
import network from '@/assets/images/marketplaceIcon/network.svg';
import observability from '@/assets/images/marketplaceIcon/observability.svg';
import security from '@/assets/images/marketplaceIcon/security.svg';
import storage from '@/assets/images/marketplaceIcon/storage.svg';
import allCategory from '@/assets/images/marketplaceIcon/allCategory.svg';
import turboCertificate from '@/assets/images/turboCertificate.png';
import enginePlug from '@/assets/images/enginePlug.png';
import turboIcon from '@/assets/images/turboIcon.png';
import { useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import turboBGCBack from '@/assets/images/turboBackground.png';
import turboBGCBackDark from '@/assets/images/turboBackgroundDark.png';
import turboBGCBackHover from '@/assets/images/turboBackgroundHover.png';
import { MarketBlock } from './MarketBlock';
import { ResponseCode } from '@/common/constants';
import { useTranslation } from 'react-i18next';

const limitSearch = 53; // Why?

export default function Market() {
  const { t } = useTranslation('appMarket');
  const categoryList = [
    {
      icon: aiml,
      title: t('category.artificialIntelligence'),
      value: 'artificial-intelligence',
    },
    {
      icon: computing,
      title: t('category.computing'),
      value: 'computing',
    },
    {
      icon: database,
      title: t('category.database'),
      value: 'database',
    },
    {
      icon: devTool,
      title: t('category.developerTool'),
      value: 'developer-tool',
    },
    {
      icon: cicd,
      title: t('category.cicd'),
      value: 'CI/CD',
    },
    {
      icon: monitoring,
      title: t('category.monitor'),
      value: 'monitor',
    },
    {
      icon: logging,
      title: t('category.log'),
      value: 'log',
    },
    {
      icon: network,
      title: t('category.network'),
      value: 'network',
    },
    {
      icon: observability,
      title: t('category.observability'),
      value: 'observability',
    },
    {
      icon: security,
      title: t('category.security'),
      value: 'security',
    },
    {
      icon: storage,
      title: t('category.storage'),
      value: 'storage',
    },
    {
      icon: allCategory,
      title: t('category.title'),
      value: 'null',
    },
  ];
  const history = useHistory();
  const [customInfo, setCustomInfo] = useState([]);
  const [recommendInfo, setRecommendInfo] = useState([]);
  const [turboInfo, setTurboInfo] = useState([]);
  const [recommendLoaded, setRecommendLoaded] = useState(false); //
  const [turboLoaded, setTurboLoaded] = useState(false);
  const [customeLoaded, setCustomeLoaded] = useState(false);
  const [recommendError, setRecommendError] = useState(false); // 是否错误
  const [turboError, setTurboError] = useState(false);
  const [customeError, setCustomeError] = useState(false);
  const themeStore = useStore('theme');

  const loadHelmChartList = async ({ fetchFn, extractItems, setList, setLoaded, setError }) => {
    setLoaded(false);
    try {
      const res = await fetchFn();
      if (res.status === ResponseCode.OK) {
        setList(extractItems(res));
      } else {
        setList([]);
        setError(true);
      }
    } catch {
      setList([]);
      setError(true);
    }
    setLoaded(true);
  };

  const getRecommendHelmChartListInfo = () =>
    loadHelmChartList({
      fetchFn: () => getRecommendHelmChartList(),
      extractItems: (res) => (res.data?.data?.length && res.data?.data?.[0]?.Charts ? res.data.data[0].Charts.slice(0, 4) : []),
      setList: setRecommendInfo,
      setLoaded: setRecommendLoaded,
      setError: setRecommendError,
    });

  const getCustomedHelmChartListInfo = () =>
    loadHelmChartList({
      fetchFn: () => getExtensionHelmChartList('openfuyao-extension'),
      extractItems: (res) => (res.data?.data?.items?.length ? res.data.data.items.slice(0, 4) : []),
      setList: setCustomInfo,
      setLoaded: setCustomeLoaded,
      setError: setCustomeError,
    });

  const getTurboHelmChartListInfo = () =>
    loadHelmChartList({
      fetchFn: () => getRecommendHelmChartList('openfuyao-accelerator'),
      extractItems: (res) => (res.data?.data?.length && res.data?.data?.[0]?.Charts ? res.data.data[0].Charts.slice(0, 4) : []),
      setList: setTurboInfo,
      setLoaded: setTurboLoaded,
      setError: setTurboError,
    });

  const handleInput = (e) => {
    if (e.target.value.length >= limitSearch) {
      message.error(t('market.searchLimit', { limit: limitSearch }));
    }
  };

  const handleSearch = (e) => {
    history.push({
      pathname: `/${containerRouterPrefix}/appMarket/marketCategory/null`,
      state: {
        isQuery: e,
      },
    });
  };
  useEffect(() => {
    getRecommendHelmChartListInfo();
    getCustomedHelmChartListInfo();
    getTurboHelmChartListInfo();
  }, []);
  return (
    <ConfigProvider theme={{ algorithm: themeStore.$s.theme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <div style={{ height: 'calc(100vh - 112px)', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#171a1f', marginBottom: 0 }}>
        <Banner />
        <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34', padding: '28px 0' }}>
          <div className='market_recommend'><Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.app', { ns: 'common' }) })} onChange={handleInput} onSearch={handleSearch} maxLength={limitSearch}/></div>
          <MarketBlock
            title={t('market.recommend')}
            loaded={recommendLoaded}
            link={`/${containerRouterPrefix}/appMarket/marketCategory/null`}
            data={recommendInfo}
          >
            <div style={{ display: 'flex', justifyContent: 'center', border: '1px solid #dcdcdcff', borderRadius: '4px', padding: '32px 0' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}> <img src={enginePlug} /></div>
                <div style={{ color: '#89939b', textAlign: 'center' }}>{recommendError ? t('market.fetchError') : t('market.noData')}</div>
              </div>
            </div>
          </MarketBlock>
        </div>

        <div className='recommend_source_category_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#171a1f' }}>
          <div className='recommend_source_category'>
            {
              categoryList.map((el) => (
                <MarketCategoryItem
                  key={el.value}
                  icon={el.icon}
                  categoryTitle={el.title}
                  tipIcon={el.tipIcon}
                  value={el.value}
                />
              ))
            }
          </div>
        </div>

        <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34', padding: '28px 0' }}>
          <MarketBlock
            title={t('market.accelerate')}
            loaded={turboLoaded}
            link={`/${containerRouterPrefix}/appMarket/marketCategory/null/false/true`}
            data={turboInfo}
            pic={enginePlug}
          >
            <div style={{ display: 'flex', justifyContent: 'center', border: '1px solid #dcdcdcff', borderRadius: '4px', padding: '32px 0' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}> <img src={enginePlug} /></div>
                <div style={{ color: '#89939b', textAlign: 'center' }}>{turboError ? t('market.fetchError') : t('market.noData')}</div>
              </div>
            </div>
          </MarketBlock>

          <MarketBlock
            title={t('market.extension')}
            link={`/${containerRouterPrefix}/appMarket/marketCategory/null/true`}
            data={customInfo}
            loaded={customeLoaded}
          >
            <div style={{ display: 'flex', justifyContent: 'center', border: '1px solid #dcdcdcff', borderRadius: '4px', padding: '32px 0' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}> <img src={enginePlug} /></div>
                <div style={{ color: '#89939b', textAlign: 'center' }}>{customeError ? t('market.fetchError') : t('market.noData')}</div>
              </div>
            </div>
          </MarketBlock>
        </div>

        <div className='turbo_certificate' style={{ display: 'none' }}>
          <div className='turbo_title'>
            {t('market.turbo.title')}
          </div>
          <div className='turbo_body'>
            <div className='turbo_bgc'>
              <img className='turbo_bgc_back' src={themeStore.$s.theme === 'light' ? turboBGCBack : turboBGCBackDark} alt='' />
              <img className='turbo_bgc_back_hover' src={turboBGCBackHover} alt='' />
              <div className='turbo_bgc_clip'>
                <div className='turbo_bgc_frame'>
                  <div className='turbo_bgc_icon'><img src={turboIcon} alt='turbo' /></div>
                  <div className='turbo_bgc_title'>{t('market.turbo.whatIs')}</div>
                  <div className='turbo_bgc_text'>
                    {t('market.turbo.description')}
                  </div>
                </div>
              </div>
            </div>
            <div className='turbo_bgc'>
              <img className='turbo_bgc_back' src={themeStore.$s.theme === 'light' ? turboBGCBack : turboBGCBackDark} alt='' />
              <img className='turbo_bgc_back_hover' src={turboBGCBackHover} alt='' />
              <div className='turbo_bgc_clip'>
                <div className='turbo_bgc_frame'>
                  <div className='turbo_bgc_icon'><img src={turboCertificate} alt='turbo' /></div>
                  <div className='turbo_bgc_title'>{t('market.turbo.howToGet')}</div>
                  <div className='turbo_bgc_text'>
                    <li>{t('market.turbo.step1')}</li>
                    <li>{t('market.turbo.step2')}</li>
                    <li>{t('market.turbo.step3')}</li>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className='turbo_trip'>
            {t('market.turbo.comingSoon')}
          </div>
        </div>
      </div >
    </ConfigProvider >
  );
}
