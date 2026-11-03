/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import DataAreaChart from '@/components/echarts/DataAreaChart';
import { solveBytesToMiB, solveUnitMonitor, solveRate } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useTranslation } from 'react-i18next';

export default function Addon({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.addon.request')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.requestLatency')}
            data={solveBytesToMiB(data.coredns_requests_latency, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_requests_latency"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.requestCount')}
            rangeColorProps={propsColor}
            data={solveBytesToMiB(data.coredns_requests_rate, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="coredns_requests_rate"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.nxdomainRate')}
            data={solveRate(data.coredns_NXDOMAIN_response_rate)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.addon.nxdomainTooltip')}</p>
            </div>}
            id="coredns_NXDOMAIN_response_rate"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.panicCount')}
            data={solveBytesToMiB(data.coredns_panic_count, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_panic_count"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.addon.responseSize')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.udpResponse', { unit: solveUnitMonitor(data?.coredns_udp_responses_size_bytes) })}
            data={solveBytesToMiB(data?.coredns_udp_responses_size_bytes)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_udp_responses_size_bytes"
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.addon.udpTooltip')}</p>
            </div>}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.tcpResponse', { unit: solveUnitMonitor(data?.coredns_tcp_responses_size_bytes) })}
            data={solveBytesToMiB(data?.coredns_tcp_responses_size_bytes)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_tcp_response"
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.addon.tcpTooltip')}</p>
            </div>}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.addon.cacheTitle')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.cacheSize')}
            data={solveBytesToMiB(data.coredns_cache_size, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_cache_size"
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.addon.cacheSizeTooltip')}</p>
            </div>}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.addon.cacheHitrate')}
            data={solveRate(data.coredns_cache_hitrate)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="coredns_cache_hitrate"
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.addon.cacheHitrateTooltip')}</p>
            </div>}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
