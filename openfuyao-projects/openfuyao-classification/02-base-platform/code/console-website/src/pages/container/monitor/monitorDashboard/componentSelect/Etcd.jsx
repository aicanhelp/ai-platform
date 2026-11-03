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
import { solveBytesToMiB, solveUnitMonitor } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function Etcd({
  etcdData,
  loaded,
  propsColor,
  refreshFn,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('chart.etcd.leaderTitle')}</p>
      {loaded
        ? <DataAreaChart
          id="status"
          title={t('chart.etcd.leaderStatus')}
          rangeColorProps={propsColor}
          data={[
            ...(etcdData.etcd_server_has_leader ? etcdData.etcd_server_has_leader : []),
            ...(etcdData.etcd_server_is_leader ? etcdData.etcd_server_is_leader : []),
          ]}
          tooltip={<div className='tooltip_box'>
            <p>{t('chart.etcd.leaderTooltip1')}</p>
            <p>{t('chart.etcd.leaderTooltip2')}</p>
            <p>{t('chart.etcd.leaderTooltip3')}</p>
          </div>}
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      {loaded
        ? <DataAreaChart
          title={t('chart.etcd.leaderChanges')}
          data={etcdData.etcd_server_leader_changes_seen_1day}
          type="area"
          id="etcd_server_leader_changes_seen_1day"
          rangeColorProps={propsColor}
          refreshFn={refreshFn}
          tooltip={<div className='tooltip_box'>
            <p>{t('chart.etcd.leaderChangesTooltip')}</p>
          </div>}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('chart.etcd.keysTotal')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.etcd.keysTotalTitle')}
          data={etcdData.etcd_debugging_mvcc_keys_total}
          type="line"
          id="debug"
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />}
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('chart.etcd.latency')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.etcd.latencyTitle')}
          data={etcdData.etcd_kube_api_request_latency}
          type="line"
          id="kube_api"
          rangeColorProps={propsColor}
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('chart.etcd.resource')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.etcd.cpuUsage')}
            data={solveBytesToMiB(etcdData.etcd_cpu_usage, false)}
            type="line"
            id="cpu"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.etcd.memoryUsage', { unit: solveUnitMonitor(etcdData.etcd_memory_usage) })}
            data={solveBytesToMiB(etcdData.etcd_memory_usage)}
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
            type="line"
            id="memory"
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('chart.etcd.diskSize')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.etcd.diskSizeTitle', {
            unit: solveUnitMonitor([
              ...(etcdData.etcd_backend_db_usage ? etcdData.etcd_backend_db_usage : []),
              ...(etcdData.etcd_backend_db_total ? etcdData.etcd_backend_db_total : []),
            ]),
          })}
          data={solveBytesToMiB([
            ...(etcdData.etcd_backend_db_usage ? etcdData.etcd_backend_db_usage : []),
            ...(etcdData.etcd_backend_db_total ? etcdData.etcd_backend_db_total : []),
          ])}
          type="area"
          id="disk"
          rangeColorProps={['#43CBB8', '#4B8BEA']}
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
  </>;
}
