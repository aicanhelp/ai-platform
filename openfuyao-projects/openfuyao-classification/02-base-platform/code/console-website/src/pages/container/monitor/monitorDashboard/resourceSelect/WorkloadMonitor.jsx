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

export default function WorkloadMonitor({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.workload.cpu')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.workload.cpuUsage')}
          data={solveBytesToMiB(data.workload_cpu_usage, false)}
          type="line"
          id="workload_cpu_usage"
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.workload.memory')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.workload.memoryUsage', { unit: solveUnitMonitor(data.workload_memory_usage) })}
          data={solveBytesToMiB(data.workload_memory_usage)}
          type="line"
          id="workload_memory_usage"
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.workload.packetRate')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.workload.receivedRate')}
            data={solveBytesToMiB(data.workload_network_packets_received, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="workload_network_packets_received"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.workload.transmittedRate')}
            data={solveBytesToMiB(data.workload_network_packets_transmitted, false)}
            type="line"
            id="workload_network_packets_transmitted"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.workload.droppedPacketRate')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.workload.receivedDroppedRate')}
            data={solveBytesToMiB(data.workload_network_packets_received_dropped, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="workload_network_packets_received_dropped"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.workload.transmittedDroppedRate')}
            data={solveBytesToMiB(data.workload_network_packets_transmitted_dropped, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="workload_network_packets_transmitted_dropped"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
