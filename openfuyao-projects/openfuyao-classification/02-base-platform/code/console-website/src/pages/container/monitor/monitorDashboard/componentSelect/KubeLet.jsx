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
import { useTranslation } from 'react-i18next';

export default function KubeLet({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubelet.storageOperation')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubelet.storageRate')}
            data={solveBytesToMiB(data.kubelet_storage_operation_rate, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="kubelet_storage_rate"
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.kubelet.storageRateTooltip')}</p>
            </div>}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded ?
          <DataAreaChart
            title={t('chart.kubelet.storageErrorRate')}
            data={solveBytesToMiB(data.kubelet_storage_operation_error_rate, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.kubelet.storageErrorRateTooltip')}</p>
            </div>}
            id="kubelet_storage_error_rate"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubelet.plegInterval')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubelet.plegIntervalTitle')}
          data={solveBytesToMiB(data.kubelet_pleg_relist_interval, false)}
          type="line"
          rangeColorProps={propsColor}
          id="kubelet_pleg_relist"
          tooltip={<div className='tooltip_box'>
            <p>{t('chart.kubelet.plegTooltip')}</p>
          </div>}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubelet.qps')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubelet.qpsTitle')}
          data={solveBytesToMiB(data.requestRate, false)}
          type="line"
          rangeColorProps={propsColor}
          id="kubelet_kube_api"
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubelet.resource')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubelet.cpuUsage')}
            data={solveBytesToMiB(data.kubelet_cpu_usage, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="kubelet_cpu_usage"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubelet.memoryUsage', { unit: solveUnitMonitor(data?.kubelet_memory_usage) })}
            data={solveBytesToMiB(data?.kubelet_memory_usage)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="kubelet_memory_usage"
            refreshFn={refreshFn}
            className='KubeLet'
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container KubeLet'>
      <div className='monitor_half each_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubelet.goroutines')}
            data={data.kubelet_goroutines}
            type="line"
            className='KubeLet'
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="kubelet_goroutines"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
