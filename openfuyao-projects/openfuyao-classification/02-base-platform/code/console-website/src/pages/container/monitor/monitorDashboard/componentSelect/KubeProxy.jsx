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

export default function KubeProxy({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeProxy.networkProgramming')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubeProxy.networkProgrammingTitle')}
          data={data.kube_proxy_network_programming_rate}
          type="area"
          rangeColorProps={propsColor}
          id="kube_proxy_network_programming_rate"
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeProxy.qps')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeProxy.qpsTitle')}
            data={solveBytesToMiB(data.requestRate, false)}
            type="line"
            rangeColorProps={propsColor}
            id="kube_proxy_kube_api"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container KubeProxy'>
      <p className='chart_title'>{t('chart.kubeProxy.resource')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            className='KubeProxy'
            title={t('chart.kubeProxy.cpuUsage')}
            data={solveBytesToMiB(data.kube_proxy_cpu_usage, false)}
            type="line" id="proxy_kube_proxy_cpu_usage"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeProxy.memoryUsage', { unit: solveUnitMonitor(data?.kube_proxy_memory_usage) })}
            data={solveBytesToMiB(data?.kube_proxy_memory_usage)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="proxy_kube_proxy_memory_usage"
            className='KubeProxy'
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container KubeProxy'>
      <div className='monitor_half each_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeProxy.goroutines')}
            data={data.kube_proxy_goroutines}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            className='KubeProxy'
            id="proxy_kube_proxy_goroutines"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
