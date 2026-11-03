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

export default function KubeScheduler({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeScheduler.overview')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubeScheduler.pendingPods')}
          data={data.kube_scheduler_pending_pods_count}
          type="area"
          rangeColorProps={propsColor}
          tooltip={<div className='tooltip_box'>
            <p>{t('chart.kubeScheduler.pendingPodsTooltip1')}</p>
            <p>{t('chart.kubeScheduler.pendingPodsTooltip2')}</p>
            <p>{t('chart.kubeScheduler.pendingPodsTooltip3')}</p>
            <p>{t('chart.kubeScheduler.pendingPodsTooltip4')}</p>
          </div>}
          id="kube_scheduler_pending_pods_count"
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubeScheduler.schedulingAttempts')}
          data={data.kube_scheduler_pod_scheduling_attempts}
          type="area"
          rangeColorProps={propsColor}
          tooltip={<div className='tooltip_box'>
            <p>{t('chart.kubeScheduler.schedulingAttemptsTooltip')}</p>
          </div>}
          id="kube_scheduler_pod_scheduling_attempts"
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeScheduler.qps')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubeScheduler.qpsTitle')}
          data={solveBytesToMiB(data.requestRate, false)}
          type="line"
          rangeColorProps={propsColor}
          id="kube_scheduler_kube_api_request_rate"
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeScheduler.resource')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeScheduler.cpuUsage')}
            data={solveBytesToMiB(data.kube_scheduler_cpu_usage, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            className='KubeScheduler'
            id="resource_cpu"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeScheduler.memoryUsage', { unit: solveUnitMonitor(data?.kube_scheduler_memory_usage) })}
            data={solveBytesToMiB(data?.kube_scheduler_memory_usage)}
            type="line"
            className='KubeScheduler'
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="resource_memory"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
