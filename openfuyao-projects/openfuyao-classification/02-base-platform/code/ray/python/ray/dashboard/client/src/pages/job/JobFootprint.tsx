import { EChartsOption } from "echarts";
import ReactECharts from "echarts-for-react";
import React from "react";

export type JobFootprintProps = {
  footprint: { [node_id: string]: { [name: string]: number } };
};

export const JobFootprint = ({ footprint }: JobFootprintProps) => {
  const node_id_list = Object.keys(footprint);
  const name_set = new Set<string>();
  for (const node_id in footprint) {
    for (const name in footprint[node_id]) {
      name_set.add(name);
    }
  }
  const series: echarts.SeriesOption[] = [];
  name_set.forEach((name: string) => {
    series.push({
      name: name,
      type: "bar",
      stack: "total",
      label: {
        show: false,
      },
      emphasis: {
        focus: "series",
      },
      data: node_id_list.map((node_id: string) => {
        if (footprint) {
          return footprint[node_id][name];
        } else {
          return 0;
        }
      }),
    });
  });

  const option: EChartsOption = {
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
    },
    legend: {},
    // animation: false,
    grid: {
      left: "3%",
      right: "4%",
      bottom: "3%",
      containLabel: true,
    },
    xAxis: {
      type: "value",
    },
    yAxis: {
      type: "category",
      data: node_id_list,
    },
    series: series,
  };
  const height = node_id_list.length * 50 + 100 + "px";
  return <ReactECharts option={option} style={{ height: height }} />;
};

export default JobFootprint;
