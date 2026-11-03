/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import EchartsInit from '@/components/EchartsInit';
import { Fragment, useCallback, useEffect, useState } from 'openinula';
import { getMappingValueArrayOfKey, getMappingValueArrayOfKeyChart, repeatFilter } from '@/tools/utils';
import dayjs from 'dayjs';
import { fixedColor } from '@/tools/utils';
import * as echarts from 'echarts';
import '@/styles/components/echartsCom.less';
import NumberAimCom from '@/components/NumberAimCom';
import { MoreOutlined, ExclamationCircleFilled } from '@ant-design/icons';
import { Popconfirm, Popover, message, Space, Button, Tooltip } from 'antd';
import { useHistory, useParams } from 'inula-router';
import { useIntl } from 'inula-intl';
import { deleteWidgetReousrce, getIndicatorsList } from '@/api/monitorApi';
import {
  requestResource,
  displayTooltip,
  solveEchartsDashboardLength,
  solveDashboardValue,
  escapeHtml,
} from '@/tools/utils';
import { ResponseCode } from '@/common/constants';
import useLocalStorage from '@/hooks/useLocalStorage';

export default function EchartsCom({
  data,
  uuid = '', // 组件标识
  type = 'line',
  id = 'preview_chart',
  isPreview = true,
  isAimExist = false,
  dashboardMax = 100,
  dashboardMin = 0,
  name = '--',
  isMore = false,
  widgets = {},
  refreshFn = () => {},
}) {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const history = useHistory();
  const { dashboardID, dashboardName } = useParams();
  const [options, setOptions] = useState({}); // 配置
  const [messageApi, contextHolder] = message.useMessage();
  const [optionsDashBoard, setOptionsDashboard] = useState([]); // 仪表盘
  const [numberAimList, setNumberAimList] = useState([]); // 数字指标
  const [totalTypeList, setTotalTypeList] = useState([]);
  const [popOpen, setPopOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const solveStyleDount = () => {
    let left = 0;
    if (window.innerWidth <= 1400) {
      left = -150;
    } else if (window.innerWidth > 1400 && window.innerWidth < 1920) {
      left = -200;
    } else {
      left = -350;
    }
    return left;
  };

  const renderLineOptions = (typeList, dataList, isArea = false) => {
    let timestampList = [];
    [...new Set(getMappingValueArrayOfKey(dataList, 'timeStamp'))].map((item) => {
      timestampList.push(item);
    });
    if (timestampList.filter((item) => item !== '--').length) {
      timestampList = timestampList.filter((item) => item !== '--');
    }
    let dataSeries = [];
    typeList.map((item, index) => {
      let obj = {};
      obj.name = item;
      obj.type = 'line';
      obj.symbol = 'none';
      obj.smooth = true;
      isArea
        ? (obj.areaStyle = {
            opacity: 0.3,
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {
                offset: 0,
                color: fixedColor()[index],
              },
              {
                offset: 1,
                color: `${fixedColor()[index]}00`,
              },
            ]),
          })
        : '';
      obj.data = getMappingValueArrayOfKeyChart(
        dataList.filter((filterItem) => filterItem.type === item),
        'value',
      );
      dataSeries.push(obj);
    });
    let option = {
      color: fixedColor(),
      title: {
        textStyle: {
          color: '#333333',
          fontWeight: 400,
          fontSize: 14,
          lineHeight: 20,
        },
      },
      tooltip: {
        className: 'custom-tooltip',
        trigger: 'axis',
        appendToBody: true,
        confine: true,
        enterable: true,
        extraCssText: `max-width:400px;max-height:400px;overflow:auto;
        ${currentTheme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        formatter: (params) => {
          let result = `<span style=${currentTheme === 'light' ? '#666' : '#89939B'}>${dayjs(params[0].name * 1000).format('YYYY-MM-DD HH:mm:ss')}</span>`;
          params.forEach((item, index) => {
            if (!isNaN(item.value)) {
              let { color } = item;
              let dom = `<span style="display:inline-block;margin-right:5px;margin-bottom:5px;border-radius:1px;width:14px;height:4px;background-color:${color};"></span>
              <span style=${currentTheme === 'light' ? '#666' : '#f7f7f7'}>${item.seriesName}: ${item.value}</span><br/>
              <span style="color:#40444D;position:relative;top:10px;">${displayTooltip(item.data.aimList)}</span>`;
              result += `<br /> ${dom}`;
            }
          });
          return result;
        },
      },
      legend: {
        type: 'scroll',
        orient: 'vertical',
        right: 0,
        top: 'center',
        icon: 'rect',
        itemHeight: 4,
        itemWidth: 16,
        textStyle: {
          color: currentTheme === 'light' ? '#333333' : '#f7f7f7',
          rich: {
            normal: {
              fontSize: 12,
              lineHeight: 16,
            },
            label: {
              padding: [4, 0, 0, 0],
            },
            value: {
              fontSize: 12,
              lineHeight: 16,
            },
          },
        },
        formatter: (legendItem) => {
          let tip1 = '';
          let tip = '';
          let le = legendItem.length;
          let length = 30;
          if (window.innerWidth < 1280) {
            length = 15;
          }
          if (window.innerWidth > 1460 && window.innerWidth <= 1740) {
            length = 20;
          }
          if (window.innerWidth >= 1280 && window.innerWidth <= 1460) {
            length = 18;
          }

          if (le > length) {
            let l = Math.ceil(le / length);
            for (let i = 1; i <= l; i++) {
              if (i < l) {
                tip1 += `${legendItem.slice(i * length - length, i * length)}\n`;
              } else {
                tip = tip1 + legendItem.slice((l - 1) * length, le);
              }
            }
            return `{value|${tip}}`;
          } else {
            tip = legendItem;
            return `{normal|${tip}}`;
          }
        },
        data: typeList,
      },
      grid: {
        left: '2%',
        top: '5%',
        right: '15%',
        bottom: '5%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: timestampList,
        axisLabel: {
          formatter: (e) => {
            return dayjs(e * 1000).format('MM/DD\nHH:mm:ss');
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          show: !typeList.length ? false : true,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: '#dcdcdc',
          },
        },
        min: 0,
        max: !typeList.length ? 500 : null,
      },
      series: dataSeries,
    };
    return option;
  };

  const renderDountOptions = (typeList, dataList) => {
    let newData = [];
    if (isAimExist) {
      // 元素去重结合
      newData = repeatFilter(typeList, dataList);
    } else {
      newData = [];
      typeList = [];
    }

    let total = 0;
    newData.map((item) => {
      total += Number(item.value);
    });
    let option = {
      color: fixedColor(),
      tooltip: {
        trigger: 'item',
        appendToBody: true,
        extraCssText: `max-width:400px;max-height:400px;overflow:auto;
        ${currentTheme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        formatter: (param) => {
          let result = '';
          let { color } = param;
          let dom = `<span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;background-color:${color};"></span>
              <span style=${currentTheme === 'light' ? '#666' : '#f7f7f7'}><b>${escapeHtml(param.data.type)}</b>: ${param.value}</span><br/>
              <span style="color:#40444D;;position:relative;top:10px;">${displayTooltip(param.data.aimList)}`;
          result += `${dom}`;
          return result;
        },
      },
      legend: {
        type: 'scroll',
        orient: 'vertical',
        left: '55%',
        top: 'center',
        icon: 'circle',
        itemHeight: 12,
        itemWidth: 12,
        textStyle: {
          color: currentTheme === 'light' ? '#333333' : '#f7f7f7',
          rich: {
            normal: {
              fontSize: 12,
              lineHeight: 16,
            },
            label: {
              padding: [4, 0, 0, 0],
            },
            value: {
              fontSize: 12,
              lineHeight: 16,
            },
            number: {
              fontSize: 12,
              lineHeight: 16,
              padding: [0, 0, 0, 20],
            },
          },
        },
        formatter: (legendItem) => {
          let tip1 = '';
          let tip = '';
          let signal = newData.filter((item) => item.name === legendItem);
          let le = legendItem.length + signal[0].value.toString().length + 5;
          let length = 35;
          if (window.innerWidth === 1280) {
            length = 20;
          }
          if (window.innerWidth > 1280 && window.innerWidth <= 1380) {
            length = 15;
          }
          if (window.innerWidth > 1380 && window.innerWidth <= 1480) {
            length = 20;
          }
          if (window.innerWidth > 1480 && window.innerWidth <= 1580) {
            length = 22;
          }
          if (window.innerWidth > 1580 && window.innerWidth <= 1680) {
            length = 30;
          }
          if (window.innerWidth > 1680 && window.innerWidth <= 1880) {
            length = 35;
          }
          if (le > length) {
            let l = Math.ceil(le / length);
            for (let i = 1; i <= l; i++) {
              if (i < l) {
                tip1 += `${legendItem.slice(i * length - length, i * length)}\n`;
              } else {
                tip = tip1 + legendItem.slice((l - 1) * length, le);
              }
            }
            return `{value|${tip}}{number|${signal[0].value}}`;
          } else {
            tip = legendItem;
            return `{normal|${tip}}{number|${signal[0].value}}`;
          }
        },
        data: typeList,
      },
      series: [
        {
          type: 'pie',
          radius: ['60%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 4,
            borderColor: '#fff',
            borderWidth: 2,
          },
          left: solveStyleDount(),
          label: {
            show: true,
            position: 'center',
            formatter: `{a|${solveDashboardValue(total)}} \n {b|${intl.formatMessage({ id: 'chart.total' })}}`,
            rich: {
              a: {
                padding: [32, 0, 0, 0],
                fontWeight: 700,
                fontSize: 36,
                align: 'center',
                color: currentTheme === 'light' ? '#333333' : '#f7f7f7',
              },
              b: {
                padding: [16, 0, 0, 0],
                fontWeight: 400,
                fontSize: 14,
                align: 'center',
                color: '#89939B',
              },
            },
          },
          labelLine: {
            show: false,
          },
          data: newData,
        },
      ],
    };
    return option;
  };

  const renderBarOptions = (typeList, dataList) => {
    let newData = [];
    if (isAimExist) {
      // 元素去重结合
      newData = repeatFilter(typeList, dataList);
    } else {
      newData = [];
      typeList = [];
    }
    let option = {
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        extraCssText: `max-width:400px;max-height:400px;overflow:auto;
        ${currentTheme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        formatter: (params) => {
          let result = '';
          params.forEach((item, index) => {
            let dom = `<span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;background-color:${item.color.colorStops[0].color};"></span>
              <span style=${currentTheme === 'light' ? '#666' : '#f7f7f7'}><b>${escapeHtml(item.seriesName)}</b>: ${item.value}</span><br/>
              <span style="color:#40444D;position:relative;top:10px;">${displayTooltip(item.data.aimList)}`;
            result += `${dom}`;
          });
          return result;
        },
      },
      legend: {
        data: typeList,
      },
      xAxis: {
        type: 'category',
        data: typeList,
        triggerEvent: true,
        axisLabel: {
          interval: 0,
          formatter: (value) => {
            if (value.length > 20) {
              return `${value.slice(0, 20)}...`;
            } else {
              return value;
            }
          },
        },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: !typeList.length ? 500 : null,
      },
      series: [
        {
          data: getMappingValueArrayOfKeyChart(newData, 'value'),
          type: 'bar',
          barMaxWidth: 60,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#346AC4' },
              { offset: 1, color: '#78AEF8' },
            ]),
          },
        },
      ],
    };
    return option;
  };

  const renderDashBoardOptions = (typeList, dataList, dashboardMaxVal, dashboardMinVal) => {
    let newData = [];
    let maxValue = dashboardMaxVal || 0;
    let minValue = dashboardMinVal || 0;
    if (maxValue <= minValue || minValue >= maxValue) {
      maxValue = 100;
      minValue = 0;
    }
    if (isAimExist) {
      // 元素去重结合
      newData = repeatFilter(typeList, dataList);
    } else {
      newData = [{ value: NaN }];
      typeList = [];
    }
    let options = [];
    newData.map((item) => {
      let option = null;
      option = {
        radius: '100%',
        tooltip: {
          appendToBody: true,
          formatter: `${item.type}: {c}`,
        },
        series: [
          {
            type: 'gauge',
            zlevel: 0,
            z: 2,
            min: minValue,
            max: maxValue,
            splitNumber: 10,
            center: ['50%', '50%'],
            progress: {
              show: true,
              width: 15,
            },
            axisLine: {
              lineStyle: {
                width: 16,
              },
            },
            axisTick: {
              show: false,
            },
            splitLine: {
              show: isNaN(item.value) ? false : true,
              length: 8,
              lineStyle: {
                width: 2,
                color: '#999',
              },
            },
            pointer: {
              show: item.value < minValue || isNaN(item.value) ? false : true,
              width: 3,
              length: '50%',
            },
            axisLabel: {
              formatter: (value) => {
                if (value === minValue || value === maxValue) {
                  return solveDashboardValue(value);
                } else {
                  return '';
                }
              },
              distance: 25,
              color: '#999',
              fontSize: 15,
            },
            title: {
              show: false,
            },
            anchor: {
              show: item.value < minValue || isNaN(item.value) ? false : true,
              showAbove: true,
              size: 15,
              itemStyle: {
                borderWidth: 5,
                borderColor: item.value > maxValue || item.value < minValue ? '#E7434A' : '#3f66f5',
              },
            },
            detail: {
              valueAnimation: true,
              fontSize: 30,
              color: currentTheme === 'light' ? '#333333' : '#f7f7f7',
              formatter: (d) => (isNaN(d) ? '--' : solveDashboardValue(d)),
              offsetCenter: [0, '90%'],
            },
            itemStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops:
                  item.value > maxValue || item.value < minValue
                    ? [
                        {
                          offset: 0,
                          color: '#F4840C', // 0% 处的颜色
                        },
                        {
                          offset: 1,
                          color: '#E7434AFF', // 100% 处的颜色
                        },
                      ]
                    : [
                        {
                          offset: 0,
                          color: '#477DD8FF',
                        },
                        {
                          offset: 1,
                          color: '#234C9EFF',
                        },
                      ],
              },
            },
            data: [
              {
                value: item.value ?? '--',
                name: '',
              },
            ],
          },
        ],
      };
      options.push({ option, type: item.type });
    });
    return options;
  };

  const renderNumber = (typeList, dataList) => {
    const allNumberList = [];
    if (isAimExist) {
      typeList.map((item) => {
        let data = dataList.filter((filterItem) => filterItem.type === item);
        let numberDataList = getMappingValueArrayOfKey(data, 'value');
        let obj = {};
        obj.type = item;
        obj.number = numberDataList.length ? numberDataList[numberDataList.length - 1] : '--';
        obj.data = data;
        allNumberList.push(obj);
      });
    } else {
      allNumberList.push({
        type: '--',
        number: '--',
        data: [
          {
            timeStamp: 0,
            value: 0,
          },
          {
            timeStamp: 1,
            value: 0,
          },
          {
            timeStamp: 2,
            value: 0,
          },
        ],
      });
    }
    setNumberAimList([...allNumberList]);
  };

  const handleDelete = async (passName) => {
    const [error, res] = await requestResource(deleteWidgetReousrce(widgets.widgetID));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'monitor.deleteFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'monitor.deleteSuccess' }));
        setDeleteOpen(false);
        /// 刷新页面
        refreshFn();
      }
    }
  };

  const handleModify = async (temporyType, temporyUuid) => {
    //
    const [error, res] = await requestResource(getIndicatorsList(widgets.dataSource));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'monitor.prometheusUrlUnreachable' }));
    }
    if (res) {
      const aimList = res.data.data;
      history.push({
        pathname: `/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}/editMonitorComponent/${temporyUuid}`,
        state: { type: temporyType, aimList, dataSource: widgets.dataSource, widgets },
      });
    }
  };

  const prefixElement = (passName, isShowMore, temporyType, uuid = widgets.widgetID) => (
    <div className={`chart_header ${temporyType}_chart_header`}>
      <Tooltip title={passName}>
        <p>{passName}</p>
      </Tooltip>
      {isShowMore && (
        <Popover
          trigger={'click'}
          arrow={false}
          getPopupContainer={(triggerNode) => {
            if (triggerNode && triggerNode.parentNode) {
              return triggerNode.parentNode;
            }
            return document.body;
          }}
          open={popOpen}
          placement="bottom"
          onOpenChange={(e) => (e ? setPopOpen(true) : setPopOpen(false))}
          content={
            <Space style={{ display: 'flex', flexDirection: 'column' }}>
              <Button className="monitor_link_btn" type="link" onClick={() => handleModify(temporyType, uuid)}>
                {intl.formatMessage({ id: 'common.edit' })}
              </Button>
              <Button
                className="monitor_link_btn"
                type="link"
                onClick={() => {
                  setPopOpen(false);
                  setDeleteOpen(true);
                }}
              >
                {intl.formatMessage({ id: 'common.delete' })}
              </Button>
            </Space>
          }
        >
          <Popconfirm
            placement="bottomLeft"
            title={intl.formatMessage({ id: 'monitor.deleteComponentConfirm' }, { name: passName })}
            open={deleteOpen}
            arrow={false}
            className="delete_component"
            icon={<ExclamationCircleFilled style={{ color: '#E7434a' }} />}
            onCancel={() => handleDelete(passName)}
            onConfirm={() => setDeleteOpen(false)}
            onOpenChange={(e) => (e && popOpen ? '' : setDeleteOpen(false))}
            okButtonProps={{
              danger: false,
              type: 'default',
            }}
            okText={intl.formatMessage({ id: 'common.cancel' })}
            cancelButtonProps={{
              danger: true,
            }}
            cancelText={intl.formatMessage({ id: 'common.delete' })}
          >
            <MoreOutlined style={{ color: '#346AC4' }} onClick={() => setPopOpen(true)} />
          </Popconfirm>
        </Popover>
      )}
    </div>
  );

  const deployEchartsOption = useCallback(() => {
    let typeList = [];
    data.map((item) => {
      if (item.type) {
        if (!typeList.includes(item.type)) {
          typeList.push(item.type);
        }
      }
    });
    setTotalTypeList([...typeList]);
    let hashMap = {
      line: () => renderLineOptions(typeList, data),
      area: () => renderLineOptions(typeList, data, true),
      number: () => renderNumber(typeList, data),
      dashboard: () => renderDashBoardOptions(typeList, data, dashboardMax, dashboardMin),
      dount: () => renderDountOptions(typeList, data),
      bar: () => renderBarOptions(typeList, data),
    };
    if (type === 'dashboard') {
      setOptionsDashboard(hashMap[type]);
    } else {
      setOptions(hashMap[type]);
    }
  }, [type, JSON.stringify(data), dashboardMax, dashboardMin, currentTheme]);

  useEffect(() => {
    deployEchartsOption();
    window.addEventListener('resize', () => {
      deployEchartsOption();
    });
  }, [deployEchartsOption]);

  return (
    <Fragment>
      {contextHolder}
      {type !== 'dashboard' && type !== 'number' && type !== 'dount' && (
        <div className={`${isPreview ? 'line_list' : 'line_display_list'}`}>
          {prefixElement(name, isMore, type)}
          <EchartsInit
            option={options}
            type={type}
            id={id}
            preview={isPreview ? true : false}
            isAimExist={isAimExist}
          />
        </div>
      )}
      {type === 'dount' && (
        <div className={`${isPreview ? 'dount_list' : 'dount_display_list'}`}>
          {prefixElement(name, isMore, type)}
          <EchartsInit
            option={options}
            type={type}
            id={id}
            preview={isPreview ? true : false}
            isAimExist={isAimExist}
          />
        </div>
      )}
      {type === 'dashboard' && (
        <div className={`${isPreview ? 'dashboard_list' : 'dashboard_display_list'}`}>
          {prefixElement(name, isMore, type)}
          <div className="dashboard_row">
            {optionsDashBoard.map((item, index) => (
              <EchartsInit
                option={item.option}
                type={type}
                id={`${id}_${index}`}
                preview={isPreview ? true : false}
                isAimExist={isAimExist}
                typeAim={item.type}
              />
            ))}
          </div>
        </div>
      )}
      {type === 'number' && (
        <div className={`${isPreview ? 'number_list' : 'number_display_list'}`}>
          {prefixElement(name, isMore, type)}
          <div className="number_row">
            {numberAimList.map((item) => (
              <NumberAimCom
                data={item.data}
                aimType={item.type}
                isAimExist={isAimExist}
                number={item.number}
                isDefault={item.type === '--' ? true : false}
              />
            ))}
          </div>
        </div>
      )}
    </Fragment>
  );
}
