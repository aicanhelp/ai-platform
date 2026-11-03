/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/pages/monitor.less';
import { Button, Form, Select, DatePicker, Input, Table, Checkbox } from 'antd';
import { Fragment, useCallback, useEffect, useRef, useState, useStore, useMemo } from 'openinula';
import Dayjs from 'dayjs';
import { SyncOutlined, DeleteOutlined, UpOutlined, DownOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import { containerRouterPrefix } from '@/constant';
import {
  timePeriodOptionsKeys,
  refreshTimeOptionsKeys,
  stepList,
  ResponseCode,
} from '@/common/constants';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { randomUuid, solveDataAreaNumPalettePlus, solveQueryMonitor, rgbToHexOut } from '@/tools/utils';
import DataAreaChart from '@/components/echarts/DataAreaChart';
import { message, FloatButton } from 'antd';
import { getCustomizeQuery } from '@/api/monitorApi';
import EmptyData from '@/components/EmptyData';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import infoStore from '@/store/infoStore';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

const { TextArea } = Input;
export default function CustomizeMonitorQuery() {
  const { t, i18n } = useTranslation('monitoring');
  const [timeForm] = Form.useForm();

  const [messageApi, contextHolder] = message.useMessage();
  const infoStoreExample = infoStore();
  const refreshTimeValue = Form.useWatch('refreshTime', timeForm);
  const [rangeColorList, setRangeColorList] = useState([]);
  const [intervalId, setInterValId] = useState(0);
  const [queryElementIDList, setQueryElementIDList] = useState([]);
  const [seletedMarkId, setSelectMarkId] = useState([]);
  const [inputList, setInputList] = useState([]);
  // 轮询回调读最新列表/查询函数，避免 setInterval 闭包捕获 stale state
  const inputListRef = useRef(inputList);
  const handleSearchMonitorRef = useRef(null);
  useEffect(() => {
    inputListRef.current = inputList;
  }, [inputList]);
  useEffect(() => {
    handleSearchMonitorRef.current = handleSearchMonitor;
  });
  const [selectIdList, setSelectIdList] = useState([]);
  const [loading, setLoading] = useState(true);
  const totalIDListRef = useRef({}); // 各查询框勾选的序列 id
  const totalColorIDListRef = useRef({}); // 各查询框序列配色
  const [refreshLoading, setRefreshLoading] = useButtonLoading();
  const [queryLoadingMap, setQueryLoadingMap] = useState({});
  const [isCheck, setIsCheck] = useState(false);
  const themeStore = useStore('theme');
  const endTimeValue = Form.useWatch('endTime', timeForm);
  const handleResetMonitor = useCallback(async (isActive = false) => {
    if (isActive) {
      setRefreshLoading(true);
      try {
        await Promise.all(inputList.map(item => handleSearchMonitorRef.current(item.id)));
      } finally {
        setRefreshLoading(false);
      }
    }
  }, [inputList]);

  const filterColorList = (data, objMap) => {
    const colorMap = new Map(objMap.map(item => [item.type, item.color]));
    const uniqueItems = [...new Map(data.map(item => [item.type, item])).values()];
    const colors = uniqueItems.map(({ type }) => colorMap.get(type) ?? null);
    return colors;
  };

  const handleCancelPaint = (uuid, selfId, dataSource) => {
    if (!Object.keys(totalIDListRef.current).includes(uuid.toString())) {
      totalIDListRef.current[uuid] = [];
    }
    let temporyId = totalIDListRef.current[uuid];
    if (!totalIDListRef.current[uuid].includes(selfId)) {
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        newQuery.map(item => {
          if (item.id === uuid) {
            let chartDataListTempory = item.chartDataList;
            chartDataListTempory = chartDataListTempory.concat(item.totalData.filter(filterItem => filterItem.type === selfId));
            item.chartDataList = chartDataListTempory;
            item.rangeColor = filterColorList(chartDataListTempory, totalColorIDListRef.current[uuid]);
          }
        });
        return [...newQuery];
      });
      const [element] = document.getElementsByClassName(`${uuid}_${selfId}_color`);
      element.style.background = element.style.color;
      temporyId.push(selfId);
    } else {
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        newQuery.map(item => {
          if (item.id === uuid) {
            let chartDataListTempory = item.totalData;
            chartDataListTempory = chartDataListTempory.filter(filterItem => filterItem.type !== selfId);
            item.chartDataList = chartDataListTempory;
          }
        });
        return [...newQuery];
      });
      const [element] = document.getElementsByClassName(`${uuid}_${selfId}_color`);
      element.style.background = '#ccc';
      temporyId = temporyId.filter(temporyItem => temporyItem !== selfId);
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        newQuery.map(item => {
          if (item.id === uuid) {
            const rgbValue = window.getComputedStyle(element).color;
            const [r, g, b] = rgbValue.match(/\d+/g).map(Number);
            item.rangeColor = item.rangeColor.filter(filterItem => (filterItem !== element.style.color && filterItem !== rgbToHexOut(r, g, b)));
          }
        });
        return [...newQuery];
      });
    }
    totalIDListRef.current[uuid] = [...temporyId];
    const checkBox = document.querySelector(`#checkbox_${uuid}_current`);
    if (totalIDListRef.current[uuid].length !== dataSource.length) {
      checkBox.value = false;
      if (checkBox.parentNode.classList.contains('container-platform-checkbox-checked')) {
        checkBox.parentNode.classList.remove('container-platform-checkbox-checked');
      }
    } else {
      checkBox.parentNode.classList.add('container-platform-checkbox-checked');
    }
    setSelectIdList([...temporyId]);
  };

  const onChange = (uuid, dataSource) => {
    if (!totalIDListRef.current[uuid].length || totalIDListRef.current[uuid].length !== dataSource.length) {
      if (totalIDListRef.current[uuid].length !== dataSource.length) {
        setTimeout(() => {
          const checkBox = document.querySelector(`#checkbox_${uuid}_current`);
          checkBox.parentNode.classList.add('container-platform-checkbox-checked');
        }, 0);
      }
      dataSource.map(item => {
        const [element] = document.getElementsByClassName(`${uuid}_${item.idSelf}_color`);
        if (element) {
          element.style.background = element.style.color;
        }
        totalIDListRef.current[uuid].push(item.idSelf);
      });
      totalIDListRef.current[uuid] = Array.from(new Set([...totalIDListRef.current[uuid]]));
      setIsCheck(true);
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        preQuery.map(item => {
          if (item.id === uuid) {
            item.chartDataList = item.totalData;
          }
        });
        return [...newQuery];
      });
    } else {
      totalIDListRef.current[uuid] = [];
      setTimeout(() => {
        const checkBox = document.querySelector(`#checkbox_${uuid}_current`);
        checkBox.parentNode.classList.remove('container-platform-checkbox-checked');
      }, 0);
      dataSource.map(item => {
        const [element] = document.getElementsByClassName(`${uuid}_${item.idSelf}_color`);
        if (element) {
          element.style.background = '#ccc';
        }
      });
      setIsCheck(true);
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        newQuery.map(item => {
          if (item.id === uuid) {
            item.chartDataList = [];
          }
        });
        return [...newQuery];
      });
    }
  };

  const renderColumns = useCallback((id, tableColumns, dataSource, selectIdTempory = []) => {
    let selectIdListTempory = selectIdTempory || selectIdList;
    let columns = [];
    tableColumns.map(item => {
      let element;
      if (item === 'idSelf') {
        element = {
          title: <Fragment>
            <Checkbox
              id={`checkbox_${id}_current`}
              defaultChecked={true}
              onChange={() => onChange(id, dataSource)}>
            </Checkbox>
          </Fragment>,
          key: 'color',
          render: (_, record) => <div className='legend_group'>
            <span className={`legend_color ${id}_${record.idSelf}_color`}
              style={{ background: `${!selectIdListTempory.includes(record.idSelf) ? '#ccc' : record.color}`, cursor: 'pointer', color: record.color }}
              onClick={() => handleCancelPaint(id, record.idSelf, dataSource)}></span>
          </div>,
        };
      } else {
        let title = item;
        if (item === 'timestamp') {
          title = t('table.collectionTime');
        }
        if (item === 'value') {
          title = t('table.value');
        }
        element = {
          title,
          key: item,
          render: (_, record) => item !== 'timestamp' ? (record[item] || '--') : (Dayjs(record[item] * 1000).format('YYYY-MM-DD HH:mm:ss') || '--'),
        };
      }
      columns.push(element);
    });
    return columns;
  }, [selectIdList, t]);

  const handleSearchMonitor = async (id, silent = false) => {
    if (!silent) {
      setQueryLoadingMap(m => ({ ...m, [id]: true }));
    }
    try {
      const [{ value }] = inputList.filter(item => item.id === id);
      let errorMessage = '';
      const formValues = timeForm.getFieldsValue();
      const endDay = Dayjs(formValues.endTime);
      if (!endDay.isValid()) {
        messageApi.warning(t('form.invalidEndTime'));
        return;
      }
      let conditions = {
        step: stepList[formValues.period],
        start: endDay.subtract(parseInt(formValues.period), formValues.period.slice(-1)),
        end: endDay,
      };
      const offset = infoStoreExample.$s.offsetCompareLocal;
      if (offset !== 0) {
        if (offset > 0) {
          conditions.start = conditions.start.subtract(offset, 'minute');
          conditions.end = conditions.end.subtract(offset, 'minute');
        } else {
          conditions.start = conditions.start.add(Math.abs(offset), 'minute');
          conditions.end = conditions.end.add(Math.abs(offset), 'minute');
        }
      }
      conditions.start = conditions.start.format('YYYY-MM-DD HH:mm:ss');
      conditions.end = conditions.end.format('YYYY-MM-DD HH:mm:ss');
      let chartDataListTempory = [];
      let chartTableTempory = {};

      const res = await getCustomizeQuery(value, conditions.start, conditions.end, conditions.step);
      if (res.status === ResponseCode.OK) {
        if (res.data.code === 0) {
          const oldData = res.data.data.result;
          let { tableColumns, chartDataList, chartTableList } = solveQueryMonitor(oldData);
          chartDataListTempory = chartDataList;
          let typeList = [];
          chartDataListTempory.map(item => {
            if (!typeList.includes(item.type)) {
              typeList.push(item.type);
            }
          });

          let dataSource = [];
          typeList.map((index) => {
            let obj = {};
            tableColumns.map(columnsItem => {
              obj[columnsItem] = chartTableList[index][columnsItem];
            });
            obj.key = index;
            obj.color = rangeColorList[index];
            dataSource.push(obj);
          });

          if (!Object.keys(totalIDListRef.current).includes(id.toString())) {
            totalIDListRef.current[id] = [];
            totalColorIDListRef.current[id] = [];
            if (!totalIDListRef.current[id].length) {
              dataSource.map(dataSourceItem => {
                totalIDListRef.current[id].push(dataSourceItem.idSelf);
              });
              setSelectIdList(totalIDListRef.current[id]);
              totalIDListRef.current[id].map(item => {
                const [element] = document.getElementsByClassName(`${id}_${item}_color`);
                if (element) {
                  element.style.background = element.style.color;
                }
              });
            }
            if (!totalColorIDListRef.current[id].length) {
              dataSource.map((dataSourceItem, dataSourceIndex) => {
                totalColorIDListRef.current[id].push({ type: typeList[dataSourceIndex], color: dataSourceItem.color });
              });
            }
          }
          chartDataListTempory = chartDataListTempory.filter(chartDataFilterItem => totalIDListRef.current[id].includes(chartDataFilterItem.type));
          const columns = renderColumns(id, tableColumns, dataSource, totalIDListRef.current[id]);
          chartTableTempory = {
            columns,
            dataSource,
          };
        } else {
          errorMessage = {
            reason: res.data.code,
            info: res.data.msg,
          };
        }
      }
      setQueryElementIDList(preQuery => {
        let newQuery = preQuery;
        preQuery.map(item => {
          if (item.id === id) {
            item.totalData = chartDataListTempory;
            item.inputValue = value;
            item.chartDataList = chartDataListTempory;
            item.dataSource = chartTableTempory.dataSource;
            item.tableColumns = chartTableTempory.columns;
            item.rangeColor = rangeColorList;
            item.errorMessage = errorMessage;
          }
        });
        return [...newQuery];
      });

      if (!seletedMarkId.includes(id)) {
        setSelectMarkId(preID => {
          return [...preID, id];
        });
      }
      setLoading(false);
    } finally {
      if (!silent) {
        setQueryLoadingMap(m => ({ ...m, [id]: false }));
      }
    }
  };

  const toTop = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  };

  const toDown = () => {
    const target = document.querySelector('.container_content');
    target.scrollTo({ top: target.scrollHeight - target.clientHeight, left: 0, behavior: 'smooth' });
  };

  const handleDeleteQuery = (id) => {
    const inputArr = inputList.filter(item => item.id !== id);
    const newList = queryElementIDList.filter(item => item.id !== id);
    setQueryElementIDList([...newList]);
    setInputList([...inputArr]);
  };

  const handleSolveMarkIdAdd = (id) => {
    setSelectMarkId(preID => {
      return [...preID, id];
    });
  };
  const handleSolveMarkIdDelete = (id) => {
    const newList = seletedMarkId.filter(item => item !== id);
    setSelectMarkId([...newList]);
  };

  const renderElement = useCallback(() => {
    const markID = randomUuid();
    let limit = false;
    setQueryElementIDList(preQuery => {
      if (preQuery.length === 10) {
        limit = true;
        return [...preQuery];
      } else {
        return [...preQuery, {
          id: markID,
          inputValue: '',
          errorMessage: {
            reason: '',
            info: '',
          },
          chartDataList: [],
          tableColumns: [],
          dataSource: [],
        }];
      }
    });
    setInputList(prev => {
      if (prev.length) {
        return [...prev, { id: markID, value: '' }];
      }
      return [{ id: markID, value: '' }];
    });
    if (limit) {
      messageApi.error(t('customizeQuery.maxQueryLimit'));
    }
  }, [t, messageApi]);

  const handleAddQuery = () => {
    renderElement();
  };

  const handleQueryKeyDown = (e, id) => {
    if (e.nativeEvent.isComposing || e.key !== 'Enter') {
      return;
    }
    if (e.shiftKey) {
      e.preventDefault();
      handleSearchMonitor(id);
    }
  };

  const handleChange = (e, id) => {
    let newList = [...inputList];
    const arr = newList.filter(item => item.id === id);
    if (arr.length) {
      newList.map(newItem => {
        if (newItem.id === id) {
          newItem.value = e.target.value;
        }
      });
    } else {
      newList.push({
        id,
        value: e.target.value,
      });
    }
    let queryList = [...queryElementIDList];
    queryList.map(queryItem => {
      if (queryItem.id === id) {
        queryItem.inputValue = e.target.value;
      }
    });
    setQueryElementIDList(queryList);
    setInputList(newList);
  };

  useEffect(() => {
    const markID = randomUuid();
    setQueryElementIDList([{
      id: markID,
      inputValue: '',
      errorMessage: { reason: '', info: '' },
      chartDataList: [],
      tableColumns: [],
      dataSource: [],
    }]);
    setInputList([{ id: markID, value: '' }]);
  }, []);

  useEffect(() => {
    const rangeBirthColorList = solveDataAreaNumPalettePlus(1000);
    setRangeColorList(rangeBirthColorList);
  }, []);

  const setupAutoRefresh = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setInterValId(0);
    }
    const refreshSec = Number(refreshTimeValue);
    if (!Number.isFinite(refreshSec) || refreshSec <= 0) {
      return;
    }
    const delayMs = refreshSec * 1000;
    const id = setInterval(() => {
      timeForm.setFieldValue('endTime', Dayjs());
      inputListRef.current.map(item => {
        totalIDListRef.current = {};
        const checkBox = document.querySelector(`#checkbox_${item.id}_current`);
        checkBox?.parentNode?.classList.add('container-platform-checkbox-checked');
        handleSearchMonitorRef.current?.(item.id, true);
      });
    }, delayMs);
    setInterValId(id);
  }, [refreshTimeValue]);

  const handleValueChange = (changeValues) => {
    if (Object.prototype.hasOwnProperty.call(changeValues, 'endTime')) {
      timeForm.setFieldValue('refreshTime', '0');
    }
    if (!Object.prototype.hasOwnProperty.call(changeValues, 'refreshTime')) {
      inputList.map(item => {
        handleSearchMonitor(item.id, true);
      });
    }
  };

  const handleTableChange = (uuid, totalData) => {
    setTimeout(() => {
      let totalList = [];
      totalData.map(queryTotalItem => {
        totalList.push(queryTotalItem.type);
      });
      const temporyList = new Set(totalIDListRef.current[uuid]);
      let filtered = totalList.filter(totalFilterItem => !temporyList.has(totalFilterItem));
      filtered = [...new Set(filtered)];
      filtered.map(filteredItem => {
        const [element] = document.getElementsByClassName(`${uuid}_${filteredItem}_color`);
        if (element) {
          element.style.background = '#ccc';
        }
      });
    }, 0);
  };

  useEffect(() => {
    setupAutoRefresh();
  }, [setupAutoRefresh]);

  useEffect(() => {
    handleResetMonitor();
  }, [handleResetMonitor]);

  useEffect(() => {
    return () => clearInterval(intervalId);
  }, [intervalId]);

  const timePeriodOptions = useMemo(() => {
    return timePeriodOptionsKeys.map(item => ({
      label: t(`timePeriod.${item}`, { ns: 'common' }),
      value: item,
    }));
  }, [t, i18n.language]);

  const refreshTimeOptions = useMemo(() => {
    return refreshTimeOptionsKeys.map(item => ({
      label: t(`refreshTime.${item}`, { ns: 'common' }),
      value: item,
      disabled: !(item === '0' || Dayjs(endTimeValue).isSame(Dayjs(), 's')),
    }));
  }, [t, i18n.language, endTimeValue]);

  return <div className="child_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/monitor/monitorDashboard`, disabled: true },
      { title: t('module.dashboard'), path: '/' },
      { title: t('module.customizeQuery'), path: '/customize/query' },
    ]} />
    <div className="monitor_query_container">
      <div className='monitor_query_header'>
        <Button
          className={'primary_btn'}
          onClick={handleAddQuery}
        >
          {t('actions.addQuery')}
        </Button>
        <Form form={timeForm}
          layout='inline'
          initialValues={{
            period: '10m',
            endTime: Dayjs(),
            refreshTime: '30',
          }}
          onValuesChange={handleValueChange}
        >
          <Form.Item name="period" className='CustomizeMonitorQuery'>
            <Select options={timePeriodOptions} style={{ minWidth: '100px' }} />
          </Form.Item>
          <Form.Item name="endTime" label={t('form.endTime')} className='CustomizeMonitorQuery'>
            <DatePicker showTime />
          </Form.Item>
          <Form.Item name="refreshTime" className='CustomizeMonitorQuery'>
            <Select className='refresh_time_select' options={refreshTimeOptions} />
          </Form.Item>
          <Form.Item className='CustomizeMonitorQuery'>
            <Button icon={<SyncOutlined />} onClick={() => handleResetMonitor(true)} loading={refreshLoading} className="reset_btn"></Button>
          </Form.Item>
        </Form>
      </div>
      <div className='monitor_query_box_all'>
        {queryElementIDList.map(item => (
          <div key={item.id} className={`monitor_query_box ${item.id}`}>
            <div className={`monitor_query_total ${item.id}`}>
              <div className={`monitor_query_item ${item.id}`}>
                {!seletedMarkId.includes(item.id) ?
                  <UpOutlined className='arrow' onClick={() => handleSolveMarkIdAdd(item.id)} />
                  : <DownOutlined className='arrow' onClick={() => handleSolveMarkIdDelete(item.id)} />
                }
                <TextArea
                  onChange={e => handleChange(e, item.id)}
                  onKeyDown={e => handleQueryKeyDown(e, item.id)}
                  placeholder={t('customizeQuery.placeholder')}
                  value={item.inputValue}
                  autoSize={{ minRows: 1, maxRows: 16 }}
                  style={{ resize: 'none', minHeight: 32 }}
                  allowClear
                />
                <Button className='primary_btn monitor_query_btn' loading={!!queryLoadingMap[item.id]} onClick={() => handleSearchMonitor(item.id)}>{t('actions.query')}</Button>
                <DeleteOutlined className='delete' onClick={() => handleDeleteQuery(item.id)} />
              </div>
              {!!(seletedMarkId.includes(item.id) && item.errorMessage.reason) && <div className='monitor_container'>
                <div className='monitor_error_box'>
                  <ExclamationCircleOutlined />
                  <div className='info_box'>
                    <p>{item.errorMessage.reason}</p>
                    <p>{item.errorMessage.info}</p>
                  </div>
                </div>
              </div>}
              {!!(seletedMarkId.includes(item.id) && item.chartDataList && (item.chartDataList.length || isCheck)) && <div className='chart_table'>
                <DataAreaChart id={`${item.id}_monitor_query_chart`} title={''} data={item.chartDataList} type="line" rangeColorProps={item.rangeColor} isShowScreen={false} isShowLegend={false} />
                <Table
                  className="monitor_table"
                  loading={loading}
                  columns={item.tableColumns}
                  dataSource={item.dataSource}
                  onChange={() => handleTableChange(item.id, item.totalData)}
                  pagination={{
                    className: 'page',
                    showTotal: (total) => t('table.total', { total }),
                    showSizeChanger: true,
                    showQuickJumper: true,
                    pageSizeOptions: [5, 10],
                  }} />
              </div>
              }
              {!!seletedMarkId.includes(item.id) && item.chartDataList.length === 0 && !item.errorMessage.reason &&
                <div className='chart_table' >
                  <EmptyData />
                </div>
              }
            </div>
          </div>
        ))}
      </div>
    </div>
    <FloatButton.Group
      className='CustomizeMonitorQuery float_btn_right'
      shape="square"
      style={{ right: 20 }}>
      <FloatButton icon={<ArrowUpOutlined />} onClick={toTop} />
      <FloatButton icon={<ArrowDownOutlined />} onClick={toDown} />
    </FloatButton.Group>
  </div>;
}
