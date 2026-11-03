/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Form, Input, Select, Table, message, Button, InputNumber } from 'antd';
import { useParams, useLocation, useHistory } from 'inula-router';
import '@/styles/pages/addMonitorComponent.less';
import { containerRouterPrefix, ResponseCode, componentTypeOptions, widgetTypeList } from '@/common/constants';
import FormTimeCom from '@/components/FormTimeCom';
import dayjs from 'dayjs';
import EchartsCom from '@/components/EchartsCom';
import { useEffect, useState, useCallback, useRef } from 'openinula';
import { PlusOutlined } from '@ant-design/icons';
import { getInstanceList, getInstanceEachData, createWidgetResource, updateWidgetResource } from '@/api/monitorApi';
import {
  getDataRowKeysInstance,
  getReturnAims,
  randomUuid,
  requestResource,
  solveInstanceData,
  getAllKeys,
  getKeyMap,
  getMappingValue,
  cleanFilterTableData,
  birthDefaultTimestamp,
  filterTableData,
  solveTableDataToInstance,
  createWidgetDataChart,
  gainAimType,
  solveSelectedRowKeys,
  solveDisabledData,
} from '@/tools/utils';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { useIntl } from 'inula-intl';
import { stepList } from '@/common/constants';

export default function AddMonitorComponent() {
  const intl = useIntl();
  const { dashboardID, dashboardName, componentID } = useParams();
  const formRef = useRef(null);
  const [comForm] = Form.useForm();
  const history = useHistory();
  const {
    state: { type = 'line', aimList, dataSource, widgets },
  } = useLocation();
  const [passType, setPassType] = useState(type);
  const [messageApi, contextHolder] = message.useMessage();
  const [data, setData] = useState(); // 数据
  const [intervalId, setInterValId] = useState(0); // 定时器id
  const refreshTimeValue = Form.useWatch('refreshTime', comForm); // 刷新间隔
  const componentName = Form.useWatch('name', comForm);
  const [aimTableData, setAimTableData] = useState([]); // 指标表格数据
  const [dashBoardMinValue, setDashBoardMinValue] = useState('');
  const [dashBoardMaxValue, setDashBoardMaxValue] = useState();
  const [cancelModalOpen, setCancelModalOpen] = useState(false);
  const [selectAimRowKeys, setSelectAimRowKeys] = useState([]);
  const [aimOptionsList, setAimOptionsList] = useState([]);
  const [selectAimList, setSelectAimList] = useState([]);

  const handleResetMonitor = () => {
    timeoutSearchTask();
  };

  const solveEditDataShow = useCallback(async () => {
    const aimTemporyList = gainAimType(widgets.instances);
    const backFillData = [];
    for await (let item of aimTemporyList) {
      const needIntance = widgets.instances.filter((filterItem) => filterItem.metric.__name__ === item);
      const [error, res] = await requestResource(getInstanceList(item, widgets.dataSource));
      if (error) {
        messageApi.error(intl.formatMessage({ id: 'monitor.fetchInstanceListFailed' }));
      }
      if (res) {
        if (res.data.code === ResponseCode.OK) {
          // 需要添加的项目
          const addDisabledData = solveDisabledData(res.data.data, needIntance);
          backFillData.push({
            aimName: item,
            tableData: [
              ...solveInstanceData(res.data.data),
              ...solveInstanceData(addDisabledData, res.data.data.instances.length),
            ],
            selectedRowKeys: solveSelectedRowKeys(needIntance, {
              instances: [...res.data.data.instances, ...addDisabledData.instances],
              displayName: res.data.data.displayName,
            }),
            labelGroup: [
              {
                id: randomUuid(),
                key: null,
                value: null,
              },
            ],
            labelOptions: getAllKeys(res.data.data),
            keyMap: getKeyMap(res.data.data),
            valueOptions: [],
            totalTableData: res.data.data,
            selectInstances: solveInstanceData({ instances: needIntance, displayName: item }),
          });
        }
      }
    }
    setDashBoardMaxValue(widgets.graphType?.boardMax || 0);
    setDashBoardMinValue(widgets.graphType?.boardMin || 0);
    setSelectAimRowKeys([...backFillData]);
    timeoutSearchTask(backFillData);
    setSelectAimList([...aimTemporyList]);
    comForm.setFieldValue('name', widgets.displayName);
  }, [widgets]);

  useEffect(() => {
    if (componentID) {
      solveEditDataShow();
    } else {
      let defaultData = birthDefaultTimestamp();
      setData(defaultData);
    }
  }, [componentID, solveEditDataShow]);

  const aimColumns = [
    {
      title: intl.formatMessage({ id: 'common.instance' }),
      key: 'instance',
      render: (_, record) => <div>{record.instance}</div>,
    },
  ];

  const handleCreate = async () => {
    let stopFlag = false;
    if (!comForm.getFieldValue('name')) {
      stopFlag = true;
      messageApi.error(intl.formatMessage({ id: 'monitor.fillComponentName' }));
    }
    if (passType === 'dashboard' && (!dashBoardMaxValue || dashBoardMinValue === '')) {
      stopFlag = true;
      messageApi.error(intl.formatMessage({ id: 'monitor.fillGaugeRange' }));
    }
    if (
      passType === 'dashboard' &&
      (dashBoardMaxValue < dashBoardMinValue || dashBoardMaxValue === dashBoardMinValue)
    ) {
      stopFlag = true;
      messageApi.error(intl.formatMessage({ id: 'monitor.gaugeRangeInvalid' }));
    }
    if (!Boolean(getDataRowKeysInstance(selectAimRowKeys))) {
      stopFlag = true;
      messageApi.error(intl.formatMessage({ id: 'monitor.selectMetricInstance' }));
    }
    if (!solveTableDataToInstance(selectAimRowKeys).length) {
      stopFlag = true;
      messageApi.error(intl.formatMessage({ id: 'monitor.fetchMetricInstanceError' }));
    }
    const values = await comForm.validateFields();
    if (!stopFlag) {
      let jsonData = {
        dashboardID,
        displayName: comForm.getFieldValue('name'),
        widgetType: widgetTypeList[passType],
        dataSource,
        boardMin: String(dashBoardMinValue),
        boardMax: String(dashBoardMaxValue),
        instances: solveTableDataToInstance(selectAimRowKeys),
      };
      if (componentID) {
        const [error, res] = await requestResource(updateWidgetResource(widgets.widgetID, jsonData));
        if (error) {
          messageApi.error(intl.formatMessage({ id: 'monitor.updateFailed' }));
        }
        if (res) {
          if (res.data.code === ResponseCode.OK) {
            messageApi.success(intl.formatMessage({ id: 'monitor.updateSuccess' }));
            setTimeout(() => {
              history.push(`/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}`);
            }, 2000);
          }
        }
      } else {
        const [error, res] = await requestResource(createWidgetResource(jsonData));
        if (error) {
          messageApi.error(intl.formatMessage({ id: 'monitor.createFailed' }));
        }
        if (res) {
          if (res.data.code === ResponseCode.OK) {
            messageApi.success(intl.formatMessage({ id: 'monitor.createSuccess' }));
            setTimeout(() => {
              history.push(`/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}`);
            }, 2000);
          }
        }
      }
    }
  };

  const handleValueChange = (changeValues) => {
    if (Object.prototype.hasOwnProperty.call(changeValues, 'type')) {
      if (changeValues.type === 'area' && getDataRowKeysInstance(selectAimRowKeys) > 3) {
        messageApi.error(intl.formatMessage({ id: 'monitor.areaMaxInstances' }));
      } else {
        setPassType(changeValues.type);
      }
    }
    if (Object.prototype.hasOwnProperty.call(changeValues, 'endTime')) {
      if (dayjs(changeValues.endTime).format('YYYY-MM-DD HH:mm:ss') === dayjs().format('YYYY-MM-DD HH:mm:ss')) {
        if (comForm.getFieldValue('refreshTime') === 0) {
          comForm.setFieldValue('refreshTime', 30);
        }
      } else {
        comForm.setFieldValue('refreshTime', 0);
      }
    }
    if (Object.prototype.hasOwnProperty.call(changeValues, 'refreshTime')) {
      if (comForm.getFieldValue('refreshTime') !== 0) {
        comForm.setFieldValue('endTime', dayjs());
        timeoutSearchTask();
      }
    }
    if (!Object.prototype.hasOwnProperty.call(changeValues, 'refreshTime')) {
      if (
        !Object.prototype.hasOwnProperty.call(changeValues, 'type') &&
        !Object.prototype.hasOwnProperty.call(changeValues, 'name')
      ) {
        timeoutSearchTask();
      }
    }
  };

  const handleCancel = () => {
    if (componentID) {
      setCancelModalOpen(true);
    } else {
      history.push(`/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}`);
    }
  };

  const handleAddLabelGroup = (name) => {
    let labelGroup = {
      id: randomUuid(),
      key: null,
      value: null,
    };
    setSelectAimRowKeys((preSelect) => {
      preSelect.map((item) => {
        if (item.aimName === name) {
          item.labelGroup.push(labelGroup);
        }
      });
      return [...preSelect];
    });
  };

  const handleCancelModalCancel = () => {
    history.push(`/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}`);
  };

  const handleCancelMonitorConfirm = () => {
    setCancelModalOpen(false);
  };

  const handleSelectChange = async (value) => {
    if (value.length > 8) {
      messageApi.error(intl.formatMessage({ id: 'monitor.maxMetricsSelected' }));
    } else {
      if (!value.length) {
        setSelectAimRowKeys([]);
        setSelectAimList([...value]);
        setData([]);
      } else {
        // 判断是否被删除
        const newArr = selectAimRowKeys.filter((item) => value.includes(item.aimName));
        let rowKeys = [...newArr];
        await addAsync(value, rowKeys);
        setSelectAimRowKeys([...rowKeys]);
        setSelectAimList([...value]);
      }
    }
  };

  const addAsync = async (value, rowKeys) => {
    for await (let item of value) {
      if (!rowKeys.filter((filterItem) => filterItem.aimName === item).length) {
        const [error, res] = await requestResource(getInstanceList(item, dataSource));
        if (error) {
          messageApi.error(intl.formatMessage({ id: 'monitor.fetchInstanceListFailed' }));
        }
        if (res) {
          if (res.data.code === ResponseCode.OK) {
            rowKeys.push({
              aimName: item,
              tableData: solveInstanceData(res.data.data),
              selectedRowKeys: [],
              labelGroup: [
                {
                  id: randomUuid(),
                  key: null,
                  value: null,
                },
              ],
              labelOptions: getAllKeys(res.data.data),
              keyMap: getKeyMap(res.data.data),
              valueOptions: [],
              totalTableData: res.data.data,
              selectInstances: [],
            });
          }
        }
      }
    }
  };

  const handleDeleteLabel = (id, name) => {
    setSelectAimRowKeys((pre) => {
      pre.map((item) => {
        if (item.aimName === name) {
          const searchIndex = item.labelGroup.findIndex((filterItem) => filterItem.id === id);
          item.labelGroup.splice(searchIndex, 1);
          item.tableData = filterTableData(item.labelGroup, solveInstanceData(item.totalTableData));
        }
      });
      return [...pre];
    });
  };

  const handleSelectKey = (e, id, name) => {
    setSelectAimRowKeys((pre) => {
      pre.map((item) => {
        if (item.aimName === name) {
          const searchIndex = item.labelGroup.findIndex((filterItem) => filterItem.id === id);
          item.labelGroup[searchIndex] = { ...item.labelGroup[searchIndex], key: e };
          item.valueOptions = getMappingValue(e, item.keyMap);
          item.tableData = filterTableData(item.labelGroup, solveInstanceData(item.totalTableData));
        }
      });
      return [...pre];
    });
  };

  const handleSelectValue = (e, id, name) => {
    setSelectAimRowKeys((pre) => {
      pre.map((item) => {
        if (item.aimName === name) {
          const searchIndex = item.labelGroup.findIndex((filterItem) => filterItem.id === id);
          item.labelGroup[searchIndex] = { ...item.labelGroup[searchIndex], value: e };
          item.tableData = filterTableData(item.labelGroup, solveInstanceData(item.totalTableData));
        }
      });
      return [...pre];
    });
  };

  const renderComFilter = (labelItem, labelOptions, valueOptions, id, name) => (
    <div className="flex_filter_group">
      <Select
        className="flex_select key_select"
        placeholder={intl.formatMessage({ id: 'monitor.selectKey' })}
        value={labelItem.key}
        options={labelItem.labelOptions || labelOptions}
        onChange={(e) => handleSelectKey(e, id, name)}
      />
      <span className="medium equal_icon">=</span>
      <Select
        className="flex_select value_select"
        placeholder={intl.formatMessage({ id: 'monitor.selectValue' })}
        value={labelItem.value}
        options={labelItem.valueOptions || valueOptions}
        onChange={(e) => handleSelectValue(e, id, name)}
      />
      <span className="medium close_icon" onClick={() => handleDeleteLabel(id, name)}>
        x
      </span>
    </div>
  );

  // 运行查询
  const handleRunSearch = async (isShow = true) => {
    if (!Boolean(getDataRowKeysInstance(selectAimRowKeys))) {
      isShow ? messageApi.error(intl.formatMessage({ id: 'monitor.selectMetricInstance' })) : '';
    } else {
      timeoutSearchTask();
    }
  };

  const timeoutSearchTask = async (tableInstance = selectAimRowKeys) => {
    const instances = solveTableDataToInstance(tableInstance);
    if (instances.length) {
      const periodTime = comForm.getFieldValue('period');
      const endTimeForm = comForm.getFieldValue('endTime');
      const startTime = dayjs(endTimeForm).subtract(parseInt(periodTime), periodTime.slice(-1)).utc().format();
      const endTime = dayjs(endTimeForm).utc().format();
      const [error, res] = await requestResource(
        getInstanceEachData(dataSource, startTime, endTime, stepList[periodTime], instances),
      );
      if (error) {
        messageApi.error(intl.formatMessage({ id: 'monitor.fetchInstanceListFailed' }));
      }
      if (res) {
        if (res.data.code === ResponseCode.OK) {
          setData(createWidgetDataChart(res.data.data));
        }
      }
    }
  };

  useEffect(() => {
    let aimTemporyList = [];
    aimList.map((item) => {
      aimTemporyList.push({
        label: item,
        value: item,
      });
    });
    setAimOptionsList([...aimTemporyList]);
  }, [aimList]);

  const validateLengthName = async (_rule, value) => {
    if (value.length > 50) {
      messageApi.error(intl.formatMessage({ id: 'monitor.nameTooLong' }));
      return Promise.reject(new Error(''));
    } else {
      return Promise.resolve();
    }
  };

  // 自动刷新
  const getWebSockect = useCallback(() => {
    if (intervalId) {
      clearTimeout(intervalId); // 清除计时器
      setInterValId(0);
    }
    if (refreshTimeValue) {
      const id = setTimeout(() => {
        // 操作
        comForm.setFieldValue('endTime', dayjs()); // 时间赋值
        // 请求接口
        initProgressBar();
        handleRunSearch(false);
        getWebSockect();
        // 请求数据
      }, refreshTimeValue * 1000);
      setInterValId(id);
    }
  }, [refreshTimeValue, JSON.stringify(selectAimRowKeys)]);

  const initProgressBar = useCallback(() => {
    const refreshDom = document.querySelector('#border_group_refresh');
    const maxWidth = window.getComputedStyle(refreshDom).width;
    const progressBar = document.querySelector('#progress_bar');
    progressBar.style.width = maxWidth;
  }, []);

  useEffect(() => {
    return () => clearTimeout(intervalId);
  }, [intervalId]);

  useEffect(() => {
    getWebSockect();
  }, [getWebSockect]);

  return (
    <div className="child_content">
      {contextHolder}
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'dashboard.title' }), path: `/dashboard` },
          {
            title: intl.formatMessage({ id: 'dashboard.monitorBoard' }),
            path: `/bigDataDashboard/detail/${dashboardID}/${dashboardName}`,
          },
          {
            title: componentID
              ? intl.formatMessage({ id: 'monitor.editMonitorComponent' })
              : intl.formatMessage({ id: 'monitor.newMonitorComponent' }),
            path: componentID ? `/editMonitorComponent` : '/addMonitorComponent',
          },
        ]}
      />
      <div className="monitor_com_box">
        <Form
          form={comForm}
          autoComplete={false}
          className="monitor_com_header"
          layout="inline"
          onValuesChange={handleValueChange}
          initialValues={{
            type,
            period: '10m',
            endTime: dayjs(),
            refreshTime: 30,
          }}
        >
          <Form.Item
            className="component_name"
            label={intl.formatMessage({ id: 'monitor.componentName' })}
            name="name"
            rules={[
              {
                required: true,
                message: intl.formatMessage({ id: 'monitor.componentNameRequired' }),
              },
            ]}
            colon={false}
            style={{
              flex: 1,
            }}
          >
            <Input
              className="component_name_input_monitor"
              placeholder={intl.formatMessage({ id: 'common.inputPlaceholder' })}
              maxLength={50}
              showCount
            />
          </Form.Item>
          <Form.Item name="type" className="monitor_type">
            <Select
              options={componentTypeOptions.map((opt) => ({ ...opt, label: intl.formatMessage({ id: opt.label }) }))}
              style={{ width: '120px' }}
            />
          </Form.Item>
          <FormTimeCom refreshTimeValue={refreshTimeValue} handleResetMonitorFn={handleResetMonitor} />
        </Form>
        <div className={`preview_box preview_${passType}_box`}>
          <EchartsCom
            type={passType}
            data={data}
            name={componentName}
            isAimExist={Boolean(getDataRowKeysInstance(selectAimRowKeys))}
            dashboardMax={dashBoardMaxValue}
            dashboardMin={dashBoardMinValue}
          />
        </div>
        <div className="select_chart_content">
          <h2>{intl.formatMessage({ id: 'dashboard.selectChartContent' })}</h2>
          <div className="form_aim">
            {passType === 'dashboard' && (
              <>
                <div className="dashboard_arround_word">{intl.formatMessage({ id: 'dashboard.gaugeRange' })}</div>
                <div className="dashboard_arround_area">
                  <InputNumber
                    controls={false}
                    placeholder={intl.formatMessage({ id: 'monitor.minValue' })}
                    className="dashboard_arround_input"
                    value={dashBoardMinValue}
                    onChange={(value) => setDashBoardMinValue(value)}
                  />
                  <span className="dashboard_split">~</span>
                  <InputNumber
                    controls={false}
                    placeholder={intl.formatMessage({ id: 'monitor.maxValue' })}
                    className="dashboard_arround_input"
                    value={dashBoardMaxValue}
                    onChange={(value) => setDashBoardMaxValue(value)}
                  />
                </div>
              </>
            )}
            <div className="aim">{intl.formatMessage({ id: 'dashboard.metric' })}</div>
            <div className="content_aim">
              <Select
                showSearch
                mode="multiple"
                value={selectAimList}
                options={aimOptionsList}
                className="aim_select"
                placeholder={intl.formatMessage({ id: 'common.selectPlaceholder' })}
                onChange={handleSelectChange}
              />
              <Button className="monitor_primary_btn" onClick={handleRunSearch}>
                {intl.formatMessage({ id: 'monitor.runQuery' })}
              </Button>
            </div>
          </div>
        </div>
        <div className="aim_table_box">
          {selectAimRowKeys.map((item) => (
            <div className="aim_content">
              <div className="aim_header">
                <h2>{item.aimName}</h2>
                <div
                  style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: '10px' }}
                >
                  {item.labelGroup.map((labelItem) =>
                    renderComFilter(labelItem, item.labelOptions, item.valueOptions, labelItem.id, item.aimName),
                  )}
                  <Button icon={<PlusOutlined />} onClick={() => handleAddLabelGroup(item.aimName)}>
                    {intl.formatMessage({ id: 'monitor.labelFilter' })}
                  </Button>
                </div>
              </div>
              <Table
                rowKey="id"
                rowSelection={{
                  type: 'checkbox',
                  selectedRowKeys: [...item.selectedRowKeys],
                  onChange: (newSelectedRowKeys, selectedRows) => {
                    let temporySelectRowKesys = selectedRows.length; // 当前表的选择数
                    setSelectAimRowKeys((preSelectAim) => {
                      preSelectAim.map((preItem) => {
                        if (preItem.aimName === item.aimName) {
                          const numCount = getDataRowKeysInstance(preSelectAim, temporySelectRowKesys, item.aimName);
                          if (numCount > 8) {
                            messageApi.error(intl.formatMessage({ id: 'monitor.maxMetrics' }));
                          } else {
                            if (passType === 'dashboard' && (dashBoardMinValue === '' || !dashBoardMaxValue)) {
                              messageApi.error(intl.formatMessage({ id: 'monitor.fillGaugeRange' }));
                            } else {
                              if (numCount > 3 && passType === 'area') {
                                messageApi.error(intl.formatMessage({ id: 'monitor.stackedAreaMaxMetrics' }));
                              } else {
                                !numCount ? setData([]) : '';
                                preItem.selectedRowKeys = [...newSelectedRowKeys];
                                preItem.selectInstances = [...selectedRows];
                              }
                            }
                          }
                        }
                      });
                      return [...preSelectAim];
                    });
                  },
                  getCheckboxProps: (record) => ({
                    disabled: record.disabled,
                  }),
                }}
                className="aim_table"
                columns={aimColumns}
                dataSource={item.tableData}
                pagination={false}
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
              />
            </div>
          ))}
        </div>
      </div>
      <div className="monitor_footer">
        {Boolean(getDataRowKeysInstance(selectAimRowKeys)) && (
          <div className="descirption">
            {intl.formatMessage(
              { id: 'monitor.selectedSummary' },
              { aims: getReturnAims(selectAimRowKeys), instances: getDataRowKeysInstance(selectAimRowKeys) },
            )}
          </div>
        )}
        <Button className="two_word_btn monitor_cancel_btn" onClick={() => handleCancel()}>
          {intl.formatMessage({ id: 'common.cancel' })}
        </Button>
        <Button className="monitor_primary_btn two_word_btn" style={{ marginLeft: '16px' }} onClick={handleCreate}>
          {componentID ? intl.formatMessage({ id: 'common.ok' }) : intl.formatMessage({ id: 'common.create' })}
        </Button>
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'monitor.abandonOperation' })}
        open={cancelModalOpen}
        confirmText={intl.formatMessage({ id: 'monitor.continueEditing' })}
        cancelText={intl.formatMessage({ id: 'monitor.confirmAbandon' })}
        confirmBtnID="monitor_cancel_handle_continue"
        cancelBtnID="monitor_cancel_handle_confirm"
        cancelFn={handleCancelModalCancel}
        content={[
          intl.formatMessage({ id: 'monitor.abandonWarning' }),
          intl.formatMessage({ id: 'monitor.confirmAbandonOperation' }),
        ]}
        showCheck={false}
        confirmFn={handleCancelMonitorConfirm}
      />
    </div>
  );
}
