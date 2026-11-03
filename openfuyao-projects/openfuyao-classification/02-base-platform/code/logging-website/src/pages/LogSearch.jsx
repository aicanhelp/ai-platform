/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix, levelOptions } from '@/common/constants';
import noSearch from '@/assets/images/noSearch.png';
import { Radio, Tooltip, DatePicker, Select, Input, Form, Button, message } from 'antd';
import {
  QuestionCircleOutlined,
  ExportOutlined,
  RightOutlined,
  DownOutlined,
  FileTextOutlined,
  CopyOutlined,
} from '@ant-design/icons';
import { ResponseCode } from '@/common/constants';
import {
  getLogOptions,
  getLogFilenameOptions,
  getLogSearchData,
  getLogSearchDataBycollecting,
  getLogOptionsAll,
} from '@/api/containerLogApi';
import { solveAnnotation, exportHelmLogOutPut } from '@/tools/utils';
import dayjs from 'dayjs';
import { useHistory } from 'inula-router';
import copy from 'copy-to-clipboard';

export default function Log() {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [selectType, setSelectType] = useState('resource');
  const { RangePicker } = DatePicker;
  const [namespaceOptions, setNamespaceOptions] = useState([]);
  const [podOptions, setPodOptions] = useState([]);
  const [containerOptions, setContainerOptions] = useState([]);
  const [beginTime, setBegintime] = useState(dayjs().add(-7, 'd').valueOf());
  const [endTime, setEndtime] = useState(dayjs().valueOf());
  const [filenameOptions, setFilenameOptions] = useState([]);
  const [rangeValue, setRangeValue] = useState([dayjs().add(-7, 'd'), dayjs()]);
  const [keywords, setKeywords] = useState('');
  const [logdata, setLogdata] = useState([]);
  const [context, setContext] = useState('');
  const [isSearch, setIsSearch] = useState('false');
  const [exportString, setExportString] = useState('');
  const [outputFilename, setOutputFilename] = useState('');
  const [ifPodDisabled, setIfPodDisabled] = useState(true);
  const [ifContainerDisabled, setIfContainerDisabled] = useState(true);
  const history = useHistory();
  const onChangeSelectType = (e) => {
    form.resetFields();
    if (e.target.value === 'resource') {
      form.setFieldsValue({ filename: '', level: '', keywords: '' });
    } else {
      form.setFieldsValue({ namespace: '', pod: '', container: '', level: '', keywords: '' });
    }
    setLogdata([]);
    setRangeValue([dayjs().add(-7, 'd'), dayjs()]);
    setSelectType(e.target.value);
  };
  const handleChangeNamespace = (value) => {
    getLogAllOptions(value, 'namespace');
  };
  const handleChangePod = (value) => {
    getLogAllOptions(value, 'pod');
  };
  const handleChangeContainer = (value) => {
    getLogAllOptions(value, 'container');
  };
  const handleChangeLevel = (value) => {};
  const handleChangeFilename = (value) => {};
  const handleChangeKeywords = (value) => {
    setKeywords(value);
  };
  const [isView, setIsView] = useState('');
  function handleViewLog(value) {
    setIsView(value);
  }
  const handleExportLog = () => {
    if (exportString) {
      exportHelmLogOutPut(outputFilename, exportString);
      messageApi.success(intl.formatMessage({ id: 'logging.exportSuccess' }));
    } else {
      messageApi.error(intl.formatMessage({ id: 'logging.noLogCannotExport' }));
    }
  };
  const onRangeChange = (dates, dateStrings) => {
    if (dates) {
      setRangeValue([dayjs(dateStrings[0]), dayjs(dateStrings[1])]);
      setBegintime(dayjs(dateStrings[0]).valueOf());
      setEndtime(dayjs(dateStrings[1]).valueOf());
    }
  };
  const rangePresets = [
    {
      label: intl.formatMessage({ id: 'time.last5Minutes' }),
      value: [dayjs().add(-5, 'm'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last15Minutes' }),
      value: [dayjs().add(-15, 'm'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last30Minutes' }),
      value: [dayjs().add(-30, 'm'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last1Hour' }),
      value: [dayjs().add(-1, 'h'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last2Hours' }),
      value: [dayjs().add(-2, 'h'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last6Hours' }),
      value: [dayjs().add(-6, 'h'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last12Hours' }),
      value: [dayjs().add(-12, 'h'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last1Day' }),
      value: [dayjs().add(-1, 'd'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last2Days' }),
      value: [dayjs().add(-2, 'd'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last3Days' }),
      value: [dayjs().add(-3, 'd'), dayjs()],
    },
    {
      label: intl.formatMessage({ id: 'time.last7Days' }),
      value: [dayjs().add(-7, 'd'), dayjs()],
    },
  ];
  function onFinish() {
    if (selectType === 'resource') {
      let _param = {
        namespace: form.getFieldsValue().namespace,
        pod: form.getFieldsValue().pod,
        container: form.getFieldsValue().container,
        start: parseInt(beginTime / 1000),
        end: parseInt(endTime / 1000),
        level: form.getFieldsValue().level,
        keyword: form.getFieldsValue().keywords,
        size: 500,
      };
      let msg = '';
      if (_param.pod && !_param.namespace) {
        msg = intl.formatMessage({ id: 'logging.mustSelectNamespace' });
      } else if (_param.container && !_param.pod) {
        msg = intl.formatMessage({ id: 'logging.mustSelectPod' });
      } else if (_param.container && !_param.namespace) {
        msg = intl.formatMessage({ id: 'logging.mustSelectNamespace' });
      } else {
        msg = '';
      }
      if (!msg) {
        if (_param.namespace) {
          getLogData(_param);
          setOutputFilename([_param.namespace, _param.pod, _param.container].filter((item) => item).join('_'));
        } else {
          messageApi.error(intl.formatMessage({ id: 'logging.selectNamespace' }));
        }
      } else {
        messageApi.error(msg);
      }
    } else {
      let _collectingParam = {
        filename: decodeURI(form.getFieldsValue().filename),
        start: parseInt(beginTime / 1000),
        end: parseInt(endTime / 1000),
        level: form.getFieldsValue().level,
        keyword: form.getFieldsValue().keywords,
        size: 500,
      };
      if (_collectingParam.filename && _collectingParam.filename !== 'undefined') {
        getLogData(_collectingParam);
        if (_collectingParam.filename.startsWith('/var/log/pods/')) {
          let [_namespacePod, _container, _filename] = _collectingParam.filename.split('/').slice(4);
          setOutputFilename([..._namespacePod.split('_').splice(0, 2), _container, _filename.split('.')[0]].join('_'));
        } else {
          let _logFilename = _collectingParam.filename.split('/').slice(-1)[0].replace('.log', '');
          setOutputFilename(_logFilename);
        }
      } else {
        messageApi.error(intl.formatMessage({ id: 'logging.selectFilename' }));
      }
    }
  }
  function onReset() {
    setLogdata([]);
    form.resetFields();
    getLogAllOptions();
    setBegintime(dayjs().add(-7, 'd').valueOf());
    setEndtime(dayjs().valueOf());
    setRangeValue([dayjs().add(-7, 'd'), dayjs()]);
  }
  const getFilenameOptions = async () => {
    const res = await getLogFilenameOptions();
    if (res.status === ResponseCode.OK) {
      let _filenameList = [];
      res.data.data.data.map((item) => {
        _filenameList.push({
          label: item,
          value: item,
        });
      });
      setFilenameOptions(_filenameList);
    }
  };
  const getLogAllOptions = async (value, type) => {
    const res = await getLogOptions(value, type);
    if (res.status === ResponseCode.OK) {
      let _namespaceList1 = [];
      let _podList1 = [];
      let _containerList1 = [];
      res.data.data.namespaces.map((item) => {
        _namespaceList1.push({
          label: item,
          value: item,
        });
      });
      res.data.data.pods.map((item) => {
        _podList1.push({
          label: item,
          value: item,
        });
      });
      res.data.data.containers.map((item) => {
        _containerList1.push({
          label: item,
          value: item,
        });
      });
      setNamespaceOptions(_namespaceList1);
      setPodOptions(_podList1);
      setContainerOptions(_containerList1);
    }
  };
  const getLogData = async (queryData) => {
    setIsSearch(true);
    let res;
    if (selectType === 'resource') {
      res = await getLogSearchData(queryData);
    } else {
      res = await getLogSearchDataBycollecting(queryData);
    }
    if (res.status === ResponseCode.OK) {
      let arr = [];
      let _mockdata = res.data.data.data.result;
      let _streams = [];
      let _logs = [];
      _mockdata.map((item) => {
        _streams.push({
          stream: item.stream,
        });
        _logs.push({
          value: item.values,
        });
      });
      for (let i = 0; i < _logs.length; i++) {
        let _item = _logs[i].value;
        for (let j = 0; j < _item.length; j++) {
          arr.push({
            streams: _streams[i].stream,
            timestarmp: Number(_item[j][0]),
            stream: solveAnnotation(_streams[i].stream),
            time: Number(_item[j][0] / 1000000),
            timestamp: Number(_item[j][0]),
            log: _item[j][1],
          });
        }
      }
      // Loki 多 stream（stdout/stderr、日志轮转文件）仅保证 stream 内部有序，
      // 必须按时间戳全局降序排序，否则跨 stream 拼接后列表时间会出现回跳
      arr.sort((a, b) => b.timestarmp - a.timestarmp);
      let _exportString = '';
      arr.map((item) => {
        _exportString += `${dayjs(item.time).format('YYYY-MM-DD HH:mm:ss')}  ` + `   ` + `${item.log}` + `\n`;
      });
      setExportString(_exportString);
      setLogdata(arr);
    }
  };

  const getOptionsSet = async (values) => {
    let namespaceTempory = '';
    let podTempory = '';
    let containerTempory = '';

    if (values) {
      namespaceTempory = form.getFieldValue('namespace');
      podTempory = values.pod || form.getFieldValue('pod');
      containerTempory = values.container || form.getFieldValue('container');
    }
    const res = await getLogOptionsAll(namespaceTempory, podTempory, containerTempory);
    if (res.status === ResponseCode.OK) {
      let _namespaceList = [];
      let _podList = [];
      let _containerList = [];
      res.data.data.namespaces.map((item) => {
        _namespaceList.push({
          label: item,
          value: item,
        });
      });
      res.data.data.pods.map((item) => {
        _podList.push({
          label: item,
          value: item,
        });
      });
      res.data.data.containers.map((item) => {
        _containerList.push({
          label: item,
          value: item,
        });
      });
      setPodOptions(_podList);
      setNamespaceOptions(_namespaceList);
      setContainerOptions(_containerList);
    }
  };

  const handleCopyLog = (logText) => {
    copy(logText);
    messageApi.success(intl.formatMessage({ id: 'common.copySuccess' }));
  };

  const handleContextAccess = (item) => {
    history.push({ pathname: `/logSearch/detail`, state: { currentlog: item } });
  };

  // 表单修改
  const handleFormChange = (changeValues) => {
    getOptionsSet(changeValues);
  };
  useEffect(() => {
    getLogAllOptions();
    getFilenameOptions();
  }, []);
  useEffect(() => {
    setIfPodDisabled(!form.getFieldValue('namespace'));
    if (form.getFieldValue('namespace')) {
      setIfPodDisabled(false);
    } else {
      form.setFieldValue('pod', '');
      form.setFieldValue('container', '');
      setIfPodDisabled(true);
    }
    if (form.getFieldValue('namespace') && form.getFieldValue('pod')) {
      setIfContainerDisabled(false);
    } else {
      form.setFieldValue('container', '');
      setIfContainerDisabled(true);
    }
  }, [form.getFieldValue('namespace'), form.getFieldValue('pod')]);
  return (
    <div className="logs_search child_content">
      {contextHolder}
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'logging.title' }), path: `/${containerRouterPrefix}` },
          { title: intl.formatMessage({ id: 'logging.logQuery' }), path: `log/logSearch` },
        ]}
      />
      <div className="search_content">
        <div className="radio_item">
          <Radio.Group onChange={onChangeSelectType} value={selectType} className="flex_between" style={{ gap: '8px' }}>
            <Radio value={'resource'}>{intl.formatMessage({ id: 'logging.queryByResource' })}</Radio>
            <Radio value={'collect'}>{intl.formatMessage({ id: 'logging.queryByCollection' })}</Radio>
          </Radio.Group>
          <span>
            <Tooltip title={intl.formatMessage({ id: 'logging.queryByFilename' })}>
              <QuestionCircleOutlined style={{ color: '#89939B' }} />
            </Tooltip>
          </span>
        </div>
        <Form form={form} name="control-hooks" onValuesChange={handleFormChange}>
          <Form.Item
            name="filename"
            label="filename"
            rules={[{ required: true, message: intl.formatMessage({ id: 'logging.filenamePlaceholder' }) }]}
            style={{ display: selectType !== 'resource' ? 'block' : 'none', marginBottom: '40px' }}
          >
            <Select onChange={handleChangeFilename} options={filenameOptions} showSearch={true} />
          </Form.Item>
          <div className="select_item">
            <Form.Item
              name="namespace"
              className="marginBottom40"
              label={intl.formatMessage({ id: 'logging.namespace' })}
              rules={[{ required: true, message: intl.formatMessage({ id: 'logging.namespacePlaceholder' }) }]}
              style={{ display: selectType === 'resource' ? 'block' : 'none' }}
            >
              <Select allowClear options={namespaceOptions} showSearch={true} />
            </Form.Item>
            <Form.Item
              name="pod"
              className="marginBottom40"
              label="Pod"
              style={{ display: selectType === 'resource' ? 'block' : 'none' }}
            >
              <Select allowClear disabled={ifPodDisabled} options={podOptions} showSearch={true} />
            </Form.Item>
            <Form.Item
              name="container"
              className="marginBottom40"
              label={intl.formatMessage({ id: 'logging.container' })}
              style={{ display: selectType === 'resource' ? 'block' : 'none' }}
            >
              <Select allowClear disabled={ifContainerDisabled} options={containerOptions} showSearch={true} />
            </Form.Item>
            <Form.Item name="level" label={intl.formatMessage({ id: 'logging.logLevel' })}>
              <Select allowClear onChange={handleChangeLevel} options={levelOptions} showSearch={true} />
            </Form.Item>
            <div style={{ display: 'flex', gap: '8px', marginTop: '0px' }}>
              <div style={{ lineHeight: '32px' }}>{intl.formatMessage({ id: 'logging.time' })}</div>
              <div style={{ flex: 1 }}>
                <RangePicker
                  showTime
                  presets={rangePresets}
                  allowClear={false}
                  onChange={onRangeChange}
                  value={rangeValue}
                />
              </div>
            </div>
            <Form.Item name="keywords" label={intl.formatMessage({ id: 'logging.keyword' })}>
              <Input
                placeholder={intl.formatMessage({ id: 'logging.keywordPlaceholder' })}
                value={keywords}
                onChange={handleChangeKeywords}
              />
            </Form.Item>
          </div>
          <div className="btn_item">
            <Button className="cancel_btn" htmlType="button" onClick={onReset}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" htmlType="submit" onClick={onFinish}>
              {intl.formatMessage({ id: 'common.query' })}
            </Button>
          </div>
        </Form>
      </div>
      <div className="log_result" style={{ display: logdata.length ? 'block' : 'none' }}>
        <div className="flex_between" style={{ marginBottom: '20px' }}>
          <div className="log_title">{intl.formatMessage({ id: 'logging.title' })}</div>
          <div
            className="exports hover"
            style={{ display: 'flex', gap: '8px', alignItems: 'center' }}
            onClick={handleExportLog}
          >
            <ExportOutlined />
            <span>{intl.formatMessage({ id: 'logging.export' })}</span>
          </div>
        </div>
        <div className="logItems" style={{ border: '1px solid #dcdcdcff' }}>
          {logdata.map((item) => (
            <div>
              <div className="logItem" onMouseEnter={() => setContext(item.time)}>
                {isView !== item.time ? (
                  <div className="right_arrow" onClick={() => handleViewLog(item.time)}>
                    <RightOutlined />
                  </div>
                ) : (
                  <div className="right_arrow" onClick={handleViewLog}>
                    <DownOutlined />
                  </div>
                )}
                <div className="log_time">{dayjs(item.time).format('YYYY-MM-DD HH:mm:ss')}</div>
                <div className="log_content">{item.log}</div>
                <div className="log_option" style={{ opacity: context === item.time ? 1 : 0 }}>
                  <FileTextOutlined style={{ color: '#3f66f5' }} />
                  <Button type="link" onClick={() => handleContextAccess(item)}>
                    {intl.formatMessage({ id: 'logging.context' })}
                  </Button>
                  <CopyOutlined style={{ color: '#3f66f5' }} />
                  <Button type="link" onClick={() => handleCopyLog(item.log)}>
                    {intl.formatMessage({ id: 'common.copy' })}
                  </Button>
                </div>
              </div>
              {isView === item.time ? (
                <div className="log_details">
                  <p className="logDetail_common">fields</p>
                  {item.stream.map((subitem) => (
                    <div style={{ display: 'flex' }}>
                      <p className="logDetail_common logDetail_padddingleft">{subitem.key}</p>
                      <p className="logDetail_value">{subitem.value}</p>
                    </div>
                  ))}
                </div>
              ) : (
                ''
              )}
            </div>
          ))}
        </div>
      </div>
      <div className="log_result" style={{ display: logdata.length ? 'none' : 'flex', flexDirection: 'column' }}>
        <div style={{ marginBottom: '20px' }}>
          <div className="log_title">{intl.formatMessage({ id: 'logging.title' })}</div>
          <div className="noSearch">
            <img src={noSearch} alt="" className="search_img" />
            <p className="search_title">
              {isSearch
                ? intl.formatMessage({ id: 'logging.noData' })
                : intl.formatMessage({ id: 'logging.pleaseQuery' })}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
