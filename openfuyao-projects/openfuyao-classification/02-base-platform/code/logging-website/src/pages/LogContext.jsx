/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/log.less';
import { useIntl } from 'inula-intl';
import { useLocation } from 'inula-router';
import { useEffect, useState } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import dayjs from 'dayjs';
import { ResponseCode, levelOptions } from '@/common/constants';
import { Select, Input, Form, Button, message } from 'antd';
import { ExportOutlined, RightOutlined, DownOutlined } from '@ant-design/icons';
import { getLogDetail, getLogSearch } from '@/api/containerLogApi';
import { solveAnnotation, exportHelmLogOutPut, getLogcontextBk } from '@/tools/utils';
import useLocalStorage from '@/hooks/useLocalStorage';
export default function LogContext() {
  const intl = useIntl();
  const location = useLocation();
  const [logContextForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [logdata, setLogdata] = useState([]);
  const [currentNum, setCurrentNum] = useState();
  const [exportString, setExportString] = useState('');
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const timeOptions = [
    { label: '5min', value: '5min' },
    { label: '15min', value: '15min' },
    { label: '30min', value: '30min' },
    { label: '1hour', value: '1hour' },
    { label: '2hour', value: '2hour' },
    { label: '12hour', value: '12hour' },
    { label: '1day', value: '1day' },
    { label: '2day', value: '2day' },
    { label: '3day', value: '3day' },
  ];
  const handleExportLog = () => {
    if (exportString) {
      let { namespace, pod, container, filename } = location.state.currentlog.streams;
      let outputFilename;
      if (namespace) {
        outputFilename = [namespace, pod, container].join('_');
      } else {
        outputFilename = filename.split('/').slice(-1)[0].replace('.log', '');
      }
      exportHelmLogOutPut(`${outputFilename}_context`, exportString);
      messageApi.success(intl.formatMessage({ id: 'logging.exportSuccess' }));
    } else {
      messageApi.error(intl.formatMessage({ id: 'logging.noLogCannotExport' }));
    }
  };
  const param = {
    filename: location.state.currentlog.streams.filename,
    level: '',
    timePeriod: '3day',
    keyword: '',
    size: 500,
    timestamp: location.state.currentlog.timestamp,
  };
  const [isView, setIsView] = useState('');
  function handleViewLog(value) {
    setIsView(value);
  }
  const onReset = () => {
    logContextForm.resetFields();
    logContextForm.setFieldsValue({ time: '3day' });
  };
  const onSubmit = async () => {
    const { time, keywords, level } = logContextForm.getFieldsValue();
    let timePeriod = time;
    let keyword = keywords;
    let newConditions = {
      fileName: param.filename,
      timestamp: param.timestamp,
      timePeriod,
      keyword,
      level,
      size: param.size,
    };
    const res = await getLogSearch(newConditions);
    if (res.status === ResponseCode.OK) {
      let arr1 = [];
      let _mockdata1 = res.data.data.results[0].data.result;
      let _streams1 = [];
      let _logs1 = [];
      _mockdata1.map((item) => {
        _streams1.push({
          stream: item.stream,
        });
        _logs1.push({
          value: item.values,
        });
      });
      for (let i = 0; i < _logs1.length; i++) {
        let _item = _logs1[i].value;
        for (let j = 0; j < _item.length; j++) {
          arr1.push({
            streams: _streams1[i].stream,
            timestarmp: Number(_item[j][0]),
            stream: solveAnnotation(_streams1[i].stream),
            time: Number(_item[j][0] / 1000000),
            log: _item[j][1],
          });
        }
      }
      let arr2 = [];
      let _mockdata2 = res.data.data.results[1].data.result;
      let _streams2 = [];
      let _logs2 = [];
      _mockdata2.map((item) => {
        _streams2.push({
          stream: item.stream,
        });
        _logs2.push({
          value: item.values,
        });
      });
      for (let i = 0; i < _logs2.length; i++) {
        let _item = _logs2[i].value;
        for (let j = 0; j < _item.length; j++) {
          arr2.push({
            streams: _streams2[i].stream,
            timestarmp: Number(_item[j][0]),
            stream: solveAnnotation(_streams2[i].stream),
            time: Number(_item[j][0] / 1000000),
            log: _item[j][1],
          });
        }
      }
      let currentArray = [location.state.currentlog];
      setCurrentNum(arr1.length);
      arr1 = arr1.sort((a, b) => a.timestarmp - b.timestarmp);
      arr2 = arr2.sort((a, b) => a.timestarmp - b.timestarmp);
      let _exportString = '';
      arr1.concat(currentArray, arr2).map((item) => {
        _exportString += `${dayjs(item.time).format('YYYY-MM-DD HH:mm:ss')}  ` + `   ` + `${item.log}` + `\n`;
      });
      setExportString(_exportString);
      setLogdata(arr1.concat(currentArray, arr2));
    }
  };
  const getLogData = async (queryData) => {
    const res = await getLogDetail(queryData);
    if (res.status === ResponseCode.OK) {
      let getLogArr1 = [];
      let _mockGetLogdata1 = res.data.data.results[0].data.result;
      let _streamsGetLog1 = [];
      let _logsGetLog1 = [];
      _mockGetLogdata1.map((item) => {
        _streamsGetLog1.push({
          stream: item.stream,
        });
        _logsGetLog1.push({
          value: item.values,
        });
      });
      for (let i = 0; i < _logsGetLog1.length; i++) {
        let _item = _logsGetLog1[i].value;
        for (let j = 0; j < _item.length; j++) {
          getLogArr1.push({
            streams: _streamsGetLog1[i].stream,
            timestarmp: Number(_item[j][0]),
            stream: solveAnnotation(_streamsGetLog1[i].stream),
            time: Number(_item[j][0] / 1000000),
            log: _item[j][1],
          });
        }
      }
      let getLogArr2 = [];
      let _mockGetLogdata2 = res.data.data.results[1].data.result;
      let _streamsGetLog2 = [];
      let _logsGetLog2 = [];
      _mockGetLogdata2.map((item) => {
        _streamsGetLog2.push({
          stream: item.stream,
        });
        _logsGetLog2.push({
          value: item.values,
        });
      });
      for (let i = 0; i < _logsGetLog2.length; i++) {
        let _item = _logsGetLog2[i].value;
        for (let j = 0; j < _item.length; j++) {
          getLogArr2.push({
            streams: _streamsGetLog2[i].stream,
            timestarmp: Number(_item[j][0]),
            stream: solveAnnotation(_streamsGetLog2[i].stream),
            time: Number(_item[j][0] / 1000000),
            log: _item[j][1],
          });
        }
      }
      let currentArray = [location.state.currentlog];
      setCurrentNum(getLogArr1.length);
      getLogArr1 = getLogArr1.sort((a, b) => a.timestarmp - b.timestarmp);
      getLogArr2 = getLogArr2.sort((a, b) => a.timestarmp - b.timestarmp);
      let _exportString = '';
      getLogArr1.concat(currentArray, getLogArr2).map((item) => {
        _exportString += `${dayjs(item.time).format('YYYY-MM-DD HH:mm:ss')}  ` + `   ` + `${item.log}` + `\n`;
      });
      setExportString(_exportString);
      setLogdata(getLogArr1.concat(currentArray, getLogArr2));
    }
  };
  useEffect(() => {
    getLogData(param);
  }, []);
  return (
    <div className="logContext">
      {contextHolder}
      <div className="logContext_top">
        <BreadCrumbCom
          items={[
            { title: intl.formatMessage({ id: 'logging.logQuery' }), path: '/logSearch' },
            { title: intl.formatMessage({ id: 'logging.context' }), path: '/logSearch/detail' },
          ]}
        />
        <div className="logContext_top_info">
          <div className="logContext_top_info_content">
            <p className="logContext_top_info_content_label">{intl.formatMessage({ id: 'logging.namespace' })}</p>
            <p className="logContext_top_info_content_value">{location.state.currentlog.streams.namespace}</p>
          </div>
          <div className="logContext_top_info_content">
            <p className="logContext_top_info_content_label">{intl.formatMessage({ id: 'context.pod' })}</p>
            <p className="logContext_top_info_content_value">{location.state.currentlog.streams.pod}</p>
          </div>
          <div className="logContext_top_info_content">
            <p className="logContext_top_info_content_label">{intl.formatMessage({ id: 'logging.container' })}</p>
            <p className="logContext_top_info_content_value">{location.state.currentlog.streams.container}</p>
          </div>
        </div>
      </div>
      <div className="logContext_search_content">
        <Form className="context_form" form={logContextForm} name="control-hooks">
          <div className="select_item">
            <Form.Item name="level" label={intl.formatMessage({ id: 'logging.logLevel' })}>
              <Select allowClear options={levelOptions} />
            </Form.Item>
            <Form.Item name="time" label="time" initialValue={'3day'}>
              <Select options={timeOptions} />
            </Form.Item>
            <Form.Item name="keywords" label={intl.formatMessage({ id: 'logging.keyword' })}>
              <Input placeholder={intl.formatMessage({ id: 'logging.keywordPlaceholder' })} />
            </Form.Item>
          </div>
          <div className="btn_item">
            <Button htmlType="button" className="resetBtn" onClick={onReset} style={{ margin: '0 16px' }}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn_log" htmlType="submit" onClick={onSubmit}>
              {intl.formatMessage({ id: 'common.query' })}
            </Button>
          </div>
        </Form>
      </div>
      <div className="log_result logContext">
        <div className="flex_between" style={{ marginBottom: '20px' }}>
          <div className="log_title">{intl.formatMessage({ id: 'logging.title' })}</div>
          <div
            className="exports hover logContext"
            style={{ display: 'flex', gap: '8px', alignItems: 'center' }}
            onClick={handleExportLog}
          >
            <ExportOutlined />
            <span>{intl.formatMessage({ id: 'logging.export' })}</span>
          </div>
        </div>
        <div className="logItems logContext" style={{ border: '1px solid #dcdcdcff' }}>
          {logdata.map((item, index) => (
            <div>
              <div className="logItem">
                {isView !== item.time ? (
                  <div className="right_arrow" onClick={() => handleViewLog(item.time)}>
                    <RightOutlined />
                  </div>
                ) : (
                  <div className="right_arrow" onClick={handleViewLog}>
                    <DownOutlined />
                  </div>
                )}
                <div className="log_time logContext">{dayjs(item.time).format('YYYY-MM-DD HH:mm:ss')}</div>
                <div className="log_content" style={{ background: getLogcontextBk(index, currentNum, currentTheme) }}>
                  {item.log}
                </div>
              </div>
              {isView === item.time ? (
                <div className="log_details">
                  <p className="logDetail_common logContext">fields</p>
                  {item.stream.map((subitem) => (
                    <div style={{ display: 'flex' }}>
                      <p className="logDetail_common logDetail_padddingleft">{subitem.key}</p>
                      <p className="logDetail_value logContext">{subitem.value}</p>
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
    </div>
  );
}
