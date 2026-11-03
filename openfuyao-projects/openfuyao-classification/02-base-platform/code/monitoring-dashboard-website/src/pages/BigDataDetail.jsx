/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useParams, useHistory } from 'inula-router';
import '@/styles/pages/bigDataDetail.less';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { Form, Button, message, Dropdown, Space, Tooltip } from 'antd';
import dayjs from 'dayjs';
import { DownOutlined, UpOutlined } from '@ant-design/icons';
import { useEffect, useState, useCallback, useRef } from 'openinula';
import { useIntl } from 'inula-intl';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import EmptyHandle from '@/components/EmptyHandle';
import emptyComponentImg from '@/assets/images/chart/emptyComponent.png';
import CreateMonitorComponentModal from '@/components/CreateMonitorComponentModal';
import FormTimeCom from '@/components/FormTimeCom';
import { getDashboardResource, updateEachDashboard, deleteDashboard, getIndicatorsList } from '@/api/monitorApi';
import DisplayBigData from '@/components/DisplayBigData';
import CreateMonitorBoardModal from '@/components/CreateMonitorBoardModal';
import utc from 'dayjs/plugin/utc';
import { requestResource } from '@/tools/utils';
import { stepList } from '@/common/constants';

dayjs.extend(utc);
export default function BigDataDetail() {
  const intl = useIntl();
  const { dashboardID, dashboardName } = useParams();
  const history = useHistory();
  const [loading, setLoading] = useState(false); // 加载中
  const [searchForm] = Form.useForm();
  const [intervalId, setInterValId] = useState(0); // 定时器id
  const refreshTimeValue = Form.useWatch('refreshTime', searchForm); // 刷新间隔
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [isMonitorDelCheck, setIsMonitorDelCheck] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [expand, setExpand] = useState(false);
  const [dashBoardData, setDashBoardData] = useState([]); // 看板数据
  const [isCreateMonitorComponentModalOpen, setIsCreateMonitorComponentModal] = useState(false);
  const [isCreateMonitorDashboardModalOpen, setIsCreateMonitorDashboardModal] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false); // 是否加载完成数据
  const [sourceURL, setSourceURL] = useState('');

  const handleValueChange = (changeValues) => {
    if (Object.prototype.hasOwnProperty.call(changeValues, 'endTime')) {
      if (dayjs(changeValues.endTime).format('YYYY-MM-DD HH:mm:ss') === dayjs().format('YYYY-MM-DD HH:mm:ss')) {
        if (searchForm.getFieldValue('refreshTime') === 0) {
          searchForm.setFieldValue('refreshTime', 30);
        }
      } else {
        searchForm.setFieldValue('refreshTime', 0);
      }
    }
    if (Object.prototype.hasOwnProperty.call(changeValues, 'refreshTime')) {
      if (searchForm.getFieldValue('refreshTime') !== 0) {
        searchForm.setFieldValue('endTime', dayjs());
      }
    }
    getWidgetList();
  };

  const handleResetMonitor = () => {
    getWidgetList();
  };
  const handleDeleteDashboard = () => {
    setDeleteModalOpen(true);
  };

  const handleEditDashboard = () => {
    setIsCreateMonitorDashboardModal(true);
  };

  const handleConfirmMonitorDashboard = async (name) => {
    const data = {
      displayName: name,
    };
    const [error, res] = await requestResource(updateEachDashboard(dashboardID, data));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'monitor.updateFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'monitor.updateSuccess' }));
        setIsCreateMonitorDashboardModal(false);
        history.replace(`/dashboard/bigDataDashboard/detail/${dashboardID}/${name}`);
      }
    }
  };

  const handleDelModalCancel = () => {
    setDeleteModalOpen(false);
    setIsMonitorDelCheck(false);
  };

  const handleCancelMonitorDashboard = () => {
    setIsCreateMonitorDashboardModal(false);
  };

  const handleMonitorCheckFn = (e) => {
    setIsMonitorDelCheck(e.target.checked);
  };

  const handleDelMonitorConfirm = async () => {
    const [error, res] = await requestResource(deleteDashboard(dashboardID));
    if (error) {
      messageApi.error(intl.formatMessage({ id: 'monitor.deleteFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'monitor.deleteSuccess' }));
        setDeleteModalOpen(false);
        setTimeout(() => {
          history.push('/dashboard');
        }, 2000);
      }
    }
  };

  const items = [
    {
      key: 'edit',
      label: (
        <Button type="link" className="monitor_link_btn" onClick={handleEditDashboard}>
          {intl.formatMessage({ id: 'common.edit' })}
        </Button>
      ),
    },
    {
      key: 'delete',
      label: (
        <Button type="link" className="monitor_link_btn" onClick={handleDeleteDashboard}>
          {intl.formatMessage({ id: 'common.delete' })}
        </Button>
      ),
    },
  ];

  const handleDropDown = () => {
    setExpand(!expand);
  };

  const handleCreateMonitorComponent = () => {
    setIsCreateMonitorComponentModal(true);
  };

  const handleConfirmMonitorComponent = async (type) => {
    // 下一步跳转
    if (sourceURL) {
      let flag = true;
      const reg = /^http:\/\/[^/]+:(?<name>\d{1,5})$/;
      const match = reg.exec(sourceURL);
      if (match) {
        if (!match.groups.name) {
          flag = false;
          messageApi.error(intl.formatMessage({ id: 'monitor.prometheusUrlInvalid' }));
        }
      } else {
        flag = false;
        messageApi.error(intl.formatMessage({ id: 'monitor.prometheusUrlInvalid' }));
      }
      if (flag) {
        // 获取指标列表
        setLoading(true);
        const [error, res] = await requestResource(getIndicatorsList(sourceURL));
        if (error) {
          messageApi.error(intl.formatMessage({ id: 'monitor.prometheusUrlUnreachable' }));
          setLoading(false);
        }
        if (res) {
          setLoading(false);
          const aimList = res.data.data;
          history.push({
            pathname: `/dashboard/bigDataDashboard/detail/${dashboardID}/${dashboardName}/addMonitorComponent`,
            state: { type, aimList, dataSource: sourceURL },
          });
        }
      }
    } else {
      messageApi.error(intl.formatMessage({ id: 'monitor.prometheusUrlRequired' }));
    }
  };

  const handleCancelMonitorComponent = () => {
    setIsCreateMonitorComponentModal(false);
  };

  const handleChangeURL = (e) => {
    setSourceURL(e.target.value);
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
        searchForm.setFieldValue('endTime', dayjs()); // 时间赋值
        // 请求接口
        initProgressBar();
        getWidgetList();
        getWebSockect();
        // 请求数据
      }, refreshTimeValue * 1000);
      setInterValId(id);
    }
  }, [refreshTimeValue]);

  const getWidgetList = useCallback(async () => {
    const periodTime = searchForm.getFieldValue('period');
    const endTimeForm = searchForm.getFieldValue('endTime');
    const startTime = dayjs(endTimeForm).subtract(parseInt(periodTime), periodTime.slice(-1)).utc().format();
    const endTime = dayjs(endTimeForm).utc().format();
    const [error, res] = await requestResource(
      getDashboardResource(dashboardID, startTime, endTime, stepList[periodTime]),
    );
    if (error) {
      setDashBoardData([]);
      setDataLoaded(true);
      messageApi.error(intl.formatMessage({ id: 'dashboard.fetchDataFailed' }));
    }
    if (res) {
      if (res.data.code === ResponseCode.OK) {
        setDashBoardData(res.data.data.widgets || []);
        setDataLoaded(true);
      }
    }
  }, []);

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

  useEffect(() => {
    getWidgetList();
  }, []);

  return (
    <div className="child_content">
      {contextHolder}
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'dashboard.title' }), path: `/dashboard` },
          { title: intl.formatMessage({ id: 'dashboard.monitorBoard' }), path: `/bigDataDashboard/detail` },
        ]}
      />
      <div className="big_data_dashboard_box">
        <div className="big_data_header">
          <Tooltip title={decodeURIComponent(dashboardName)}>
            <h2>{decodeURIComponent(dashboardName)}</h2>
          </Tooltip>
          <Form
            form={searchForm}
            layout="inline"
            onValuesChange={handleValueChange}
            initialValues={{
              period: '10m',
              endTime: dayjs(),
              refreshTime: 30,
            }}
          >
            <FormTimeCom refreshTimeValue={refreshTimeValue} handleResetMonitorFn={handleResetMonitor} />
            <Form.Item>
              <Dropdown
                menu={{ items }}
                trigger={['click']}
                getPopupContainer={(triggerNode) => {
                  if (triggerNode && triggerNode.parentNode) {
                    return triggerNode.parentNode;
                  }
                  return document.body;
                }}
                onOpenChange={handleDropDown}
                onClick={handleDropDown}
              >
                <Button className="monitor_primary_btn">
                  <Space>
                    {intl.formatMessage({ id: 'common.operations' })}
                    {expand ? <UpOutlined /> : <DownOutlined />}
                  </Space>
                </Button>
              </Dropdown>
              <Button
                className="monitor_primary_btn"
                style={{ marginLeft: '16px' }}
                onClick={handleCreateMonitorComponent}
              >
                {intl.formatMessage({ id: 'dashboard.createComponent' })}
              </Button>
            </Form.Item>
          </Form>
        </div>
        {dataLoaded ? (
          <div className="monitor_component_content">
            {dashBoardData.length ? (
              <DisplayBigData data={dashBoardData} refreshFn={getWidgetList} />
            ) : (
              <EmptyHandle imgSrc={emptyComponentImg} description={intl.formatMessage({ id: 'dashboard.noComponent' })}>
                {' '}
                <Button onClick={handleCreateMonitorComponent}>
                  {intl.formatMessage({ id: 'dashboard.createComponent' })}
                </Button>
              </EmptyHandle>
            )}
          </div>
        ) : (
          <div className="monitor_component_content" />
        )}
      </div>
      <CreateMonitorComponentModal
        open={isCreateMonitorComponentModalOpen}
        okFn={handleConfirmMonitorComponent}
        cancelFn={handleCancelMonitorComponent}
        URL={sourceURL}
        loading={loading}
        changeURL={handleChangeURL}
      />
      <CreateMonitorBoardModal
        title={intl.formatMessage({ id: 'dashboard.editBoard' })}
        open={isCreateMonitorDashboardModalOpen}
        name={decodeURIComponent(dashboardName)}
        cancelFn={handleCancelMonitorDashboard}
        okFn={handleConfirmMonitorDashboard}
      />
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'dashboard.deleteBoard' })}
        open={deleteModalOpen}
        cancelFn={handleDelModalCancel}
        content={[
          intl.formatMessage({ id: 'monitor.deleteBoardWarning' }),
          intl.formatMessage({ id: 'monitor.confirmDeleteBoard' }, { name: decodeURIComponent(dashboardName) }),
        ]}
        isCheck={isMonitorDelCheck}
        showCheck={true}
        checkFn={handleMonitorCheckFn}
        confirmFn={handleDelMonitorConfirm}
      />
    </div>
  );
}
