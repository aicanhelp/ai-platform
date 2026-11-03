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
import { Button, Popover, Space, Tabs, message } from 'antd';
import { useEffect, useRef, useState } from 'openinula';
import { DownOutlined } from '@ant-design/icons';
import PriorityDetail from '@/pages/SchedulingPolicyConfiguration/PriorityDetail';
import PriorityYaml from '@/pages/SchedulingPolicyConfiguration/PriorityYaml';
import { useHistory, useLocation } from 'inula-router';
import { getSpecifiedPriorityClasses } from '@/api/schedulingApi';
import { solveAnnotation, solveAnnotationOrLabelDiff } from '@/tools/utils';
import AnnotationModal from '@/components/AnnotationModal';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { deleteSpecifiedPriorityClasses, editAnnotationsOrLabels } from '@/api/schedulingApi';
import { ResponseCode } from '@/common/constants';
import '@/styles/priorityClassDetails.less';
import { jsonToYaml } from '@/tools/utils';
import { useIntl } from 'inula-intl';

export default function PriorityClassDetails() {
  const intl = useIntl();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [popOpen, setPopOpen] = useState(false);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const [detailsInfo, setDetailsInfo] = useState({});
  const priorityName = queryParams.get('priorityName');
  const [isEdit, setIsEdit] = useState(queryParams.get('isEdit') === 'true');
  const [isAnnotationModalOpen, setIsAnnotationModalOpen] = useState(false);
  const [isLabelModalOpen, setIsLabelModalOpen] = useState(false);
  const [oldAnnotations, setOldAnnotataions] = useState('');
  const [oldLabels, setOldLabels] = useState('');
  const buttonRef = useRef(null);
  const [priorityModalOpen, setPriorityModalOpen] = useState(false); // 删除对话框展示
  const [isPriorityDelCheck, setIsPriorityDelCheck] = useState(false); // 是否选中
  const [isReadyOnly, setIsReadyOnly] = useState(true); // 是否只读
  const [activeKey, setActiveKey] = useState('1');
  const [yamlData, setYamlData] = useState('');
  const [priorityLoading, setPriorityLoading] = useState(false);

  const handlePopOpenChange = () => {
    setPopOpen((open) => !open);
  };

  const getSpecifiedPriorityClassesInfo = async (name) => {
    setPriorityLoading(false);
    try {
      const res = await getSpecifiedPriorityClasses(name);
      setYamlData(jsonToYaml(JSON.stringify(res.data)));
      res.data.metadata.labels = solveAnnotation(res.data.metadata.labels);
      res.data.metadata.annotations = solveAnnotation(res.data.metadata.annotations);
      setDetailsInfo(res.data);
      setPriorityLoading(true);
    } catch (error) {
      message.error(error.response.data.message);
    }
  };

  // 注解成功回调
  const handleAnnotationOk = async (data) => {
    if (JSON.stringify(oldAnnotations) === JSON.stringify(data)) {
      messageApi.info(intl.formatMessage({ id: 'annotation.noChanges' }));
      setIsAnnotationModalOpen(false);
    } else {
      const keyArrPriorityClassDetailsAnnotation = [];
      data.map((item) => keyArrPriorityClassDetailsAnnotation.push(item.key));
      if (
        keyArrPriorityClassDetailsAnnotation.filter(
          (item, index) => keyArrPriorityClassDetailsAnnotation.indexOf(item) !== index,
        ).length
      ) {
        messageApi.error(intl.formatMessage({ id: 'annotation.duplicateKey' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addAnnotationList = solveAnnotationOrLabelDiff(oldAnnotations, data, 'annotation');
        try {
          const res = await editAnnotationsOrLabels(priorityName, addAnnotationList);
          if (res.status === ResponseCode.OK) {
            message.success(intl.formatMessage({ id: 'annotation.editSuccess' }));
            setTimeout(() => {
              getSpecifiedPriorityClassesInfo(priorityName);
              setIsAnnotationModalOpen(false);
            }, 1000);
          }
        } catch (error) {
          if (error.response.status === 403) {
            message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
          } else {
            message.error(`${intl.formatMessage({ id: 'annotation.editFailed' })}${error.response.data.message}`);
          }
        }
      }
    }
  };
  // 注解失败回调
  const handleAnnotationCancel = () => {
    setIsAnnotationModalOpen(false);
  };

  // 标签成功回调
  const handleLabelOk = async (data) => {
    if (JSON.stringify(oldLabels) === JSON.stringify(data)) {
      messageApi.info(intl.formatMessage({ id: 'annotation.noLabelChanges' }));
      setIsLabelModalOpen(false);
    } else {
      const keyArr = [];
      data.map((item) => keyArr.push(item.key));
      if (keyArr.filter((item, index) => keyArr.indexOf(item) !== index).length) {
        messageApi.error(intl.formatMessage({ id: 'annotation.duplicateKey' }));
      } else {
        // 请求接口添加
        // 比较前后是否一致 返回处理后的注解
        const addLabelList = solveAnnotationOrLabelDiff(oldLabels, data, 'label');
        try {
          const res = await editAnnotationsOrLabels(priorityName, addLabelList);
          if (res.status === ResponseCode.OK) {
            message.success(intl.formatMessage({ id: 'annotation.editLabelSuccess' }));
            setTimeout(() => {
              getSpecifiedPriorityClassesInfo(priorityName);
              setIsLabelModalOpen(false);
            }, 1000);
          }
        } catch (error) {
          if (error.response.status === 403) {
            message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
          } else {
            message.error(`${intl.formatMessage({ id: 'annotation.editLabelFailed' })}${error.response.data.message}`);
          }
        }
      }
    }
  };

  // 标签失败回调
  const handleLabelCancel = () => {
    setIsLabelModalOpen(false);
  };

  const handlePriorityCheckFn = (e) => {
    setIsPriorityDelCheck(e.target.checked);
  };

  const handleEdit = () => {
    setPopOpen(false);
    setActiveKey('2');
    setIsReadyOnly(false);
  };

  // 取消删除
  const handleDelCancelPriorityIndex = () => {
    setPriorityModalOpen(false);
  };

  const handleDelPriorityConfirm = async () => {
    try {
      const resPriorityClassDetails = await deleteSpecifiedPriorityClasses(priorityName);
      if (resPriorityClassDetails.status === ResponseCode.OK) {
        message.success(intl.formatMessage({ id: 'priority.deleteSuccess' }));
        setPriorityModalOpen(false);
        setIsPriorityDelCheck(false);
        setTimeout(() => {
          history.push('/SchedulingPolicyConfiguration/priorityClass');
        }, 1000);
      }
    } catch (error) {
      if (error.response.status === 403) {
        message.error(intl.formatMessage({ id: 'common.permissionDenied' }));
      } else {
        message.error(error.response?.data.message);
      }
    }
  };

  const handleReadyOnly = (bool) => {
    setIsReadyOnly(bool);
  };

  const handleRefresh = () => {
    getSpecifiedPriorityClassesInfo(priorityName);
  };

  // 删除按钮
  const handleDeletePriorityIndex = (name) => {
    // 隐藏气泡框
    setPopOpen(false);
    setPriorityModalOpen(true); // 打开弹窗
    setIsPriorityDelCheck(false);
  };

  const items = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'priority.details' }),
      children: <PriorityDetail details={detailsInfo} refreshFn={handleRefresh} />,
    },
    {
      key: '2',
      label: 'YAML',
      children: (
        <PriorityYaml
          priorityName={priorityName}
          details={yamlData}
          readOnly={isReadyOnly}
          handleEditFn={handleReadyOnly}
        />
      ),
    },
  ];

  const onChange = (e) => {
    setActiveKey(e);
  };

  useEffect(() => {
    getSpecifiedPriorityClassesInfo(priorityName);
  }, []);
  return (
    <div className="priorityClassDetails_container" id="priority">
      <div>
        <BreadCrumbCom
          items={[
            { title: intl.formatMessage({ id: 'scheduling.schedulingPolicyConfig' }), disabled: true },
            { title: 'PriorityClass', path: '/SchedulingPolicyConfiguration/PriorityClass' },
            {
              title: intl.formatMessage({ id: 'priority.details' }),
              path: '/SchedulingPolicyConfiguration/priorityClassDetails',
            },
          ]}
        />
      </div>
      <div className="priority_header">
        <div className="priority_title">{priorityName}</div>
        <div ref={buttonRef}>
          <Popover
            placement="bottom"
            getPopupContainer={() => document.getElementById('priority')}
            content={
              <Space className="column_pop" getContainer={false}>
                <Button type="link" onClick={handleEdit}>
                  {intl.formatMessage({ id: 'priority.modify' })}
                </Button>
                <Button
                  type="link"
                  onClick={() => {
                    setIsLabelModalOpen(true);
                    setPopOpen(false);
                  }}
                >
                  {intl.formatMessage({ id: 'priority.modifyLabels' })}
                </Button>
                <Button
                  type="link"
                  onClick={() => {
                    setIsAnnotationModalOpen(true);
                    setPopOpen(false);
                  }}
                >
                  {intl.formatMessage({ id: 'priority.modifyAnnotations' })}
                </Button>
                <Button type="link" onClick={() => handleDeletePriorityIndex(priorityName)}>
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </Space>
            }
            trigger="click"
            open={popOpen}
            onOpenChange={handlePopOpenChange}
          >
            <Button className="primary_btn">
              {intl.formatMessage({ id: 'common.action' })} <DownOutlined className="small_margin_adjust" />
            </Button>
          </Popover>
        </div>
      </div>
      {priorityLoading && (
        <Tabs items={items} style={{ backgroundColor: '#fff' }} activeKey={activeKey} onChange={onChange} />
      )}
      {priorityLoading && (
        <AnnotationModal
          open={isLabelModalOpen}
          type="label"
          dataList={detailsInfo.metadata?.labels}
          callbackOk={handleLabelOk}
          callbackCancel={handleLabelCancel}
        />
      )}
      {priorityLoading && (
        <AnnotationModal
          open={isAnnotationModalOpen}
          type="annotation"
          dataList={detailsInfo.metadata?.annotations}
          callbackOk={handleAnnotationOk}
          callbackCancel={handleAnnotationCancel}
        />
      )}
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'priority.deletePriorityClass' })}
        open={priorityModalOpen}
        className="PriorityClassDetails"
        cancelFn={handleDelCancelPriorityIndex}
        content={[
          intl.formatMessage({ id: 'priority.deleteWarning' }),
          intl.formatMessage({ id: 'priority.confirmDelete' }, { name: priorityName }),
        ]}
        isCheck={isPriorityDelCheck}
        showCheck={true}
        confirmFn={handleDelPriorityConfirm}
        checkFn={handlePriorityCheckFn}
      />
    </div>
  );
}
